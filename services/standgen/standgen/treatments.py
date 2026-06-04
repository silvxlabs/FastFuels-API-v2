"""
standgen/treatments.py

Silvicultural treatment processing.

Applies thinning treatments to tree DataFrames, mapping the v2 treatment schema
(``InventoryDiameterTreatment`` / ``InventoryBasalAreaTreatment``) onto
fastfuels-core's thinning classes. :func:`apply_treatments` takes and returns a
lazy dask DataFrame and chooses, per treatment, how much to pull into memory:

- **Diameter** treatments are row-local filters (even when spatially scoped), so
  they stay lazy and stream per-partition — no materialization.
- **Basal-area** treatments are GLOBAL stand reductions (a whole-population sort
  + cumulative basal area), so the treated population must be in memory at once.
  Only the *treated region* is materialized — the whole domain for an
  inventory-wide treatment, or just the in-polygon trees for a spatially scoped
  one (the rest stays lazy) — bounded by :data:`MAX_TREATMENT_AREA_SQ_KM`.

Spatial scoping reuses ``modifications.py``: ``resolve_spatial_conditions`` /
``_has_spatial_condition`` run in the handler (network I/O off the per-row path)
and ``build_condition_mask`` builds the in-region tree mask here.
"""

import logging

import dask.dataframe as dd
import geopandas as gpd
import numpy as np
import pandas as pd
import pint
from fastfuels_core.treatments import (
    DirectionalThinToDiameterLimit,
    DirectionalThinToStandBasalArea,
    ProportionalThinToBasalArea,
    ThinningDirection,
)

from lib.config import SUPPORT_EMAIL
from lib.errors import ProcessingError
from standgen.modifications import build_condition_mask

logger = logging.getLogger(__name__)

ureg = pint.UnitRegistry()
Q_ = ureg.Quantity

# v2 schema diameter column (fastfuels-core defaults to "DIA").
DIA_COLUMN = "dbh"

# Native unit per metric — the unit `value` is stored/validated in at create time.
NATIVE_UNITS = {
    "diameter": "cm",
    "basal_area": "m**2/ha",
}

_METHOD_TO_DIRECTION = {
    "from_below": ThinningDirection.BELOW,
    "from_above": ThinningDirection.ABOVE,
}

# A basal-area treatment holds its entire treated population in memory at once
# (global diameter sort + cumulative basal area), so the treated area is capped
# to keep that materialization within the worker's memory. This is a MEMORY
# bound, independent of the domain area limit. Diameter treatments (lazy) and the
# in-polygon path below are unaffected. The API enforces the same limit eagerly
# for inventory-wide treatments; standgen is the backstop for every treatment.
MAX_TREATMENT_AREA_SQ_KM = 16.0


def convert_treatment_value(metric: str, value: float, unit: str | None) -> float:
    """Convert ``value`` to the metric's native unit when a unit is given.

    The schema already guaranteed the unit is canonical and dimensionally
    compatible; this only performs the magnitude conversion.
    """
    if unit is None:
        return value
    native = NATIVE_UNITS[metric]
    return Q_(value, unit).to(native).magnitude


def build_thinner(treatment: dict, target_basal_area_m2: float | None):
    """Map a treatment dict to a fastfuels-core thinning object.

    For diameter, the limit is converted here. For basal area, the caller passes
    the already-computed total-m² target (it depends on the treated region area).
    """
    metric = treatment["metric"]
    method = treatment["method"]

    if metric == "diameter":
        limit_cm = convert_treatment_value(
            "diameter", treatment["value"], treatment.get("unit")
        )
        return DirectionalThinToDiameterLimit(limit_cm, _direction(method))

    if metric == "basal_area":
        if method == "proportional":
            return ProportionalThinToBasalArea(target_basal_area_m2)
        return DirectionalThinToStandBasalArea(target_basal_area_m2, _direction(method))

    raise ProcessingError(
        code="UNSUPPORTED_TREATMENT_METRIC",
        message=f"Treatment metric '{metric}' is not supported.",
        suggestion="Supported metrics: diameter, basal_area.",
    )


def _direction(method: str) -> ThinningDirection:
    """Map a directional method to a fastfuels-core ThinningDirection."""
    try:
        return _METHOD_TO_DIRECTION[method]
    except KeyError:
        raise ProcessingError(
            code="INVALID_TREATMENT_METHOD",
            message=f"Treatment method '{method}' is not a directional method.",
            suggestion="Use 'from_below' or 'from_above'.",
        )


def compute_region_area_ha(
    domain_gdf: gpd.GeoDataFrame, conditions: list[dict]
) -> float:
    """Area (hectares) of the domain clipped to a treatment's spatial conditions.

    The domain is in a projected UTM CRS (meters), so polygon ``.area`` is m² and
    dividing by 10,000 yields hectares. Each condition's geometry was pre-resolved
    into the domain CRS by ``resolve_spatial_conditions``. Conditions are ANDed:
    ``within``/``intersects`` clip the region to the geometry, ``outside``
    subtracts it. With no conditions the region is the whole domain.
    """
    region = domain_gdf.geometry.union_all()

    for cond in conditions:
        geom = cond.get("_resolved_geometry")
        if geom is None:
            raise ProcessingError(
                code="SPATIAL_CONDITION_UNRESOLVED",
                message=(
                    "Treatment spatial condition was not resolved before area "
                    "computation (missing '_resolved_geometry')."
                ),
                suggestion="Call resolve_spatial_conditions() in the handler first.",
            )
        op = cond["operator"]
        if op in ("within", "intersects"):
            region = region.intersection(geom)
        elif op == "outside":
            region = region.difference(geom)
        else:
            raise ProcessingError(
                code="INVALID_SPATIAL_OPERATOR",
                message=f"Unknown spatial operator '{op}'.",
                suggestion="Use 'within', 'outside', or 'intersects'.",
            )

    return region.area / 10_000.0


def apply_single_treatment(
    df: pd.DataFrame, treatment: dict, domain_gdf: gpd.GeoDataFrame
) -> pd.DataFrame:
    """Apply one treatment, scoped to its spatial conditions, to the DataFrame."""
    if df.empty:
        return df

    conditions = treatment.get("conditions", [])

    # Basal-area targets are per-hectare → total m² over the treated region.
    target_m2 = None
    if treatment["metric"] == "basal_area":
        area_ha = compute_region_area_ha(domain_gdf, conditions)
        value_per_ha = convert_treatment_value(
            "basal_area", treatment["value"], treatment.get("unit")
        )
        target_m2 = value_per_ha * area_ha

    thinner = build_thinner(treatment, target_m2)

    # Inventory-wide: thin the whole population.
    if not conditions:
        return thinner.apply(df, dia_column_name=DIA_COLUMN).reset_index(drop=True)

    # Spatially scoped: thin only the in-region subset, keep the rest untouched.
    mask = build_condition_mask(df, conditions)
    subset = df[mask]
    rest = df[~mask]

    if subset.empty:
        return df.reset_index(drop=True)

    thinned = thinner.apply(subset, dia_column_name=DIA_COLUMN)
    return pd.concat([thinned, rest], ignore_index=True)


def _enforce_area_limit(region_area_ha: float, scoped: bool) -> None:
    """Reject a basal-area treatment whose treated region is too large to hold
    in memory. ``scoped`` selects the wording (spatial region vs whole domain)."""
    area_sq_km = region_area_ha / 100.0
    if area_sq_km <= MAX_TREATMENT_AREA_SQ_KM:
        return
    where = "The treatment's spatial region" if scoped else "The inventory's domain"
    raise ProcessingError(
        code="TREATMENT_AREA_TOO_LARGE",
        message=(
            f"A basal-area treatment thins the whole stand at once, so its entire "
            f"treated population is held in memory. {where} covers "
            f"{area_sq_km:.1f} km², above the "
            f"{MAX_TREATMENT_AREA_SQ_KM:.0f} km² limit for a single "
            f"basal-area treatment."
        ),
        suggestion=(
            f"Scope the treatment to a smaller area with a spatial condition, or "
            f"contact {SUPPORT_EMAIL} for help processing a larger area."
        ),
    )


def apply_treatments(
    ddf: dd.DataFrame,
    treatments: list[dict],
    domain_gdf: gpd.GeoDataFrame,
    seed: int | None = None,
) -> dd.DataFrame:
    """Apply treatments to a lazy dask DataFrame, returning a lazy dask DataFrame.

    Treatments compose: each sees the output of the previous. Diameter treatments
    stay lazy (row-local, streamed per-partition); basal-area treatments
    materialize only their treated region (see the module docstring). The global
    numpy RNG is seeded once so ``proportional`` thinning (which uses
    ``np.random`` during the eager basal-area step) is reproducible from the
    inventory's stored seed; diameter thinning consumes no RNG.
    """
    if not treatments:
        return ddf
    if seed is not None:
        np.random.seed(seed)
    for treatment in treatments:
        ddf = _apply_treatment_to_ddf(ddf, treatment, domain_gdf)
    return ddf


def _apply_treatment_to_ddf(
    ddf: dd.DataFrame, treatment: dict, domain_gdf: gpd.GeoDataFrame
) -> dd.DataFrame:
    """Apply one treatment, choosing the lazy or materialized path by metric."""
    # Diameter is a row-local filter (even when spatially scoped): apply it
    # per-partition so the streaming write is preserved. domain_gdf is unused for
    # diameter, so pass None to keep it off the serialized worker payload.
    if treatment["metric"] == "diameter":
        return ddf.map_partitions(
            apply_single_treatment, treatment, None, meta=ddf._meta
        )

    # Basal area is a global reduction. Size the treated region (pure geometry)
    # and enforce the in-memory limit before pulling any trees into RAM.
    conditions = treatment.get("conditions", [])
    region_area_ha = compute_region_area_ha(domain_gdf, conditions)
    _enforce_area_limit(region_area_ha, scoped=bool(conditions))

    # Inventory-wide: the population is the whole domain (already <= the limit).
    if not conditions:
        df = ddf.compute().reset_index(drop=True)
        thinned = apply_single_treatment(df, treatment, domain_gdf)
        return dd.from_pandas(thinned, npartitions=max(1, len(thinned) // 100_000))

    # Spatially scoped: materialize only the in-region trees; thin them; concat
    # with the untouched out-of-region trees, kept lazy/streaming.
    mask = ddf.map_partitions(
        build_condition_mask, conditions, meta=pd.Series(dtype=bool)
    )
    inside = ddf[mask].compute().reset_index(drop=True)
    if inside.empty:
        return ddf
    outside = ddf[~mask]
    thinned = apply_single_treatment(inside, treatment, domain_gdf)
    inside_ddf = dd.from_pandas(
        thinned, npartitions=max(1, len(thinned) // 100_000)
    ).clear_divisions()
    return dd.concat([inside_ddf, outside], axis=0)
