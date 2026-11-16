"""
standgen/treatments.py

Silvicultural treatment processing.

Applies thinning treatments to tree DataFrames, mapping the v2 treatment schema
(``InventoryDiameterTreatment`` / ``InventoryBasalAreaTreatment`` /
``InventoryCrownSpacingTreatment``) onto fastfuels-core's thinning classes.
:func:`apply_treatments` takes and returns a lazy dask DataFrame and chooses, per
treatment, how much to pull into memory:

- **Diameter** treatments are row-local filters (even when spatially scoped), so
  they stay lazy and stream per-partition — no materialization.
- **Basal-area** treatments are GLOBAL stand reductions (a whole-population sort
  + cumulative basal area), so the treated population must be in memory at once.
  Only the *treated region* is materialized — the whole domain for an
  inventory-wide treatment, or just the in-polygon trees for a spatially scoped
  one (the rest stays lazy) — bounded by :data:`MAX_TREATMENT_AREA_SQ_KM`.
- **Crown-spacing** treatments compare every tree with its neighbours, so they
  are global too, under the same area limit. A spatially scoped one materializes
  the region plus the trees near enough for their crowns to touch a crown in it:
  those neighbours are never removed, but trees in the region must clear them.

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
import shapely
from fastfuels_core.canopy_fuel.crown_radius import max_crown_radius
from fastfuels_core.treatments import (
    DirectionalThinToDiameterLimit,
    DirectionalThinToStandBasalArea,
    ProportionalThinToBasalArea,
    ThinningDirection,
    ThinToCrownSpacing,
)

from lib.config import SUPPORT_EMAIL
from lib.errors import ProcessingError
from lib.inventory_io import CROWN_RADIUS_EQUATION_COLUMNS, treatment_required_columns
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
    "crown_spacing": "m",
}

# Metrics that thin the whole treated population at once, in memory.
IN_MEMORY_METRICS = {"basal_area", "crown_spacing"}

# Temporary column holding each tree's crown radius during a crown-spacing thin.
_RADIUS_COLUMN = "_crown_spacing_radius"

_METHOD_TO_DIRECTION = {
    "from_below": ThinningDirection.BELOW,
    "from_above": ThinningDirection.ABOVE,
}

# Basal-area and crown-spacing treatments hold their entire treated population in
# memory at once (a global diameter sort, or a neighbour search), so the treated
# area is capped to keep that materialization within the worker's memory. This is a MEMORY
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

    For diameter and crown spacing, the value is converted here. For basal area,
    the caller passes the already-computed total-m² target (it depends on the
    treated region area).
    """
    metric = treatment["metric"]

    if metric == "diameter":
        limit_cm = convert_treatment_value(
            "diameter", treatment["value"], treatment.get("unit")
        )
        return DirectionalThinToDiameterLimit(limit_cm, _direction(treatment["method"]))

    if metric == "basal_area":
        method = treatment["method"]
        if method == "proportional":
            return ProportionalThinToBasalArea(target_basal_area_m2)
        return DirectionalThinToStandBasalArea(target_basal_area_m2, _direction(method))

    if metric == "crown_spacing":
        return ThinToCrownSpacing(
            crown_spacing_m(treatment), rank_by=treatment["rank_by"]
        )

    raise ProcessingError(
        code="UNSUPPORTED_TREATMENT_METRIC",
        message=f"Treatment metric '{metric}' is not supported.",
        suggestion="Supported metrics: diameter, basal_area, crown_spacing.",
    )


def crown_spacing_m(treatment: dict) -> float:
    """A crown-spacing treatment's minimum gap, in meters."""
    return convert_treatment_value(
        "crown_spacing", treatment["value"], treatment.get("unit")
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
    dividing by 10,000 yields hectares.
    """
    return compute_region(domain_gdf, conditions).area / 10_000.0


def compute_region(
    domain_gdf: gpd.GeoDataFrame, conditions: list[dict]
) -> shapely.Geometry:
    """The domain clipped to a treatment's spatial conditions, in the domain CRS.

    Each condition's geometry was pre-resolved into the domain CRS by
    ``resolve_spatial_conditions``. Conditions are ANDed: ``within``/``intersects``
    clip the region to the geometry, ``outside`` subtracts it. With no conditions
    the region is the whole domain.
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

    return region


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


def _enforce_area_limit(region_area_ha: float, scoped: bool, metric: str) -> None:
    """Reject a basal-area or crown-spacing treatment whose treated region is too
    large to hold in memory. ``scoped`` selects the wording (spatial region vs
    whole domain)."""
    area_sq_km = region_area_ha / 100.0
    if area_sq_km <= MAX_TREATMENT_AREA_SQ_KM:
        return
    where = "The treatment's spatial region" if scoped else "The inventory's domain"
    kind = metric.replace("_", "-")
    raise ProcessingError(
        code="TREATMENT_AREA_TOO_LARGE",
        message=(
            f"A {kind} treatment thins the whole stand at once, so its entire "
            f"treated population is held in memory. {where} covers "
            f"{area_sq_km:.1f} km², above the "
            f"{MAX_TREATMENT_AREA_SQ_KM:.0f} km² limit for a single "
            f"{kind} treatment."
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
    stay lazy (row-local, streamed per-partition); basal-area and crown-spacing
    treatments materialize only their treated region (see the module docstring).
    Every column a treatment reads is checked before any data is computed. The
    global
    numpy RNG is seeded once so ``proportional`` thinning (which uses
    ``np.random`` during the eager basal-area step) is reproducible from the
    inventory's stored seed; diameter thinning consumes no RNG.
    """
    if not treatments:
        return ddf
    validate_treatment_columns(treatments, set(ddf.columns))
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

    # Basal area and crown spacing are global. Size the treated region (pure
    # geometry) and enforce the in-memory limit before pulling any trees into RAM.
    conditions = treatment.get("conditions", [])
    region = compute_region(domain_gdf, conditions)
    _enforce_area_limit(
        region.area / 10_000.0, scoped=bool(conditions), metric=treatment["metric"]
    )

    if treatment["metric"] == "crown_spacing":
        return _apply_crown_spacing_to_ddf(ddf, treatment, region)

    # Inventory-wide: the population is the whole domain (already <= the limit).
    if not conditions:
        df = ddf.compute().reset_index(drop=True)
        thinned = apply_single_treatment(df, treatment, domain_gdf)
        return _from_pandas(thinned)

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
    return dd.concat([_from_pandas(thinned).clear_divisions(), outside], axis=0)


def _apply_crown_spacing_to_ddf(
    ddf: dd.DataFrame, treatment: dict, region: shapely.Geometry
) -> dd.DataFrame:
    """Apply a crown-spacing treatment, materializing only the trees it can touch.

    Inventory-wide, that is every tree. With conditions, only trees in the region
    can be removed, but they must clear every crown near enough to touch theirs:
    any stem within ``spacing + 2 × max radius`` of the region. That
    neighbourhood is materialized and thinned; the rest stays lazy. Sizing it
    takes one extra pass over the data, for the largest crown radius.
    """
    conditions = treatment.get("conditions", [])
    if not conditions:
        df = ddf.compute().reset_index(drop=True)
        return _from_pandas(apply_crown_spacing(df, treatment))

    radius_source = treatment["max_crown_radius_source"]
    max_radius = (
        ddf.map_partitions(_crown_radius_series, radius_source, meta=(None, "f8"))
        .max()
        .compute()
    )
    if not np.isfinite(max_radius):
        return ddf  # no trees

    reach = crown_spacing_m(treatment) + 2 * max_radius
    near = ddf.map_partitions(
        _neighbourhood_mask,
        conditions,
        region.buffer(reach),
        meta=pd.Series(dtype=bool),
    )
    neighbourhood = ddf[near].compute().reset_index(drop=True)
    if neighbourhood.empty:
        return ddf
    thinned = apply_crown_spacing(neighbourhood, treatment)
    return dd.concat([_from_pandas(thinned).clear_divisions(), ddf[~near]], axis=0)


def _neighbourhood_mask(
    df: pd.DataFrame, conditions: list[dict], near: shapely.Geometry
) -> pd.Series:
    """Trees a scoped crown-spacing thin must see: every tree it may remove,
    plus every stem within reach of its region."""
    points = gpd.GeoSeries(gpd.points_from_xy(df["x"], df["y"]), index=df.index)
    return build_condition_mask(df, conditions) | points.within(near)


def apply_crown_spacing(df: pd.DataFrame, treatment: dict) -> pd.DataFrame:
    """Thin a materialized set of trees to the treatment's crown spacing.

    Trees whose stems meet the treatment's conditions may be removed; the others
    are fixed neighbours that the removable trees must clear.
    """
    if df.empty:
        return df
    eligible = build_condition_mask(df, treatment.get("conditions", []))
    radius = crown_radii(df, treatment["max_crown_radius_source"])
    thinner = build_thinner(treatment, None)
    try:
        kept = thinner.apply(
            df.assign(**{_RADIUS_COLUMN: radius}),
            eligible=eligible.to_numpy(dtype=bool),
            radius_column=_RADIUS_COLUMN,
        )
    except ValueError as e:
        raise ProcessingError(
            code="INVALID_TREATMENT_INPUT",
            message=f"Crown-spacing treatment could not be applied: {e}",
            suggestion=(
                "Fill or correct the column named above, or remove those trees "
                "with a modification before the treatment."
            ),
        ) from e
    return kept.drop(columns=_RADIUS_COLUMN).reset_index(drop=True)


def crown_radii(df: pd.DataFrame, radius_source: dict) -> np.ndarray:
    """Per-tree crown radius (m) from a treatment's ``max_crown_radius_source``.

    A radius column's null values fall back to the Purves radius, as tree
    voxelization and inventory canopy grids do.
    """
    if radius_source["type"] == "allometry":
        return _allometric_radii(df, radius_source["equations"])

    column = radius_source["column"]
    radius = df[column].to_numpy(dtype=np.float64, na_value=np.nan)
    missing = np.isnan(radius)
    if missing.any():
        radius[missing] = _allometric_radii(df[missing], "purves", fallback_for=column)
        logger.info(
            f"{int(missing.sum())} trees have no {column}; using their Purves "
            "crown radius"
        )
    return radius


def _crown_radius_series(df: pd.DataFrame, radius_source: dict) -> pd.Series:
    """:func:`crown_radii` as a Series, for ``map_partitions``."""
    return pd.Series(crown_radii(df, radius_source), index=df.index, dtype="f8")


def _allometric_radii(
    df: pd.DataFrame, equations: str, fallback_for: str | None = None
) -> np.ndarray:
    """Crown radii (m) from allometry, failing on trees it cannot size.

    The allometry needs every one of its input columns on every tree; core would
    otherwise give a tree with no ``dbh`` a radius of 0. ``fallback_for`` names
    the radius column whose null values are being filled, for the error.
    """
    inputs = sorted(CROWN_RADIUS_EQUATION_COLUMNS[equations])
    which = (
        f"{len(df)} trees have no {fallback_for!r} value, so they need a "
        f"Purves crown radius, which"
        if fallback_for
        else f"The {equations!r} crown radius"
    )
    suggestion = (
        f"Fill {fallback_for!r} for every tree, or remove the trees without one "
        f"with a modification."
        if fallback_for
        else "Remove those trees with a modification, or read crown radius from "
        "an inventory column."
    )

    absent = [c for c in inputs if c not in df.columns]
    if absent:
        raise ProcessingError(
            code="CROWN_RADIUS_UNAVAILABLE",
            message=f"{which} reads {absent}, which this inventory does not have.",
            suggestion=suggestion,
        )
    incomplete = [c for c in inputs if df[c].isna().any()]
    if incomplete:
        raise ProcessingError(
            code="CROWN_RADIUS_UNAVAILABLE",
            message=f"{which} reads {incomplete}, which some trees lack.",
            suggestion=suggestion,
        )
    try:
        return max_crown_radius(df, equations=equations)
    except KeyError as e:
        raise ProcessingError(
            code="CROWN_RADIUS_UNAVAILABLE",
            message=f"{which} has no equation for species code {e}.",
            suggestion=suggestion,
        ) from e
    except ValueError as e:
        raise ProcessingError(
            code="CROWN_RADIUS_UNAVAILABLE",
            message=f"{which} could not be computed: {e}",
            suggestion=suggestion,
        ) from e


def validate_treatment_columns(treatments: list[dict], columns: set[str]) -> None:
    """Reject treatments that read a column the data does not have.

    The API checks this at request time from the inventory document; this is the
    backstop against the actual data, before anything is computed.
    """
    for index, treatment in enumerate(treatments):
        missing = sorted(treatment_required_columns(treatment) - columns)
        if missing:
            raise ProcessingError(
                code="TREATMENT_COLUMNS_MISSING",
                message=(
                    f"Treatment {index} ({treatment['metric']}) reads {missing}, "
                    f"which this inventory's data does not have."
                ),
                suggestion=(
                    "Diameter and basal-area treatments need dbh. For an "
                    "inventory without it, use a crown_spacing treatment ranked "
                    "by height that reads a crown radius column."
                ),
            )


def _from_pandas(df: pd.DataFrame) -> dd.DataFrame:
    """Wrap a materialized result as a dask DataFrame for the partitioned write."""
    return dd.from_pandas(df, npartitions=max(1, len(df) // 100_000))
