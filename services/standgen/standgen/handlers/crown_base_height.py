"""
Crown base height allometry handler.

Computes each tree's crown base height from the species, crown ratio, dbh and
height already on a tree inventory with the core crown base height model the
source names. No GDAM call is made.

Without ``overwrite`` existing crown base heights are kept and only empty cells
are filled; with it every tree is recomputed. Species the model does not cover
get NaN.
"""

import logging

import pandas as pd
from fastfuels_core.allometry.crown_ratio import FIA_NATIONAL_LOGIT_2026

from lib.errors import ProcessingError
from standgen.crown_base import CROWN_BASE_INPUTS, fill_crown_base_height
from standgen.region import in_pacific_northwest
from standgen.storage import load_inventory_parquet, save_parquet_with_summary

logger = logging.getLogger(__name__)


def _result_meta(ddf) -> pd.DataFrame:
    """Empty DataFrame describing ``fill_crown_base_height``'s output for dask:
    the source columns with ``crown_base_height`` as float64, appended when the
    source lacks it."""
    meta = ddf._meta.copy()
    meta["crown_base_height"] = pd.Series([], dtype="float64", index=meta.index)
    return meta


def handle_crown_base_height(
    inventory: dict, source: dict, domain_gdf, progress
) -> dict:
    """Fill (or, with ``overwrite``, recompute) a tree inventory's crown base
    heights from the crown base height model.

    Dict with 'georeference', 'columns' with per-column summary statistics,
    and 'forestry_metrics' with stand-level forestry scalars or None.
    """
    inventory_id = inventory["id"]
    source_id = source["source_tree_inventory_id"]
    overwrite = bool(source.get("overwrite", False))
    model = (source.get("model") or {}).get("name", FIA_NATIONAL_LOGIT_2026)

    progress("Loading source inventory...", 10)
    try:
        ddf = load_inventory_parquet(source_id)
    except FileNotFoundError as e:
        raise ProcessingError(
            code="SOURCE_INVENTORY_NOT_FOUND",
            message=f"Source tree inventory '{source_id}' has no data.",
            suggestion=(
                "Ensure the source inventory completed before computing crown "
                "base heights."
            ),
        ) from e

    missing_inputs = [c for c in CROWN_BASE_INPUTS if c not in ddf.columns]
    if missing_inputs:
        raise ProcessingError(
            code="MISSING_CROWN_BASE_INPUTS",
            message=(
                f"Crown base height needs column(s) {missing_inputs}, which the "
                f"source inventory lacks."
            ),
            suggestion="Fill the missing column(s) with the GDAM allometry route.",
        )

    if not overwrite and "crown_base_height" in ddf.columns:
        null_count = int(ddf["crown_base_height"].isna().sum().compute())
        if null_count == 0:
            raise ProcessingError(
                code="NOTHING_TO_FILL",
                message="Every tree already has a crown base height.",
                suggestion=(
                    "Send overwrite: true to recompute every tree's crown base height."
                ),
            )

    progress("Computing crown base heights...", 40)
    pacific_northwest = in_pacific_northwest(domain_gdf)
    result_ddf = ddf.map_partitions(
        fill_crown_base_height,
        pacific_northwest,
        model,
        overwrite,
        meta=_result_meta(ddf),
    )

    progress("Writing inventory...", 90)
    _, stats, forestry_metrics = save_parquet_with_summary(
        inventory_id, result_ddf, inventory["columns"], inventory["type"], domain_gdf
    )
    logger.info(
        f"Computed crown base heights (model {model}, overwrite={overwrite})",
        extra={"inventory_id": inventory_id},
    )

    georeference = {
        "crs": str(domain_gdf.crs),
        "bounds": [float(b) for b in domain_gdf.total_bounds],
    }
    progress("Complete", 100)
    return {
        "georeference": georeference,
        "columns": [
            {**col, "summary": stats.get(col["key"])} for col in inventory["columns"]
        ],
        "forestry_metrics": forestry_metrics,
    }
