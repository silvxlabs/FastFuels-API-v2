"""
In-place inventory modifications for standgen.

Applies the modification delta queued by the most recent
``POST .../{inventory_id}/modifications`` call to the inventory's own current
Parquet data, writing the result back under the same inventory ID.
"""

import logging

import geopandas as gpd

from standgen.modifications import (
    _has_spatial_condition,
    apply_modifications,
    resolve_spatial_conditions,
)
from standgen.storage import load_inventory_parquet, save_parquet_replace_with_summary

logger = logging.getLogger(__name__)


def apply_in_place_modifications(
    inventory: dict, domain_gdf: gpd.GeoDataFrame, progress
) -> dict:
    """Apply the inventory's pending modification delta to its own data, in place.

    The API appends new rules to the cumulative ``modifications`` ledger and
    queues only that delta in ``pending_modifications``. This loads the
    inventory's current Parquet, applies only the delta, and writes it back
    under the same ID. The ``modifications`` ledger and ``georeference`` are
    unchanged — modifications filter/transform rows but never move the grid
    footprint.

    Args:
        inventory: Full inventory document from Firestore.
        domain_gdf: Domain geometry as GeoDataFrame.
        progress: Callback for progress reporting.

    Returns:
        Dict with 'georeference', 'columns' with per-column summary statistics,
        and 'forestry_metrics' with stand-level forestry scalars or None.
    """
    inventory_id = inventory["id"]
    modifications = inventory.get("pending_modifications", [])

    # Load the inventory's own current data as a dask DataFrame.
    progress("Loading inventory...", 10)
    ddf = load_inventory_parquet(inventory_id)

    # Apply only the new delta. Resolve spatial-condition geometries once here
    # (off the per-partition path) when any are present.
    progress("Applying modifications...", 40)
    if _has_spatial_condition(modifications):
        modifications = resolve_spatial_conditions(
            modifications, inventory["domain_id"], domain_gdf.crs
        )
    ddf = ddf.map_partitions(apply_modifications, modifications)

    # Replace the inventory's Parquet in place (staging swap — see storage.py).
    progress("Writing modified inventory...", 70)
    _, stats, forestry_metrics = save_parquet_replace_with_summary(
        inventory_id, ddf, inventory["columns"], inventory["type"], domain_gdf
    )

    progress("Complete", 100)

    return {
        "georeference": inventory.get("georeference"),
        "columns": [
            {**col, "summary": stats.get(col["key"])} for col in inventory["columns"]
        ],
        "forestry_metrics": forestry_metrics,
    }
