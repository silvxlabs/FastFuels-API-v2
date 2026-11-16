"""
In-place inventory treatments for standgen.

Applies the treatment delta queued by the most recent
``POST .../{inventory_id}/treatments`` call to the inventory's own current
Parquet data, writing the result back under the same inventory ID.
"""

import logging

import geopandas as gpd

from standgen.modifications import (
    _has_spatial_condition,
    resolve_spatial_conditions,
)
from standgen.storage import load_inventory_parquet, save_parquet_replace_with_summary
from standgen.treatments import apply_treatments

logger = logging.getLogger(__name__)


def apply_in_place_treatments(
    inventory: dict, domain_gdf: gpd.GeoDataFrame, progress
) -> dict:
    """Apply the inventory's pending treatment delta to its own data, in place.

    The API appends new treatments to the cumulative ``treatments`` ledger and
    queues only that delta in ``pending_treatments``. This loads the inventory's
    current Parquet, applies only the delta, and writes it back under the same
    ID. The ``treatments`` ledger and ``georeference`` are unchanged — thinning
    removes trees but never moves the grid footprint.

    Args:
        inventory: Full inventory document from Firestore.
        domain_gdf: Domain geometry as GeoDataFrame.
        progress: Callback for progress reporting.

    Returns:
        Dict with 'georeference', 'columns' with per-column summary statistics,
        and 'forestry_metrics' with stand-level forestry scalars or None.
    """
    inventory_id = inventory["id"]
    treatments = inventory.get("pending_treatments", [])
    # Reuse the inventory's stored expansion seed so ``proportional`` thinning is
    # reproducible (diameter/directional thinning consumes no RNG).
    seed = inventory.get("source", {}).get("seed")

    # Load the inventory's own current data as a dask DataFrame.
    progress("Loading inventory...", 10)
    ddf = load_inventory_parquet(inventory_id)

    # Apply only the new delta. Resolve spatial-condition geometries once here
    # (off the per-partition path) when any are present. apply_treatments checks
    # the columns each treatment reads against the actual Parquet schema before
    # any compute, so a stale document fails with an actionable error.
    progress("Applying treatments...", 40)
    if _has_spatial_condition(treatments):
        treatments = resolve_spatial_conditions(
            treatments, inventory["domain_id"], domain_gdf.crs
        )
    ddf = apply_treatments(ddf, treatments, domain_gdf, seed=seed)

    # Replace the inventory's Parquet in place (staging swap — see storage.py).
    progress("Writing treated inventory...", 70)
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
