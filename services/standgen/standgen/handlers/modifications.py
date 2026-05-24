"""
Handler for inventory modifications source type.

Loads an existing inventory's parquet data, applies modifications,
and saves as a new inventory.
"""

import logging

import geopandas as gpd

from standgen.modifications import (
    _has_spatial_condition,
    apply_modifications,
    resolve_spatial_conditions,
)
from standgen.storage import load_inventory_parquet, save_parquet

logger = logging.getLogger(__name__)


def handle_modifications(
    inventory: dict, source: dict, domain_gdf: gpd.GeoDataFrame, progress
) -> dict:
    """Process a modifications inventory request.

    Args:
        inventory: Full inventory document from Firestore
        source: Source dict with modifications-specific fields
        domain_gdf: Domain geometry as GeoDataFrame
        progress: Callback for progress reporting

    Returns:
        Dict with 'georeference' key
    """
    inventory_id = inventory["id"]
    source_inventory_id = source["source_inventory_id"]
    modifications = source["modifications"]

    # Load source inventory parquet as dask DataFrame
    progress("Loading source inventory...", 10)
    ddf = load_inventory_parquet(source_inventory_id)

    # Apply modifications. Resolve spatial-condition geometries once here (off
    # the per-partition path) when any are present.
    progress("Applying modifications...", 30)
    if _has_spatial_condition(modifications):
        modifications = resolve_spatial_conditions(
            modifications, inventory["domain_id"], domain_gdf.crs
        )
    ddf = ddf.map_partitions(apply_modifications, modifications)

    # Save to new inventory path
    progress("Writing modified inventory...", 70)
    save_parquet(inventory_id, ddf)

    # Copy georeference from source inventory
    progress("Computing georeference...", 95)
    from standgen.handlers.pim import compute_georeference

    georeference = compute_georeference(domain_gdf)

    progress("Complete", 100)

    return {"georeference": georeference}
