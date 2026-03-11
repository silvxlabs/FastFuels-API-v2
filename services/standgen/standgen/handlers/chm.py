"""
CHM (Canopy Height Model) extraction handler.

Generates tree inventories by applying stem isolation algorithms (like LMF)
to Canopy Height Model grids.
"""

import logging

import dask.dataframe as dd
import geopandas as gpd
import pandas as pd

from lib.config import GRIDS_COLLECTION
from lib.firestore import DocumentNotFoundError, get_document
from standgen.columns import BASE_COLUMNS
from standgen.errors import ProcessingError
from standgen.handlers.utils import find_treetops_lmf
from standgen.modifications import apply_modifications
from standgen.storage import load_grid, save_parquet

logger = logging.getLogger(__name__)


def handle_chm(
    inventory: dict, source: dict, domain_gdf: gpd.GeoDataFrame, progress
) -> dict:
    """Process a CHM extraction inventory request.

    Args:
        inventory: Full inventory document from Firestore
        source: Source dict with chm-specific fields
        domain_gdf: Domain geometry as GeoDataFrame
        progress: Callback for progress reporting

    Returns:
        Dict with 'georeference' key
    """
    inventory_id = inventory["id"]
    source_chm_grid_id = source["source_chm_grid_id"]
    algorithm_config = source.get("algorithm", {})

    # Load source CHM grid document from Firestore
    try:
        _, grid_snapshot = get_document(GRIDS_COLLECTION, source_chm_grid_id)
    except DocumentNotFoundError:
        raise ProcessingError(
            code="SOURCE_GRID_NOT_FOUND",
            message=f"Source CHM grid '{source_chm_grid_id}' not found.",
            suggestion="Ensure the source CHM grid exists and has been processed.",
        )

    # Load CHM grid data
    progress("Loading CHM grid data...", 10)
    grid_ds = load_grid(source_chm_grid_id)

    if "chm" not in grid_ds.data_vars:
        raise ProcessingError(
            code="MISSING_BAND",
            message="Source grid is missing the required 'chm' band.",
            suggestion="Ensure the grid is a valid Canopy Height Model.",
        )

    chm_da = grid_ds["chm"]

    # Route to the appropriate stem isolation algorithm
    progress(
        f"Running {algorithm_config.get('name', 'lmf').upper()} stem isolation...", 30
    )

    if algorithm_config.get("name") == "lmf":
        try:
            treetops_gdf = find_treetops_lmf(
                chm_da=chm_da,
                min_height=algorithm_config.get("min_height", 2.0),
                footprint_size=algorithm_config.get("footprint_size", 3),
            )
        except ValueError as e:
            raise ProcessingError(code="INVALID_ALGORITHM_PARAMS", message=str(e))
    else:
        raise ProcessingError(
            code="UNSUPPORTED_ALGORITHM",
            message=f"Algorithm '{algorithm_config.get('name')}' is not supported.",
        )

    progress("Clipping trees to domain boundary...", 60)

    # Ensure CRSs match before clipping (they should, but it's safe practice)
    if treetops_gdf.crs != domain_gdf.crs:
        treetops_gdf = treetops_gdf.to_crs(domain_gdf.crs)


    logger.info(
        f"Extracted {len(treetops_gdf)} trees from CHM",
        extra={"inventory_id": inventory_id},
    )

    # Convert to standard inventory DataFrame
    progress("Formatting inventory attributes...", 70)

    # Extract coordinates
    df = pd.DataFrame(
        {
            "x": treetops_gdf.geometry.x,
            "y": treetops_gdf.geometry.y,
            "height": treetops_gdf["height"],
            "dbh": None,
            "fia_species_code": None,
            "fia_status_code": None,
            "crown_ratio": None,
        }
    )

    # Convert to Dask DataFrame (partitioned arbitrarily, e.g., by chunks of 100k)
    # This matches the signature required by your save_parquet function
    chunk_size = 100_000
    npartitions = max(1, len(df) // chunk_size)
    ddf = dd.from_pandas(df, npartitions=npartitions)

    # Ensure column order matches standard
    ddf = ddf[[col for col in BASE_COLUMNS if col in ddf.columns]]

    # Apply modifications if present
    modifications = inventory.get("modifications", [])
    if modifications:
        progress("Applying modifications...", 80)
        ddf = ddf.map_partitions(apply_modifications, modifications)

    # Write Parquet to GCS
    progress("Writing inventory data...", 90)
    save_parquet(inventory_id, ddf)

    # Compute georeference
    progress("Computing georeference...", 95)
    bounds = domain_gdf.total_bounds
    georeference = {
        "crs": str(domain_gdf.crs),
        "bounds": [float(b) for b in bounds],
    }

    progress("Complete", 100)
    return {"georeference": georeference}
