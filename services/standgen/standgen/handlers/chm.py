"""
CHM (Canopy Height Model) extraction handler.

Generates tree inventories by applying stem isolation algorithms (LMF or VWF)
to Canopy Height Model grids.
"""

import logging

import geopandas as gpd
import pandas as pd

# --- FASTFUELS CORE IMPORTS ---
from fastfuels_core.itd.local_maxima_filter import (
    fixed_window_filter,
    variable_window_filter,
)

from lib.config import GRIDS_COLLECTION
from lib.errors import ProcessingError
from lib.firestore import DocumentNotFoundError, get_document
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

    # --- 1. DATA LOADING ---
    try:
        _, grid_snapshot = get_document(GRIDS_COLLECTION, source_chm_grid_id)
    except DocumentNotFoundError:
        raise ProcessingError(
            code="SOURCE_GRID_NOT_FOUND",
            message=f"Source CHM grid '{source_chm_grid_id}' not found.",
            suggestion="Ensure the source CHM grid exists and has been processed.",
        )

    progress("Loading CHM grid data...", 10)
    grid_ds = load_grid(source_chm_grid_id)

    if "chm" not in grid_ds.data_vars:
        raise ProcessingError(
            code="MISSING_BAND",
            message="Source grid is missing the required 'chm' band.",
        )

    chm_da = grid_ds["chm"]

    # Dynamically extract spatial resolution if not explicitly provided
    spatial_res = algorithm_config.get("spatial_resolution")
    if spatial_res is None:
        spatial_res = abs(chm_da.rio.resolution()[0])

    # --- 2. ALGORITHM ROUTING ---
    alg_name = algorithm_config.get("name", "lmf")
    progress(f"Running {alg_name.upper()} stem isolation...", 30)

    if alg_name == "lmf":
        try:
            # Safely convert API schema pixel parameters to meter parameters for the core logic
            footprint_pixels = algorithm_config.get("footprint_size", 3)
            window_meters = footprint_pixels * spatial_res

            ddf = fixed_window_filter(
                chm_da=chm_da,
                min_height=algorithm_config.get("min_height", 2.0),
                spatial_resolution=spatial_res,
                window_size_meters=window_meters,
            )
        except ValueError as e:
            raise ProcessingError(code="INVALID_ALGORITHM_PARAMS", message=str(e))

    elif alg_name == "vwf":
        try:
            ddf = variable_window_filter(
                chm_da=chm_da,
                min_height=algorithm_config.get("min_height", 2.0),
                spatial_resolution=spatial_res,
                crown_ratio=algorithm_config.get("crown_ratio", 0.10),
                crown_offset=algorithm_config.get("crown_offset", 1.0),
            )
        except ValueError as e:
            raise ProcessingError(code="INVALID_ALGORITHM_PARAMS", message=str(e))

    else:
        raise ProcessingError(
            code="UNSUPPORTED_ALGORITHM",
            message=f"Algorithm '{alg_name}' is not supported.",
        )

    # --- 3. DISTRIBUTED SPATIAL PROCESSING ---
    progress("Checking spatial reference systems...", 60)
    source_crs = chm_da.rio.crs
    target_crs = domain_gdf.crs

    # Only reproject if necessary, and do it safely across Dask workers
    if source_crs != target_crs:
        # Convert CRSs to strings so they safely serialize via pickle to worker nodes
        source_crs_str = source_crs.to_string()
        target_crs_str = str(target_crs)

        def reproject_partition(df: pd.DataFrame) -> pd.DataFrame:
            import pyproj

            # Instantiate the C-bindings inside the worker to prevent crashes
            transformer = pyproj.Transformer.from_crs(
                source_crs_str, target_crs_str, always_xy=True
            )
            df["x"], df["y"] = transformer.transform(df["x"].values, df["y"].values)
            return df

        ddf = ddf.map_partitions(
            reproject_partition, meta={"x": "f8", "y": "f8", "height": "f8"}
        )

    # --- 4. FORMATTING & STORAGE ---
    tree_count = len(ddf)
    logger.info(
        f"Extracted {tree_count} trees from CHM", extra={"inventory_id": inventory_id}
    )

    modifications = inventory.get("modifications", [])
    if modifications:
        progress("Applying modifications...", 80)
        ddf = ddf.map_partitions(apply_modifications, modifications)

    progress("Writing inventory data...", 90)
    save_parquet(inventory_id, ddf)

    progress("Computing georeference...", 95)
    bounds = domain_gdf.total_bounds
    georeference = {
        "crs": str(domain_gdf.crs),
        "bounds": [float(b) for b in bounds],
    }

    progress("Complete", 100)
    return {"georeference": georeference}
