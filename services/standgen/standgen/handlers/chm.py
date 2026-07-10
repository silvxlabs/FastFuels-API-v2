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
from standgen.modifications import (
    _has_spatial_condition,
    apply_modifications,
    resolve_spatial_conditions,
)
from standgen.storage import count_inventory_rows, load_grid, save_parquet_with_summary

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
        Dict with 'georeference', 'columns' with per-column summary statistics,
        and 'forestry_metrics' with stand-level forestry scalars or None.
    """
    inventory_id = inventory["id"]

    # Treatments thin against tree diameter; CHM stem isolation produces only
    # height and position (x, y, height). The API rejects this at create time —
    # this is a defensive guard in case a treatment-bearing CHM document reaches
    # standgen.
    if inventory.get("treatments"):
        raise ProcessingError(
            code="TREATMENTS_NOT_SUPPORTED_FOR_CHM",
            message=(
                "Silvicultural treatments require a tree diameter (dbh) to thin "
                "against, which CHM extraction does not produce (only height and "
                "position)."
            ),
            suggestion="Remove 'treatments' from this CHM inventory.",
        )

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

    # Drop implausibly tall CHM returns (e.g. LiDAR noise spikes) before detection
    # so they can't seed a treetop or, via VWF's height-scaled search window,
    # suppress real nearby maxima. Only over-max pixels are zeroed — below
    # min_height, so never a treetop — leaving every valid pixel and existing
    # nodata untouched; 0 (not NaN) keeps the window-size math finite.
    max_height = algorithm_config.get("max_height")
    if max_height is not None:
        chm_da = chm_da.where(~(chm_da > max_height), 0.0)

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
            # VWF runs one bounded `da.unique(...).compute()` internally to discover
            # the window sizes present in the CHM. That is an intentional one-time
            # scan of a small derived array during graph construction — it is not the
            # treetop graph, so it doesn't break the single-compute guarantee below.
            # Don't precompute `unique_windows` to avoid it: that just moves the scan.
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
    # Apply modifications (lazily) before the single write. Resolve spatial-condition
    # geometries once here, off the per-partition path, when any are present.
    modifications = inventory.get("modifications", [])
    if modifications:
        progress("Applying modifications...", 80)
        if _has_spatial_condition(modifications):
            modifications = resolve_spatial_conditions(
                modifications, inventory["domain_id"], domain_gdf.crs
            )
        ddf = ddf.map_partitions(apply_modifications, modifications)

    # save_parquet_with_summary is the single execution of the lazy ITD graph.
    # Both the parquet write and the summary reductions are fused into one
    # dask.compute call — don't trigger any separate compute (e.g. `len(ddf)`)
    # as that would run the entire local-maxima graph an extra time over the
    # full CHM.
    progress("Writing inventory data...", 90)
    _, stats, forestry_metrics = save_parquet_with_summary(
        inventory_id, ddf, inventory["columns"], inventory["type"], domain_gdf
    )

    # Read the tree count from the written Parquet footer (footer-only, no recompute).
    # This reflects rows actually persisted (post-modification), which is the
    # meaningful figure to report.
    tree_count = count_inventory_rows(inventory_id)
    if tree_count is not None:
        logger.info(
            f"Extracted {tree_count} trees from CHM",
            extra={"inventory_id": inventory_id},
        )

    progress("Computing georeference...", 95)
    bounds = domain_gdf.total_bounds
    georeference = {
        "crs": str(domain_gdf.crs),
        "bounds": [float(b) for b in bounds],
    }

    progress("Complete", 100)
    return {
        "georeference": georeference,
        "columns": [
            {**col, "summary": stats.get(col["key"])} for col in inventory["columns"]
        ],
        "forestry_metrics": forestry_metrics,
    }
