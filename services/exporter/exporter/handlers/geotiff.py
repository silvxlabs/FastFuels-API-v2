"""
GeoTIFF export handler.

Loads a grid from Zarr, optionally selects a band subset,
and writes a GeoTIFF to GCS.
"""

import logging
import traceback
from collections.abc import Callable

import rasterio
import rioxarray  # noqa: F401

from exporter.errors import ProcessingError
from exporter.filename import sanitize_filename
from exporter.storage import load_grid_zarr
from lib.config import EXPORTS_BUCKET

logger = logging.getLogger(__name__)


def export_geotiff(
    export: dict,
    source: dict,
    progress: Callable[[str, int | None], None],
) -> str:
    """Export a grid to GeoTIFF format.

    Args:
        export: Export document from Firestore
        source: Source configuration (grid_id, bands)
        progress: Progress callback (message, percent)

    Returns:
        GCS path to the exported GeoTIFF file

    Raises:
        ProcessingError: If the grid cannot be loaded or converted
    """
    grid_ids = source["grid_ids"]
    band_subset = source.get("bands")
    export_id = export["id"]

    # Load grid data from Zarr
    progress("Loading grid data...", 30)
    try:
        ds = load_grid_zarr(grid_ids[0])
    except Exception as e:
        raise ProcessingError(
            code="GRID_LOAD_ERROR",
            message=f"Failed to load grid {grid_ids[0]}: {e}",
            suggestion="Ensure the grid exists and has completed processing.",
            traceback=traceback.format_exc(),
        )

    # TODO: Support multi-grid exports (load and merge additional grids)
    if len(grid_ids) > 1:
        logger.warning(
            "Multi-grid GeoTIFF export not yet implemented, "
            f"exporting first grid only: {grid_ids[0]}"
        )

    # Select band subset if specified
    if band_subset:
        progress("Selecting bands...", 50)
        missing = [b for b in band_subset if b not in ds.data_vars]
        if missing:
            raise ProcessingError(
                code="BAND_NOT_FOUND",
                message=f"Bands not found in grid: {missing}",
                suggestion=f"Available bands: {list(ds.data_vars)}",
            )
        ds = ds[band_subset]

    # Write GeoTIFF directly to GCS
    progress("Writing GeoTIFF...", 70)
    filename = sanitize_filename(export.get("name", ""), ".tif")
    gcs_path = f"gs://{EXPORTS_BUCKET}/{export_id}/{filename}"
    try:
        with rasterio.Env(CPL_VSIL_USE_TEMP_FILE_FOR_RANDOM_WRITE="YES"):
            ds.rio.to_raster(gcs_path, driver="GTiff", windowed=True)
    except Exception as e:
        raise ProcessingError(
            code="GEOTIFF_WRITE_ERROR",
            message=f"Failed to write GeoTIFF: {e}",
            suggestion="This may indicate an issue with the grid's spatial metadata.",
            traceback=traceback.format_exc(),
        )

    return gcs_path
