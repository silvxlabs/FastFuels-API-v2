"""
Grid export handlers.

Loads a grid from Zarr, optionally selects a band subset,
and writes the requested format to GCS.
"""

import logging
import os
import shutil
import tempfile
import traceback
from collections.abc import Callable

import rasterio
import rioxarray  # noqa: F401
import xarray as xr

from exporter.errors import ProcessingError
from exporter.filename import sanitize_filename
from exporter.storage import load_grid_zarr
from lib.config import EXPORTS_BUCKET

logger = logging.getLogger(__name__)


def _load_and_select_bands(
    source: dict,
    progress: Callable[[str, int | None], None],
) -> xr.Dataset:
    """Load a grid from Zarr and optionally select a band subset.

    Args:
        source: Source configuration with grid_id and optional bands
        progress: Progress callback (message, percent)

    Returns:
        Dataset with grid data, optionally subset to requested bands
    """
    grid_id = source["grid_id"]
    band_subset = source.get("bands")

    progress("Loading grid data...", 30)
    try:
        ds = load_grid_zarr(grid_id)
    except Exception as e:
        raise ProcessingError(
            code="GRID_LOAD_ERROR",
            message=f"Failed to load grid {grid_id}: {e}",
            suggestion="Ensure the grid exists and has completed processing.",
            traceback=traceback.format_exc(),
        )

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

    return ds


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
    """
    export_id = export["id"]
    ds = _load_and_select_bands(source, progress)

    progress("Writing GeoTIFF...", 70)
    filename = sanitize_filename(export.get("name", ""), ".tif")
    gcs_path = f"gs://{EXPORTS_BUCKET}/{export_id}/{filename}"
    try:
        with rasterio.Env(CPL_VSIL_USE_TEMP_FILE_FOR_RANDOM_WRITE="YES"):
            ds.rio.to_raster(gcs_path, driver="GTiff")
    except Exception as e:
        raise ProcessingError(
            code="GEOTIFF_WRITE_ERROR",
            message=f"Failed to write GeoTIFF: {e}",
            suggestion="This may indicate an issue with the grid's spatial metadata.",
            traceback=traceback.format_exc(),
        )

    return gcs_path


def export_zarr(
    export: dict,
    source: dict,
    progress: Callable[[str, int | None], None],
) -> str:
    """Export a grid to zipped Zarr format.

    Args:
        export: Export document from Firestore
        source: Source configuration (grid_id, bands)
        progress: Progress callback (message, percent)

    Returns:
        GCS path to the exported zip file
    """
    export_id = export["id"]
    ds = _load_and_select_bands(source, progress)

    progress("Writing Zarr...", 70)
    filename = sanitize_filename(export.get("name", ""), ".zip")
    gcs_path = f"gs://{EXPORTS_BUCKET}/{export_id}/{filename}"

    tmp_dir = tempfile.mkdtemp()
    try:
        zarr_dir = os.path.join(tmp_dir, "export.zarr")
        ds.to_zarr(zarr_dir)

        progress("Zipping Zarr...", 85)
        zip_path = os.path.join(tmp_dir, "export")
        zip_file = shutil.make_archive(zip_path, "zip", tmp_dir, "export.zarr")

        progress("Uploading...", 90)
        from google.cloud import storage as gcs_storage

        without_scheme = gcs_path.removeprefix("gs://")
        bucket_name, blob_path = without_scheme.split("/", 1)
        client = gcs_storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        blob.upload_from_filename(zip_file)
    except ProcessingError:
        raise
    except Exception as e:
        raise ProcessingError(
            code="ZARR_WRITE_ERROR",
            message=f"Failed to write Zarr export: {e}",
            suggestion="This may indicate an issue with the grid data.",
            traceback=traceback.format_exc(),
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return gcs_path
