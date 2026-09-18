"""
Storage operations for exporter.

Handles loading grid data from Zarr and uploading export files to GCS.
"""

import logging

import pandas as pd
import xarray as xr

from lib.config import EXPORTS_BUCKET, GRIDS_BUCKET, INVENTORIES_BUCKET
from lib.gcs import delete_directory
from lib.gcs.signed_urls import generate_download_signed_url
from lib.zarr_utils import load_zarr as _load_zarr

logger = logging.getLogger(__name__)


def load_grid_zarr(grid_id: str) -> xr.Dataset:
    """Load grid data from Zarr in Cloud Storage.

    Args:
        grid_id: The grid document ID

    Returns:
        Dataset with grid data
    """
    path = f"gs://{GRIDS_BUCKET}/{grid_id}"
    return _load_zarr(path)


def load_inventory_parquet(inventory_id: str) -> pd.DataFrame:
    """Load inventory data from partitioned Parquet in Cloud Storage.

    Args:
        inventory_id: The inventory document ID

    Returns:
        DataFrame with inventory data
    """
    path = f"gs://{INVENTORIES_BUCKET}/{inventory_id}"
    return pd.read_parquet(path)


def generate_signed_download(gcs_path: str, expiration_days: int) -> str:
    """Generate a signed download URL for an export file.

    The URL forces a download (rather than inline browser rendering) by signing
    a ``Content-Disposition: attachment`` response header with the export's
    filename. Without it, browser-renderable formats (CSV, GeoJSON) open in the
    tab instead of downloading. The filename is the export object's own basename,
    which already carries the correct per-format extension (see #603).

    Args:
        gcs_path: Full GCS path (gs://bucket/path/to/file)
        expiration_days: Number of days until the URL expires

    Returns:
        Signed URL string for GET request.
    """
    # Parse "gs://bucket-name/blob/path" into bucket and blob
    without_scheme = gcs_path.removeprefix("gs://")
    bucket_name, blob_path = without_scheme.split("/", 1)
    filename = blob_path.rsplit("/", 1)[-1]
    disposition = f'attachment; filename="{filename}"'
    return generate_download_signed_url(
        bucket_name,
        blob_path,
        expiration_days,
        response_disposition=disposition,
    )


def delete_export_files(export_id: str) -> None:
    """Delete export files from GCS.

    Used for cleanup during cancellation. Best-effort.

    Args:
        export_id: The export document ID
    """
    path = f"gs://{EXPORTS_BUCKET}/{export_id}"
    try:
        delete_directory(path)
        logger.info(f"Deleted export files at {path}")
    except Exception as e:
        logger.warning(f"Failed to delete export files at {path}: {e}")
