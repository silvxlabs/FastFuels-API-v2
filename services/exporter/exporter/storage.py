"""
Storage operations for exporter.

Handles loading grid data from Zarr and uploading export files to GCS.
"""

import logging
import mimetypes
from collections.abc import Iterator
from contextlib import contextmanager
from typing import BinaryIO

import pandas as pd
import xarray as xr
from google.cloud import storage as gcs_storage

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

    Signs a ``Content-Disposition: attachment`` header so the file downloads
    instead of opening inline; without it, CSV and GeoJSON render in the browser
    tab (#603). The filename is the object basename, which already carries the
    correct per-format extension.

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


def _blob(gcs_path: str) -> gcs_storage.Blob:
    """Return the GCS blob for a gs://bucket/path URI."""
    bucket_name, blob_path = gcs_path.removeprefix("gs://").split("/", 1)
    return gcs_storage.Client().bucket(bucket_name).blob(blob_path)


def _content_type(gcs_path: str) -> str:
    """Content type guessed from the file extension, as gcsfs does."""
    return mimetypes.guess_type(gcs_path)[0] or "application/octet-stream"


@contextmanager
def open_upload(gcs_path: str) -> Iterator[BinaryIO]:
    """Open a streaming, unseekable writer for a GCS object.

    An exception inside the block cancels the upload, so no partial object
    is left behind.

    Args:
        gcs_path: Full GCS path (gs://bucket/path/to/file)
    """
    # zipfile and pandas call flush(); the GCS writer raises on flush()
    # unless told to ignore it.
    with _blob(gcs_path).open(
        "wb", content_type=_content_type(gcs_path), ignore_flush=True
    ) as f:
        yield f


def upload_file(local_path: str, gcs_path: str) -> None:
    """Upload a local file to GCS.

    Args:
        local_path: Path of the file to upload
        gcs_path: Full GCS path (gs://bucket/path/to/file)
    """
    _blob(gcs_path).upload_from_filename(
        local_path, content_type=_content_type(gcs_path)
    )
