"""
Zarr storage for grid data.
"""

import logging

import xarray as xr

from lib.config import GRIDS_BUCKET
from lib.gcs import delete_directory
from lib.zarr_utils import load_zarr as _load_zarr
from lib.zarr_utils import save_zarr as _save_zarr

logger = logging.getLogger(__name__)


def save_zarr(grid_id: str, data: xr.Dataset, chunk_shape: tuple[int, int]) -> str:
    """Save grid data to Zarr in Cloud Storage.

    Args:
        grid_id: The grid document ID
        data: Dataset with named 2D (y, x) variables and spatial metadata.
        chunk_shape: (height, width) chunk shape for on-disk Zarr chunks.

    Returns:
        GCS path where data was written
    """
    path = f"gs://{GRIDS_BUCKET}/{grid_id}"
    _save_zarr(path, data, chunk_shape=chunk_shape)
    logger.info(f"Saved grid data to {path}")
    return path


def load_zarr(grid_id: str) -> xr.Dataset:
    """Load grid data from Zarr in Cloud Storage.

    Args:
        grid_id: The grid document ID

    Returns:
        Dataset with spatial metadata preserved
    """
    path = f"gs://{GRIDS_BUCKET}/{grid_id}"
    return _load_zarr(path)


def delete_zarr(grid_id: str) -> None:
    """Delete Zarr store from Cloud Storage.

    Used for cleanup during cancellation. Best-effort - exceptions are logged
    but not raised.

    Args:
        grid_id: The grid document ID
    """
    path = f"gs://{GRIDS_BUCKET}/{grid_id}"
    try:
        delete_directory(path)
        logger.info(f"Deleted grid data at {path}")
    except Exception as e:
        logger.warning(f"Failed to delete grid data at {path}: {e}")
