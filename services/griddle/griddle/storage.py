"""
Zarr storage for grid data.
"""

import logging

import xarray as xr

from lib.config import GRIDS_BUCKET
from lib.errors import ProcessingError
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


def load_band(grid_id: str, band: str) -> xr.DataArray:
    """Load one band of a stored grid, as stored.

    Args:
        grid_id: The grid document ID
        band: The band key to take from it

    Returns:
        The band with its dtype, nodata and lattice unchanged.

    Raises:
        ProcessingError: The grid or the band can't be loaded.
    """
    try:
        dataset = load_zarr(grid_id)
    except FileNotFoundError as e:
        raise ProcessingError(
            code="SOURCE_GRID_UNAVAILABLE",
            message=f"Could not load source grid '{grid_id}': {e}",
            suggestion="Ensure the grid exists and has been processed.",
        ) from e

    if band not in dataset.data_vars:
        raise ProcessingError(
            code="SOURCE_BAND_NOT_FOUND",
            message=(
                f"Source grid '{grid_id}' has no band '{band}'. "
                f"Available bands: {sorted(dataset.data_vars)}."
            ),
        )

    return dataset[band]


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
