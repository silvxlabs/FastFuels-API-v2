"""
Zarr storage utilities for xarray Datasets.

Provides save/load with correct decode_coords handling to preserve
rioxarray spatial metadata (CRS, transform, spatial_ref coordinate).
"""

import xarray as xr


def save_zarr(path: str, data: xr.Dataset, chunk_shape: tuple[int, int]) -> str:
    """Save an xarray Dataset to a Zarr store.

    Only accepts Dataset — not DataArray. This enforces the convention
    that all grid data uses named 2D (y, x) variables, catching bugs
    where a handler accidentally returns a DataArray or drops variables.

    Args:
        path: Local or GCS path for the Zarr store
        data: Dataset with named variables and spatial metadata
        chunk_shape: (height, width) chunk shape for on-disk Zarr chunks.
            The Dataset is rechunked via xarray's .chunk() before writing —
            xarray uses dask chunk sizes as Zarr on-disk chunks automatically.

    Returns:
        The path where data was written

    Raises:
        TypeError: If data is not an xr.Dataset
    """
    if not isinstance(data, xr.Dataset):
        raise TypeError(
            f"save_zarr requires xr.Dataset, got {type(data).__name__}. "
            "Handlers must return Dataset with named 2D (y, x) variables."
        )
    data = data.chunk({"y": chunk_shape[0], "x": chunk_shape[1]})
    data.to_zarr(path, mode="w", consolidated=True)
    return path


def load_zarr(path: str) -> xr.Dataset:
    """Load a Zarr store as an xarray Dataset.

    Uses decode_coords="all" to ensure spatial_ref is loaded as a
    coordinate (not a data variable), which rioxarray requires for
    CRS/transform access via .rio accessors.

    Args:
        path: Local or GCS path to the Zarr store

    Returns:
        Dataset with spatial metadata preserved
    """
    return xr.open_zarr(path, decode_coords="all")
