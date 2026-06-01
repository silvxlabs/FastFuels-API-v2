"""
Zarr storage utilities for xarray Datasets.

Provides save/load with correct decode_coords handling to preserve
rioxarray spatial metadata (CRS, transform, spatial_ref coordinate).

Uses consolidated metadata for fast remote reads. This is experimental
for zarr v3 stores (zarr-specs#309) but stable in zarr-python and a
good fit for our write-once, read-many grids on GCS.

## CRS metadata convention

This module **never** passes the `encoding=` kwarg to `to_zarr`. The
kwarg fully replaces each variable's encoding dict, which would wipe the
`grid_mapping` field that `rio.write_crs` / `decode_coords="all"` placed
there. Without `grid_mapping` in encoding, xarray's CF encoder cannot
write a CRS-reference attribute to the on-disk zarr, and downstream
readers see `ds.rio.crs is None`.

If you need per-variable encoding (compression, fill values, dtype),
mutate `var.encoding[...]` directly **before** calling `save_zarr` and
leave the kwarg unused. See `services/lib/lib/cf_utils.py` for the full
convention and the corresponding netCDF write-side pattern in
`services/exporter/exporter/handlers/netcdf.py`.

`load_zarr` uses `decode_coords="all"` so callers always get a Dataset
where `spatial_ref` is a coord and `ds.rio.crs` is populated, regardless
of whether the on-disk zarr stored `grid_mapping` in attrs (treevox
writer) or in encoding (anything written via `save_zarr`).
"""

import warnings

import xarray as xr


def save_zarr(
    path: str,
    data: xr.Dataset,
    chunk_shape: tuple[int, int] | tuple[int, int, int],
) -> str:
    """Save an xarray Dataset to a Zarr store.

    Only accepts Dataset — not DataArray. This enforces the convention
    that all grid data uses named 2D (y, x) or 3D (z, y, x) variables,
    catching bugs where a handler accidentally returns a DataArray or
    drops variables.

    Args:
        path: Local or GCS path for the Zarr store
        data: Dataset with named variables and spatial metadata
        chunk_shape: Chunk shape for on-disk Zarr chunks. Pass `(y, x)`
            for 2D datasets, or `(z, y, x)` for 3D datasets. The Dataset
            is rechunked via xarray's .chunk() before writing — xarray
            uses dask chunk sizes as Zarr on-disk chunks automatically.

    Returns:
        The path where data was written

    Raises:
        TypeError: If data is not an xr.Dataset
        ValueError: If chunk_shape length doesn't match dataset rank
    """
    if not isinstance(data, xr.Dataset):
        raise TypeError(
            f"save_zarr requires xr.Dataset, got {type(data).__name__}. "
            "Handlers must return Dataset with named 2D (y, x) variables."
        )

    if len(chunk_shape) == 2:
        chunks = {"y": chunk_shape[0], "x": chunk_shape[1]}
    elif len(chunk_shape) == 3:
        chunks = {"z": chunk_shape[0], "y": chunk_shape[1], "x": chunk_shape[2]}
    else:
        raise ValueError(
            f"chunk_shape must be (y, x) or (z, y, x); got length {len(chunk_shape)}."
        )

    data = data.chunk(chunks)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="Consolidated metadata", category=UserWarning
        )
        data.to_zarr(path, mode="w", consolidated=True)
    return path


def load_zarr(path: str) -> xr.Dataset:
    """Load a Zarr store as an xarray Dataset.

    Uses decode_coords="all" to ensure spatial_ref is loaded as a
    coordinate (not a data variable), which rioxarray requires for
    CRS/transform access via .rio accessors.

    Uses mask_and_scale=False so grids load back exactly as they were
    written, matching how the write side reads its sources
    (``rioxarray.open_rasterio`` defaults to ``mask_and_scale=False``) and
    following rioxarray's own NoData guidance. Without it, xarray's CF
    decoding masks each band's ``_FillValue`` to NaN and promotes integer
    grids (categorical FBFM/TreeMap codes) to float on load. Keeping it off
    preserves dtype and the nodata sentinel so ``rio.nodata`` round-trips
    faithfully; consumers that want gaps as NaN mask explicitly against
    ``rio.nodata`` (e.g. ``da.where(da != da.rio.nodata)``). See issue #290.

    griddle never relies on CF scale_factor/add_offset decoding — every
    handler reads sources raw and applies any scaling eagerly — so disabling
    mask_and_scale changes only the nodata-masking behavior, not values.

    Args:
        path: Local or GCS path to the Zarr store

    Returns:
        Dataset with spatial metadata preserved
    """
    return xr.open_zarr(path, decode_coords="all", mask_and_scale=False)
