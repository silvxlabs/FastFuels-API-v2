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

The one encoding key `save_zarr` manages itself is `chunks`: it rechunks
to the caller's `chunk_shape` and clears any stale per-variable
`encoding['chunks']`/`preferred_chunks` first, so a hint left by a prior
read cannot disagree with the dask chunks and trip xarray's overlap guard
(issue #417). Do not set `encoding['chunks']` expecting it to survive.

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
            Any stale per-variable `encoding['chunks']` (e.g. left by a
            prior `load_zarr`) is cleared first so it cannot conflict with
            these dask chunks (issue #417).

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

    # Clear any stale per-variable `encoding['chunks']` (and the paired
    # `preferred_chunks`) carried over from a prior read: `load_zarr` stamps
    # every variable with the source store's on-disk chunk sizes, and a source
    # raster's native tiling lands there too. We have just rechunked to
    # `chunk_shape`, so those dask chunks are the intended on-disk layout; a
    # leftover `encoding['chunks']` that disagrees makes xarray's Zarr writer
    # raise "chunks ... would overlap multiple Dask chunks" and, under a
    # parallel Dask write, risks corrupting the array (issue #417). Dropping
    # the hint lets xarray derive the on-disk chunks from the dask chunks, as
    # this function's contract promises. Every other encoding key — notably
    # `grid_mapping` — is left intact, honoring the "mutate `var.encoding`
    # before write, never pass `encoding=`" convention above.
    for variable in data.variables.values():
        variable.encoding.pop("chunks", None)
        variable.encoding.pop("preferred_chunks", None)

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
