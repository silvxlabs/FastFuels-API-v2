"""
Band summary statistics for completed grids.

Computes scalar stats (min, max, mean, std, count, nodata_count, unique_count)
per band from an xr.Dataset without loading the full array into memory.
Called from main.py after dispatch_handler returns, before save_zarr.
"""

from __future__ import annotations

import math

import numpy as np
import xarray as xr

CHUNK_SHAPE = (512, 512)


def summarize_dataset(
    ds: xr.Dataset,
    bands: list[dict],
) -> dict[str, dict]:
    """Return a summary dict for every band in *bands*.

    Args:
        ds:    Dataset returned by a griddle handler.  Variable names must
               match the ``key`` field of each band dict.
        bands: Band dicts from the Firestore grid document.  Each must have
               at least ``key`` (str) and ``type`` ("continuous" |
               "categorical").

    Returns:
        Mapping of band key -> summary dict, ready to merge into Firestore.
    """
    result: dict[str, dict] = {}
    for band in bands:
        key = band["key"]
        band_type = band["type"]
        # Resolve dot-notation keys — layerset bands are stored as
        # `<var>.<band_coord>` (e.g. "herb.loading") where the variable
        # is 3D (band, y, x) and the band coord selects the 2D slice.
        if key in ds.data_vars:
            da = ds[key]
        elif "." in key:
            var, band_coord = key.rsplit(".", 1)
            da = ds[var].sel(band=band_coord)
        else:
            raise KeyError(f"Band key '{key}' not found in dataset")

        if band_type == "continuous":
            result[key] = _summarize_continuous(da)
        elif band_type == "categorical":
            result[key] = _summarize_categorical(da)
        else:
            raise ValueError(f"Unknown band type '{band_type}' for key '{key}'")
    return result


def _nodata_mask(arr: np.ndarray, nodata) -> np.ndarray:
    """Return a boolean mask of nodata cells.

    Args:
        arr:    Numpy array of pixel values.
        nodata: Nodata value from da.rio.nodata, or None.

    Returns:
        Boolean array, True where cells are nodata.
    """
    if nodata is None:
        return np.zeros(arr.shape, dtype=bool)
    if np.isnan(float(nodata)):
        return np.isnan(arr.astype(np.float64))
    return arr == nodata


def _summarize_continuous(da: xr.DataArray) -> dict:
    """Single-pass accumulation of continuous stats over all chunks."""
    nodata = da.rio.nodata

    count = 0
    nodata_count = 0
    running_min = math.inf
    running_max = -math.inf
    running_sum = 0.0
    running_sum_sq = 0.0

    for chunk in _iter_chunks(da):
        arr = np.asarray(chunk, dtype=np.float64)
        mask = _nodata_mask(arr, nodata)
        valid = arr[~mask]

        n_valid = valid.size
        n_nodata = arr.size - n_valid

        count += n_valid
        nodata_count += n_nodata

        if n_valid > 0:
            running_min = min(running_min, float(valid.min()))
            running_max = max(running_max, float(valid.max()))
            running_sum += float(valid.sum())
            running_sum_sq += float((valid**2).sum())

    if count == 0:
        return {
            "type": "continuous",
            "count": 0,
            "nodata_count": nodata_count,
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
        }

    mean = running_sum / count
    variance = max(running_sum_sq / count - mean**2, 0.0)
    std = math.sqrt(variance)

    return {
        "type": "continuous",
        "count": count,
        "nodata_count": nodata_count,
        "min": running_min,
        "max": running_max,
        "mean": mean,
        "std": std,
    }


def _summarize_categorical(da: xr.DataArray) -> dict:
    """Single-pass accumulation of categorical stats over all chunks."""
    nodata = da.rio.nodata

    count = 0
    nodata_count = 0
    unique_values: set = set()

    for chunk in _iter_chunks(da):
        arr = np.asarray(chunk)
        mask = _nodata_mask(arr, nodata)
        valid = arr[~mask]

        count += valid.size
        nodata_count += int(mask.sum())
        unique_values.update(np.unique(valid).tolist())

    return {
        "type": "categorical",
        "count": count,
        "nodata_count": nodata_count,
        "unique_count": len(unique_values),
    }


def _iter_chunks(da: xr.DataArray):
    """Yield numpy arrays one chunk at a time.

    If the DataArray is Dask-backed, iterate Dask blocks directly so we
    never materialise the whole array.  Otherwise slice the in-memory array
    using CHUNK_SHAPE to keep the same memory profile.
    """
    try:
        import dask.array as dsa  # noqa: PLC0415

        if isinstance(da.data, dsa.Array):
            for block in da.data.blocks.ravel():
                yield block.compute()
            return
    except ImportError:
        pass

    arr = np.asarray(da)
    h, w = arr.shape[-2], arr.shape[-1]
    ch, cw = CHUNK_SHAPE
    for row in range(0, h, ch):
        for col in range(0, w, cw):
            yield arr[..., row : row + ch, col : col + cw]
