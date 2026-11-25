"""
Grid export handlers.

Loads a grid from Zarr, optionally selects a band subset,
and writes the requested format to GCS.
"""

import logging
import math
import os
import tempfile
import threading
import traceback
import zipfile
from collections.abc import Callable

import numpy as np
import rioxarray  # noqa: F401
import xarray as xr
from zarr.codecs import ZstdCodec

from exporter.errors import ProcessingError
from exporter.filename import sanitize_filename
from exporter.storage import load_grid_zarr, open_upload, upload_file
from lib.config import EXPORTS_BUCKET

logger = logging.getLogger(__name__)

# Shared GeoTIFF nodata for integer bands whose own nodata values disagree.
_SHARED_NODATA = -9999

# GeoTIFF tile edge. Each dask chunk is one tile of one band, written once.
_TILE = 512

# Chunk codec of the exported Zarr store, pinned like zarr_format so a library
# default change can't alter what users receive.
_ZARR_COMPRESSORS = (ZstdCodec(level=0, checksum=False),)


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


def _is_nan(value) -> bool:
    """True if value is a NaN float; tolerant of non-float nodata (int, None)."""
    try:
        return math.isnan(value)
    except (TypeError, ValueError):
        return False


def _nodata_is_uniform(ds: xr.Dataset) -> bool:
    """True when every band shares one nodata value (NaN counts as a match).

    A GeoTIFF carries a single file-level nodata, and rioxarray's
    ``Dataset.to_raster`` only handles a uniform nodata across bands — it
    raises on a heterogeneous list (e.g. distinct per-band integer sentinels,
    now preserved because grids load raw). When bands disagree we fall back to
    a stacked, uniform-dtype write instead.
    """
    nodatas = [ds[v].rio.nodata for v in ds.data_vars]
    if len(nodatas) <= 1:
        return True
    first = nodatas[0]
    for n in nodatas[1:]:
        same = (n is None and first is None) or (
            n is not None
            and first is not None
            and ((_is_nan(n) and _is_nan(first)) or n == first)
        )
        if not same:
            return False
    return True


def _stack_with_shared_nodata(ds: xr.Dataset) -> xr.DataArray:
    """Stack bands into one array that carries a single nodata value.

    Each band's cells at its own nodata are set to a shared sentinel: NaN when
    the stacked dtype is floating, else -9999 (integer dtypes are widened to
    a signed type that holds it).
    """
    dtype = np.result_type(*(ds[v].dtype for v in ds.data_vars))
    if np.issubdtype(dtype, np.integer):
        dtype = np.result_type(dtype, np.int16)
    sentinel = np.nan if np.issubdtype(dtype, np.floating) else _SHARED_NODATA

    bands = {}
    for name in ds.data_vars:
        da = ds[name]
        nodata = da.rio.nodata
        if nodata is None:
            is_nodata = xr.zeros_like(da, dtype=bool)
        elif _is_nan(nodata):
            is_nodata = da.isnull()
        else:
            is_nodata = da == nodata
        bands[name] = da.astype(dtype).where(~is_nodata, sentinel)
        bands[name].encoding.pop("_FillValue", None)
        bands[name].attrs.pop("_FillValue", None)

    stacked = ds.assign(bands).to_array(dim="band")
    # rioxarray writes long_name as the GeoTIFF band descriptions.
    stacked.attrs["long_name"] = tuple(str(v) for v in ds.data_vars)
    return stacked.rio.write_nodata(sentinel, encoded=False)


def _geotiff_options(dtype: np.dtype) -> dict:
    """GeoTIFF creation options: band-interleaved 512 x 512 tiles, DEFLATE.

    Band interleave lets each tile of each band be written and compressed
    once; pixel interleave would rewrite every tile once per band (#608).
    """
    return {
        "driver": "GTiff",
        "tiled": True,
        "blockxsize": _TILE,
        "blockysize": _TILE,
        "interleave": "band",
        "compress": "DEFLATE",
        "predictor": 3 if np.issubdtype(dtype, np.floating) else 2,
        "lock": threading.Lock(),
    }


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

    # GeoTIFF is a 2D raster format; a 3D voxel grid (z, y, x) can't be written.
    # The API rejects this at request time (#243); this guards the case where a
    # 3D grid reaches the exporter another way, with a clear message instead of
    # the catch-all GEOTIFF_WRITE_ERROR below.
    if "z" in ds.dims:
        raise ProcessingError(
            code="GEOTIFF_UNSUPPORTED_DIMENSIONALITY",
            message="GeoTIFF is a 2D raster format and cannot represent a 3D voxel grid.",
            suggestion="Use the `netcdf` or `zarr` export format for 3D fuel grids.",
        )

    progress("Writing GeoTIFF...", 70)
    filename = sanitize_filename(export.get("name", ""), ".tif")
    gcs_path = f"gs://{EXPORTS_BUCKET}/{export_id}/{filename}"
    # Written locally (tmpfs on Cloud Run, so the compressed size counts toward
    # memory) and uploaded only once complete. rioxarray reopens the file in
    # r+ mode per chunk from dask threads, which /vsigs/ cannot do.
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            local_path = os.path.join(tmp_dir, "export.tif")
            if _nodata_is_uniform(ds):
                dtype = np.result_type(*(ds[v].dtype for v in ds.data_vars))
                tiles = ds.chunk({"y": _TILE, "x": _TILE})
                tiles.rio.to_raster(local_path, **_geotiff_options(dtype))
            else:
                # Bands carry distinct nodata (e.g. LANDFIRE aspect -1 next to
                # elevation 32767). A GeoTIFF has one dtype and one file-level
                # nodata, so map every band's nodata onto a shared value.
                stacked = _stack_with_shared_nodata(ds)
                tiles = stacked.chunk({"band": 1, "y": _TILE, "x": _TILE})
                tiles.rio.to_raster(local_path, **_geotiff_options(stacked.dtype))

            upload_file(local_path, gcs_path)
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

    for variable in ds.variables.values():
        variable.encoding.pop("compressor", None)
        variable.encoding["compressors"] = _ZARR_COMPRESSORS

    # The store is staged in /tmp (tmpfs); the zip streams straight to GCS.
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            zarr_dir = os.path.join(tmp_dir, "export.zarr")
            # The format version is part of the export contract: users open this
            # store with their own tooling, and zarr-python 2.x cannot read v3.
            # Pinned rather than left to the library default so a future default
            # flip can't silently change what users receive.
            ds.to_zarr(zarr_dir, zarr_format=3)

            progress("Uploading...", 85)
            # Same entries as shutil.make_archive. The chunks are already
            # zstd-compressed, so they are stored; only the JSON is deflated.
            with (
                open_upload(gcs_path) as f,
                zipfile.ZipFile(f, "w", zipfile.ZIP_DEFLATED) as zf,
            ):
                zf.write(zarr_dir, "export.zarr")
                for root, dirs, files in os.walk(zarr_dir):
                    dirs.sort()
                    for name in [*dirs, *sorted(files)]:
                        path = os.path.join(root, name)
                        stored = os.path.isfile(path) and not name.endswith(".json")
                        zf.write(
                            path,
                            os.path.relpath(path, tmp_dir),
                            compress_type=zipfile.ZIP_STORED if stored else None,
                        )
    except ProcessingError:
        raise
    except Exception as e:
        raise ProcessingError(
            code="ZARR_WRITE_ERROR",
            message=f"Failed to write Zarr export: {e}",
            suggestion="This may indicate an issue with the grid data.",
            traceback=traceback.format_exc(),
        )

    return gcs_path
