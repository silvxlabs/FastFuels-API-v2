"""xarray-backed 3D zarr I/O for treevox.

Orchestrator-only module — never imported by spawned workers.

The surface is built around one xarray Dataset per grid with one variable
per requested band. Per-variable dtypes and fill values are preserved via
`encoding` on the initial `to_zarr(compute=False)` call. Workers fill chunk
regions by returning numpy buffers; the orchestrator merges them into a
halo-extended union Dataset and writes the region back with
`to_zarr(region=...)`.

See TREEVOX.md "Concurrency & Runtime Constraints" for the ten design
constraints this file enforces.
"""

from __future__ import annotations

import logging
import warnings

import dask.array as da
import numpy as np
import rioxarray  # noqa: F401 — registers `.rio` accessor on xr.Dataset
import xarray as xr
import zarr

from lib.config import GRIDS_BUCKET
from lib.gcs import delete_directory

logger = logging.getLogger(__name__)


# (dtype, fill_value) per band.
BAND_SPECS: dict[str, tuple[str, float | int]] = {
    "volume_fraction": ("float32", 0.0),
    "bulk_density.foliage.live": ("float32", 0.0),
    "bulk_density.foliage.dead": ("float32", 0.0),
    "bulk_density.branchwood.live": ("float32", 0.0),
    "bulk_density.branchwood.dead": ("float32", 0.0),
    "bulk_density.fine.live": ("float32", 0.0),
    "bulk_density.fine.dead": ("float32", 0.0),
    "leaf_area_density": ("float32", 0.0),
    "savr.foliage": ("float32", 0.0),
    "fuel_moisture.live": ("float32", 0.0),
    "fuel_moisture.dead": ("float32", 0.0),
    "spcd": ("uint16", 0),
    "tree_id": ("int32", -1),
}

ADDITIVE_BANDS = frozenset(
    {
        "volume_fraction",
        "bulk_density.foliage.live",
        "bulk_density.foliage.dead",
        "bulk_density.branchwood.live",
        "bulk_density.branchwood.dead",
        "bulk_density.fine.live",
        "bulk_density.fine.dead",
        "leaf_area_density",
    }
)


def gcs_path(grid_id: str) -> str:
    """Build the GCS URI for a grid's zarr store."""
    return f"gs://{GRIDS_BUCKET}/{grid_id}"


def init_store(
    path: str,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    z_coords: np.ndarray,
    hr: float,
    vr: float,
    crs: str,
    z_origin: float,
    requested_keys: list[str],
    chunk_shape: tuple[int, int, int],
) -> None:
    """Create an empty zarr store with per-band dtypes, fill values, and chunks.

    Uses `to_zarr(compute=False)` with dask-array dummies so only metadata is
    written — data is filled in later via region writes. The resulting store
    can be opened and written to in parallel from disjoint regions.

    Writes consolidated metadata (`.zmetadata`) as part of this single call.
    That lets subsequent `open_zarr(consolidated=True)` fetch one small file
    instead of listing the store prefix and fetching per-variable `.zarray` /
    `.zattrs` on every batch — a major win on GCS where small-op latency
    dominates. Region writes during the batch loop only touch data chunks, so
    the consolidated metadata stays valid for the whole job.

    Coords: z, y, x (cell-center arrays), plus `spatial_ref` added by
    rioxarray. These become index coordinates and are protected from overwrite
    on subsequent region writes.
    """
    nx, ny, nz = len(x_coords), len(y_coords), len(z_coords)
    data_vars: dict = {}
    encoding: dict = {}
    for key in requested_keys:
        dtype, fill = BAND_SPECS[key]
        data_vars[key] = (
            ("z", "y", "x"),
            da.full((nz, ny, nx), fill, dtype=dtype, chunks=chunk_shape),
        )
        encoding[key] = {"fill_value": fill}

    ds = xr.Dataset(
        data_vars,
        coords={"z": z_coords, "y": y_coords, "x": x_coords},
    )
    ds = ds.rio.write_crs(crs)

    # CF §5.6: link each data var to the spatial_ref coord so
    # `xr.open_zarr(..., decode_coords="all")` promotes spatial_ref back
    # to a coord on reopen — without this, rioxarray.open_rasterio-style
    # auto-discovery fails and `.rio.crs` returns None.
    for v in ds.data_vars:
        ds[v].attrs["grid_mapping"] = "spatial_ref"

    # Derive transform from cell-center coords and resolution.
    x_origin = float(x_coords[0]) - hr / 2
    y_origin = float(y_coords[0]) + hr / 2
    ds.attrs["transform"] = [hr, 0.0, x_origin, 0.0, -hr, y_origin]
    ds.attrs["z_origin"] = float(z_origin)
    ds.attrs["z_resolution"] = float(vr)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="Consolidated metadata", category=UserWarning
        )
        ds.to_zarr(path, mode="w", compute=False, encoding=encoding, consolidated=True)


def read_union(path: str, y_slice: slice, x_slice: slice) -> xr.Dataset:
    """Load a halo-extended region fully into memory.

    Workers receive numpy arrays, never lazy dask xarray. The caller asserts
    `not ds.chunks` before splitting into payloads.

    Uses `consolidated=True` because `init_store` writes consolidated
    metadata at job start — one `.zmetadata` GET per batch instead of a
    directory listing + per-variable `.zarray`/`.zattrs` fetches.
    """
    ds = xr.open_zarr(path, consolidated=True).isel(y=y_slice, x=x_slice).load()
    return ds


def write_union(path: str, ds: xr.Dataset, y_slice: slice, x_slice: slice) -> None:
    """Write a halo-extended region back via a single region write.

    Halo unions never align with on-disk chunks, so `align_chunks=True` lets
    xarray rechunk the write to match. Coord variables are dropped because
    xarray rejects overwriting index coords on region writes.
    """
    ds_to_write = ds.drop_vars(["x", "y", "z", "spatial_ref"], errors="ignore")
    ds_to_write.to_zarr(
        path,
        region={"y": y_slice, "x": x_slice},
        consolidated=False,
        align_chunks=True,
    )


def masked_merge(
    union_ds: xr.Dataset,
    chunk_results: list[dict],
    union_y: slice,
    union_x: slice,
) -> xr.Dataset:
    """Merge worker results into the union Dataset.

    Each worker starts from the same pre-batch union slice. For additive bands
    (`volume_fraction`, `bulk_density.*`) merge only the worker's delta above
    that original slice, so overlapping halo cells sum contributions from every
    chunk instead of keeping the last worker's full buffer. For overwrite bands
    (`spcd`, `tree_id`, `savr.*`, `fuel_moisture.*`) only cells changed by that
    worker overwrite the merged union, preserving last-writer-wins semantics
    without letting an untouched neighboring halo revert a previous write.

    Like the previous fill-value mask, comparing against the original union
    cannot distinguish a real write whose value exactly equals the baseline.
    That is acceptable for current bands: real `tree_id` never equals -1, and
    species code / moisture / SAV writes equal to the baseline are no-ops.
    """
    merged = union_ds.copy(deep=True)
    for result in chunk_results:
        if "error" in result:
            raise RuntimeError(
                f"chunk {result['chunk_location']} failed: {result['error']}"
            )
        y_slice: slice = result["y_slice"]
        x_slice: slice = result["x_slice"]
        rel_y = slice(
            y_slice.start - union_y.start,
            y_slice.stop - union_y.start,
        )
        rel_x = slice(
            x_slice.start - union_x.start,
            x_slice.stop - union_x.start,
        )
        for key, buffer in result["buffers"].items():
            original = union_ds[key].values[:, rel_y, rel_x]
            target = merged[key].values[:, rel_y, rel_x]
            if key in ADDITIVE_BANDS:
                target += buffer - original
            else:
                mask = buffer != original
                target[mask] = buffer[mask]
            merged[key].values[:, rel_y, rel_x] = target
    return merged


def consolidate_metadata(path: str) -> None:
    """Reconsolidate zarr metadata for an existing store.

    Not used in the normal voxelization flow — `init_store` already writes
    consolidated metadata, and region writes during the batch loop only touch
    data chunks, so `.zmetadata` stays valid for the whole job. Exposed for
    ad-hoc reconsolidation if the schema is ever mutated after init.
    """
    zarr.consolidate_metadata(path)


def delete_zarr(path: str) -> None:
    """Best-effort zarr store deletion.

    Called on cancellation or failure to avoid leaving a partially-written
    store on GCS.
    """
    try:
        delete_directory(path)
        logger.info(f"Deleted grid data at {path}")
    except Exception as e:
        logger.warning(f"Failed to delete grid data at {path}: {e}")
