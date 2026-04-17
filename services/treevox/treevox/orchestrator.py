"""Voxelization job orchestration: dispatch + the inventory voxelization job.

Distinguishes carefully between:
- *inventory* (tabular tree data — a parquet of rows); handled by
  `treevox.inventory_io`.
- *voxelization* (the 3D gridding job this module runs); the thing treevox
  actually does.

Decomposed into small testable stages so each pure-ish step (plan layout,
prepare tree chunks, build payloads, process a batch) can be unit-tested
independently of multiprocessing and GCS I/O.
"""

from __future__ import annotations

import logging
import math
import multiprocessing
import os
import tempfile
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from treevox import storage, voxelize
from treevox._worker import run as worker_run
from treevox.errors import ProcessingError
from treevox.inventory_io import assign_tree_ids, download_inventory, filter_live

logger = logging.getLogger(__name__)

DEFAULT_MAX_WORKERS = 4
WORKER_MEMORY_ESTIMATE_BYTES = 500 * 1024 * 1024  # ~500 MB budget per worker


@dataclass
class VoxelizationResult:
    """Returned by the voxelization job for persistence in Firestore."""

    gcs_path: str
    georeference: dict
    chunk_shape: list[int]


@dataclass
class GridLayout:
    """Derived layout for a voxelization job.

    Bundles the grid dimensions, requested band keys, chunk sizing, and the
    deterministic order in which chunks will be processed. Constructed once by
    `_plan_grid_layout` and threaded through the rest of the stages so
    downstream functions take one object instead of ~8 scalars.
    """

    dims: dict
    requested_keys: list[str]
    chunk_xy: int
    chunk_shape: tuple[int, int, int]
    chunk_locations: list[tuple[int, int]]


def _pick_worker_count() -> int:
    """Cap workers by CPU affinity and available memory.

    Cloud Run container memory fluctuates; read `MemAvailable` at startup
    rather than `MemTotal` so we don't oversubscribe.
    """
    try:
        cpu_cap = max(1, min(DEFAULT_MAX_WORKERS, len(os.sched_getaffinity(0))))
    except AttributeError:
        cpu_cap = max(1, min(DEFAULT_MAX_WORKERS, os.cpu_count() or 1))
    try:
        with open("/proc/meminfo") as f:
            avail_kb = next(
                int(ln.split()[1]) for ln in f if ln.startswith("MemAvailable")
            )
        mem_cap = max(1, (avail_kb * 1024) // WORKER_MEMORY_ESTIMATE_BYTES)
    except (FileNotFoundError, StopIteration):
        # Non-Linux (local dev): fall back to CPU cap only.
        mem_cap = DEFAULT_MAX_WORKERS
    return int(min(cpu_cap, mem_cap))


# Voxelization stages


def _load_inventory_dataframe(
    source: dict, progress: Callable[[str, int | None], None]
) -> pd.DataFrame:
    """Download the parquet, filter to live trees, and assign tree IDs."""
    progress("Loading inventory...", 5)
    with tempfile.TemporaryDirectory() as tmp:
        df = download_inventory(source["source_inventory_id"], tmp)
    df = filter_live(df, source.get("biomass_column"))
    df = assign_tree_ids(df)
    if df.empty:
        raise ProcessingError(
            code="EMPTY_INVENTORY",
            message="Inventory has no live trees with complete measurements.",
            suggestion="Verify the inventory contains rows with fia_status_code == 1 "
            "and non-null dbh / height / crown_ratio.",
        )
    return df


def _plan_grid_layout(grid: dict, domain_gdf, df: pd.DataFrame) -> GridLayout:
    """Compute grid dimensions, chunk sizing, and the chunk processing order."""
    try:
        dims = voxelize.compute_grid_dimensions(
            domain_gdf, df, grid["source"]["resolution"]
        )
    except voxelize.InvalidResolutionError as e:
        raise ProcessingError(
            code="INVALID_RESOLUTION",
            message=str(e),
            suggestion="Check the grid resolution against the domain bounds and "
            "inventory tree heights.",
        ) from e

    requested_keys = [b["key"] for b in grid["bands"]]
    hr = dims["hr"]
    nx, ny, nz = dims["nx"], dims["ny"], dims["nz"]
    chunk_xy = min(max(1, int(voxelize.CHUNK_LENGTH_METERS / hr)), nx, ny)
    chunk_shape: tuple[int, int, int] = (nz, chunk_xy, chunk_xy)

    num_row_chunks = int(math.ceil(ny / chunk_xy))
    num_col_chunks = int(math.ceil(nx / chunk_xy))
    # 2x2 block-order: chunks in the same 2x2 neighborhood process together so
    # adjacent halos stay warm in the same batch's union read.
    chunk_locations = sorted(
        [(r, c) for r in range(num_row_chunks) for c in range(num_col_chunks)],
        key=lambda loc: (loc[0] // 2, loc[1] // 2, loc[0], loc[1]),
    )
    return GridLayout(
        dims=dims,
        requested_keys=requested_keys,
        chunk_xy=chunk_xy,
        chunk_shape=chunk_shape,
        chunk_locations=chunk_locations,
    )


def _prepare_tree_chunks(df: pd.DataFrame, layout: GridLayout) -> pd.DataFrame:
    """Attach cache keys and assign each tree to its (row_chunk, col_chunk)."""
    df = df.copy()
    df["_cache_key"] = voxelize.compute_cache_keys(df)
    df = voxelize.assign_trees_to_chunks(
        df,
        layout.dims["x_origin"],
        layout.dims["y_origin"],
        layout.dims["hr"],
        layout.dims["nx"],
        layout.dims["ny"],
        layout.chunk_xy,
    )
    return df


def _build_payloads(
    batch: list[tuple[int, int]],
    union_ds,
    union_y: slice,
    union_x: slice,
    df: pd.DataFrame,
    layout: GridLayout,
    source_config: dict,
    grid_id: str,
) -> list[dict]:
    """Split a halo-extended union Dataset into per-chunk worker payloads.

    Each payload carries numpy buffers and scalar grid params only (no xarray
    or custom objects) so it pickles cheaply into spawned workers.
    """
    dims = layout.dims
    ny, nx, nz = dims["ny"], dims["nx"], dims["nz"]
    hr, vr = dims["hr"], dims["vr"]
    x_origin, y_origin = dims["x_origin"], dims["y_origin"]

    payloads: list[dict] = []
    for row, col in batch:
        chunk_y, chunk_x = voxelize.chunk_slice(
            (row, col), ny, nx, layout.chunk_xy, overlap_cells=voxelize.OVERLAP_CELLS
        )
        # Relative slice into the union buffer.
        rel_y = slice(chunk_y.start - union_y.start, chunk_y.stop - union_y.start)
        rel_x = slice(chunk_x.start - union_x.start, chunk_x.stop - union_x.start)

        buffers: dict = {}
        for key in layout.requested_keys:
            dtype, fill = storage.BAND_SPECS[key]
            expected_shape = (
                nz,
                chunk_y.stop - chunk_y.start,
                chunk_x.stop - chunk_x.start,
            )
            existing = union_ds[key].values[:, rel_y, rel_x]
            buf = np.array(existing, dtype=dtype, copy=True)
            if buf.shape != expected_shape:
                resized = np.full(expected_shape, fill, dtype=dtype)
                z_n = min(expected_shape[0], buf.shape[0])
                y_n = min(expected_shape[1], buf.shape[1])
                x_n = min(expected_shape[2], buf.shape[2])
                resized[:z_n, :y_n, :x_n] = buf[:z_n, :y_n, :x_n]
                buf = resized
            buffers[key] = buf

        trees_in_chunk = df[(df["row_chunk"] == row) & (df["col_chunk"] == col)]
        rng_seed = abs(hash((grid_id, row, col))) & 0xFFFFFFFF

        payloads.append(
            {
                "chunk_location": (row, col),
                "buffers": buffers,
                "trees": trees_in_chunk,
                "hr": hr,
                "vr": vr,
                "x_origin": x_origin,
                "y_origin": y_origin,
                "source_config": source_config,
                "chunk_y_start": chunk_y.start,
                "chunk_x_start": chunk_x.start,
                "y_slice": chunk_y,
                "x_slice": chunk_x,
                "rng_seed": int(rng_seed),
            }
        )
    return payloads


def _process_batch(
    pool,
    batch: list[tuple[int, int]],
    df: pd.DataFrame,
    layout: GridLayout,
    source: dict,
    grid_id: str,
    path: str,
) -> None:
    """Run one batch: union-read → split payloads → pool.map → merge → write."""
    dims = layout.dims
    union_y, union_x = voxelize.batch_union_slices(
        batch,
        dims["ny"],
        dims["nx"],
        layout.chunk_xy,
        overlap_cells=voxelize.OVERLAP_CELLS,
    )
    union_ds = storage.read_union(path, union_y, union_x)
    assert not union_ds.chunks, (
        "read_union must materialize — workers cannot receive lazy dask arrays"
    )

    payloads = _build_payloads(
        batch, union_ds, union_y, union_x, df, layout, source, grid_id
    )
    results = pool.map(worker_run, payloads)

    for r in results:
        if "error" in r:
            raise ProcessingError(
                code="VOXELIZATION_FAILED",
                message=f"Chunk {r['chunk_location']} failed during voxelization.",
                suggestion="Check service logs for the worker traceback.",
                traceback=r["error"],
            )

    merged = storage.masked_merge(union_ds, results, union_y, union_x)
    storage.write_union(path, merged, union_y, union_x)


def _run_voxelization_batches(
    df: pd.DataFrame,
    layout: GridLayout,
    source: dict,
    grid_id: str,
    path: str,
    progress: Callable[[str, int | None], None],
) -> None:
    """Create one persistent Pool and drive every batch through `_process_batch`."""
    num_workers = _pick_worker_count()
    batch_size = max(1, num_workers)
    num_batches = max(1, int(math.ceil(len(layout.chunk_locations) / batch_size)))
    ctx = multiprocessing.get_context("spawn")

    # One Pool for all batches — per-batch spawning would re-import
    # fastfuels_core N times and dominate wall time.
    with ctx.Pool(processes=num_workers) as pool:
        for i in range(num_batches):
            batch = layout.chunk_locations[i * batch_size : (i + 1) * batch_size]
            if not batch:
                continue
            _process_batch(pool, batch, df, layout, source, grid_id, path)
            pct = 15 + int(75 * (i + 1) / num_batches)
            progress(f"Voxelizing batch {i + 1}/{num_batches}...", pct)


def _build_voxelization_result(layout: GridLayout, path: str) -> VoxelizationResult:
    dims = layout.dims
    georeference = {
        "crs": dims["crs"],
        "transform": list(dims["transform"]),
        "shape": [dims["nz"], dims["ny"], dims["nx"]],
        "z_origin": dims["z_origin"],
        "z_resolution": dims["vr"],
    }
    return VoxelizationResult(
        gcs_path=path,
        georeference=georeference,
        chunk_shape=list(layout.chunk_shape),
    )


# Job entry


def voxelize_inventory(
    grid: dict,
    domain_gdf,
    progress: Callable[[str, int | None], None],
) -> VoxelizationResult:
    """Voxelize a tree inventory into a 3D zarr grid on GCS.

    Stages (each testable in isolation):
      1. _load_inventory_dataframe — download + filter + id.
      2. _plan_grid_layout         — dims, chunk sizing, chunk order.
      3. storage.init_store        — write empty zarr metadata.
      4. _prepare_tree_chunks      — cache keys + per-chunk assignment + sort.
      5. _run_voxelization_batches — persistent Pool, batch loop.
      6. storage.consolidate_metadata + _build_voxelization_result.
    """
    grid_id = grid["id"]
    source = grid["source"]
    path = storage.gcs_path(grid_id)

    df = _load_inventory_dataframe(source, progress)

    progress("Computing grid extent...", 10)
    layout = _plan_grid_layout(grid, domain_gdf, df)

    progress("Initializing zarr store...", 15)
    storage.init_store(
        path,
        x_coords=layout.dims["x_coords"],
        y_coords=layout.dims["y_coords"],
        z_coords=layout.dims["z_coords"],
        hr=layout.dims["hr"],
        vr=layout.dims["vr"],
        crs=layout.dims["crs"],
        z_origin=layout.dims["z_origin"],
        requested_keys=layout.requested_keys,
        chunk_shape=layout.chunk_shape,
    )

    df = _prepare_tree_chunks(df, layout)
    _run_voxelization_batches(df, layout, source, grid_id, path, progress)

    progress("Finalizing...", 95)
    storage.consolidate_metadata(path)
    return _build_voxelization_result(layout, path)


# Dispatch


def dispatch_handler(
    grid: dict,
    domain_gdf,
    progress: Callable[[str, int | None], None],
) -> VoxelizationResult:
    """Route on `grid['source']['name']`. Single-source today; extensible later."""
    source_name = grid["source"]["name"]
    match source_name:
        case "inventory":
            return voxelize_inventory(grid, domain_gdf, progress)
        case _:
            raise ProcessingError(
                code="UNKNOWN_SOURCE",
                message=f"Unknown tree grid source: {source_name!r}",
                suggestion="Supported sources today: 'inventory'.",
            )
