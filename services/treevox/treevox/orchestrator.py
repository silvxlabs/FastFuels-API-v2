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
import time
import zlib
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import xarray as xr

from treevox import storage, voxelize
from treevox._worker import run as worker_run
from treevox.errors import ProcessingError
from treevox.inventory_io import assign_tree_ids, drop_null_rows, read_inventory

logger = logging.getLogger(__name__)

DEFAULT_MAX_WORKERS = 4
WORKER_MEMORY_ESTIMATE_BYTES = 500 * 1024 * 1024  # ~500 MB budget per worker


@dataclass
class BatchStats:
    """Per-job accumulator for timings and counts — mirrors v1's stats dict.

    Times are wall-clock seconds aggregated across batches. `num_trees` sums
    the tree count processed in each chunk's worker; `empty_chunks` counts
    chunks with zero trees dispatched (those still do a union read/write).
    """

    read_time: float = 0.0
    process_time: float = 0.0
    write_time: float = 0.0
    num_trees: int = 0
    empty_chunks: int = 0
    per_batch: list[dict] = field(default_factory=list)


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
    """Read the parquet from GCS, filter to live trees, and assign tree IDs.

    Reads directly from GCS (no tmpfile staging) with column projection and a
    `fia_status_code == 1` predicate pushdown — see `read_inventory`.
    """
    progress("Loading inventory...", 5)
    biomass_column = voxelize.foliage_inventory_column(source)
    crown_radius_column = voxelize.max_crown_radius_inventory_column(source)
    df = read_inventory(
        source["source_inventory_id"], biomass_column, crown_radius_column
    )
    df = drop_null_rows(df, biomass_column, crown_radius_column)
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
    res = grid["source"]["resolution"]
    if isinstance(res, dict):
        resolution_tuple = (res["horizontal"], res["horizontal"], res["vertical"])
    else:
        resolution_tuple = tuple(res)
    try:
        dims = voxelize.compute_grid_dimensions(domain_gdf, df, resolution_tuple)
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
    chunk_xy = min(voxelize.CHUNK_SIZE_HORIZONTAL, nx, ny)
    chunk_shape: tuple[int, int, int] = (nz, chunk_xy, chunk_xy)

    num_row_chunks = int(math.ceil(ny / chunk_xy))
    num_col_chunks = int(math.ceil(nx / chunk_xy))
    # 2x2 block-order: chunks in the same 2x2 neighborhood process together so
    # adjacent halos stay warm in the same batch's union read.
    chunk_locations = sorted(
        [(r, c) for r in range(num_row_chunks) for c in range(num_col_chunks)],
        key=lambda loc: (loc[0] // 2, loc[1] // 2, loc[0], loc[1]),
    )
    logger.info(
        f"Grid dimensions: {nz}x{ny}x{nx} (z, y, x); "
        f"chunk_xy={chunk_xy} ({num_row_chunks}x{num_col_chunks} chunks, "
        f"{num_row_chunks * num_col_chunks} total); "
        f"hr={hr}m, vr={dims['vr']}m",
        extra={"grid_id": grid["id"]},
    )
    return GridLayout(
        dims=dims,
        requested_keys=requested_keys,
        chunk_xy=chunk_xy,
        chunk_shape=chunk_shape,
        chunk_locations=chunk_locations,
    )


def _prepare_tree_chunks(
    df: pd.DataFrame, layout: GridLayout, source: dict | None = None
) -> pd.DataFrame:
    """Attach cache keys and assign each tree to its (row_chunk, col_chunk).

    Uses `DataFrame.assign` to add `_cache_key` without a full block-manager
    copy of the caller's frame; `assign_trees_to_chunks` then returns a fresh
    sorted DataFrame so the caller's input is never mutated.
    """
    df = df.assign(_cache_key=voxelize.compute_cache_keys(df, source))
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


def _chunk_relative_slices(
    chunk_y: slice,
    chunk_x: slice,
    union_y: slice,
    union_x: slice,
) -> tuple[slice, slice]:
    """Translate a chunk's absolute grid slices into union-relative indices.

    Inputs
    ------
    chunk_y, chunk_x
        Absolute y/x cell-index ranges for one chunk's halo-extended region
        in the full grid. Produced by `voxelize.chunk_slice`. E.g. for chunk
        (1, 0) at chunk_xy=1000 with 10-cell overlap: `slice(990, 2010)`.
    union_y, union_x
        Absolute y/x cell-index ranges for the halo-extended union covering
        every chunk in the current batch. Produced by
        `voxelize.batch_union_slices`. Guaranteed by construction to contain
        every chunk in the batch (the union is the outer bound of those
        chunks' halos).

    What it does
    ------------
    Subtracts `union.start` from both endpoints of the chunk slice on each
    axis, producing slices that can index directly into the numpy array that
    `storage.read_union` materialized for this union region.

    Validates the containment invariant (chunk ⊆ union). If it fails, the
    upstream `chunk_slice` and `batch_union_slices` implementations have
    drifted apart — there is no correct numpy slice to return, so we raise
    rather than silently produce a negative-start slice that numpy would
    turn into an empty array.

    Output
    ------
    `(rel_y, rel_x)`: slices into the union's numpy buffer. Each has
    `stop - start == chunk.stop - chunk.start`, so indexing preserves the
    chunk's halo-extended shape. Example: chunk=(990, 2010), union=(0, 2010)
    → `rel_y = slice(990, 2010)`.
    """
    if chunk_y.start < union_y.start or chunk_y.stop > union_y.stop:
        raise ProcessingError(
            code="BATCH_SLICE_MISMATCH",
            message=(
                f"Chunk y-slice {chunk_y} is not contained within union "
                f"y-slice {union_y}. chunk_slice and batch_union_slices "
                f"must share the same overlap_cells value."
            ),
        )
    if chunk_x.start < union_x.start or chunk_x.stop > union_x.stop:
        raise ProcessingError(
            code="BATCH_SLICE_MISMATCH",
            message=(
                f"Chunk x-slice {chunk_x} is not contained within union "
                f"x-slice {union_x}."
            ),
        )
    return (
        slice(chunk_y.start - union_y.start, chunk_y.stop - union_y.start),
        slice(chunk_x.start - union_x.start, chunk_x.stop - union_x.start),
    )


def _materialize_chunk_buffer(
    union_ds: xr.Dataset,
    key: str,
    rel_y: slice,
    rel_x: slice,
    expected_shape: tuple[int, int, int],
) -> np.ndarray:
    """Extract one band's slice from the union Dataset as a writable numpy buffer.

    Inputs
    ------
    union_ds
        In-memory xarray Dataset produced by `storage.read_union` — one
        variable per requested band, each shaped `(nz, union_y_span,
        union_x_span)`. Already `.load()`-ed (not lazy dask). Values reflect
        current on-disk state so prior-batch writes carry forward into this
        chunk's halo.
    key
        Band name matching a key in `storage.BAND_SPECS`
        (e.g. `"volume_fraction"`, `"tree_id"`). Determines output dtype
        and fill value.
    rel_y, rel_x
        Slices into `union_ds`'s numpy arrays covering this chunk's halo,
        from `_chunk_relative_slices`.
    expected_shape
        `(nz, chunk_y_span, chunk_x_span)` — the exact shape the worker
        expects for its chunk buffer. Derived from `chunk_y.stop -
        chunk_y.start` etc. in the caller.

    What it does
    ------------
    Slices the band variable out of `union_ds`, copies it into a fresh array
    coerced to the band's declared dtype from `BAND_SPECS` (so workers
    receive e.g. `int32` for `tree_id` even if the stored zarr variable drifts).
    When the slice's shape differs from `expected_shape`:

    - **smaller** on any axis (can happen at grid edges where the union was
      clipped at the boundary): pad the trailing cells with the band's fill
      value up to `expected_shape` and log a warning. Pad direction matters —
      only the trailing positions are padded so the chunk's origin cell
      remains aligned with the union's origin.
    - **larger** on any axis: raise `ProcessingError(UNION_SHAPE_MISMATCH)`.
      A larger slice means `_chunk_relative_slices` returned indices wider
      than `expected_shape`, which would require `chunk_slice` to be
      inconsistent with itself. Silently truncating would discard real data.

    Output
    ------
    `np.ndarray` of exactly `expected_shape` with the band's declared dtype.
    Writable (workers mutate in place). Cells outside the union-overlap
    region (for edge chunks) carry the band's fill value.
    """
    dtype, fill = storage.BAND_SPECS[key]
    existing = union_ds[key].values[:, rel_y, rel_x]
    buf = np.array(existing, dtype=dtype, copy=True)
    if buf.shape == expected_shape:
        return buf

    for axis, (got, want) in enumerate(zip(buf.shape, expected_shape)):
        if got > want:
            raise ProcessingError(
                code="UNION_SHAPE_MISMATCH",
                message=(
                    f"Band {key!r} union slice shape {buf.shape} exceeds "
                    f"expected chunk shape {expected_shape} on axis {axis}; "
                    f"refusing to truncate."
                ),
            )

    logger.warning(
        f"Band {key!r} union slice shape {buf.shape} smaller than expected "
        f"{expected_shape}; padding trailing cells with fill={fill!r}."
    )
    padded = np.full(expected_shape, fill, dtype=dtype)
    padded[: buf.shape[0], : buf.shape[1], : buf.shape[2]] = buf
    return padded


def _resolve_base_seed(source_config: dict, grid_id: str) -> int:
    """Extract the job-level base seed from the grid's source config.

    Inputs
    ------
    source_config
        The grid document's `source` sub-dict. Carries `seed: int` for grids
        created after the seed field was added to the API schema.
    grid_id
        Firestore document ID of the grid. Only used as a fallback input for
        legacy docs that predate the seed field.

    What it does
    ------------
    Returns `source_config["seed"]` when present. For legacy grid docs that
    lack the field, logs a warning and falls back to `zlib.crc32(grid_id)`
    so those jobs remain processable and still deterministic against their
    own `grid_id`. New grids always carry a seed — either user-supplied or
    API-autogenerated at create time.

    Output
    ------
    `int`. Fed into `_chunk_rng_seed` as the per-job base for every chunk's
    RNG seed.
    """
    seed = source_config.get("seed")
    if seed is not None:
        return int(seed)
    logger.warning(
        f"Grid {grid_id!r} source config has no `seed` field; falling back "
        f"to deterministic hash of grid_id. This grid was created before "
        f"`source.seed` was added to the API schema."
    )
    return zlib.crc32(grid_id.encode())


def _chunk_rng_seed(base_seed: int, row: int, col: int) -> int:
    """Deterministic 32-bit seed for one chunk's stochastic sampling.

    Inputs
    ------
    base_seed
        The job-level seed from `source.seed` (resolved by
        `_resolve_base_seed`). Same for every chunk in a given grid; makes
        the whole job re-run-reproducible against a single user-visible
        integer.
    row, col
        Chunk row/column indices within the grid.

    What it does
    ------------
    Combines the three components into a stable 32-bit integer via
    `zlib.crc32`. Unlike Python's built-in `hash()`, CRC32 is NOT affected
    by `PYTHONHASHSEED` randomization — the output is bit-identical across
    Python invocations, Cloud Function cold starts, and machine/platform
    boundaries.

    Output
    ------
    `int` in `[0, 2^32)`. Passed to `np.random.default_rng(seed=...)` inside
    the worker. The same `(base_seed, row, col)` triple always produces the
    same seed, so re-running a job with the same `source.seed` yields
    bit-identical voxelized output (given the same inventory).
    """
    return zlib.crc32(f"{base_seed}:{row}:{col}".encode())


def _build_payloads(
    batch: list[tuple[int, int]],
    union_ds,
    union_y: slice,
    union_x: slice,
    df: pd.DataFrame,
    chunk_indices: dict,
    layout: GridLayout,
    source_config: dict,
    grid_id: str,
) -> list[dict]:
    """Split a halo-extended union Dataset into per-chunk worker payloads.

    Each payload carries numpy buffers and scalar grid params only (no xarray
    or custom objects) so it pickles cheaply into spawned workers.

    `chunk_indices` maps `(row, col) -> np.ndarray[int64]` of row positions
    into `df` (built once in `_run_voxelization_batches` via
    `df.groupby([...]).indices`). We look up each chunk's trees in O(k)
    instead of materializing a full-length boolean mask per chunk per batch.
    """
    dims = layout.dims
    ny, nx, nz = dims["ny"], dims["nx"], dims["nz"]
    hr, vr = dims["hr"], dims["vr"]
    x_origin, y_origin = dims["x_origin"], dims["y_origin"]

    base_seed = _resolve_base_seed(source_config, grid_id)

    payloads: list[dict] = []
    for row, col in batch:
        chunk_y, chunk_x = voxelize.chunk_slice(
            (row, col), ny, nx, layout.chunk_xy, overlap_cells=voxelize.OVERLAP_CELLS
        )
        rel_y, rel_x = _chunk_relative_slices(chunk_y, chunk_x, union_y, union_x)
        expected_shape = (
            nz,
            chunk_y.stop - chunk_y.start,
            chunk_x.stop - chunk_x.start,
        )
        buffers = {
            key: _materialize_chunk_buffer(union_ds, key, rel_y, rel_x, expected_shape)
            for key in layout.requested_keys
        }

        indices = chunk_indices.get((row, col))
        trees_in_chunk = df.iloc[indices] if indices is not None else df.iloc[0:0]

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
                "rng_seed": _chunk_rng_seed(base_seed, row, col),
            }
        )
    return payloads


def _process_batch(
    pool,
    batch: list[tuple[int, int]],
    batch_idx: int,
    num_batches: int,
    df: pd.DataFrame,
    chunk_indices: dict,
    layout: GridLayout,
    source: dict,
    grid_id: str,
    path: str,
) -> dict:
    """Run one batch: union-read → split payloads → pool.map → merge → write.

    Logs three per-batch phase timings (read / process / write) and one
    line per chunk with worker pid + tree count — matches v1's log shape
    so profiling across versions is directly comparable.
    """
    dims = layout.dims
    union_y, union_x = voxelize.batch_union_slices(
        batch,
        dims["ny"],
        dims["nx"],
        layout.chunk_xy,
        overlap_cells=voxelize.OVERLAP_CELLS,
    )

    read_start = time.monotonic()
    union_ds = storage.read_union(path, union_y, union_x)
    assert not union_ds.chunks, (
        "read_union must materialize — workers cannot receive lazy dask arrays"
    )
    read_time = time.monotonic() - read_start
    logger.info(
        f"Combined reading of batch {batch_idx}/{num_batches} completed in "
        f"{read_time:.2f}s",
        extra={"grid_id": grid_id},
    )

    payloads = _build_payloads(
        batch, union_ds, union_y, union_x, df, chunk_indices, layout, source, grid_id
    )
    num_trees = sum(len(p["trees"]) for p in payloads)
    empty_chunks = sum(1 for p in payloads if len(p["trees"]) == 0)

    process_start = time.monotonic()
    results = pool.map(worker_run, payloads)
    process_time = time.monotonic() - process_start

    for r in results:
        if "error" in r:
            if r.get("error_code") == "BIOMASS_COMPONENT_NOT_IMPLEMENTED":
                raise ProcessingError(
                    code=r["error_code"],
                    message=r.get(
                        "error_message",
                        "Treevox does not yet support this biomass component.",
                    ),
                    suggestion=(
                        "Request foliage biomass outputs for now, or wait for "
                        "fastfuels-core branchwood/fine component support."
                    ),
                    traceback=r["error"],
                )
            raise ProcessingError(
                code="VOXELIZATION_FAILED",
                message=f"Chunk {r['chunk_location']} failed during voxelization.",
                suggestion="Check service logs for the worker traceback.",
                traceback=r["error"],
            )
        logger.info(
            f"Process {r.get('pid', '?')}: batch {batch_idx}, chunk {r['chunk_location']} with "
            f"{r.get('num_trees', 0)} trees completed in "
            f"{r.get('process_time_s', 0.0):.2f}s",
            extra={"grid_id": grid_id},
        )
    logger.info(
        f"Parallel processing of batch {batch_idx}/{num_batches} completed in "
        f"{process_time:.2f}s",
        extra={"grid_id": grid_id},
    )

    write_start = time.monotonic()
    merged = storage.masked_merge(union_ds, results, union_y, union_x)
    storage.write_union(path, merged, union_y, union_x)
    write_time = time.monotonic() - write_start
    logger.info(
        f"Combined writing of batch {batch_idx}/{num_batches} completed in "
        f"{write_time:.2f}s",
        extra={"grid_id": grid_id},
    )

    return {
        "read_time": read_time,
        "process_time": process_time,
        "write_time": write_time,
        "num_trees": num_trees,
        "empty_chunks": empty_chunks,
        "num_chunks": len(batch),
    }


def _run_voxelization_batches(
    df: pd.DataFrame,
    layout: GridLayout,
    source: dict,
    grid_id: str,
    path: str,
    progress: Callable[[str, int | None], None],
) -> BatchStats:
    """Create one persistent Pool and drive every batch through `_process_batch`.

    `chunk_indices` is precomputed once via `groupby(...).indices` so each
    chunk lookup in `_build_payloads` is O(k) instead of allocating a
    full-length boolean mask per chunk per batch.

    Returns accumulated `BatchStats` for the whole voxelization job so the
    caller can log totals. Per-batch timings are also logged as they happen.
    """
    chunk_indices = df.groupby(["row_chunk", "col_chunk"], sort=False).indices

    num_workers = _pick_worker_count()
    batch_size = max(1, num_workers)
    total_chunks = len(layout.chunk_locations)
    num_batches = max(1, int(math.ceil(total_chunks / batch_size)))
    ctx = multiprocessing.get_context("spawn")

    logger.info(
        f"Processing {total_chunks} chunks in {num_batches} batches with "
        f"{num_workers} parallel workers (batch_size={batch_size})",
        extra={"grid_id": grid_id},
    )

    stats = BatchStats()

    # One Pool for all batches — per-batch spawning would re-import
    # fastfuels_core N times and dominate wall time.
    with ctx.Pool(processes=num_workers) as pool:
        for i in range(num_batches):
            batch = layout.chunk_locations[i * batch_size : (i + 1) * batch_size]
            if not batch:
                continue

            logger.info(
                f"Starting batch {i + 1}/{num_batches} with {len(batch)} chunks",
                extra={"grid_id": grid_id},
            )
            batch_stats = _process_batch(
                pool,
                batch,
                i + 1,
                num_batches,
                df,
                chunk_indices,
                layout,
                source,
                grid_id,
                path,
            )

            stats.read_time += batch_stats["read_time"]
            stats.process_time += batch_stats["process_time"]
            stats.write_time += batch_stats["write_time"]
            stats.num_trees += batch_stats["num_trees"]
            stats.empty_chunks += batch_stats["empty_chunks"]
            stats.per_batch.append(batch_stats)

            pct = 15 + int(75 * (i + 1) / num_batches)
            progress(f"Voxelizing batch {i + 1}/{num_batches}...", pct)

    return stats


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

    job_start = time.monotonic()
    logger.info(
        f"Starting voxelization for grid {grid_id}",
        extra={"grid_id": grid_id},
    )

    df = _load_inventory_dataframe(source, progress)
    logger.info(
        f"Inventory loaded: {len(df)} trees",
        extra={"grid_id": grid_id},
    )

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

    df = _prepare_tree_chunks(df, layout, source)
    stats = _run_voxelization_batches(df, layout, source, grid_id, path, progress)

    progress("Finalizing...", 95)
    # Metadata was consolidated at init_store time; region writes during the
    # batch loop only modify data chunks, so no reconsolidation is needed.

    total_time = time.monotonic() - job_start
    total_chunks = len(layout.chunk_locations)
    phase_total = stats.read_time + stats.process_time + stats.write_time
    extra = {"grid_id": grid_id}
    logger.info(f"Read time: {stats.read_time:.2f} seconds", extra=extra)
    logger.info(f"Processing time: {stats.process_time:.2f} seconds", extra=extra)
    logger.info(f"Write time: {stats.write_time:.2f} seconds", extra=extra)
    logger.info(f"Total trees processed: {stats.num_trees}", extra=extra)
    logger.info(
        f"Empty chunks: {stats.empty_chunks} out of {total_chunks}", extra=extra
    )
    logger.info(f"Total phase time: {phase_total:.2f} seconds", extra=extra)
    logger.info(f"Voxelization job completed in {total_time:.2f} seconds", extra=extra)

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
