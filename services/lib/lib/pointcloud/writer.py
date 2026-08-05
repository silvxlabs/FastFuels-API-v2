"""Spatially-partitioned, LOD-tagged Parquet.

The output is a Hive-partitioned dataset -- ``tile_x=<i>/tile_y=<j>/part-*.parquet``
plus a ``_metadata`` footer and a ``_manifest.json`` -- rather than one file. A
reader that wants a box opens ``_metadata``, which carries every row group's
statistics, and touches only the parts that overlap.

Each part is written one row group per LOD level, so a ``lod <= k`` filter prunes
on row-group statistics. Writing a single row group spanning every level made the
pyramid correct but useless: pushdown prunes row groups, not rows.

The LOD is a real pyramid: level k's cell is ``extent / (2**k * CELL_COUNT)``,
the union of every level-k COPC node's sampling grid. Cells are voxels, not
columns -- see LodAssigner.

Coordinates stay as LAS scaled int32s with scale/offset in the file metadata,
which keeps them small and lossless rather than exploding to float64.

Work is spread across processes, not threads. The encode is GIL-bound: measured
on threads in the parent, Arrow encode plus zstd took 38% of the job while
leaving most cores idle.

The layout this writes is defined in `lib.pointcloud.schema`, and read back by
`lib.pointcloud.reader`.
"""

import io
import json
import math
import multiprocessing
import threading

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from lib.config import LAKITU_WRITE_QUEUE_DEPTH, LAKITU_WRITE_WORKERS
from lib.pointcloud.schema import (
    CELL_COUNT,
    DELTA_COLUMNS,
    DICT_COLUMNS,
    LOD_LEVELS,
    choose_tile_m,
    columns_for,
)

# How much routed-but-unflushed point data the parent holds, and how big a tile
# has to be before it is worth flushing. File size is roughly
# BUFFER_BUDGET / active tiles, so the budget is what sets it.
BUFFER_BUDGET = 192 << 20
MIN_FLUSH_BYTES = 32 << 20  # prefer flushing tiles that have grown big
MAX_TILE_BYTES = 96 << 20  # flush a tile at this size regardless of budget


# Cells per level per tile. Voxels cost side**2 * nz against a column's side**2,
# so this is what stops the deep levels exploding -- at a 500 m tile, 4 km extent
# and 300 m of relief, level 3 wants 1.26M cells and level 4 wants 10.1M. Past
# the budget a level keeps as many z bins as fit instead of going flat.
#
# It binds the int32 claim scratch at 4 bytes per cell, more tightly than the
# bool grids, so the claim is shared across every tile a worker owns rather than
# allocated per tile: 4.2 MB per worker, not per tile.
#
# 1 << 20 keeps a tile's grids near 3.3 MB against 0.35 MB for flat columns.
# Raising it mostly buys z bins at levels 4-5, where the XY cell is already
# sub-metre and vertical collapse is barely visible; 1 << 22 cost 9.4 MB a tile.
GRID_CELL_BUDGET = 1 << 20


class GcsSink:
    """Writes objects under a prefix. Rebuilt per process, never pickled.

    Deliberately not `lib.gcs`, which wraps a module-level gcsfs client: this is
    constructed inside forkserver children, where a fresh
    `google.cloud.storage.Client` is the safe thing to hold.
    """

    def __init__(self, bucket, prefix):
        from google.cloud import storage

        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket)
        self.prefix = prefix.rstrip("/")
        self.bytes_written = 0

    def put(self, name, data):
        blob = self.bucket.blob(f"{self.prefix}/{name}")
        blob.upload_from_string(data, content_type="application/octet-stream")
        self.bytes_written += len(data)


class LodAssigner:
    """Per-tile LOD occupancy grids, sort-free.

    Cell size stays the GLOBAL one -- extent/(2**k * CELL_COUNT) -- so levels
    mean the same thing everywhere; only the extent covered is tile-local. Tile
    boundaries fall on cell boundaries at every level (a 500 m tile is exactly 16
    level-0 cells at a 4 km extent), so a tile-local grid is a clean subset of
    the global one and the pyramid is unchanged.

    Cells are cubic voxels, not columns, on every level whose grid fits
    GRID_CELL_BUDGET. A column would put exactly one point in a 31 x 31 m cell at
    level 0 and pick it by EPT arrival order, which is unrelated to height, so a
    coarse preview of a forest was an arbitrary vertical draw rather than
    ground-plus-canopy -- measured on a real tile, level 0's Z spread was *wider*
    than the full cloud's. It also made the pyramid a quadtree, where COPC's is
    an octree, so the levels did not mean what a COPC-shaped viewer expects.

    Making this per-tile is what lets LOD assignment leave the main process:
    tiles are independent, so their assignments share no mutable state.

    First-writer-wins scatter: claim slots in a scratch array, gather to see who
    won, reset only the touched entries.
    """

    def __init__(
        self, origin, tile_m, extent, z_origin, z_span, levels=LOD_LEVELS, claim=None
    ):
        self.origin = np.asarray(origin, dtype=float)
        self.z_origin = float(z_origin)
        self.extent = float(extent)
        self.levels = levels
        self.cell = [self.extent / (CELL_COUNT * (2**k)) for k in range(levels)]
        self.side = [max(1, int(round(tile_m / c))) for c in self.cell]
        # Cubic voxels where the budget allows, and where it does not, as many
        # bins as fit rather than falling back to a flat column. Collapsing to
        # 2D was the first cut and it was too brittle: z_span is the raw header
        # range, so a handful of noise points hundreds of metres up would push a
        # level over budget and cost it every bit of vertical structure. Taller
        # bins still separate ground from canopy; one bin does not.
        self.nz = [
            max(1, min(int(math.ceil(z_span / c)), GRID_CELL_BUDGET // (s * s)))
            for s, c in zip(self.side, self.cell)
        ]
        # Bin height, which equals the cell size until the budget caps a level.
        self.zcell = [z_span / n if n > 1 else float("inf") for n in self.nz]
        self.grids = [
            np.zeros(s * s * n, dtype=bool) for s, n in zip(self.side, self.nz)
        ]
        # Reusable across tiles: it is scratch, reset before it is handed back,
        # and every tile in a job shares one tile_m/extent/z_span so one size
        # fits all. Callers that assign tiles concurrently must not share it.
        need = max(s * s * n for s, n in zip(self.side, self.nz))
        self.claim = (
            claim
            if claim is not None and claim.size >= need
            else np.full(need, -1, dtype=np.int32)
        )

    def assign(self, x, y, z):
        lod = np.full(len(x), self.levels, dtype=np.uint8)
        remaining = np.arange(len(x), dtype=np.int64)
        claim = self.claim
        for k in range(self.levels):
            if remaining.size == 0:
                break
            side = self.side[k]
            nz = self.nz[k]
            w = self.cell[k]
            xi = np.clip(
                ((x[remaining] - self.origin[0]) / w).astype(np.int64), 0, side - 1
            )
            yi = np.clip(
                ((y[remaining] - self.origin[1]) / w).astype(np.int64), 0, side - 1
            )
            flat = xi * side + yi
            del xi, yi
            if nz > 1:
                zi = np.clip(
                    ((z[remaining] - self.z_origin) / self.zcell[k]).astype(np.int64),
                    0,
                    nz - 1,
                )
                flat = flat * nz + zi
                del zi
            grid = self.grids[k]
            cand = np.flatnonzero(~grid[flat])
            if cand.size:
                f = flat[cand]
                n = f.size
                claim[f[::-1]] = np.arange(n - 1, -1, -1, dtype=np.int32)
                won = claim[f] == np.arange(n, dtype=np.int32)
                claim[f] = -1
                winners = cand[won]
                lod[remaining[winners]] = k
                grid[f[won]] = True
                keep = np.ones(remaining.size, dtype=bool)
                keep[winners] = False
                remaining = remaining[keep]
        return lod


def _file_metadata(data, path):
    """FileMetaData for the just-written bytes, tagged with its dataset path."""
    md = pq.ParquetFile(io.BytesIO(data)).metadata
    md.set_file_path(path)
    return md


def _encode(records, lod, scales, offsets):
    """One row group per LOD level, X/Y-sorted within each.

    Sorts on a single packed int64 key per level instead of a 3-key lexsort;
    tile-local millimetre coordinates fit well inside 21 bits each.

    Morton (Z-order) interleaving was tried, to make X and Y both locally
    coherent rather than leaving X near-sorted and Y sawtoothing. It was worth
    3.6% while gps_time was still stored, but only 1.3% without it, against 62%
    more encode time for the bit-spreading -- a bad trade in a path where worker
    backpressure is what binds.
    """
    # Derived from the records rather than fixed, so a cloud with colour and one
    # without each write exactly the columns they have.
    columns = columns_for(records.dtype)
    buf = io.BytesIO()
    writer = None
    try:
        for level in range(LOD_LEVELS + 1):
            sel = np.flatnonzero(lod == level)
            if sel.size == 0:
                continue
            r = records[sel]
            dx = (r["X"] - r["X"].min()).astype(np.int64)
            dy = (r["Y"] - r["Y"].min()).astype(np.int64)
            r = r[np.argsort((dx << 21) | dy, kind="stable")]
            arrays = []
            for c in columns:
                col = (
                    np.full(sel.size, level, dtype=np.uint8)
                    if c == "lod"
                    else np.ascontiguousarray(r[c])
                )
                arrays.append(pa.array(col))
            del r
            table = pa.Table.from_arrays(arrays, names=list(columns))
            del arrays
            if writer is None:
                meta = {
                    b"scales": json.dumps(list(map(float, scales))).encode(),
                    b"offsets": json.dumps(list(map(float, offsets))).encode(),
                }
                table = table.replace_schema_metadata(meta)
                writer = pq.ParquetWriter(
                    buf,
                    table.schema,
                    compression="zstd",
                    use_dictionary=[c for c in DICT_COLUMNS if c in columns],
                    column_encoding=DELTA_COLUMNS,
                    write_statistics=True,
                )
            writer.write_table(table, row_group_size=table.num_rows)
            del table
    finally:
        if writer is not None:
            writer.close()
    return buf.getvalue()


# Per-process state for the write workers, set once by the initializer so each
# task carries only its points.
_W = {}


def _worker_init(bucket, prefix, scales, offsets):
    # An exception here kills the worker and the parent would only see a dead
    # queue, so say what actually failed before going down.
    try:
        _W["sink"] = GcsSink(bucket, prefix)
        _W["scales"] = np.asarray(scales)
        _W["offsets"] = np.asarray(offsets)
    except BaseException:
        import sys
        import traceback

        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        raise


def _pin(tile, n):
    """Which worker owns a tile. Any stable spread works; this is cheap."""
    return ((tile[0] * 73856093) ^ (tile[1] * 19349663)) % n


def _write_worker(in_q, out_q, init_args, grid_args):
    """Own a subset of tiles end to end: LOD, encode, compress, upload.

    A tile's LOD grids have to survive its repeated flushes, which is why this
    exists instead of a plain ProcessPoolExecutor: tasks are routed by `_pin`, so
    every flush of a tile reaches the same process and finds its grids. One FIFO
    queue per worker also preserves flush order within a tile, which
    first-writer-wins assignment depends on.
    """
    try:
        _worker_init(*init_args)
        mins, tile_m, extent, z_span = grid_args
        scales, offsets = _W["scales"], _W["offsets"]
        assigners = {}
        shared_claim = None
        while True:
            item = in_q.get()
            if item is None:
                break
            tile, recs, rel = item
            try:
                assigner = assigners.get(tile)
                if assigner is None:
                    origin = (mins[0] + tile[0] * tile_m, mins[1] + tile[1] * tile_m)
                    assigner = assigners[tile] = LodAssigner(
                        origin, tile_m, extent, mins[2], z_span, claim=shared_claim
                    )
                    # Safe here and only here: this loop assigns one tile at a
                    # time, so no two assigners touch the scratch at once.
                    shared_claim = assigner.claim
                x = recs["X"] * scales[0] + offsets[0]
                y = recs["Y"] * scales[1] + offsets[1]
                z = recs["Z"] * scales[2] + offsets[2]
                lod = assigner.assign(x, y, z)
                del x, y, z
                data = _encode(recs, lod, scales, offsets)
                _W["sink"].put(rel, data)
                md = _file_metadata(data, rel)
                buf = io.BytesIO()
                # FileMetaData does not pickle; the parent revives the bytes.
                md.write_metadata_file(buf)
                out_q.put(("ok", len(data), buf.getvalue(), rel))
            except BaseException as e:
                import traceback

                out_q.put(("err", repr(e), traceback.format_exc(), rel))
    except BaseException as e:
        import traceback

        out_q.put(("err", repr(e), traceback.format_exc(), "<worker init>"))
    finally:
        out_q.put(None)


class _PinnedPool:
    """Tile-pinned worker processes with a collector thread.

    The parent must never block on a full input queue while nobody drains the
    output queue, so collection runs on its own thread. Input queues are shallow
    on purpose: they are the backpressure.
    """

    def __init__(self, n, ctx, init_args, grid_args, on_result, depth):
        self.n = n
        self.out_q = ctx.Queue()
        self.in_qs = [ctx.Queue(maxsize=depth) for _ in range(n)]
        self.procs = [
            ctx.Process(
                target=_write_worker,
                args=(self.in_qs[i], self.out_q, init_args, grid_args),
                daemon=True,
            )
            for i in range(n)
        ]
        for p in self.procs:
            p.start()
        self.error = None
        self._on_result = on_result
        self._done = 0
        self._collector = threading.Thread(target=self._collect, daemon=True)
        self._collector.start()

    def _collect(self):
        while self._done < self.n:
            item = self.out_q.get()
            if item is None:
                self._done += 1
                continue
            if item[0] == "err":
                if self.error is None:
                    self.error = RuntimeError(f"{item[3]}: {item[1]}\n{item[2]}")
                continue
            self._on_result(item[1], item[2])

    def submit(self, tile, recs, rel):
        if self.error is not None:
            raise self.error
        self.in_qs[_pin(tile, self.n)].put((tile, recs, rel))

    def close(self):
        for q in self.in_qs:
            q.put(None)
        self._collector.join()
        for p in self.procs:
            p.join(timeout=30)
        if self.error is not None:
            raise self.error


def write_parquet(
    records,
    info,
    bucket,
    prefix,
    *,
    tile_m=None,
    workers=LAKITU_WRITE_WORKERS,
    depth=LAKITU_WRITE_QUEUE_DEPTH,
):
    """Route records into tiles and write a partitioned Parquet dataset.

    Args:
        records: Iterable of PDRF-6 structured arrays, in any order.
        info: Source bounds and scaling -- ``mins``, ``maxs``, ``scales``,
            ``offsets``, each a length-3 sequence.
        bucket: Destination GCS bucket name.
        prefix: Destination prefix within the bucket. The dataset is written
            directly under it.
        tile_m: Tile edge in metres. Defaults to `choose_tile_m`.
        workers: Write processes. Must not exceed the vCPU allocation --
            oversubscribing measured 2.4x the CPU for identical output.
        depth: Per-worker queue depth, which is the parent's backpressure.

    Returns:
        Dict with ``points``, ``tiles``, ``files`` and ``output_bytes``.

    Raises:
        RuntimeError: If any write worker failed.
    """
    mins, maxs = np.asarray(info["mins"]), np.asarray(info["maxs"])
    scales, offsets = np.asarray(info["scales"]), np.asarray(info["offsets"])

    extent = float(max(maxs[:2] - mins[:2]))
    if extent <= 0:
        # Every point at one location, or a single point. An upload can be
        # either. The pyramid means nothing over no area, so any positive extent
        # will do; this keeps the cell arithmetic from dividing by zero.
        extent = 1.0
    z_span = float(maxs[2] - mins[2])
    if tile_m is None:
        tile_m = choose_tile_m(extent)
    n_tx = int(np.ceil((maxs[1] - mins[1]) / tile_m)) + 1

    buffers, sizes, counts, nparts = {}, {}, {}, {}
    buffered = 0
    total = 0
    lock = threading.Lock()
    stats = {"written_bytes": 0, "files": 0}
    footers = []

    def on_result(nbytes_out, md_bytes):
        md = pq.read_metadata(io.BytesIO(md_bytes))
        with lock:
            stats["written_bytes"] += nbytes_out
            stats["files"] += 1
            footers.append(md)

    # forkserver, not fork: this process has a collector thread, and forking a
    # threaded process can inherit a lock held by a thread the child lacks.
    pool = _PinnedPool(
        workers,
        multiprocessing.get_context("forkserver"),
        (bucket, prefix, scales, offsets),
        (mins, tile_m, extent, z_span),
        on_result,
        depth,
    )

    def flush(tile):
        nonlocal buffered
        parts = buffers.pop(tile, None)
        if not parts:
            return
        buffered -= sizes.pop(tile)
        seq = nparts.get(tile, 0)
        nparts[tile] = seq + 1
        rel = f"tile_x={tile[0]}/tile_y={tile[1]}/part-{seq:05d}.parquet"
        # Blocking here is the parent waiting on this tile's owner, which is the
        # intended backpressure.
        pool.submit(tile, np.concatenate(parts), rel)

    try:
        for out in records:
            x = out["X"] * scales[0] + offsets[0]
            y = out["Y"] * scales[1] + offsets[1]
            tx = np.maximum(((x - mins[0]) / tile_m).astype(np.int32), 0)
            ty = np.maximum(((y - mins[1]) / tile_m).astype(np.int32), 0)
            tid = tx * n_tx + ty
            # Small bounded range -> numpy radix-sorts int16 under kind="stable".
            order = (
                np.argsort(tid.astype(np.int16), kind="stable")
                if tid.max() < 32767
                else np.argsort(tid, kind="stable")
            )
            sids = tid[order]
            edges = np.flatnonzero(np.diff(sids)) + 1
            for s, e in zip(
                np.concatenate([[0], edges]), np.concatenate([edges, [sids.size]])
            ):
                sel = order[s:e]
                key = (int(tx[sel[0]]), int(ty[sel[0]]))
                part = out[sel]
                buffers.setdefault(key, []).append(part)
                sizes[key] = sizes.get(key, 0) + part.nbytes
                counts[key] = counts.get(key, 0) + part.size
                buffered += part.nbytes
            for big_tile in [k for k, n in sizes.items() if n >= MAX_TILE_BYTES]:
                flush(big_tile)
            while buffered > BUFFER_BUDGET:
                big = [t for t, n in sizes.items() if n >= MIN_FLUSH_BYTES]
                flush(max(big, key=sizes.get) if big else max(sizes, key=sizes.get))
            total += out.size
        for tile in list(buffers):
            flush(tile)
    finally:
        pool.close()

    manifest = {
        "tiles": len(counts),
        "tile_m": tile_m,
        "points": total,
        "lod_levels": LOD_LEVELS,
        "n_tx": n_tx,
        "scales": list(map(float, scales)),
        "offsets": list(map(float, offsets)),
        "mins": list(map(float, mins)),
        "maxs": list(map(float, maxs)),
    }
    sink = GcsSink(bucket, prefix)
    sink.put("_manifest.json", json.dumps(manifest).encode())
    if footers:
        combined = footers[0]
        for md in footers[1:]:
            combined.append_row_groups(md)
        mbuf = io.BytesIO()
        combined.write_metadata_file(mbuf)
        sink.put("_metadata", mbuf.getvalue())

    return {
        "points": total,
        "tiles": len(counts),
        "files": stats["files"],
        "output_bytes": stats["written_bytes"] + sink.bytes_written,
    }
