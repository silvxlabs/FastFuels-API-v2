"""Spatially-partitioned, LOD-tagged Parquet.

The output is a Hive-partitioned dataset -- ``tile_x=<i>/tile_y=<j>/part-*.parquet``
plus a ``_metadata`` footer and a ``_manifest.json`` -- rather than one file. A
reader that wants a box opens ``_metadata``, which carries every row group's
statistics, and touches only the parts that overlap.

Each part is written one row group per LOD level, so a ``lod <= k`` filter prunes
on row-group statistics. Writing a single row group spanning every level made the
pyramid correct but useless: pushdown prunes row groups, not rows.

The LOD is a stride: level k holds 1 in 4**(L-1-k) of a tile's points, so
``lod <= k`` is a nested, unbiased subsample and the deepest level is the whole
tile -- see assign_lod.

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
import multiprocessing
import threading

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from lib.config import LAKITU_WRITE_QUEUE_DEPTH, LAKITU_WRITE_WORKERS
from lib.pointcloud.schema import (
    DELTA_COLUMNS,
    DICT_COLUMNS,
    LOD_LEVELS,
    choose_tile_m,
    columns_for,
)

# How much routed-but-unflushed point data the parent holds, and how big a tile
# has to be before it is worth flushing.
#
# A tile is written once per flush, so file size is roughly
# BUFFER_BUDGET / active tiles. Two things set the count: how many tiles the
# arriving points touch at once, and how much the parent may hold. Feeding
# points in spatial order took 16 km2 from 464 files to 272 on its own.
#
# Raising the budget buys fewer files and costs some wall clock. Measured with
# the network removed and three runs per budget, at the real 343.6M-point scale:
# 384 MiB gave +3.8% wall, 768 MiB gave +15.0%, against a 3-6% spread within a
# budget. An earlier single-run pass reported +23% and +36% for the same two,
# which was the harness rather than the writer.
#
# The cost is ramp-up and tail, not memory pressure: the pool idles while the
# parent fills the first buffer and the parent idles while the pool drains the
# last one. Both are fixed, so the penalty falls as the job grows -- 768 MiB cost
# +30.9% over 60M points and +15.0% over 343.6M. A larger domain can afford a
# larger budget, and flushing a tile once the locality sweep has passed it would
# retire the trade-off rather than tune it.
BUFFER_BUDGET = 192 << 20
MAX_TILE_BYTES = 96 << 20  # flush a tile at this size regardless of budget


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


def assign_lod(count, levels=LOD_LEVELS):
    """Assign each point a pyramid level by stride: level k keeps 1 in 4**(L-1-k).

    Levels are nested -- ``lod <= k`` is a 1-in-4**(L-1-k) sample of the tile,
    and each level is a strict superset of the one above -- so a reader gets a
    geometric ladder of point counts and the deepest level is the whole tile.
    There is no residual: every point belongs to a level.

    Deliberately not voxel sampling. An occupancy grid keeps at most one point
    per cell, which equalises density, but measured against this on a real
    7.8M-point tile the difference is not worth what it costs:

    * both cover 100% of the tile's occupied 15 m cells at every level (99.8% at
      the very coarsest, 7,663 points), so there are no holes to fix;
    * a stride is an unbiased sample, so a preview's class mix and height
      distribution match the full cloud exactly -- ground stays at 30.0% at every
      level, where voxel sampling drifted it to 35.2%;
    * a grid costs `side**2 * nz` cells that have to persist for the life of a
      tile, because a tile is flushed many times. Sized to reach point spacing
      over a 500 m tile that is 25 MB per tile, or 6.4 GB across a 64 km2
      domain. A stride costs nothing and needs no state at all.

    The order this strides is the order points arrived, which is not spatially
    sorted. That is the pessimistic case and it was measured that way.

    Args:
        count: Number of points to assign.
        levels: Number of pyramid levels; level ``levels - 1`` holds everything.

    Returns:
        uint8 array of levels, one per point.
    """
    lod = np.full(count, levels - 1, dtype=np.uint8)
    index = np.arange(count)
    # Coarse last so it wins: a point taken by level k must not be re-taken by a
    # finer level, which is what makes the levels nested.
    for k in range(levels - 2, -1, -1):
        lod[index % (4 ** (levels - 1 - k)) == 0] = k
    return lod


def _file_metadata(data, path):
    """FileMetaData for the just-written bytes, tagged with its dataset path."""
    md = pq.ParquetFile(io.BytesIO(data)).metadata
    md.set_file_path(path)
    return md


# Cells per axis in the sort key. Must keep the packed key inside a uint16:
# above 256 numpy stops radix-sorting it and the sort costs what an exact one
# does. See _grid_key.
_SORT_CELLS = 256


def _grid_key(xs, ys):
    """Each point's cell in a _SORT_CELLS square over the extent, packed to uint16."""

    def cell(v):
        lo, hi = int(v.min()), int(v.max())
        # The +1 puts the maximum in the last cell rather than one past the end.
        return (v - lo).astype(np.uint64) * _SORT_CELLS // (hi - lo + 1)

    return (cell(xs) * _SORT_CELLS + cell(ys)).astype(np.uint16)


def _encode(records, lod, scales, offsets):
    """One row group per LOD level, spatially ordered within each.

    The ordering is for compression alone. Row-group statistics are min/max,
    which no permutation changes, so pushdown gets nothing from it; what it buys
    is DELTA_BINARY_PACKED on X/Y/Z, worth 16% of the file on a real dense tile.

    Points are ordered by their cell in a coarse grid over the level's extent.
    Two things make that the right key. It fits a uint16, which numpy radix-sorts
    in a single pass instead of falling back to merge sort, and it keeps X and Y
    coherent together -- sorting on a packed (x, y) key orders by X and leaves Y
    sawtoothing inside each run. Measured against that packed key on a real tile:
    45% less CPU for 0.9% more file.

    That is the Z-order benefit without Z-order's cost. Morton interleaving was
    tried directly and rejected -- 62% more encode time for the bit-spreading.
    Finer grids compress better still (512 cells gave 4.9% less file) but need a
    uint32 key, which hands the radix sort back.
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
            r = r[np.argsort(_grid_key(r["X"], r["Y"]), kind="stable")]
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


def _write_worker(in_q, out_q, init_args):
    """Take any flush off the shared queue and carry it to GCS: LOD, encode, upload.

    Workers used to own a fixed subset of tiles, because a tile's LOD grids had
    to survive its repeated flushes. A stride carries no state between flushes,
    so nothing is owned and every worker draws from one queue. That matters once
    points arrive in spatial order: consecutive flushes are then neighbouring
    tiles, which under a hash pinning would land on the same worker and block the
    parent on one queue while the rest idled. Part numbers come from the parent,
    so they stay in sequence whoever writes them.
    """
    try:
        _worker_init(*init_args)
        scales, offsets = _W["scales"], _W["offsets"]
        while True:
            item = in_q.get()
            if item is None:
                break
            tile, recs, rel = item
            try:
                data = _encode(recs, assign_lod(len(recs)), scales, offsets)
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


class _WritePool:
    """Write worker processes drawing from one queue, with a collector thread.

    The parent must never block on a full input queue while nobody drains the
    output queue, so collection runs on its own thread. The queue is shallow on
    purpose: it is the backpressure. Sharing it means a slow flush occupies one
    worker rather than stalling everything routed behind it.
    """

    def __init__(self, n, ctx, init_args, on_result, depth):
        self.n = n
        self.out_q = ctx.Queue()
        self.in_q = ctx.Queue(maxsize=depth * n)
        self.procs = [
            ctx.Process(
                target=_write_worker,
                args=(self.in_q, self.out_q, init_args),
                daemon=True,
            )
            for _ in range(n)
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
        self.in_q.put((tile, recs, rel))

    def close(self):
        for _ in range(self.n):
            self.in_q.put(None)
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
    pool = _WritePool(
        workers,
        multiprocessing.get_context("forkserver"),
        (bucket, prefix, scales, offsets),
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
            # Largest first. Evicting the least-recently-touched tile instead
            # was tried, on the theory that spatial ordering makes the stalest
            # tile a finished one: it produced 441 files against 274, because a
            # stale tile is often a sliver a coarse node clipped, and freeing
            # almost nothing per flush means flushing many of them.
            while buffered > BUFFER_BUDGET:
                flush(max(sizes, key=sizes.get))
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
