"""Spatially-partitioned, LOD-tagged Parquet.

The output is a Hive-partitioned dataset -- ``tile_x=<i>/tile_y=<j>/part-*.parquet``
plus a ``_manifest.json`` -- rather than one file. A reader that wants a box
prunes on the partition columns, which come from the paths, so it touches only
the parts that overlap.

There is no combined ``_metadata`` footer. One was written and nothing ever read
it: `reader.open_dataset` discovers by listing, and replacing the footer with the
bytes "GARBAGE" left the dataset opening and counting rows fine. What it saves is
one LIST per job, against a `FileMetaData` parse per file on the parent, which is
this path's ceiling. An index of what a tile holds belongs in ``_manifest.json``,
which every reader fetches anyway.

Each part is written one row group per LOD level *that has points*, so a
``lod <= k`` filter prunes on row-group statistics. A level with none is skipped,
so a group's position does not say which level it is and no count may be read off
it. Writing a single row group spanning every level made the pyramid correct but
useless: pushdown prunes row groups, not rows.

A tile should be one part file. It is written when the last node that can reach
it has been routed -- see the ``schedule`` argument to `write_parquet` -- rather
than when the parent runs short of buffer. Evicting under pressure instead made
file count scale with data volume rather than with area: 64 km2 wrote 2,953 files
over 260 tiles, so the coarsest LOD of one tile was scattered across as many as
33 objects and a preview of the whole cloud had to open all 2,953.

An upload has no schedule to pass -- it is one LAS file read in chunks, with no
octree to say which tiles a chunk can still reach -- so it runs the eviction
path, and that is deliberate. Measured on two real uploads at the size cap, both
about 4.5 GB of records: 238 files over 64 tiles, and 136 over 88. The severity
is a property of the file's own point order rather than of its size or area. The
first has none, so all 64 tiles stayed live for the whole job and a tile's writes
were spread over a median of 171 other flushes; the second sweeps a diagonal
strip, so 49 of its 88 tiles were written once and never reopened. 1.5-3.7x the
ideal file count is not the 33-way split the schedule was built for, and deriving
a schedule from a first pass over the coordinates would cost a whole extra decode
of the file to buy it back.

The LOD is a stride: ``lod <= k`` is 1 in 4**(L-1-k) of a tile's points, a
nested and unbiased subsample, and reading every level gives the tile back whole
-- see assign_lod.

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
import queue
import threading

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from lib.config import (
    LAKITU_BUFFER_BUDGET_MB,
    LAKITU_WRITE_QUEUE_DEPTH,
    LAKITU_WRITE_WORKERS,
)
from lib.pointcloud.schema import (
    DELTA_COLUMNS,
    DICT_COLUMNS,
    LOD_LEVELS,
    choose_tile_m,
    columns_for,
)
from lib.pointcloud.summary import PointSummary

# How much routed-but-unflushed point data the parent holds, and how big a tile
# has to be before it is worth flushing.
#
# What the budget means depends on whether a `schedule` is supplied. Without
# one it is the flush trigger: the largest tile is evicted whenever the budget
# is exceeded, so file size is roughly BUFFER_BUDGET / active tiles and the
# budget is the only thing standing between the writer and one file per node.
#
# With a schedule it is a backstop. Tiles are written when they are finished, so
# every eviction the budget forces is a tile that was going to be one file and
# is now two. Size it to the peak the schedule needs, not to a memory floor.
#
# Measured at 64 km2 (1.045B points, 260 tiles), one image, env-switched arms,
# every arm writing an identical point count:
#
#   eviction only, 192 MiB   2,953 files   11.4/tile   2.72 GB RSS   317 s
#   schedule,      192 MiB   1,069 files    4.1/tile   2.96 GB RSS   309 s
#   schedule,      512 MiB     449 files    1.7/tile   3.92 GB RSS   365 s
#   schedule,     1024 MiB     359 files    1.4/tile   5.12 GB RSS   385 s
#
# The schedule itself is free -- 309 s against 317 s -- so the whole cost of the
# 512 MiB default is the budget: +15% wall, +1.2 GB RSS. Size a budget from
# measured RSS, not from this number: buffers hold lists of small arrays and
# np.concatenate doubles at flush, so RSS grew about 3x the budget increase.
#
# What the files buy is the read. A whole-cloud `lod <= 0` preview took 243 s
# over 3,210 files and 26.6 s over 359 -- latency against object count, not
# bytes, since the preview is ~1 M points either way.
BUFFER_BUDGET = LAKITU_BUFFER_BUDGET_MB << 20
MAX_TILE_BYTES = 96 << 20  # flush a tile at this size regardless of budget
_WORKER_POLL_SECONDS = 0.1


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


def clear_prefix(bucket, prefix):
    """Delete anything already stored under a dataset prefix.

    Part numbers restart at zero on every run, so a re-run that produces fewer
    parts for a tile than its predecessor leaves the higher-numbered files in
    place. Readers discover by listing, so those orphans come back as points --
    a silent duplicate rather than an error.

    Imported here rather than at module scope for the same reason `GcsSink`
    defers its client: forkserver children re-import this module, and none of
    them clears anything.
    """
    from lib.gcs import delete_directory, exists

    path = f"{bucket}/{prefix.rstrip('/')}"
    if exists(path):
        delete_directory(path)


def assign_lod(count, levels=LOD_LEVELS):
    """Assign each point a pyramid level by stride: ``lod <= k`` keeps 1 in 4**(L-1-k).

    The cuts are nested -- each ``lod <= k`` is a strict superset of the one
    above, and ``lod <= L-1`` is the whole tile -- so a reader gets a geometric
    ladder of point counts. A level on its own is the difference between two
    cuts rather than a fixed fraction, and the deepest holds three quarters of
    the tile. There is no residual: every point belongs to exactly one level.

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


def _tile_of(x, y, mins, tile_m):
    """Each point's tile, and a packed id that sorts points into tiles.

    Two things this must not do, because both put a point in a tile the reader
    will never look in. `schema.tile_span` floors, so this has to floor too --
    ``astype`` truncates toward zero, which folds every point in
    ``[-tile_m, 0)`` into tile 0 rather than tile -1. And neither index is
    clamped to the declared bounds: a LAS header's bounding box is a declared
    value that laspy does not re-derive from the points, so an upload cropped
    without a header rewrite has points outside it, and clamping those into the
    edge tile hides them from every box query.

    The packed id is offset by this batch's own minima rather than by the grid
    width, which keeps it injective whatever range the tiles span. Packing
    against a fixed width let a point past the declared maximum collide with a
    real tile and be written into that tile's partition.

    Args:
        x: Point x coordinates in world units.
        y: Point y coordinates in world units.
        mins: Dataset origin, ``(min_x, min_y, ...)``, which anchors the grid.
        tile_m: Tile edge in metres.

    Returns:
        ``(tile_x, tile_y, packed_id)``, one entry per point.
    """
    tx = np.floor((x - mins[0]) / tile_m).astype(np.int32)
    ty = np.floor((y - mins[1]) / tile_m).astype(np.int32)
    ty0 = ty.min()
    tid = (tx - tx.min()).astype(np.int64) * (int(ty.max() - ty0) + 1) + (ty - ty0)
    return tx, ty, tid


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
    """One row group per LOD level that has points, spatially ordered within each.

    A level with no points writes no row group, so a sparse tile ends up with
    fewer groups than levels. `lod` is a stored column and its statistics are
    what a reader selects on; a group's position carries nothing.

    The ordering is for compression alone. Row-group statistics are min/max,
    which no permutation changes, so pushdown gets nothing from it; what it buys
    is DELTA_BINARY_PACKED on X/Y/Z, worth 16% of the file on a real dense tile.

    Points are ordered by their cell in a coarse grid over the level's extent,
    which fits a uint16 and so radix-sorts in a single pass rather than falling
    back to merge sort. Measured against a packed exact (x, y) key on a real
    tile: 45% less CPU for 0.9% more file. Morton interleaving was tried directly
    and rejected -- 62% more encode time for the bit-spreading. Finer grids
    compress better still (512 cells gave 4.9% less file) but need a uint32 key,
    which hands the radix sort back.

    The key is x-cell major, so it does not treat the two axes alike: measured
    over a 16 km2 dataset, X came to 2.3% of the stored bytes against Y's 39.2%.
    """
    # Derived from the records rather than fixed, so a cloud with colour and one
    # without each write exactly the columns they have.
    columns = columns_for(records.dtype)
    buf = io.BytesIO()
    writer = None
    try:
        for level in range(LOD_LEVELS):
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


def _summarize(records):
    """Reduce one flush to what the resource reports about itself.

    Runs in the write worker, on points it is about to encode anyway, so the
    reduction is spread across the pool and lands while the data is still hot.
    Every point belongs to exactly one flush, so folding these is exact.

    Returns:
        ``(mins, maxs, classes, count)`` over the stored integers, for
        `lib.pointcloud.summary.PointSummary.fold`.
    """
    seen = np.zeros(256, dtype=bool)
    seen[records["classification"]] = True
    return (
        np.array([records[c].min() for c in ("X", "Y", "Z")], dtype=np.int64),
        np.array([records[c].max() for c in ("X", "Y", "Z")], dtype=np.int64),
        np.flatnonzero(seen),
        records.size,
    )


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
                out_q.put(("ok", len(data), rel, _summarize(recs)))
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
            try:
                item = self.out_q.get(timeout=_WORKER_POLL_SECONDS)
            except queue.Empty:
                failed = next(
                    (p for p in self.procs if p.exitcode not in (None, 0)), None
                )
                if failed is None:
                    continue
                if self.error is None:
                    self.error = RuntimeError(
                        f"write worker {failed.pid} exited unexpectedly "
                        f"with code {failed.exitcode}"
                    )
                return
            if item is None:
                self._done += 1
                continue
            if item[0] == "err":
                if self.error is None:
                    self.error = RuntimeError(f"{item[3]}: {item[1]}\n{item[2]}")
                continue
            self._on_result(item[1], item[3])

    def _put(self, item):
        while self.error is None:
            try:
                self.in_q.put(item, timeout=_WORKER_POLL_SECONDS)
                return
            except queue.Full:
                pass
        raise self.error

    def submit(self, tile, recs, rel):
        self._put((tile, recs, rel))

    def close(self):
        try:
            for _ in range(self.n):
                self._put(None)
            self._collector.join()
        finally:
            if self.error is not None:
                for p in self.procs:
                    if p.is_alive():
                        p.terminate()
            for p in self.procs:
                p.join(timeout=30)
            if self._collector.is_alive():
                self._collector.join(timeout=2 * _WORKER_POLL_SECONDS)
        if self.error is not None:
            raise self.error


def write_parquet(
    records,
    info,
    bucket,
    prefix,
    *,
    tile_m=None,
    schedule=None,
    workers=LAKITU_WRITE_WORKERS,
    depth=LAKITU_WRITE_QUEUE_DEPTH,
):
    """Route records into tiles and write a partitioned Parquet dataset.

    Args:
        records: Iterable of `schema.point_dtype` arrays, in any order. With a
            `schedule`, ``(node_index, array)`` pairs instead, so the writer can
            tell which of the scheduled nodes has landed.
        info: Source bounds and scaling -- ``mins``, ``maxs``, ``scales``,
            ``offsets``, each a length-3 sequence.
        bucket: Destination GCS bucket name.
        prefix: Destination prefix within the bucket. The dataset is written
            directly under it.
        tile_m: Tile edge in metres. Defaults to `choose_tile_m`.
        schedule: Optional ``{tile: last_node_index}`` from `plan_nodes`. Each
            tile is written once its last node has been routed, which is what
            makes a tile one file rather than one per buffer eviction. None
            falls back to evicting the largest tile under budget pressure.
        workers: Write processes. Must not exceed the vCPU allocation --
            oversubscribing measured 2.4x the CPU for identical output.
        depth: Per-worker queue depth, which is the parent's backpressure.

    Returns:
        Dict with ``points``, ``tiles``, ``files``, ``output_bytes``, and the
        ``summary`` and ``bounds`` the resource reports about itself. The last
        two are reduced in the write workers rather than by the caller, because
        every point already passes through one on its way to a file.

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

    # Before anything is written, not after: a run that dies partway must not
    # leave a mix of its own parts and its predecessor's behind.
    clear_prefix(bucket, prefix)

    buffers, sizes, counts, nparts = {}, {}, {}, {}
    buffered = 0
    lock = threading.Lock()
    stats = {"written_bytes": 0, "files": 0}
    summary = PointSummary(scales, offsets)

    def on_result(nbytes_out, folded):
        with lock:
            stats["written_bytes"] += nbytes_out
            stats["files"] += 1
            summary.fold(*folded)

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

    # Tiles owed to each node index, and how far the arrival stream is complete.
    # Nodes come back in download-completion order, not the order they were
    # scheduled in, so a tile is only final once every node up to its last one
    # has actually landed -- not merely once that one node has. `watermark` is
    # the highest index with no gap below it, which is exactly that test.
    due_at = {}
    if schedule is not None:
        for tile, last in schedule.items():
            due_at.setdefault(last, []).append(tile)
    arrived = set()
    watermark = 0
    swept = 0

    def write_finished():
        """Write every tile the arrival stream has moved past, whole."""
        nonlocal swept
        for i in range(swept, watermark):
            for tile in due_at.get(i, ()):
                flush(tile)
        swept = watermark

    try:
        for item in records:
            if schedule is None:
                out = item
            else:
                index, out = item
                arrived.add(index)
                while watermark in arrived:
                    arrived.discard(watermark)
                    watermark += 1
                # A node that contributed no points still has to advance the
                # watermark, or the tiles waiting on it are never written.
                if out is None:
                    write_finished()
                    continue
            x = out["X"] * scales[0] + offsets[0]
            y = out["Y"] * scales[1] + offsets[1]
            tx, ty, tid = _tile_of(x, y, mins, tile_m)
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
            write_finished()
            for big_tile in [k for k, n in sizes.items() if n >= MAX_TILE_BYTES]:
                flush(big_tile)
            # Largest first. Evicting the least-recently-touched tile instead
            # was tried, on the theory that spatial ordering makes the stalest
            # tile a finished one: it produced 441 files against 274, because a
            # stale tile is often a sliver a coarse node clipped, and freeing
            # almost nothing per flush means flushing many of them.
            #
            # Under a schedule this is a backstop, and each time it fires it
            # splits a tile that the schedule was about to write in one piece.
            while buffered > BUFFER_BUDGET:
                flush(max(sizes, key=sizes.get))
        for tile in list(buffers):
            flush(tile)
    finally:
        pool.close()

    manifest = {
        "tiles": len(counts),
        "tile_m": tile_m,
        "points": summary.count,
        "lod_levels": LOD_LEVELS,
        "scales": list(map(float, scales)),
        "offsets": list(map(float, offsets)),
        "mins": list(map(float, mins)),
        "maxs": list(map(float, maxs)),
    }
    sink = GcsSink(bucket, prefix)
    sink.put("_manifest.json", json.dumps(manifest).encode())

    return {
        "points": summary.count,
        "tiles": len(counts),
        "files": stats["files"],
        "output_bytes": stats["written_bytes"] + sink.bytes_written,
        "summary": summary.summary(),
        "bounds": summary.bounds(),
    }
