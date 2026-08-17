"""Unit tests for the point cloud writer's encoding.

Everything here runs in memory on synthetic records — no GCS, no processes. The
writer's own fan-out is covered by the lakitu integration tests; what these guard
is the part that decides what a file contains: the LOD stride, the spatial sort
key, and the row-group-per-level layout a reader's pushdown depends on.
"""

import io
import queue
import threading

import numpy as np
import pyarrow.parquet as pq
import pytest

from lib.pointcloud import writer as writer_module
from lib.pointcloud.schema import LOD_LEVELS, point_dtype, tile_span
from lib.pointcloud.summary import PointSummary
from lib.pointcloud.writer import (
    _encode,
    _grid_key,
    _summarize,
    _tile_of,
    _WritePool,
    assign_lod,
    write_parquet,
)

DTYPE = point_dtype(has_color=False)
SCALES = np.array([0.01, 0.01, 0.01])
OFFSETS = np.zeros(3)


class _DeadProcess:
    """Process stub that exited before its target could report completion."""

    pid = 123
    exitcode = -9

    def start(self):
        pass

    def join(self, timeout=None):
        pass

    def is_alive(self):
        return False

    def terminate(self):
        pass


class _DeadProcessContext:
    def Queue(self, maxsize=0):
        return queue.Queue(maxsize=maxsize)

    def Process(self, **_kwargs):
        return _DeadProcess()


def records(count, seed=0, span=50_000):
    """A tile's worth of points, in stored int32 counts (500 m at 0.01 m)."""
    rng = np.random.default_rng(seed)
    out = np.empty(count, dtype=DTYPE)
    out["X"] = rng.integers(0, span, count)
    out["Y"] = rng.integers(0, span, count)
    out["Z"] = rng.integers(80_000, 110_000, count)
    out["intensity"] = rng.integers(0, 1600, count)
    out["classification"] = rng.choice([1, 2, 5], count)
    return out


def test_write_pool_reports_a_worker_that_exits_without_a_sentinel():
    pool = _WritePool(1, _DeadProcessContext(), (), lambda *_: None, depth=1)
    errors = []

    def close():
        try:
            pool.close()
        except RuntimeError as error:
            errors.append(error)

    closer = threading.Thread(target=close, daemon=True)
    closer.start()
    closer.join(timeout=1)

    assert not closer.is_alive(), "close waited forever for the missing sentinel"
    assert len(errors) == 1
    assert "write worker 123 exited" in str(errors[0])


def test_grid_key_stays_inside_uint16():
    """The key must fit uint16 or numpy stops radix-sorting it, which is the point."""
    r = records(50_000)
    key = _grid_key(r["X"], r["Y"])
    assert key.dtype == np.uint16
    assert key.min() >= 0 and key.max() <= np.iinfo(np.uint16).max


def test_grid_key_handles_a_single_distinct_coordinate():
    """A sliver with no extent on an axis must not divide by zero."""
    r = records(1_000)
    r["X"] = 42
    key = _grid_key(r["X"], r["Y"])
    assert key.dtype == np.uint16 and len(key) == 1_000


def test_grid_key_groups_neighbours_together():
    """Points in one cell share a key; distant points do not."""
    xs = np.array([0, 1, 2, 40_000, 40_001], dtype="<i4")
    ys = np.array([0, 1, 2, 40_000, 40_001], dtype="<i4")
    key = _grid_key(xs, ys)
    assert key[0] == key[1] == key[2]
    assert key[3] == key[4]
    assert key[0] != key[3]


def test_assign_lod_is_a_nested_geometric_ladder():
    lod = assign_lod(4**5 * 100)
    counts = [int((lod <= k).sum()) for k in range(LOD_LEVELS)]
    assert counts[-1] == lod.size  # the deepest cut is the whole tile
    for finer, coarser in zip(counts, counts[1:]):
        assert coarser == finer * 4


def test_assign_lod_continues_across_parts():
    """A part is a flush boundary, not a tile: the stride must not restart at it."""
    whole = assign_lod(4**3)
    parts = np.concatenate(
        [
            assign_lod(hi - lo, offset=lo)
            for lo, hi in ((0, 1), (1, 5), (5, 33), (33, 4**3))
        ]
    )
    assert list(parts) == list(whole)


def test_assign_lod_of_a_sparse_tile_is_unchanged():
    """Four points keep exclusive labels [0, 5] and their cumulative ladder."""
    lod = assign_lod(4)
    assert list(lod) == [0, LOD_LEVELS - 1, LOD_LEVELS - 1, LOD_LEVELS - 1]
    assert [int((lod <= k).sum()) for k in range(LOD_LEVELS)] == [1, 1, 1, 1, 1, 4]


def read_back(data):
    return pq.ParquetFile(io.BytesIO(data)).read().to_pandas()


def test_multipart_and_single_part_encodings_share_cumulative_lod_counts():
    """What a `lod <= k` read gets back must not depend on how the tile was split."""
    r = records(5_000)
    splits = ((0, 7), (7, 1_003), (1_003, 1_004), (1_004, 5_000))

    single = read_back(_encode(r, assign_lod(len(r)), SCALES, OFFSETS))["lod"]
    multipart = np.concatenate(
        [
            read_back(
                _encode(r[lo:hi], assign_lod(hi - lo, offset=lo), SCALES, OFFSETS)
            )["lod"].to_numpy()
            for lo, hi in splits
        ]
    )

    assert len(multipart) == len(single)
    for k in range(LOD_LEVELS):
        assert int((multipart <= k).sum()) == int((single.to_numpy() <= k).sum())


def test_encode_preserves_every_point():
    """The sort is a permutation: nothing may be dropped or duplicated."""
    r = records(200_000)
    table = read_back(_encode(r, assign_lod(len(r)), SCALES, OFFSETS))
    assert len(table) == len(r)
    for column in ("X", "Y", "Z", "intensity", "classification"):
        assert sorted(table[column].tolist()) == sorted(r[column].tolist())


def test_encode_writes_one_row_group_per_level():
    """`lod <= k` prunes on row-group statistics, so levels must not share one."""
    r = records(200_000)
    meta = pq.ParquetFile(io.BytesIO(_encode(r, assign_lod(len(r)), SCALES, OFFSETS)))
    groups = [meta.metadata.row_group(i) for i in range(meta.metadata.num_row_groups)]
    assert len(groups) == LOD_LEVELS
    lod_column = meta.schema_arrow.names.index("lod")
    for index, group in enumerate(groups):
        stats = group.column(lod_column).statistics
        assert stats.min == stats.max == index


def test_summary_folds_across_flushes():
    """Per-flush reduction must equal reducing the whole cloud at once.

    The write workers each see one flush; the resource reports one set of
    figures. Splitting unevenly, because real flushes are whatever the buffer
    budget happened to evict.
    """
    r = records(100_000, seed=3)
    summary = PointSummary(SCALES, OFFSETS)
    for lo, hi in ((0, 7), (7, 40_001), (40_001, 40_002), (40_002, 100_000)):
        summary.fold(*_summarize(r[lo:hi]))

    assert summary.count == len(r)
    assert summary.summary()["point_count"] == len(r)
    assert summary.summary()["point_classes"] == sorted(
        set(r["classification"].tolist())
    )
    expected = [
        float(r[c].min()) * s + o for c, s, o in zip(("X", "Y", "Z"), SCALES, OFFSETS)
    ] + [float(r[c].max()) * s + o for c, s, o in zip(("X", "Y", "Z"), SCALES, OFFSETS)]
    assert summary.bounds() == pytest.approx(expected)


def test_summary_of_no_points_is_not_an_error():
    """A cloud with nothing in it reports zero rather than sentinel extremes."""
    summary = PointSummary(SCALES, OFFSETS)
    assert summary.summary() == {
        "point_count": 0,
        "point_classes": [],
        "density": 0.0,
    }


def test_encode_orders_points_spatially():
    """Sorted output must be more locally coherent than the input it came from."""
    r = records(200_000)
    table = read_back(_encode(r, assign_lod(len(r)), SCALES, OFFSETS))
    deepest = table[table["lod"] == LOD_LEVELS - 1]
    step = np.abs(np.diff(deepest["X"].to_numpy().astype(np.int64))).mean()
    unsorted_step = np.abs(np.diff(r["X"].astype(np.int64))).mean()
    assert step < unsorted_step / 10


# The writer places a point in a tile; `tile_span` decides which tiles a reader
# opens for a box. They have to agree everywhere, including outside the bounds
# the dataset declares -- a LAS header's bbox is declared, not measured, so an
# upload cropped without a header rewrite has points beyond it.
MINS = np.array([1000.0, 2000.0])
TILE_M = 500.0


def tile_a_reader_would_open(x, y):
    """The tile a box query at exactly (x, y) prunes to."""
    tx0, _ = tile_span(x, x, MINS[0], TILE_M)
    ty0, _ = tile_span(y, y, MINS[1], TILE_M)
    return tx0, ty0


@pytest.mark.parametrize(
    "x,y",
    [
        (1000.0, 2000.0),  # exactly the origin
        (1250.0, 2250.0),  # inside
        (999.9, 2250.0),  # a fraction below the origin, where truncation folded
        (700.0, 2250.0),  # a tile below the origin
        (-4000.0, 2250.0),  # far below
        (1250.0, 9000.0),  # far above, where the packing collided
    ],
)
def test_writer_and_reader_agree_on_which_tile_a_point_is_in(x, y):
    tx, ty, _ = _tile_of(np.array([x]), np.array([y]), MINS, TILE_M)
    assert (int(tx[0]), int(ty[0])) == tile_a_reader_would_open(x, y)


def test_points_outside_the_declared_bounds_keep_their_own_tile():
    """Clamping them into the edge tile hid them from every box query."""
    x = np.array([1250.0, 700.0, 9000.0])
    y = np.array([2250.0, 2250.0, 2250.0])
    tx, ty, _ = _tile_of(x, y, MINS, TILE_M)
    assert list(tx) == [0, -1, 16]
    assert list(ty) == [0, 0, 0]


def test_an_orphan_part_is_read_back_as_points(tmp_path):
    """Why `clear_prefix` exists, stated as the failure it prevents.

    Part numbers restart at zero each run, so a re-run producing fewer parts for
    a tile leaves the higher-numbered files behind. Nothing rejects them: the
    reader discovers by listing, so they come back as extra points.
    """
    import pyarrow as pa
    import pyarrow.dataset as pa_ds

    root = tmp_path / "cloud.parquet"
    part = root / "tile_x=0" / "tile_y=0" / "part-00000.parquet"
    part.parent.mkdir(parents=True)
    table = pa.table({"X": np.arange(10, dtype=np.int32)})
    pq.write_table(table, part)

    def rows():
        return pa_ds.dataset(
            str(root), format="parquet", partitioning="hive"
        ).count_rows()

    assert rows() == 10

    # A shorter re-run would overwrite part-00000 and never touch part-00001.
    pq.write_table(table, part.parent / "part-00001.parquet")
    assert rows() == 20, "orphan silently doubled the cloud"


# One 500 m tile at the origin, millimetre-free 0.01 m scaling, so `records`
# with a 1,000-count span lands inside tile (0, 0).
INFO = {
    "mins": [0.0, 0.0, 0.0],
    "maxs": [500.0, 500.0, 100.0],
    "scales": [0.01, 0.01, 0.01],
    "offsets": [0.0, 0.0, 0.0],
}


def parts_written(monkeypatch, chunks, **kwargs):
    """Run the parent write path with the worker pool and GCS stubbed out.

    Returns what the parent handed each worker: ``(tile, records, path, offset)``
    per flush, in submission order.
    """
    submitted = []

    class Pool:
        def __init__(self, *_args, **_kwargs):
            pass

        def submit(self, tile, recs, rel, offset):
            submitted.append((tile, recs, rel, offset))

        def close(self):
            pass

    class Sink:
        bytes_written = 0

        def __init__(self, *_args):
            pass

        def put(self, *_args):
            pass

    monkeypatch.setattr(writer_module, "_WritePool", Pool)
    monkeypatch.setattr(writer_module, "GcsSink", Sink)
    monkeypatch.setattr(writer_module, "clear_prefix", lambda *_args: None)
    write_parquet(chunks, INFO, "bucket", "prefix", tile_m=TILE_M, **kwargs)
    return submitted


def test_a_tile_flushed_in_several_parts_is_assigned_as_one(monkeypatch):
    """A buffer eviction must not change what the tile's pyramid holds."""
    monkeypatch.setattr(writer_module, "MAX_TILE_BYTES", 1)
    chunks = [records(9, seed=seed, span=1_000) for seed in range(3)]

    parts = parts_written(monkeypatch, chunks)

    assert len(parts) == 3, "the tile has to actually flush more than once"
    assert [offset for *_, offset in parts] == [0, 9, 18]
    labels = np.concatenate(
        [assign_lod(len(recs), offset=offset) for _, recs, _, offset in parts]
    )
    assert list(labels) == list(assign_lod(27))


def test_each_tile_counts_its_own_lod_offsets(monkeypatch):
    """The offset is a position within a tile, not within the job."""
    monkeypatch.setattr(writer_module, "MAX_TILE_BYTES", 1)
    chunk = records(8, span=1_000)
    chunk["X"][4:] += 60_000  # the second half sits a tile east

    parts = parts_written(monkeypatch, [chunk, chunk.copy()])

    by_tile = {}
    for tile, recs, _, offset in parts:
        by_tile.setdefault(tile, []).append((len(recs), offset))
    assert set(by_tile) == {(0, 0), (1, 0)}
    for entries in by_tile.values():
        assert entries == [(4, 0), (4, 4)]


def test_distinct_tiles_never_share_a_packed_id():
    """A collision merges two tiles under one key and writes both into one."""
    rng = np.random.default_rng(0)
    # Deliberately far outside the declared extent, in both directions.
    x = rng.uniform(-20_000, 40_000, 50_000)
    y = rng.uniform(-20_000, 40_000, 50_000)
    tx, ty, tid = _tile_of(x, y, MINS, TILE_M)
    pairs = {(int(a), int(b)) for a, b in zip(tx, ty)}
    assert len(set(tid.tolist())) == len(pairs)
    for packed in set(tid.tolist()):
        sel = tid == packed
        assert len(set(zip(tx[sel].tolist(), ty[sel].tolist()))) == 1
