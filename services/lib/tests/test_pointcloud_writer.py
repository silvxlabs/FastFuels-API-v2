"""Unit tests for the point cloud writer's encoding.

Everything here runs in memory on synthetic records — no GCS, no processes. The
writer's own fan-out is covered by the lakitu integration tests; what these guard
is the part that decides what a file contains: the LOD stride, the spatial sort
key, and the row-group-per-level layout a reader's pushdown depends on.
"""

import io

import numpy as np
import pyarrow.parquet as pq

from lib.pointcloud.schema import LOD_LEVELS, point_dtype
from lib.pointcloud.writer import _encode, _grid_key, assign_lod

DTYPE = point_dtype(has_color=False)
SCALES = np.array([0.01, 0.01, 0.01])
OFFSETS = np.zeros(3)


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
    assert counts[-1] == lod.size  # the deepest level is the whole tile
    for finer, coarser in zip(counts, counts[1:]):
        assert coarser == finer * 4


def read_back(data):
    return pq.ParquetFile(io.BytesIO(data)).read().to_pandas()


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


def test_encode_orders_points_spatially():
    """Sorted output must be more locally coherent than the input it came from."""
    r = records(200_000)
    table = read_back(_encode(r, assign_lod(len(r)), SCALES, OFFSETS))
    deepest = table[table["lod"] == LOD_LEVELS - 1]
    step = np.abs(np.diff(deepest["X"].to_numpy().astype(np.int64))).mean()
    unsorted_step = np.abs(np.diff(r["X"].astype(np.int64))).mean()
    assert step < unsorted_step / 10
