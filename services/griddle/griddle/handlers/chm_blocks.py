"""Per-block point rasterisation for the canopy height model.

Kept in its own module, and deliberately import-light at module scope, because
forkserver re-imports it in every child. `chm_point_cloud` pulls in dask,
geopandas, rioxarray and scipy; a worker needs none of them, so importing this
instead keeps a child's start-up and resident set to numpy plus what the reader
brings. Everything heavier is imported inside the worker functions.

These run in processes rather than threads because the read was throughput-bound
inside one interpreter, not latency-bound. Measured in-region at 8 vCPU with the
dataset already shared, holding the work fixed and raising dask threads 8 -> 16
-> 32 moved the ground read 99.4 s -> 102.6 s -> 103.0 s while the time each
thread spent waiting grew exactly linearly, 760 -> 1,553 -> 3,019 thread-
seconds, and CPU stayed pinned near 1.6 cores. Total throughput was constant
however many threads queued up: gcsfs is instance-cached so every block shares
one asyncio event loop, and pyarrow reached it through `FSSpecHandler`, which
re-enters Python and takes the GIL once per range read. The implied ~28 MB/s
over a 2.76 GB dataset is nowhere near a bandwidth ceiling, so the ceiling was
the interpreter.

`iter_points` no longer reads that way -- it fetches each tile's file in one GET
and decodes from memory -- which removes most of what that measurement was
measuring. Whether threads would now scale is open and untested; the process
pool stays until someone measures it in-region rather than because the reasoning
above still holds.

Blocking does not change the answer. Every pass here is cell-local: a cell's
value depends only on points inside it, and the reductions are order-
independent, so how the points are grouped cannot matter.

What follows from that differs by statistic. A maximum or a mean is also a
*fold*, so a batch can be reduced and dropped and a worker's memory stays off
the tile's point count. A percentile is not: it holds the block's returns to
the end. Blocking is exact either way -- sorting is a function of the multiset
-- but a percentile's blocks have to be sized by their points rather than by
their cells, which is `chm_point_cloud._retaining_block_cells`.
"""

import numpy as np

# ASPRS classes that can contribute to a canopy surface: never-classified (0),
# unclassified (1), ground (2), and the three vegetation classes. This
# deliberately excludes noise (7, 18), water (9) and buildings (6).
#
# Two traps here. Many 3DEP acquisitions classify only ground and unclassified,
# so vegetation lives in class 1 -- filtering *to* classes 3-5 returns nothing.
# And a genuinely unclassified upload is all class 0, so omitting it (as v1's
# `Classification >= 1 && <= 5` did) yields an empty CHM for exactly the clouds
# the derived-ground path exists to serve.
SURFACE_CLASSES = (0, 1, 2, 3, 4, 5)
GROUND_CLASS = 2

# Heights above this are not canopy. Matches the v1 sanity filter.
MAX_CANOPY_HEIGHT_M = 100.0

# The floor heights are clamped to, rather than filtered against: an unbiased
# ground has returns on both sides of it, so a return below it is ground.
MIN_CANOPY_HEIGHT_M = 0.0

# How far below the ground estimate a return may sit and still be one. Within
# this it is the estimate's own scatter and reads as zero height; deeper is a
# blunder, and dropping it keeps it from making a bare cell out of nothing.
GROUND_TOLERANCE_M = 0.5

# A point within this distance of the provisional surface is taken to be a
# ground return. Used to re-derive the ground from real measurements rather
# than from the eroded and dilated surface, which reads about 0.1 m low.
GROUND_SNAP_TOLERANCE_M = 0.5

# Per-worker state, set once by the initializer so each task carries only its
# own lattice and, where a pass needs one, its own slice of a raster.
_W = {}


def worker_init(prefix, manifest, filesystem=None):
    """Open one dataset per worker process, once, before any block runs.

    The dataset is opened here rather than per task for the reason the threaded
    version shared one handle: opening it discovers and lists the cloud's hive
    partitions, which measured 0.54 s a time.

    Args:
        prefix: Dataset prefix, e.g. ``<bucket>/<id>/cloud.parquet``.
        manifest: Its manifest, which every read needs for the tiling.
        filesystem: Override for the filesystem, defaulting to GCS. Passed in
            rather than patched because a worker is a separate interpreter --
            nothing the caller does to its own modules reaches one, so a local
            filesystem has to travel with the initializer or not at all.
    """
    # An exception here kills the worker and the parent only ever sees
    # BrokenProcessPool, so say what actually failed before going down.
    try:
        from lib.pointcloud.reader import open_dataset

        _W["dataset"] = open_dataset(prefix, filesystem=filesystem)
        _W["manifest"] = manifest
        # Kept alongside the dataset because the reader fetches each tile's file
        # itself rather than through pyarrow, so it needs the same handle.
        _W["filesystem"] = filesystem
    except Exception as e:
        print(f"chm_blocks worker init failed: {e!r}", flush=True)
        raise


def run_block(job):
    """Rasterise one block, in a worker process.

    Args:
        job: ``(kind, lattice, resolution, classes, extra)``. ``extra`` carries
            whatever raster the pass samples -- the block's slice of the
            provisional surface for ``"snap"``, and ``(slice, lattice,
            statistic)`` of the ground for ``"height"``. Both rasters are
            block-sized, so a task stays small.

    Returns:
        The block's ``(rows, cols)`` float32 raster.
    """
    from lib.pointcloud.reader import iter_points

    kind, lattice, resolution, classes, extra = job

    def reader(bounds, wanted):
        return iter_points(
            _W["dataset"], _W["manifest"], bounds, wanted, _W["filesystem"]
        )

    if kind == "mean":
        return mean_surface_block(reader, lattice, resolution, classes)
    if kind == "min":
        return min_surface_block(reader, lattice, resolution, classes)
    if kind == "snap":
        return snap_ground_block(reader, extra, lattice, resolution)
    if kind == "height":
        ground, ground_lattice, statistic = extra
        return height_block(
            reader, ground, ground_lattice, lattice, resolution, statistic
        )
    raise ValueError(f"unknown block kind {kind!r}")


def block_lattice(transform, row0: int, col0: int, rows: int, cols: int):
    """Lattice tuple for one block, in the same form the whole-grid one takes."""
    return (
        transform.c + col0 * transform.a,
        transform.f + row0 * transform.e,
        rows,
        cols,
    )


def block_bounds(lattice, resolution: float) -> tuple:
    """World bounds of a block lattice, for partition pruning.

    Open on every side but ``min_x``, so a block sitting on tile boundaries reads
    exactly the one tile it covers. Left closed, an edge that lands on a tile
    origin pulls in the whole neighbouring partition -- and once the blocks are
    cut on tile boundaries that is not an edge case but every block, doubling the
    partitions read on that axis.

    ``max_x`` and ``min_y`` are free: `cell_indices` floors, so a point on either
    already belongs to the neighbouring block. ``max_y`` is not, and it is the
    expensive one -- the row axis runs downward, so a block's top edge is closed
    while a tile's is open, and the two conventions meet on every internal seam.

    Excluding it takes a measured 2.0 partitions read per block down to 1.0, and
    is not free: a point whose ``y`` falls exactly on a seam belongs to the block
    below by `cell_indices` but lives in the tile above, which that block no
    longer reads. Diffed over the 64 km2 validation cloud, that changed 679 of
    51,307,443 finite cells -- one in 75,000, all within two rows of a seam,
    median 0.012 m and one cell by 5.6 m -- and lost no cells at all: the finite
    count was identical either way. The two passes went from 690 s to 419 s.
    """
    origin_x, origin_y, rows, cols = lattice
    return (
        origin_x,
        np.nextafter(origin_y - rows * resolution, np.inf),
        np.nextafter(origin_x + cols * resolution, -np.inf),
        np.nextafter(origin_y, -np.inf),
    )


def cell_indices(x, y, lattice, resolution):
    """Return (flat_index, in_bounds_mask) for points on the output lattice.

    A point contributes to the cell its coordinates fall in -- square
    containment, so every return inside a cell counts exactly once.
    """
    origin_x, origin_y, height, width = lattice
    col = np.floor((x - origin_x) / resolution).astype(np.int64)
    row = np.floor((origin_y - y) / resolution).astype(np.int64)
    inside = (col >= 0) & (col < width) & (row >= 0) & (row < height)
    return row * width + col, inside


def sample_bilinear(surface, x, y, lattice, resolution) -> np.ndarray:
    """Bilinearly interpolate a raster at point coordinates."""
    origin_x, origin_y, height, width = lattice
    col = (x - origin_x) / resolution - 0.5
    row = (origin_y - y) / resolution - 0.5
    col0 = np.clip(np.floor(col).astype(np.int64), 0, width - 2)
    row0 = np.clip(np.floor(row).astype(np.int64), 0, height - 2)
    tx = np.clip(col - col0, 0.0, 1.0)
    ty = np.clip(row - row0, 0.0, 1.0)
    return (
        surface[row0, col0] * (1 - tx) * (1 - ty)
        + surface[row0, col0 + 1] * tx * (1 - ty)
        + surface[row0 + 1, col0] * (1 - tx) * ty
        + surface[row0 + 1, col0 + 1] * tx * ty
    )


def mean_surface_block(reader, lattice, resolution, classes) -> np.ndarray:
    """Mean z per cell over the given classes, NaN where no point falls.

    The mean, not the minimum: a minimum is an extreme-value statistic, so it
    sits further below the surface the more returns a cell holds, and every
    height measured above it rises to match (issue #503). A mean is unbiased at
    any count.

    Cell-local, and a sum and a count are both commutative, so a block needs no
    halo and the answer cannot depend on how the points were grouped.
    """
    _, _, height, width = lattice
    size = height * width
    total, count = np.zeros(size), np.zeros(size)
    for x, y, z, _ in reader(block_bounds(lattice, resolution), classes):
        index, inside = cell_indices(x, y, lattice, resolution)
        total += np.bincount(index[inside], z[inside], size)
        count += np.bincount(index[inside], minlength=size)
    # 0/0 is the NaN an empty cell wants; nothing else divides by zero.
    with np.errstate(invalid="ignore"):
        return (total / count).reshape(height, width).astype(np.float32)


def min_surface_block(reader, lattice, resolution, classes) -> np.ndarray:
    """Lowest z per cell over the given classes, NaN where no point falls.

    The input `_pmf` opens, which is defined on a minimum surface (Zhang et al.
    2003). Not a ground estimate -- `mean_surface_block` says why a minimum is
    the wrong statistic for one.

    Cell-local: a cell's minimum depends only on points inside it, so a block
    needs no halo and blocked output is identical to rasterising the cloud whole.
    `np.minimum.at` is commutative, so the order points arrive in cannot change
    the answer either.
    """
    _, _, height, width = lattice
    surface = np.full(height * width, np.inf, dtype=np.float32)
    for x, y, z, _ in reader(block_bounds(lattice, resolution), classes):
        index, inside = cell_indices(x, y, lattice, resolution)
        np.minimum.at(surface, index[inside], z[inside].astype(np.float32))
    surface[~np.isfinite(surface)] = np.nan
    return surface.reshape(height, width)


def snap_ground_block(reader, provisional, lattice, resolution) -> np.ndarray:
    """Re-derive ground from the returns the morphological filter accepts.

    The opened surface has been eroded and dilated, so reading it directly runs
    about 0.1 m low; taking the real returns that sit near it does not.

    The accepted returns are averaged for the reason `mean_surface_block` gives.

    `provisional` is this block's slice of the filtered surface, so the lookup
    is block-local -- a point's snap test reads its own cell and no other.
    """
    _, _, height, width = lattice
    size = height * width
    total, count = np.zeros(size), np.zeros(size)
    flat = provisional.reshape(-1)
    for x, y, z, _ in reader(block_bounds(lattice, resolution), SURFACE_CLASSES):
        index, inside = cell_indices(x, y, lattice, resolution)
        z, index = z[inside], index[inside]
        near = np.abs(z - flat[index]) <= GROUND_SNAP_TOLERANCE_M
        total += np.bincount(index[near], z[near], size)
        count += np.bincount(index[near], minlength=size)
    with np.errstate(invalid="ignore"):
        return (total / count).reshape(height, width).astype(np.float32)


def point_heights(reader, ground, ground_lattice, lattice, resolution):
    """Yield ``(cell index, height above ground)`` a batch at a time.

    Ground is sampled bilinearly at each point rather than read from the point's
    own cell: on a slope a cell-constant ground under-reads uphill and
    over-reads downhill, which measured 0.27 m RMSE against a per-point ground
    versus 0.17 m for bilinear.

    `ground` is this block's slice of the ground raster grown by one cell on
    every side, with `ground_lattice` describing that slice. One cell is exactly
    what bilinear sampling of a point inside the block can reach: the sample sits
    half a cell back, so a point in the first column reads columns -1 and 0. A
    point near a block edge therefore interpolates across that edge exactly as it
    would have without blocking, while a task carries a megabyte instead of the
    whole 256 MB raster.

    The in-bounds mask is applied before sampling, not after. Points outside the
    block are discarded either way, and not sampling them is what keeps the slice
    to one cell of halo rather than the tile's full overhang.

    Heights are clamped up to zero rather than filtered against it. Ground is an
    unbiased estimate, so returns sit on both sides of it, and dropping the ones
    below cost a measured 0.9% of cells on a real cloud -- cells that are bare
    ground, not cells with nothing in them. `GROUND_TOLERANCE_M` bounds how far
    below the clamp reaches.
    """
    for x, y, z, _ in reader(block_bounds(lattice, resolution), SURFACE_CLASSES):
        index, inside = cell_indices(x, y, lattice, resolution)
        index, x, y, z = index[inside], x[inside], y[inside], z[inside]
        above = z - sample_bilinear(ground, x, y, ground_lattice, resolution)
        usable = (
            (above >= -GROUND_TOLERANCE_M)
            & (above < MAX_CANOPY_HEIGHT_M)
            & np.isfinite(above)
        )
        yield (
            index[usable],
            np.maximum(above[usable], MIN_CANOPY_HEIGHT_M).astype(np.float32),
        )


def height_block(reader, ground, ground_lattice, lattice, resolution, statistic):
    """One statistic of the heights above ground per cell, for one block.

    Args:
        reader: Per-block point reader.
        ground: This block's slice of the ground raster, grown by one cell.
        ground_lattice: Lattice of that slice.
        lattice: The block's own lattice.
        resolution: Cell size in metres.
        statistic: ``(method, percentile)``, with ``percentile`` set only for
            ``"percentile"``.

    Returns:
        The block's ``(rows, cols)`` float32 raster, NaN where no return fell.
    """
    method, percentile = statistic
    _, _, height, width = lattice
    size = height * width
    batches = point_heights(reader, ground, ground_lattice, lattice, resolution)

    if method == "max":
        chm = np.full(size, -np.inf, dtype=np.float32)
        for index, above in batches:
            np.maximum.at(chm, index, above)
        chm[~np.isfinite(chm)] = np.nan
        return chm.reshape(height, width)

    if method == "mean":
        # A sum and a count, for the reason `mean_surface_block` gives: both
        # are commutative and cell-local, so a batch can be folded and dropped
        # and the answer does not depend on how the points were grouped.
        total, count = np.zeros(size), np.zeros(size)
        for index, above in batches:
            total += np.bincount(index, above, size)
            count += np.bincount(index, minlength=size)
        with np.errstate(invalid="ignore"):
            return (total / count).reshape(height, width).astype(np.float32)

    return _percentile_block(batches, percentile, size).reshape(height, width)


def _percentile_block(batches, percentile: float, size: int) -> np.ndarray:
    """Linearly interpolated percentile per cell, over retained returns.

    A percentile is not a fold: no state smaller than the cell's whole set of
    returns answers it, because which return is the answer is not known until
    the last one has arrived. So this holds every usable return in the block --
    a cell index and a height, 8 bytes -- and reduces once at the end.

    That is why a percentile is blocked by its points rather than by its cells;
    `chm_point_cloud._retaining_block_cells` sizes the blocks that reach here.

    Order still cannot matter. Sorting is a function of the multiset, so any
    grouping of the same returns produces the same sorted run and the same
    arithmetic on it -- bit for bit, not approximately.

    The rank is numpy's default definition (`method="linear"`, Hyndman and Fan
    type 7): the percentile sits at ``(n - 1) * p / 100`` in the sorted run, and
    between two returns the height is interpolated.
    """
    out = np.full(size, np.nan, dtype=np.float32)
    indices, heights = [], []
    for index, above in batches:
        indices.append(index.astype(np.int32))
        heights.append(above)
    if not indices:
        return out

    index = np.concatenate(indices)
    height = np.concatenate(heights)
    del indices, heights
    if index.size == 0:
        return out

    order = np.lexsort((height, index))
    index, height = index[order], height[order]
    del order

    starts = np.flatnonzero(np.concatenate(([True], index[1:] != index[:-1])))
    counts = np.diff(np.concatenate((starts, [index.size])))
    virtual = (counts - 1) * (percentile / 100.0)
    low = np.floor(virtual).astype(np.int64)
    high = np.minimum(low + 1, counts - 1)
    fraction = virtual - low
    lower = height[starts + low].astype(np.float64)
    upper = height[starts + high].astype(np.float64)
    out[index[starts]] = lower + (upper - lower) * fraction
    return out
