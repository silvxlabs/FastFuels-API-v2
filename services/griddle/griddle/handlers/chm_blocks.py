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
value depends only on points inside it, and the reductions are commutative, so
how the points are grouped cannot matter -- and for the same reason a block can
be folded a batch at a time rather than read whole, which is what keeps a
worker's memory off the tile's point count.
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

# Heights outside this range are not canopy. Matches the v1 sanity filter.
MIN_CANOPY_HEIGHT_M = 0.0
MAX_CANOPY_HEIGHT_M = 100.0

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
            provisional surface for ``"snap"``, and ``(slice, lattice)`` of the
            ground for ``"max"``. Both are block-sized, so a task stays small.

    Returns:
        The block's ``(rows, cols)`` float32 raster.
    """
    from lib.pointcloud.reader import iter_points

    kind, lattice, resolution, classes, extra = job

    def reader(bounds, wanted):
        return iter_points(
            _W["dataset"], _W["manifest"], bounds, wanted, _W["filesystem"]
        )

    if kind == "min":
        return min_surface_block(reader, lattice, resolution, classes)
    if kind == "snap":
        return snap_ground_block(reader, extra, lattice, resolution)
    if kind == "max":
        ground, ground_lattice = extra
        return max_height_block(reader, ground, ground_lattice, lattice, resolution)
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


def min_surface_block(reader, lattice, resolution, classes) -> np.ndarray:
    """Lowest z per cell over the given classes, NaN where no point falls.

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

    `provisional` is this block's slice of the filtered surface, so the lookup
    is block-local -- a point's snap test reads its own cell and no other.
    """
    _, _, height, width = lattice
    ground = np.full(height * width, np.inf, dtype=np.float32)
    flat = provisional.reshape(-1)
    for x, y, z, _ in reader(block_bounds(lattice, resolution), SURFACE_CLASSES):
        index, inside = cell_indices(x, y, lattice, resolution)
        z, index = z[inside], index[inside]
        near = np.abs(z - flat[index]) <= GROUND_SNAP_TOLERANCE_M
        np.minimum.at(ground, index[near], z[near].astype(np.float32))
    ground[~np.isfinite(ground)] = np.nan
    return ground.reshape(height, width)


def max_height_block(reader, ground, ground_lattice, lattice, resolution):
    """Highest height-above-ground per cell, for one block.

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
    """
    _, _, height, width = lattice
    chm = np.full(height * width, -np.inf, dtype=np.float32)
    for x, y, z, _ in reader(block_bounds(lattice, resolution), SURFACE_CLASSES):
        index, inside = cell_indices(x, y, lattice, resolution)
        index, x, y, z = index[inside], x[inside], y[inside], z[inside]
        above = z - sample_bilinear(ground, x, y, ground_lattice, resolution)
        usable = (
            (above >= MIN_CANOPY_HEIGHT_M)
            & (above < MAX_CANOPY_HEIGHT_M)
            & np.isfinite(above)
        )
        np.maximum.at(chm, index[usable], above[usable].astype(np.float32))
    chm[~np.isfinite(chm)] = np.nan
    return chm.reshape(height, width)
