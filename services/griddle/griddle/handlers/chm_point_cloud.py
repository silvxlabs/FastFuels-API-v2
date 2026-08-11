"""
Canopy height model from a point cloud.

Rasterises height above ground onto the lattice the alignment defines, from the
point cloud's partitioned Parquet dataset.

The work is blocked over that lattice: each block reads only the cloud
partitions it overlaps and rasterises them itself, so no pass ever holds more
than one block's points and peak memory tracks the output grid rather than the
cloud. Blocking does not change the answer — the point passes are commutative
scatter-reductions, and the raster steps that are not cell-local run through a
halo at least as wide as their dependency radius.

Ground comes from the cloud's own ASPRS class 2 returns when it has them. When
it does not — user uploads carry no guarantee of a classification, let alone an
ASPRS-conformant one — the ground surface is derived from the data with a
progressive morphological filter (Zhang et al. 2003) at the parameters PDAL's
``filters.pmf`` uses. That derivation is good on forested terrain (0.08-0.61 m
RMSE against vendor-classified ground across eight 3DEP clouds) and degrades
where contiguous ground-return-free voids are wider than the filter's largest
window — large buildings, closed evergreen canopy. ``ground_coverage`` and
``max_ground_distance_m`` are reported on the grid's source so a bad result has
a visible cause.
"""

import math
import multiprocessing
from collections.abc import Callable
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from contextlib import contextmanager

import dask
import dask.array as da
import geopandas as gpd
import numpy as np
import rioxarray  # noqa: F401
import xarray as xr
from affine import Affine
from scipy.ndimage import (
    distance_transform_edt,
    grey_opening,
    maximum_filter,
    uniform_filter,
)

from griddle.handlers import chm_blocks
from griddle.handlers.chm_blocks import GROUND_CLASS, SURFACE_CLASSES
from lib.alignment import resolve_alignment_destination
from lib.config import (
    GRIDDLE_DASK_WORKERS,
    GRIDDLE_READ_WORKERS,
    POINT_CLOUDS_BUCKET,
)
from lib.crs import crs_equal
from lib.errors import ProcessingError
from lib.pointcloud.reader import read_manifest
from lib.pointcloud.schema import cloud_prefix

# Block edge in cells, before the cloud-tile floor in `_compute_block_cells`.
# Matches griddle's default storage chunk, so at 1 m and coarser the compute
# blocks and the written chunks are the same size.
DEFAULT_BLOCK_CELLS = 512

# How far `max_ground_distance_m` is measured before it saturates. The number
# only carries information near the distances the filter can bridge — 30 m of
# fill, a 33 m widest window — so measuring further costs halo for nothing.
GROUND_DISTANCE_CAP_M = 60.0

# How far a ground gap may be filled by interpolation. Beyond this a cell stays
# nodata rather than being extrapolated across a large void.
#
# In metres, converted to cells at run time, for the reason `_pmf` gives about
# its windows: this is a physical distance, and a fixed cell count would mean
# 30 m of interpolation at 1 m cells but 900 m at 30 m cells. 30 m keeps the
# behaviour the derived-ground accuracy in the module docstring was measured
# at, and sits at the same scale as the filter's largest window.
GROUND_FILL_MAX_M = 30.0

# filters.pmf's defaults, expressed in metres so they mean the same thing at
# any resolution. The window is converted to cells at run time.
PMF_MAX_WINDOW_M = 33.0
PMF_SLOPE = 1.0
PMF_INITIAL_DISTANCE_M = 0.15
PMF_MAX_DISTANCE_M = 2.5

# A cell whose height exceeds *every* neighbour by more than this is a lone
# spurious return rather than a treetop. Applied to the finished raster so the
# band stays a true maximum.
#
# The threshold sits deliberately high. Measured noise returns stood 40-80 m
# above their surroundings, while a real crown spans several cells so its peak
# is within a few metres of its neighbours. Setting it near canopy scale would
# delete isolated trees, which are real fuel — a lone 15 m tree in a meadow is
# a one-cell spike by any tighter rule. The residual risk is a single-cell tree
# more than this much taller than everything around it, which needs a crown
# narrower than one cell to arise.
SPIKE_THRESHOLD_M = 25.0

# What the shared alignment resolver falls back to when `target="domain"` names
# no resolution. The API resolves that default at create time and stores it, so
# this is unreachable for any grid the API created — it only keeps a malformed
# source from becoming a TypeError deep in the lattice math.
FALLBACK_RESOLUTION_M = 1.0

# Filesystem the block workers read through; None means GCS. A worker is a
# separate interpreter, so patching this module cannot reach one -- an override
# has to be a value that travels in the pool initializer, which is what this is.
# Only set when reading a dataset off local disk.
BLOCK_FILESYSTEM = None


def fetch_point_cloud_chm(
    roi: gpd.GeoDataFrame,
    point_cloud_id: str,
    point_classes: list[int],
    alignment: dict,
    progress: Callable[[str, int | None], None],
    target_grid_doc: dict | None = None,
    extent_buffer_cells: int = 0,
) -> tuple[xr.Dataset, dict]:
    """Build a canopy height model from a stored point cloud.

    The output lattice comes from the alignment, and the work is blocked over
    it: each block reads only the cloud partitions it overlaps and rasterises
    them itself. Peak memory is a few rasters over the output grid plus one
    block's points, so it tracks the grid rather than the cloud.

    Blocking does not change the answer. The point passes are commutative
    scatter-reductions, so their result is independent of how points are
    grouped, and the raster steps that are not cell-local run through
    `map_overlap` with a halo at least as wide as their dependency radius.

    Args:
        roi: Domain geometry, in the domain's projected CRS. The stored cloud is
            already in this CRS — the uploader reprojects on ingest and lakitu
            writes 3DEP clouds in the domain CRS — so no reprojection happens.
        point_cloud_id: Point cloud whose dataset to read.
        point_classes: ASPRS classes present in the cloud, from the point cloud
            document's ``summary.point_classes``. Decides whether ground is read
            from the classification or derived.
        alignment: Persisted alignment spec deciding the output lattice.
        progress: Progress callback (message, percent).
        target_grid_doc: Grid document named by ``alignment.grid_id``, required
            when ``alignment.target`` is ``"grid"``.
        extent_buffer_cells: Output cells of buffer around the extent.

    Returns:
        Tuple of (Dataset with the ``chm`` variable, provenance dict recording
        how ground was obtained and how well constrained it was).

    Raises:
        ProcessingError: If the alignment cannot produce a lattice this handler
            can rasterize onto, or if the cloud contributes no points to it.
    """
    with dask.config.set(scheduler="threads", num_workers=GRIDDLE_DASK_WORKERS):
        return _fetch(
            roi,
            point_cloud_id,
            point_classes,
            alignment,
            progress,
            target_grid_doc,
            extent_buffer_cells,
        )


def _fetch(
    roi: gpd.GeoDataFrame,
    point_cloud_id: str,
    point_classes: list[int],
    alignment: dict,
    progress: Callable[[str, int | None], None],
    target_grid_doc: dict | None,
    extent_buffer_cells: int,
) -> tuple[xr.Dataset, dict]:
    """Body of `fetch_point_cloud_chm`, under a pinned dask thread pool."""
    transform, (height, width) = _resolve_lattice(
        roi, alignment, target_grid_doc, extent_buffer_cells
    )
    resolution = transform.a

    progress("Reading point cloud...", 10)
    prefix = cloud_prefix(POINT_CLOUDS_BUCKET, point_cloud_id)
    manifest = read_manifest(prefix)
    fill_cells = _fill_cells(resolution)
    # The widest halo any step needs, which is what the blocking has to be able
    # to feed. PMF only runs for a cloud with no ground class, so a classified
    # cloud is not charged for its much wider reach.
    halo_cells = max(
        fill_cells,
        int(np.ceil(GROUND_DISTANCE_CAP_M / resolution)),
        0 if GROUND_CLASS in point_classes else _pmf_depth_cells(resolution),
    )
    tile_m, tile_origin = manifest["tile_m"], manifest["mins"]
    block_cells = _compute_block_cells(resolution, tile_m, halo_cells)
    # A block only has to be wide enough to feed the widest halo; anything more
    # is a perimeter block merged past a tile boundary, reading a second tile to
    # use part of it. See `_block_slices`.
    min_block_cells = halo_cells + 1
    blocks = (
        _block_slices(
            height,
            block_cells,
            _tile_cuts(height, transform.f, transform.e, tile_origin[1], tile_m),
            min_block_cells,
        ),
        _block_slices(
            width,
            block_cells,
            _tile_cuts(width, transform.c, transform.a, tile_origin[0], tile_m),
            min_block_cells,
        ),
    )

    tiles = [
        (row0, row1, col0, col1) for row0, row1 in blocks[0] for col0, col1 in blocks[1]
    ]

    # Chunks for every dask pass below, taken from the same division the reads
    # use. `da.from_array(..., chunks=block_cells)` would lay down whole blocks
    # and leave `extent % block_cells` over, and `_overlap_depth` caps the halo
    # by the *smallest* chunk in the array — so a one-cell remainder truncates
    # the halo for every block, not just its own.
    block_chunks = tuple(tuple(stop - start for start, stop in axis) for axis in blocks)

    with _block_pool(prefix, manifest, BLOCK_FILESYSTEM) as pool:

        def over_blocks(kind, classes=None, extra_for=None):
            """Rasterise every block on the pool and assemble the grid.

            `extra_for` supplies whatever raster a pass samples, per block, so a
            task carries its own slice rather than the whole grid.
            """
            return _run_blocks(
                pool,
                transform,
                tiles,
                (height, width),
                kind,
                resolution,
                classes,
                extra_for,
            )

        if GROUND_CLASS in point_classes:
            progress("Reading ground returns...", 15)
            ground = over_blocks("min", classes=(GROUND_CLASS,))
            ground_source = "classification"
        else:
            progress("Deriving ground surface...", 15)
            minimum = da.from_array(
                over_blocks("min", classes=SURFACE_CLASSES),
                chunks=block_chunks,
            )
            # Each step re-halos from the materialised intermediate, so the
            # depths are per-step rather than one accumulated worst case.
            provisional = da.map_overlap(
                _fill_gaps,
                minimum,
                depth=_overlap_depth(minimum, fill_cells),
                boundary="none",
                dtype=np.float32,
                max_cells=fill_cells,
            )
            provisional = da.map_overlap(
                _pmf,
                provisional,
                depth=_overlap_depth(provisional, _pmf_depth_cells(resolution)),
                boundary="none",
                dtype=np.float32,
                resolution=resolution,
            )
            progress("Separating ground from cover...", 30)
            provisional = np.asarray(provisional.compute())

            progress("Re-reading ground returns...", 40)
            ground = over_blocks(
                "snap",
                extra_for=lambda bl: _block_of(provisional, transform, bl, resolution),
            )
            ground_source = "derived"

        # Ground is needed whole from here: the provenance reduces over all of
        # it, and the height pass interpolates it across block edges.
        known_ground = np.isfinite(ground)
        coverage = float(known_ground.mean())

        ground_distance_m = _blocked_ground_distance(
            known_ground, block_chunks, resolution
        )

        progress("Filling ground gaps...", 45)
        ground = _blocked_fill_gaps(ground, block_chunks, fill_cells)

        progress("Rasterizing canopy heights...", 55)
        chm = over_blocks(
            "max",
            extra_for=lambda bl: _ground_window(ground, transform, bl, resolution),
        )

    blocked_chm = da.from_array(chm, chunks=block_chunks)
    chm = np.asarray(
        da.map_overlap(
            _remove_spikes,
            blocked_chm,
            depth=_overlap_depth(blocked_chm, 1),
            boundary="none",
            dtype=np.float32,
            threshold=SPIKE_THRESHOLD_M,
        ).compute()
    )

    if not np.isfinite(chm).any():
        raise ProcessingError(
            code="EMPTY_POINT_CLOUD",
            message="No point cloud returns fall inside this domain.",
            suggestion=(
                "Check that the point cloud covers the domain before creating "
                "a canopy height grid from it."
            ),
        )

    progress("Building dataset...", 85)
    ds = _to_dataset(chm, transform, roi.crs)

    provenance = {
        "ground_source": ground_source,
        "ground_coverage": round(coverage, 4),
        "max_ground_distance_m": round(ground_distance_m, 1),
    }
    return ds, provenance


def _blocked_ground_distance(known, block_chunks, resolution) -> float:
    """Furthest any cell sits from a ground return, in metres, saturating.

    Blocked with a halo as wide as the cap, so a cell whose nearest ground lies
    within the cap finds it inside its own halo and the answer below the cap is
    exact. `max` is commutative, so combining the blocks is just a reduction.

    The halo is trimmed before that reduction, which is not cosmetic. A halo
    cell is only as well served as *its* own halo, which it does not have, so
    its distance can be an overestimate; reducing over it let the answer depend
    on the chunking rather than on the data. Measured on one 2 km grid, the same
    ground read 40 m at a 512-cell block and 38 m at 500.
    """
    cap_cells = GROUND_DISTANCE_CAP_M / resolution
    blocks = da.from_array(known, chunks=block_chunks)
    distance = da.map_overlap(
        _ground_distance_cells,
        blocks,
        depth=_overlap_depth(blocks, int(np.ceil(cap_cells))),
        boundary="none",
        dtype=np.float64,
        cap_cells=cap_cells,
    )
    return float(distance.max().compute()) * resolution


class _InlineExecutor:
    """Runs each block in the calling process, for a single worker.

    A pool of one buys no concurrency and costs a whole interpreter, and reading
    in-process is also what lets a caller stub the reader -- a patch in this
    process cannot reach a forkserver child.
    """

    def submit(self, function, *args):
        future = Future()
        try:
            future.set_result(function(*args))
        except Exception as e:
            future.set_exception(e)
        return future


@contextmanager
def _block_pool(prefix: str, manifest: dict, filesystem=None):
    """One forkserver pool serving every point pass of a run.

    forkserver, not fork: gcsfs pins its event loop to the PID that built it and
    refuses to run in a forked child (`lib.gcs.blobs`), and this process has
    already built one. Each worker opens its own dataset in the initializer, so
    the handles are never shared across the fork.

    One pool rather than one per pass. Starting it costs a fresh interpreter and
    a pyarrow import per worker, which is why `chm_blocks` is import-light and
    why the pool spans the ground read, the snap and the height pass instead of
    being rebuilt between them.
    """
    if GRIDDLE_READ_WORKERS <= 1:
        chm_blocks.worker_init(prefix, manifest, filesystem)
        yield _InlineExecutor()
        return

    context = multiprocessing.get_context("forkserver")
    with ProcessPoolExecutor(
        max_workers=GRIDDLE_READ_WORKERS,
        mp_context=context,
        initializer=chm_blocks.worker_init,
        initargs=(prefix, manifest, filesystem),
    ) as pool:
        yield pool


def _run_blocks(
    pool, transform, tiles, shape, kind, resolution, classes, extra_for
) -> np.ndarray:
    """Rasterise every block on `pool` and write the results into one grid.

    Results are written into a preallocated grid as they complete rather than
    concatenated at the end, so the parent holds one output raster and not also
    a list of every block that made it.

    Raises:
        ProcessingError: If a worker dies, which surfaces from the pool as a
            bare BrokenProcessPool with no indication of what was running.
    """
    out = np.empty(shape, dtype=np.float32)
    submitted = {}
    for row0, row1, col0, col1 in tiles:
        lattice = chm_blocks.block_lattice(
            transform, row0, col0, row1 - row0, col1 - col0
        )
        extra = extra_for(lattice) if extra_for is not None else None
        job = (kind, lattice, resolution, classes, extra)
        submitted[pool.submit(chm_blocks.run_block, job)] = (row0, row1, col0, col1)

    try:
        for future in as_completed(submitted):
            row0, row1, col0, col1 = submitted[future]
            out[row0:row1, col0:col1] = future.result()
    except BrokenProcessPool as e:
        raise ProcessingError(
            code="POINT_CLOUD_UNREADABLE",
            message="This point cloud's stored data could not be read.",
            suggestion="Retry, and contact support if it fails again.",
            # A worker dying is nearly always the container running out of
            # memory, which Cloud Run does not log -- the child is killed, not
            # the server, so nothing appears above this line. GRIDDLE_READ_WORKERS
            # against the memory limit is the first thing to check.
            traceback=(
                f"block worker died during the {kind!r} pass with "
                f"{GRIDDLE_READ_WORKERS} workers, usually out of memory: {e}"
            ),
        ) from e
    return out


def _ground_window(ground, transform, block_lattice, resolution):
    """The block's slice of the ground raster grown by one cell, and its lattice.

    One cell is exactly what bilinear sampling of a point inside the block can
    reach -- the sample sits half a cell back, so a point in the first column
    reads columns -1 and 0. Clipped to the grid, which costs nothing: sampling
    clamps to the array's own edge, and clamping to a slice that stops at the
    grid edge is the same as clamping to the grid.
    """
    origin_x, origin_y, rows, cols = block_lattice
    col0 = int(round((origin_x - transform.c) / resolution))
    row0 = int(round((transform.f - origin_y) / resolution))
    grid_rows, grid_cols = ground.shape
    lo_row, lo_col = max(0, row0 - 1), max(0, col0 - 1)
    hi_row = min(grid_rows, row0 + rows + 1)
    hi_col = min(grid_cols, col0 + cols + 1)
    window = ground[lo_row:hi_row, lo_col:hi_col]
    lattice = (
        transform.c + lo_col * resolution,
        transform.f - lo_row * resolution,
        hi_row - lo_row,
        hi_col - lo_col,
    )
    return np.ascontiguousarray(window), lattice


def _blocked_fill_gaps(surface, block_chunks, max_cells: int) -> np.ndarray:
    """`_fill_gaps` over blocks, with a halo as wide as its reach.

    The only whole-grid pass left in the handler, and the reason it had to go:
    `_fill_gaps` is pure array work with no dask around it, so it ran at exactly
    1.00 core in every in-region measurement no matter what the container was
    given. At 64 km2 that was 36-82 s the allocation could not touch.

    Blocking is exact here. Each iteration extends the filled region by one cell
    and there are `max_cells` of them, so a cell's value depends only on inputs
    within `max_cells` and a halo that wide feeds it everything the whole-grid
    pass would have seen. `boundary="none"` leaves the grid's own edges to be
    handled by `uniform_filter`'s `mode="nearest"`, exactly as before.

    Most blocks also finish far sooner than the whole grid does: `_fill_gaps`
    breaks as soon as nothing is missing, and gaps are local, so a block with no
    NaN costs one pass instead of `max_cells` of them.
    """
    blocks = da.from_array(surface, chunks=block_chunks)
    filled = da.map_overlap(
        _fill_gaps,
        blocks,
        depth=_overlap_depth(blocks, max_cells),
        boundary="none",
        dtype=np.float32,
        max_cells=max_cells,
    )
    return np.asarray(filled.compute())


def _block_of(surface, transform, block_lattice, resolution) -> np.ndarray:
    """The slice of a whole-grid raster covering one block."""
    origin_x, origin_y, rows, cols = block_lattice
    col0 = int(round((origin_x - transform.c) / resolution))
    row0 = int(round((transform.f - origin_y) / resolution))
    return surface[row0 : row0 + rows, col0 : col0 + cols]


def _resolve_lattice(
    roi: gpd.GeoDataFrame,
    alignment: dict,
    target_grid_doc: dict | None,
    extent_buffer_cells: int,
) -> tuple[Affine, tuple[int, int]]:
    """Return (transform, (height, width)) for the output lattice.

    Goes through the shared resolver every other source handler uses, so the
    same request produces the same lattice here as it would from a raster
    source: ``target="domain"`` on the domain lattice, ``target="grid"``
    cell-for-cell on another grid — either matching it exactly, or keeping its
    origin at a new cell size.

    Raises:
        ProcessingError: If the alignment names no lattice this handler can
            rasterize onto, or names one in another CRS.
    """
    destination = resolve_alignment_destination(
        alignment,
        roi,
        target_grid_doc,
        FALLBACK_RESOLUTION_M,
        extent_buffer_cells=extent_buffer_cells,
    )

    # The resolver returns a bare CRS override (or nothing at all) for
    # `target="native"`, which means "reproject, keep the source's own pixel
    # anchor" — there is no source raster here to take an anchor from. The API
    # rejects native at create time; this covers a source that reached storage
    # some other way.
    if "destination_transform" not in destination:
        raise ProcessingError(
            code="UNSUPPORTED_ALIGNMENT",
            message=(
                f"alignment.target '{alignment['target']}' does not describe a "
                f"lattice a point cloud can be rasterized onto."
            ),
            suggestion="Recreate the grid with alignment.target 'domain' or 'grid'.",
        )

    # Returns are read in the domain CRS and this handler does not reproject,
    # so a target lattice in another CRS would place every cell wrongly.
    destination_crs = destination["destination_crs"]
    if not crs_equal(str(destination_crs), str(roi.crs)):
        raise ProcessingError(
            code="ALIGNMENT_CRS_MISMATCH",
            message=(
                f"The alignment target grid is in {destination_crs}, but this "
                f"domain's point clouds are stored in {roi.crs}."
            ),
            suggestion=(
                "Align to a grid in this domain's CRS, or use "
                "alignment.target 'domain'."
            ),
        )

    return destination["destination_transform"], destination["destination_shape"]


def _compute_block_cells(resolution: float, tile_m: float, halo_cells: int) -> int:
    """Block edge in cells, rounded up to a whole number of cloud tiles.

    Two floors decide the size:

    - the default, which matches griddle's storage chunk so that at 1 m and
      coarser the compute blocks and the written chunks coincide;
    - twice the widest halo. This one is correctness, not economy: a halo wider
      than a block cannot be supplied.

    A whole number of tiles because a block is only cheap to read if it stops
    where a partition stops. `iter_points` prunes on the partition columns and
    nothing finer, so a block overlapping a tile by one cell fetches that tile
    entirely — a block sized like a tile but out of step with it reads four
    where it needs one. Measured at 1 m on a 4 km domain: 4.89 partition reads
    per tile over a run, against the 2 the two passes actually need.

    The halo floor is rounded *up* to a whole tile because it is a correctness
    bound. The default is rounded to the *nearest* whole tile instead: it only
    buys matching storage chunks, which `save_zarr` restores on write anyway,
    and rounding it up would double a block's area — and a block's points are
    what peak memory tracks.
    """
    tile_cells = tile_m / resolution
    tiles_per_block = max(
        1,
        math.ceil(2 * (halo_cells + 1) / tile_cells),
        round(DEFAULT_BLOCK_CELLS / tile_cells),
    )
    return max(1, int(round(tiles_per_block * tile_cells)))


def _tile_cuts(
    count: int, anchor: float, step: float, tile_origin: float, tile_m: float
) -> list[int]:
    """Cell indices strictly inside ``(0, count)`` where a cloud tile begins.

    Args:
        count: Cells on this axis.
        anchor: World coordinate of cell zero's leading edge.
        step: World distance per cell, negative on the row axis, where the
            lattice runs north to south while the index runs downward.
        tile_origin: The cloud's own origin on this axis, which anchors its
            tiling — ``manifest["mins"]``, never the lattice.
        tile_m: Tile edge in metres.

    Returns:
        Ascending cell indices, whichever way the axis runs. Empty when no tile
        boundary falls inside the extent, which is the single-tile case.
    """
    if tile_m <= 0 or step == 0:
        return []
    low = min(anchor, anchor + step * count)
    high = max(anchor, anchor + step * count)
    first = math.floor((low - tile_origin) / tile_m) + 1
    last = math.ceil((high - tile_origin) / tile_m)
    cuts = set()
    for k in range(first, last):
        index = int(round((tile_origin + k * tile_m - anchor) / step))
        if 0 < index < count:
            cuts.add(index)
    return sorted(cuts)


def _block_slices(
    extent: int, block: int, cuts: list[int] | None = None, min_block: int | None = None
) -> list[tuple[int, int]]:
    """Half-open (start, stop) pairs tiling `extent`.

    Cuts on cloud tile boundaries where there are any, so a block reads as few
    partitions as it can. A boundary closer than `min_block` to the previous cut,
    or to the end, is skipped rather than emitted, and the short piece merges
    into its neighbour — so no block comes out too narrow to feed the halo the
    overlapped steps need.

    `min_block` is that halo floor and nothing else; it defaults to `block` only
    so a caller with no halo to protect keeps the old shape. Passing `block`
    when the halo is far smaller is expensive rather than safe: the short piece
    merges into a neighbour that then grows past a tile boundary, and since the
    lattice has no reason to start on one, that neighbour is a perimeter block
    straddling two tiles per axis to use part of the second. It reads the whole
    of both, so the widest block is also the most wasteful, and peak memory
    follows the widest block.

    Falls back to an even division when the extent holds no usable boundary.
    Even rather than greedy: a greedy split leaves a remainder block that can be
    a single cell wide, and a block narrower than the halo silently under-feeds
    those steps.
    """
    min_block = block if min_block is None else min_block
    edges = [0]
    for cut in cuts or ():
        if cut - edges[-1] >= min_block and extent - cut >= min_block:
            edges.append(cut)
    if len(edges) == 1:
        count = max(1, math.ceil(extent / block))
        edges = [round(index * extent / count) for index in range(count)]
    edges.append(extent)
    return list(zip(edges[:-1], edges[1:], strict=True))


def _overlap_depth(array, required: int) -> int:
    """Halo to ask dask for, capped by what the blocking can supply.

    The cap is taken over the *whole* array, so one narrow chunk shortens the
    halo everywhere — and a step given too little halo does not fail, it returns
    values computed without the neighbours it needed. Every caller therefore
    chunks on `_block_slices`, which divides evenly for exactly this reason;
    handing `da.from_array` a scalar block size instead leaves
    ``extent % block`` over, and a one-cell remainder takes the halo to zero.

    With that division the cap binds only on a single-block grid, where the
    block already holds every cell any step can reach and the halo is moot.
    """
    smallest = min(min(sizes) for sizes in array.chunks)
    return int(max(0, min(required, smallest - 1)))


def _pmf(surface: np.ndarray, resolution: float) -> np.ndarray:
    """Progressive morphological filter (Zhang et al. 2003).

    Grey-opening removes anything raised and narrower than the window while
    leaving broader features intact, so growing the window strips shrubs, then
    trees, then buildings. The opened value is only accepted where the drop it
    causes exceeds a slope-derived tolerance, which is what stops a wide window
    from flattening a real ridge.

    Windows are specified in metres and converted to cells here: they are a
    physical distance, and a fixed cell count would silently mean something
    different at every resolution.
    """
    max_window_cells = max(3, round(PMF_MAX_WINDOW_M / resolution))
    ground = surface.copy()
    previous = 1
    exponent = 0
    while True:
        window = 2 * (2**exponent) + 1
        if window > max_window_cells:
            break
        opened = grey_opening(ground, size=(window, window), mode="nearest")
        if window <= 3:
            tolerance = PMF_INITIAL_DISTANCE_M
        else:
            tolerance = min(
                PMF_SLOPE * (window - previous) * resolution + PMF_INITIAL_DISTANCE_M,
                PMF_MAX_DISTANCE_M,
            )
        ground = np.where(ground - opened > tolerance, opened, ground)
        previous = window
        exponent += 1
    return ground


def _pmf_depth_cells(resolution: float) -> int:
    """Halo `_pmf` needs, in cells.

    `grey_opening(W)` is an erosion followed by a dilation, so an output cell
    depends on inputs within `W - 1`. The filter reapplies that over a ladder of
    windows and each pass reads the previous one's output, so the radii add:
    ~62 cells at 1 m, and near 60 m at any resolution since the windows are
    themselves derived from a distance.
    """
    max_window_cells = max(3, round(PMF_MAX_WINDOW_M / resolution))
    depth, exponent = 0, 0
    while True:
        window = 2 * (2**exponent) + 1
        if window > max_window_cells:
            return max(1, depth)
        depth += window - 1
        exponent += 1


def _fill_cells(resolution: float) -> int:
    """``GROUND_FILL_MAX_M`` as a whole number of cells, at least one.

    A cell coarser than the reach would otherwise round to zero and disable
    the fill outright.
    """
    return max(1, round(GROUND_FILL_MAX_M / resolution))


def _fill_gaps(surface: np.ndarray, max_cells: int) -> np.ndarray:
    """Interpolate NaN cells from their finite neighbours, bounded in reach.

    Each iteration extends the filled region by one cell, so `max_cells` caps
    how far a value travels. Anything further from real data stays NaN rather
    than being extrapolated across a void the surface says nothing about.

    Takes a cell count rather than a distance so the bound stays testable in
    the units it is applied in; `_fill_cells` converts.
    """
    filled = surface.copy()
    for _ in range(max_cells):
        missing = np.isnan(filled)
        if not missing.any():
            break
        known = ~missing
        total = uniform_filter(np.where(known, filled, 0.0), size=3, mode="nearest")
        weight = uniform_filter(known.astype(np.float32), size=3, mode="nearest")
        fillable = missing & (weight > 0)
        if not fillable.any():
            break
        filled[fillable] = total[fillable] / weight[fillable]
    return filled


def _remove_spikes(chm: np.ndarray, threshold: float) -> np.ndarray:
    """Drop cells that tower over every neighbour by more than `threshold`.

    The footprint excludes its own centre so the comparison is against the
    neighbours alone, and `cval=-inf` keeps the frame from wrapping around.
    """
    footprint = np.ones((3, 3), dtype=bool)
    footprint[1, 1] = False
    filled = np.where(np.isnan(chm), -np.inf, chm)
    neighbour_max = maximum_filter(
        filled, footprint=footprint, mode="constant", cval=-np.inf
    )
    spikes = np.isfinite(chm) & np.isfinite(neighbour_max)
    spikes &= chm - neighbour_max > threshold
    despiked = chm.copy()
    despiked[spikes] = np.nan
    return despiked


def _ground_distance_cells(known: np.ndarray, cap_cells: float) -> np.ndarray:
    """How far each cell sits from a ground return, in cells, saturating at `cap_cells`.

    This is the variable that predicted ground-derivation error across the
    validation clouds — better than coverage alone, since scattered gaps
    interpolate fine while one wide void does not.

    Bounded rather than global, and blocked rather than whole-grid, because the
    number only carries information near the distances the filter can bridge:
    `GROUND_FILL_MAX_M` is 30 m and PMF's widest window is 33 m, so anything
    past the cap says "badly constrained" and nothing more. Computed on a block
    plus a halo of `cap_cells`, any cell whose true nearest ground is within the
    cap has it inside the halo, so the answer below the cap is exact.

    Per cell rather than reduced here, so the caller can drop the halo before
    reducing — see `_blocked_ground_distance` on why that matters.

    `distance_transform_edt` is called for distances only; asking it for indices
    as well is what allocates gigabytes on a large grid.
    """
    if known.all():
        return np.zeros(known.shape, dtype=np.float64)
    if not known.any():
        # No ground anywhere in reach, so every cell is at least the cap away.
        # `distance_transform_edt` has no zero to measure to in this case.
        return np.full(known.shape, float(cap_cells), dtype=np.float64)
    return np.minimum(distance_transform_edt(~known), cap_cells)


def _to_dataset(chm, transform, crs) -> xr.Dataset:
    """Wrap the CHM array as a georeferenced Dataset with a single band.

    Cell centres are read off the transform rather than recomputed from bounds,
    so the coordinates and the transform written beside them cannot disagree.
    """
    height, width = chm.shape
    x_coords = transform.c + (np.arange(width) + 0.5) * transform.a
    y_coords = transform.f + (np.arange(height) + 0.5) * transform.e

    da = xr.DataArray(
        chm.astype(np.float32), dims=["y", "x"], coords={"y": y_coords, "x": x_coords}
    )
    da = da.rio.write_nodata(np.float32("nan"))

    ds = xr.Dataset({"chm": da})
    ds = ds.rio.write_crs(crs)
    ds = ds.rio.write_transform(transform)
    return ds
