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
from collections.abc import Callable

import dask.array as da
import geopandas as gpd
import numpy as np
import rioxarray  # noqa: F401
import xarray as xr
from affine import Affine
from dask import delayed
from scipy.ndimage import (
    distance_transform_edt,
    grey_opening,
    maximum_filter,
    uniform_filter,
)

from lib.alignment import resolve_alignment_destination
from lib.config import POINT_CLOUDS_BUCKET
from lib.crs import crs_equal
from lib.errors import ProcessingError
from lib.pointcloud.reader import open_dataset, read_manifest, read_points
from lib.pointcloud.schema import cloud_prefix

# Block edge in cells, before the cloud-tile floor in `_compute_block_cells`.
# Matches griddle's default storage chunk, so at 1 m and coarser the compute
# blocks and the written chunks are the same size.
DEFAULT_BLOCK_CELLS = 512

# How far `max_ground_distance_m` is measured before it saturates. The number
# only carries information near the distances the filter can bridge — 30 m of
# fill, a 33 m widest window — so measuring further costs halo for nothing.
GROUND_DISTANCE_CAP_M = 60.0

# ASPRS classes that can contribute to a canopy surface: never-classified (0),
# unclassified (1), ground (2), and the three vegetation classes. This
# deliberately excludes noise (7, 18), water (9) and buildings (6).
#
# Two traps here. Many 3DEP acquisitions classify only ground and unclassified,
# so vegetation lives in class 1 — filtering *to* classes 3-5 returns nothing.
# And a genuinely unclassified upload is all class 0, so omitting it (as v1's
# `Classification >= 1 && <= 5` did) yields an empty CHM for exactly the clouds
# the derived-ground path exists to serve.
SURFACE_CLASSES = (0, 1, 2, 3, 4, 5)
GROUND_CLASS = 2

# Heights outside this range are not canopy. Matches the v1 sanity filter.
MIN_CANOPY_HEIGHT_M = 0.0
MAX_CANOPY_HEIGHT_M = 100.0

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

# A point within this distance of the provisional surface is taken to be a
# ground return. Used to re-derive the ground from real measurements rather
# than from the eroded and dilated surface, which reads about 0.1 m low.
GROUND_SNAP_TOLERANCE_M = 0.5

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
    transform, (height, width) = _resolve_lattice(
        roi, alignment, target_grid_doc, extent_buffer_cells
    )
    resolution = transform.a
    lattice = (transform.c, transform.f, height, width)

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
    block_cells = _compute_block_cells(resolution, manifest["tile_m"], halo_cells)
    blocks = (_block_slices(height, block_cells), _block_slices(width, block_cells))

    def reader(bounds, classes):
        # One dataset handle per call: these run on worker threads, and pyarrow
        # datasets are cheap to open next to the read they serve.
        return read_points(open_dataset(prefix), manifest, bounds, classes)

    def over_blocks(function, dtype=np.float32):
        """Assemble one dask array from a per-block function of (lattice)."""
        return da.block(
            [
                [
                    da.from_delayed(
                        delayed(function)(
                            _block_lattice(
                                transform, row0, col0, row1 - row0, col1 - col0
                            )
                        ),
                        shape=(row1 - row0, col1 - col0),
                        dtype=dtype,
                    )
                    for col0, col1 in blocks[1]
                ]
                for row0, row1 in blocks[0]
            ]
        )

    if GROUND_CLASS in point_classes:
        progress("Reading ground returns...", 15)
        ground = over_blocks(
            lambda bl: _min_surface_block(reader, bl, resolution, (GROUND_CLASS,))
        )
        ground_source = "classification"
    else:
        progress("Deriving ground surface...", 15)
        minimum = over_blocks(
            lambda bl: _min_surface_block(reader, bl, resolution, SURFACE_CLASSES)
        )
        # Each step re-halos from the materialised intermediate, so the depths
        # are per-step rather than one accumulated worst case.
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
            lambda bl: _snap_ground_block(
                reader,
                _block_of(provisional, transform, bl, resolution),
                bl,
                resolution,
            )
        )
        ground_source = "derived"

    # Ground is needed whole from here: the provenance reduces over all of it,
    # and the height pass interpolates it across block edges.
    ground = np.asarray(ground.compute())
    known_ground = np.isfinite(ground)
    coverage = float(known_ground.mean())

    ground_distance_m = _blocked_ground_distance(known_ground, block_cells, resolution)

    progress("Filling ground gaps...", 45)
    ground = _fill_gaps(ground, fill_cells)

    progress("Rasterizing canopy heights...", 55)
    chm = over_blocks(
        lambda bl: _max_height_block(reader, ground, lattice, bl, resolution)
    )
    chm = da.map_overlap(
        _remove_spikes,
        chm,
        depth=_overlap_depth(chm, 1),
        boundary="none",
        dtype=np.float32,
        threshold=SPIKE_THRESHOLD_M,
    )
    chm = np.asarray(chm.compute())

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


def _blocked_ground_distance(known, block_cells, resolution) -> float:
    """Furthest any cell sits from a ground return, in metres, saturating.

    Blocked with a halo as wide as the cap, so a cell whose nearest ground lies
    within the cap finds it inside its own halo and the answer below the cap is
    exact. `max` is commutative, so combining the blocks is just a reduction.
    """
    cap_cells = GROUND_DISTANCE_CAP_M / resolution
    blocks = da.from_array(known, chunks=(block_cells, block_cells))
    overlapped = da.overlap.overlap(
        blocks,
        depth=_overlap_depth(blocks, int(np.ceil(cap_cells))),
        boundary="none",
    )
    per_block = da.map_blocks(
        lambda block: np.array(
            [[_max_ground_distance(block, cap_cells)]], dtype=np.float64
        ),
        overlapped,
        dtype=np.float64,
        chunks=(1, 1),
    )
    return float(per_block.max().compute()) * resolution


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


def _block_lattice(transform: Affine, row0: int, col0: int, rows: int, cols: int):
    """Lattice tuple for one block, in the same form the whole-grid one takes."""
    return (
        transform.c + col0 * transform.a,
        transform.f + row0 * transform.e,
        rows,
        cols,
    )


def _block_bounds(lattice, resolution: float) -> tuple:
    """World bounds of a block lattice, for partition pruning."""
    origin_x, origin_y, rows, cols = lattice
    return (
        origin_x,
        origin_y - rows * resolution,
        origin_x + cols * resolution,
        origin_y,
    )


def _compute_block_cells(resolution: float, tile_m: float, halo_cells: int) -> int:
    """Block edge in cells.

    Three floors, in increasing order of consequence:

    - the default, which matches griddle's storage chunk so that at 1 m and
      coarser the compute blocks and the written chunks coincide;
    - one cloud tile, because a block narrower than a partition decodes that
      whole partition and keeps a fraction of it — four 256 m blocks would read
      a 500 m tile four times. Costs nothing: `save_zarr` restores the storage
      chunking when it rechunks on write;
    - twice the widest halo. This one is correctness, not economy. A halo wider
      than a block cannot be supplied, and `_block_slices` divides evenly, so a
      block can come out as small as half the requested size; sizing for twice
      the halo keeps every block able to feed it.
    """
    return max(
        DEFAULT_BLOCK_CELLS, math.ceil(tile_m / resolution), 2 * (halo_cells + 1)
    )


def _block_slices(extent: int, block: int) -> list[tuple[int, int]]:
    """Half-open (start, stop) pairs tiling `extent` in even blocks.

    Even rather than greedy: a greedy split leaves a remainder block that can be
    a single cell wide, and a block narrower than the halo silently under-feeds
    the overlapped steps.
    """
    count = max(1, math.ceil(extent / block))
    edges = [round(index * extent / count) for index in range(count + 1)]
    return list(zip(edges[:-1], edges[1:], strict=True))


def _overlap_depth(array, required: int) -> int:
    """Halo to ask dask for, capped by what the blocking can supply.

    Only binds when the grid is a single block, where the block already holds
    every cell any step can reach and the halo is moot. `_compute_block_cells`
    is what keeps it from binding anywhere it would matter.
    """
    smallest = min(min(sizes) for sizes in array.chunks)
    return int(max(0, min(required, smallest - 1)))


def _cell_indices(x, y, lattice, resolution):
    """Return (flat_index, in_bounds_mask) for points on the output lattice.

    A point contributes to the cell its coordinates fall in — square
    containment, so every return inside a cell counts exactly once.
    """
    origin_x, origin_y, height, width = lattice
    col = np.floor((x - origin_x) / resolution).astype(np.int64)
    row = np.floor((origin_y - y) / resolution).astype(np.int64)
    inside = (col >= 0) & (col < width) & (row >= 0) & (row < height)
    return row * width + col, inside


def _min_surface_block(reader, lattice, resolution, classes) -> np.ndarray:
    """Lowest z per cell over the given classes, NaN where no point falls.

    Cell-local: a cell's minimum depends only on points inside it, so a block
    needs no halo and blocked output is identical to rasterising the cloud whole.
    `np.minimum.at` is commutative, so the order points arrive in cannot change
    the answer either.
    """
    _, _, height, width = lattice
    surface = np.full(height * width, np.inf, dtype=np.float32)
    x, y, z, classification = reader(_block_bounds(lattice, resolution), classes)
    if x.size:
        index, inside = _cell_indices(x, y, lattice, resolution)
        np.minimum.at(surface, index[inside], z[inside].astype(np.float32))
    surface[~np.isfinite(surface)] = np.nan
    return surface.reshape(height, width)


def _snap_ground_block(reader, provisional, lattice, resolution) -> np.ndarray:
    """Re-derive ground from the returns the morphological filter accepts.

    The opened surface has been eroded and dilated, so reading it directly runs
    about 0.1 m low; taking the real returns that sit near it does not.

    `provisional` is this block's slice of the filtered surface, so the lookup
    is block-local — a point's snap test reads its own cell and no other.
    """
    _, _, height, width = lattice
    ground = np.full(height * width, np.inf, dtype=np.float32)
    x, y, z, classification = reader(
        _block_bounds(lattice, resolution), SURFACE_CLASSES
    )
    if x.size:
        index, inside = _cell_indices(x, y, lattice, resolution)
        z, index = z[inside], index[inside]
        near = np.abs(z - provisional.reshape(-1)[index]) <= GROUND_SNAP_TOLERANCE_M
        np.minimum.at(ground, index[near], z[near].astype(np.float32))
    ground[~np.isfinite(ground)] = np.nan
    return ground.reshape(height, width)


def _max_height_block(reader, ground, grid_lattice, block_lattice, resolution):
    """Highest height-above-ground per cell, for one block.

    Ground is sampled bilinearly at each point rather than read from the point's
    own cell: on a slope a cell-constant ground under-reads uphill and
    over-reads downhill, which measured 0.27 m RMSE against a per-point ground
    versus 0.17 m for bilinear.

    Sampling uses the whole ground raster and the whole grid's lattice, while
    output indexing uses the block's, so a point near a block edge interpolates
    across that edge exactly as it would have without blocking. Ground is a
    single raster over the output grid — small next to the cloud — so sharing it
    costs far less than haloing it would.
    """
    _, _, height, width = block_lattice
    chm = np.full(height * width, -np.inf, dtype=np.float32)
    x, y, z, _ = reader(_block_bounds(block_lattice, resolution), SURFACE_CLASSES)
    if x.size:
        above = z - _sample_bilinear(ground, x, y, grid_lattice, resolution)
        index, inside = _cell_indices(x, y, block_lattice, resolution)
        usable = (
            inside
            & (above >= MIN_CANOPY_HEIGHT_M)
            & (above < MAX_CANOPY_HEIGHT_M)
            & np.isfinite(above)
        )
        np.maximum.at(chm, index[usable], above[usable].astype(np.float32))
    chm[~np.isfinite(chm)] = np.nan
    return chm.reshape(height, width)


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


def _sample_bilinear(surface, x, y, lattice, resolution) -> np.ndarray:
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


def _max_ground_distance(known: np.ndarray, cap_cells: float) -> float:
    """Furthest any cell sits from a ground return, in cells, saturating at `cap_cells`.

    This is the variable that predicted ground-derivation error across the
    validation clouds — better than coverage alone, since scattered gaps
    interpolate fine while one wide void does not.

    Bounded rather than global, and blocked rather than whole-grid, because the
    number only carries information near the distances the filter can bridge:
    `GROUND_FILL_MAX_M` is 30 m and PMF's widest window is 33 m, so anything
    past the cap says "badly constrained" and nothing more. Computed on a block
    plus a halo of `cap_cells`, any cell whose true nearest ground is within the
    cap has it inside the halo, so the answer below the cap is exact.

    `distance_transform_edt` is called for distances only; asking it for indices
    as well is what allocates gigabytes on a large grid.
    """
    if known.all():
        return 0.0
    if not known.any():
        return float(cap_cells)
    return float(min(distance_transform_edt(~known).max(), cap_cells))


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
