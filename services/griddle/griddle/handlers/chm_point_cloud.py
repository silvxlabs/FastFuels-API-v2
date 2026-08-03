"""
Canopy height model from a point cloud.

Rasterises height above ground from a stored LAZ onto the domain lattice. The
cloud is streamed in chunks and every global computation happens on the raster,
so peak memory is set by the grid size rather than the point count — a 168M
point cloud measured 0.61 GB end to end.

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

import io
import os
from collections.abc import Callable, Iterator
from typing import IO

import geopandas as gpd
import laspy
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

from lib.alignment import resolve_alignment_destination
from lib.config import POINT_CLOUDS_BUCKET
from lib.crs import crs_equal
from lib.errors import ProcessingError
from lib.gcs import get_gcsfs_client

# Points are read in chunks of this many. Peak memory is one chunk plus the
# rasters; 2M keeps the chunk well under 100 MB for any point format.
CHUNK_POINTS = 2_000_000

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

# How far a ground gap may be filled by interpolation, in cells. Beyond this a
# cell stays nodata rather than being extrapolated across a large void.
GROUND_FILL_MAX_CELLS = 30

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

# Largest LAZ held in memory between passes, compared against the peak of the
# fetch rather than the object size. Above this the object is re-read per pass
# instead, trading network for a bounded footprint. The reachable ceiling sits
# well under the default: 200M points (LAKITU_MAX_POINTS) at a measured
# 6.9 B/point is ~1.3 GiB, and an upload is capped at 1 GiB.
MAX_BUFFERED_LAZ_BYTES = int(os.getenv("CHM_MAX_BUFFERED_LAZ_BYTES", 4 * 1024**3))

# A buffered cloud and the rasters share the worker, so the buffering decision
# subtracts the lattice's share. Peak measured at 37.2 B/cell.
RASTER_BYTES_PER_CELL = 40
MEMORY_BUDGET_BYTES = int(os.getenv("CHM_MEMORY_BUDGET_BYTES", 6 * 1024**3))

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

    Args:
        roi: Domain geometry, in the domain's projected CRS. The stored cloud is
            already in this CRS — the uploader reprojects on ingest and lakitu
            writes 3DEP clouds in the domain CRS — so no reprojection happens.
        point_cloud_id: Point cloud whose LAZ to read.
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
    cloud = _open_cloud(point_cloud_id, height * width)

    if GROUND_CLASS in point_classes:
        progress("Reading ground returns...", 15)
        ground = _min_surface(cloud, lattice, resolution, (GROUND_CLASS,))
        ground_source = "classification"
    else:
        progress("Deriving ground surface...", 15)
        ground, ground_source = _derive_ground(cloud, lattice, resolution, progress)

    known_ground = np.isfinite(ground)
    coverage = float(known_ground.mean())
    ground_distance_m = _max_ground_distance(known_ground) * resolution

    progress("Filling ground gaps...", 45)
    ground = _fill_gaps(ground, GROUND_FILL_MAX_CELLS)

    progress("Rasterizing canopy heights...", 55)
    chm = _max_height_above(cloud, ground, lattice, resolution, progress)
    chm = _remove_spikes(chm, SPIKE_THRESHOLD_M)

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


def _cloud_path(point_cloud_id: str) -> str:
    """GCS path of a point cloud's LAZ."""
    return f"{POINT_CLOUDS_BUCKET}/{point_cloud_id}/cloud.laz"


class _ReplayableBuffer(io.BytesIO):
    """An in-memory LAZ that survives being read.

    ``laspy.open`` used as a context manager closes the stream it was handed,
    but the algorithm makes two or three passes over the same bytes. Ignoring
    ``close`` is what keeps one download replayable; the buffer is released when
    the handler returns and the last reference goes away.
    """

    def close(self) -> None:
        pass


def _open_cloud(point_cloud_id: str, cells: int) -> Callable[[], IO[bytes]]:
    """Return a factory yielding a readable LAZ stream, one per pass.

    A cloud small enough to hold alongside the rasters is fetched once and
    replayed from memory, which measured ~2.6x the throughput of streaming the
    same object through gcsfs. Anything larger is re-opened per pass, since
    buffering does not scale to a multi-gigabyte cloud.

    Args:
        point_cloud_id: Point cloud whose LAZ to read.
        cells: Cells in the output lattice, which sets the raster working set.

    Returns:
        A callable returning a fresh readable stream positioned for one pass.
    """
    path = _cloud_path(point_cloud_id)
    fs = get_gcsfs_client()

    # The fetch peaks at ~2x the object, measured 2.32x: gcsfs reads the
    # response in chunks and joins them, so both copies are briefly live.
    # Wrapping the result adds nothing — BytesIO shares an exact bytes object
    # rather than copying it — so the download is the whole cost.
    headroom = MEMORY_BUDGET_BYTES - cells * RASTER_BYTES_PER_CELL
    if 2 * fs.size(path) <= min(MAX_BUFFERED_LAZ_BYTES, headroom):
        buffer = _ReplayableBuffer(fs.cat(path))
        return lambda: buffer

    return lambda: fs.open(path, "rb")


def _iter_points(open_cloud: Callable[[], IO[bytes]]) -> Iterator[tuple]:
    """Yield (x, y, z, classification) arrays a chunk at a time.

    Takes a factory rather than a stream so each pass gets its own reader:
    ``laspy.open`` closes what it is handed, which a buffered cloud survives
    and a freshly-opened remote one does not need to.
    """
    cloud = open_cloud()
    cloud.seek(0)
    with laspy.open(cloud) as reader:
        for points in reader.chunk_iterator(CHUNK_POINTS):
            yield (
                np.asarray(points.x),
                np.asarray(points.y),
                np.asarray(points.z),
                np.asarray(points.classification),
            )


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


def _min_surface(cloud, lattice, resolution, classes) -> np.ndarray:
    """Lowest z per cell over the given classes, NaN where no point falls."""
    _, _, height, width = lattice
    surface = np.full(height * width, np.inf, dtype=np.float32)
    for x, y, z, classification in _iter_points(cloud):
        keep = np.isin(classification, classes)
        index, inside = _cell_indices(x[keep], y[keep], lattice, resolution)
        np.minimum.at(surface, index[inside], z[keep][inside].astype(np.float32))
    surface[~np.isfinite(surface)] = np.nan
    return surface.reshape(height, width)


def _derive_ground(cloud, lattice, resolution, progress) -> tuple[np.ndarray, str]:
    """Derive a ground surface from a cloud with no usable classification.

    Builds the minimum surface over all returns, opens the non-ground objects
    out of it with a progressive morphological filter, then re-derives the
    surface from the points that filter accepts. That last pass matters: the
    opened surface has been eroded and dilated, and using it directly reads
    about 0.1 m low.
    """
    _, _, height, width = lattice
    minimum = _min_surface(cloud, lattice, resolution, SURFACE_CLASSES)

    progress("Separating ground from cover...", 30)
    provisional = _pmf(_fill_gaps(minimum, GROUND_FILL_MAX_CELLS), resolution)

    progress("Re-reading ground returns...", 40)
    ground = np.full(height * width, np.inf, dtype=np.float32)
    for x, y, z, classification in _iter_points(cloud):
        keep = np.isin(classification, SURFACE_CLASSES)
        x, y, z = x[keep], y[keep], z[keep]
        index, inside = _cell_indices(x, y, lattice, resolution)
        x, y, z, index = x[inside], y[inside], z[inside], index[inside]
        near = np.abs(z - provisional.reshape(-1)[index]) <= GROUND_SNAP_TOLERANCE_M
        np.minimum.at(ground, index[near], z[near].astype(np.float32))
    ground[~np.isfinite(ground)] = np.nan
    return ground.reshape(height, width), "derived"


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


def _fill_gaps(surface: np.ndarray, max_cells: int) -> np.ndarray:
    """Interpolate NaN cells from their finite neighbours, bounded in reach.

    Each iteration extends the filled region by one cell, so `max_cells` caps
    how far a value travels. Anything further from real data stays NaN rather
    than being extrapolated across a void the surface says nothing about.
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


def _max_height_above(cloud, ground, lattice, resolution, progress) -> np.ndarray:
    """Highest height-above-ground per cell.

    Ground is sampled bilinearly at each point rather than read from the point's
    own cell: on a slope a cell-constant ground under-reads uphill and
    over-reads downhill, which measured 0.27 m RMSE against a per-point ground
    versus 0.17 m for bilinear.
    """
    origin_x, origin_y, height, width = lattice
    chm = np.full(height * width, -np.inf, dtype=np.float32)

    for x, y, z, classification in _iter_points(cloud):
        keep = np.isin(classification, SURFACE_CLASSES)
        x, y, z = x[keep], y[keep], z[keep]
        above = z - _sample_bilinear(ground, x, y, lattice, resolution)
        index, inside = _cell_indices(x, y, lattice, resolution)
        usable = (
            inside
            & (above >= MIN_CANOPY_HEIGHT_M)
            & (above < MAX_CANOPY_HEIGHT_M)
            & np.isfinite(above)
        )
        np.maximum.at(chm, index[usable], above[usable].astype(np.float32))

    chm[~np.isfinite(chm)] = np.nan
    return chm.reshape(height, width)


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


def _max_ground_distance(known: np.ndarray) -> float:
    """Furthest any cell sits from a cell holding a real ground return, in cells.

    This is the variable that predicted ground-derivation error across the
    validation clouds — better than coverage alone, since scattered gaps
    interpolate fine while one wide void does not. `distance_transform_edt` is
    called for distances only; asking it for indices as well is what allocates
    gigabytes on a large grid.
    """
    if known.all():
        return 0.0
    if not known.any():
        return float("inf")
    return float(distance_transform_edt(~known).max())


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
