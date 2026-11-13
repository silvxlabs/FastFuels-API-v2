"""
3DEP (3D Elevation Program) source handlers.

Fetches elevation data from USGS 3DEP via AWS S3 COGs. Source products by
``source_resolution``:

- 1: Seamless 1-Meter (S1M); where S1M has no tiles, the 1 m lidar project
  tiles at full resolution
- 2: the 1 m lidar project tiles, read from their 2 m overview
- 10: 1/3 arc-second seamless DEM
- 30: 1 arc-second seamless DEM

Slope and aspect are computed locally from the DEM using Horn's method via
numpy.gradient.

All handlers return xr.Dataset where each variable name is a band name.
"""

import logging
import traceback
from collections.abc import Callable

import geopandas as gpd
import numpy as np
import rioxarray  # noqa: F401
import xarray as xr
from rioxarray.exceptions import NoDataInBounds
from rioxarray.merge import merge_arrays
from xarray import DataArray

from griddle.handlers.tiles import TileMetadata
from lib.alignment import RESAMPLING_METHOD_MAP, resolve_alignment_destination
from lib.errors import ProcessingError
from lib.raster import RasterConnection, cog_env
from lib.threedep import (
    TnmSearchError,
    discover_lidar_project_tiles,
    discover_s1m_tiles,
    discover_tiles_arc_second,
)

logger = logging.getLogger(__name__)

# Extra DEM cells fetched beyond the user's requested extent_buffer_cells when
# slope/aspect are requested. numpy.gradient falls back to one-sided
# differences at the boundary, so we fetch a couple extra cells, compute
# derivatives, then clip back to the user-requested extent. Internal only.
_DERIVATIVE_GRADIENT_OVERHEAD_CELLS = 3

# The TNM Access product search has no SLA and has been seen taking ~30 s and
# returning 504s, so project tile discovery gets a few bounded tries.
_TNM_TIMEOUT_SECONDS = 30.0
_TNM_ATTEMPTS = 3

# GDAL open options per source resolution. Lidar project tiles are 1 m COGs
# whose first overview (OVERVIEW_LEVEL=0) is 2 m. Opening that overview as
# the dataset makes every read, and the native cell size, 2 m.
_OPEN_OPTIONS = {2: {"OVERVIEW_LEVEL": 0}}


def fetch_topography(
    roi: gpd.GeoDataFrame,
    resolution: int,
    bands: list[str],
    progress: Callable[[str, int | None], None],
    extent_buffer_cells: int = 0,
    alignment: dict | None = None,
    target_grid_doc: dict | None = None,
) -> tuple[xr.Dataset, TileMetadata]:
    """Fetch 3DEP topographic data for a domain extent.

    Each tile's ``extract_window`` reprojects directly into the alignment
    destination lattice — that is the single reprojection. Slope/aspect
    are then computed via ``numpy.gradient`` on the aligned mosaic at the
    destination cell size. When derivatives are requested the destination
    lattice is grown by ``_DERIVATIVE_GRADIENT_OVERHEAD_CELLS`` cells per
    side so ``numpy.gradient`` has central differences at the user-requested
    boundary; those extra cells are stripped from every band before return.

    Args:
        roi: GeoDataFrame defining the region of interest
        resolution: Source product resolution in meters (1, 2, 10, or 30)
        bands: List of band names ("elevation", "slope", "aspect")
        progress: Progress callback
        extent_buffer_cells: Result-grid cells of buffer around the ROI in the
            returned dataset.
        alignment: Alignment specification dict. Defaults to
            ``{"target": "domain"}`` when omitted.
        target_grid_doc: Loaded grid document used when
            ``alignment["target"] == "grid"``.

    Returns:
        Tuple of (Dataset with named variables, tile metadata dict)

    Raises:
        ProcessingError: If no tiles found or fetch fails
    """
    alignment = alignment or {"target": "domain"}
    needs_derivatives = "slope" in bands or "aspect" in bands
    fetch_buffer = (
        extent_buffer_cells + _DERIVATIVE_GRADIENT_OVERHEAD_CELLS
        if needs_derivatives
        else extent_buffer_cells
    )

    progress(f"Discovering 3DEP {resolution}m tiles...", 10)

    if resolution in (10, 30):
        tile_urls = discover_tiles_arc_second(roi, resolution)
        tile_source = None
        native_crs = "EPSG:4326"
        acquisition_dates = None
    elif resolution == 1:
        tile_urls, tile_source, native_crs, acquisition_dates = _discover_tiles_1m(roi)
    elif resolution == 2:
        tile_urls, tile_source, native_crs, acquisition_dates = _discover_project_tiles(
            roi
        )
    else:
        raise ProcessingError(
            code="INVALID_RESOLUTION",
            message=f"Unsupported 3DEP resolution: {resolution}m",
            suggestion="Supported resolutions: 1, 2, 10, 30",
        )

    if not tile_urls:
        raise _no_tiles_error(resolution)

    progress(f"Fetching {len(tile_urls)} 3DEP tile(s)...", 20)
    dem_da, used_urls = _fetch_and_mosaic_tiles(
        roi,
        tile_urls,
        resolution,
        progress,
        fetch_buffer,
        alignment,
        target_grid_doc,
    )

    tile_metadata: TileMetadata = {
        "tiles": used_urls,
        "tile_source": tile_source,
        "tile_count": len(used_urls),
        "native_crs": native_crs,
        "acquisition_dates": acquisition_dates,
    }

    # Validate that we got actual data, not just nodata fill
    _validate_dem_has_data(dem_da, resolution)

    variables: dict[str, DataArray] = {}

    if "elevation" in bands:
        if needs_derivatives:
            variables["elevation"] = _trim_derivative_overhead(
                dem_da, alignment, roi, extent_buffer_cells
            )
        else:
            variables["elevation"] = dem_da

    if needs_derivatives:
        progress("Computing slope and aspect...", 80)
        cell_size = abs(float(dem_da.rio.transform().a))
        slope_da, aspect_da = _compute_slope_aspect(dem_da, cell_size)
        slope_da = _trim_derivative_overhead(
            slope_da, alignment, roi, extent_buffer_cells
        )
        aspect_da = _trim_derivative_overhead(
            aspect_da, alignment, roi, extent_buffer_cells
        )

        if "slope" in bands:
            variables["slope"] = slope_da
        if "aspect" in bands:
            variables["aspect"] = aspect_da

    progress("Building dataset...", 90)
    ds = _to_dataset(variables)
    return ds, tile_metadata


def _no_tiles_error(resolution: int) -> ProcessingError:
    return ProcessingError(
        code="COVERAGE_ERROR",
        message=(f"No 3DEP {resolution}m tiles found covering the domain extent."),
        suggestion=(
            "The domain may be outside 3DEP coverage. "
            "Try a different resolution or verify the domain location."
        ),
    )


def _discover_tiles_1m(
    roi: gpd.GeoDataFrame,
) -> tuple[list[str], str, str | None, list[str] | None]:
    """Discover 1m tiles: S1M (Seamless 1-Meter) first, else lidar projects.

    Returns:
        Tuple of (tile_urls, tile_source, native_crs, acquisition_dates)
    """
    s1m_urls, s1m_dates = discover_s1m_tiles(roi)
    if s1m_urls:
        return s1m_urls, "s1m", "EPSG:6350", s1m_dates

    return _discover_project_tiles(roi)


def _discover_project_tiles(
    roi: gpd.GeoDataFrame,
) -> tuple[list[str], str, None, list[str] | None]:
    """Discover 1 m lidar project tiles via the TNM Access product search.

    Tiles are ordered newest first, so where flights or UTM zones overlap
    the first-wins mosaic keeps the newest data. ``native_crs`` is None
    because tiles can come from more than one UTM zone.

    Returns:
        Tuple of (tile_urls, tile_source, native_crs, acquisition_dates)

    Raises:
        ProcessingError: TILE_SOURCE_UNAVAILABLE if the search fails after
            retries.
    """
    try:
        urls, dates = discover_lidar_project_tiles(
            roi, timeout=_TNM_TIMEOUT_SECONDS, attempts=_TNM_ATTEMPTS
        )
    except TnmSearchError as e:
        raise ProcessingError(
            code="TILE_SOURCE_UNAVAILABLE",
            message=(
                "The USGS National Map tile search could not be reached. "
                "Please try again shortly."
            ),
            suggestion="Retry the request in a few minutes.",
            traceback=traceback.format_exc(),
        ) from e
    return urls, "projects", None, dates or None


def _fetch_and_mosaic_tiles(
    roi: gpd.GeoDataFrame,
    tile_urls: list[str],
    resolution: int,
    progress: Callable[[str, int | None], None],
    extent_buffer_cells: int,
    alignment: dict,
    target_grid_doc: dict | None,
) -> tuple[DataArray, list[str]]:
    """Fetch tiles and mosaic them onto the alignment destination lattice.

    Each tile's ``extract_window`` reprojects directly into the destination
    transform/shape resolved from ``alignment`` — all tiles land on the same
    output grid, so the mosaic is a nodata-aware composite of identically
    shaped arrays with no second interpolation pass. The destination is
    resolved once, from the first tile: tiles in different CRSs (lidar
    projects span UTM zones) report slightly different native cell sizes in
    the domain CRS, and a per-tile lattice would not line up.

    Tiles are merged first-wins, in ``tile_urls`` order. Lidar project tiles
    are discovered by EPSG:4326 bounding box, which is wider than the tile, so
    a discovered tile may not reach the destination; those are skipped.

    Args:
        roi: Region of interest
        tile_urls: S3/HTTPS URLs to COG tiles
        resolution: Source product resolution in meters. Selects the GDAL
            open options (2 reads the tiles' 2 m overview); the output cell
            size comes from ``alignment``.
        progress: Progress callback
        extent_buffer_cells: Output DEM cells of buffer around the ROI in the
            destination CRS, baked into ``destination_transform``/``shape``
            for ``target='domain'`` and ``target='grid'``
        alignment: Alignment specification dict (see ``GridAlignmentSpecification``)
        target_grid_doc: Loaded grid document used as the alignment target
            when ``alignment["target"] == "grid"``. Required in that case.

    Returns:
        Tuple of (DataArray with dims (y, x) in the alignment destination CRS
        at the alignment destination resolution, URLs of the tiles read)

    Raises:
        ProcessingError: COVERAGE_ERROR if no tile reaches the destination.
    """
    n_tiles = len(tile_urls)
    tile_arrays = []
    used_urls = []
    method_name = alignment.get("method") or "bilinear"
    open_options = _OPEN_OPTIONS.get(resolution, {})
    dest = None

    with cog_env(AWS_NO_SIGN_REQUEST="YES"):
        for i, url in enumerate(tile_urls):
            pct = 20 + int(50 * (i + 1) / n_tiles)
            progress(f"Fetching 3DEP tile {i + 1}/{n_tiles}...", pct)

            raster = RasterConnection(
                url, connection_type="rioxarray", cache=False, **open_options
            )
            if dest is None:
                dest = resolve_alignment_destination(
                    alignment,
                    roi,
                    target_grid_doc,
                    raster.target_native_resolution(roi)[0],
                    extent_buffer_cells=extent_buffer_cells,
                )
            try:
                data = raster.extract_window(
                    roi=roi,
                    interpolation_padding_cells=extent_buffer_cells,
                    resampling=RESAMPLING_METHOD_MAP[method_name],
                    destination_resolution=alignment.get("resolution")
                    if alignment["target"] == "native"
                    else None,
                    **dest,
                )
            except NoDataInBounds:
                logger.info(f"3DEP tile {url} does not overlap the domain; skipping.")
                continue
            used_urls.append(url)
            tile_arrays.append(data.squeeze("band", drop=True))

    if not tile_arrays:
        raise _no_tiles_error(resolution)

    if len(tile_arrays) == 1:
        return tile_arrays[0], used_urls

    return merge_arrays(tile_arrays), used_urls


def _compute_slope_aspect(
    dem_da: DataArray,
    cell_size: float,
) -> tuple[DataArray, DataArray]:
    """Compute slope and aspect from a DEM using Horn's method.

    Uses numpy.gradient which implements a central difference approximation
    (equivalent to a simplified Horn's method).

    Args:
        dem_da: DEM DataArray with dims (y, x)
        cell_size: Cell size in meters

    Returns:
        Tuple of (slope_da, aspect_da) in degrees, same coords/CRS as input,
        NaN (and NaN nodata) wherever the stencil touches DEM nodata
    """
    values = dem_da.values.astype(np.float64)
    nodata = dem_da.rio.nodata
    if nodata is not None and not np.isnan(nodata):
        values[values == nodata] = np.nan

    # numpy.gradient computes central differences; NaN propagates to every
    # cell whose stencil touches a nodata cell
    dy, dx = np.gradient(values, cell_size)

    # Slope: arctan of the magnitude of the gradient vector
    slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
    slope_deg = np.degrees(slope_rad).astype(np.float32)

    # Aspect: direction of steepest descent, clockwise from north
    # -dx because aspect convention is CW from north, and np.arctan2 is CCW from east
    aspect_rad = np.arctan2(-dx, dy)
    aspect_deg = np.degrees(aspect_rad).astype(np.float32)
    # Normalize to [0, 360)
    aspect_deg = np.where(aspect_deg < 0, aspect_deg + 360, aspect_deg)

    slope_da = dem_da.copy(data=slope_deg).rio.write_nodata(np.nan)
    slope_da.name = "slope"

    aspect_da = dem_da.copy(data=aspect_deg).rio.write_nodata(np.nan)
    aspect_da.name = "aspect"

    return slope_da, aspect_da


def _validate_dem_has_data(dem_da: DataArray, resolution: int) -> None:
    """Raise ProcessingError if the DEM is entirely nodata.

    Tiles may exist on S3 but contain no actual elevation data for a given
    area. Detect this early rather than returning a dataset full of nodata.
    """
    nodata = dem_da.rio.nodata
    values = dem_da.values

    if nodata is not None:
        valid_mask = ~np.isnan(values) & (values != nodata)
    else:
        valid_mask = ~np.isnan(values)

    if not valid_mask.any():
        raise ProcessingError(
            code="COVERAGE_ERROR",
            message=(
                f"3DEP {resolution}m tiles were found but contain no valid "
                f"elevation data for this domain."
            ),
            suggestion=(
                "The tiles may be placeholders without survey data for this "
                "area. Try a different resolution (10m or 30m have broader "
                "coverage)."
            ),
        )


def _trim_derivative_overhead(
    da: DataArray,
    alignment: dict,
    roi: gpd.GeoDataFrame,
    extent_buffer_cells: int,
) -> DataArray:
    """Remove the gradient-overhead ring fetched for numpy.gradient.

    For ``target="domain"`` / ``target="grid"`` the destination lattice
    is exact (``lattice_from_bounds`` or the target grid's own lattice plus
    whole buffer cells), so the ring is exactly
    ``_DERIVATIVE_GRADIENT_OVERHEAD_CELLS`` rows/columns per side — we
    index-slice. For ``target="native"`` the lattice depends on where the
    pixel anchor falls relative to the buffered bounds, so the ring count
    can vary per edge; we clip-box back to the same bounds
    ``extract_window`` would use for an elevation-only fetch.
    """
    if alignment["target"] in ("domain", "grid"):
        n = _DERIVATIVE_GRADIENT_OVERHEAD_CELLS
        return da.isel(y=slice(n, -n), x=slice(n, -n))

    cell_size = abs(float(da.rio.transform().a))
    pad = extent_buffer_cells * cell_size
    minx, miny, maxx, maxy = roi.to_crs(da.rio.crs).total_bounds
    return da.rio.clip_box(
        minx=minx - pad,
        miny=miny - pad,
        maxx=maxx + pad,
        maxy=maxy + pad,
    )


def _to_dataset(variables: dict[str, DataArray]) -> xr.Dataset:
    """Build a Dataset from named DataArrays, propagating spatial metadata."""
    first = next(iter(variables.values()))
    ds = xr.Dataset(variables)
    ds = ds.rio.write_crs(first.rio.crs)
    ds = ds.rio.write_transform(first.rio.transform())
    return ds
