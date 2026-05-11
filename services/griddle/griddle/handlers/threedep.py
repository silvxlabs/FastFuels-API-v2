"""
3DEP (3D Elevation Program) source handlers.

Fetches elevation data from USGS 3DEP via AWS S3 COGs. Supports 1m, 10m,
and 30m resolutions. Slope and aspect are computed locally from the DEM
using Horn's method via numpy.gradient.

All handlers return xr.Dataset where each variable name is a band name.
"""

import logging
from collections.abc import Callable

import geopandas as gpd
import numpy as np
import rioxarray  # noqa: F401
import xarray as xr
from rasterio.enums import Resampling
from rioxarray.merge import merge_arrays
from xarray import DataArray

from griddle.errors import ProcessingError
from griddle.handlers.tiles import TileMetadata
from lib.alignment import resolve_alignment_destination
from lib.raster import RasterConnection, cog_env
from lib.threedep import discover_s1m_tiles, discover_tiles_arc_second

logger = logging.getLogger(__name__)

# Extra DEM cells fetched beyond the user's requested extent_buffer_cells when
# slope/aspect are requested. numpy.gradient falls back to one-sided
# differences at the boundary, so we fetch a couple extra cells, compute
# derivatives, then clip back to the user-requested extent. Internal only.
_DERIVATIVE_GRADIENT_OVERHEAD_CELLS = 3


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

    Tiles are fetched at native ``resolution`` and slope/aspect are computed
    via ``numpy.gradient`` at that native resolution to preserve derivative
    quality. A single end-of-pipeline ``rio.reproject`` then lands every
    band on the alignment destination.

    Args:
        roi: GeoDataFrame defining the region of interest
        resolution: Source product resolution in meters (1, 10, or 30)
        bands: List of band names ("elevation", "slope", "aspect")
        progress: Progress callback
        extent_buffer_cells: Result-grid cells of buffer around the ROI in the
            returned dataset. When slope or aspect is requested, the handler
            fetches a few extra DEM cells under the hood so numpy.gradient
            produces central differences at the boundary; those extra cells
            are clipped away before returning.
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
    else:
        raise ProcessingError(
            code="INVALID_RESOLUTION",
            message=f"Unsupported 3DEP resolution: {resolution}m",
            suggestion="Supported resolutions: 1, 10, 30",
        )

    if not tile_urls:
        raise ProcessingError(
            code="COVERAGE_ERROR",
            message=(f"No 3DEP {resolution}m tiles found covering the domain extent."),
            suggestion=(
                "The domain may be outside 3DEP coverage. "
                "Try a different resolution or verify the domain location."
            ),
        )

    tile_metadata: TileMetadata = {
        "tiles": tile_urls,
        "tile_source": tile_source,
        "tile_count": len(tile_urls),
        "native_crs": native_crs,
        "acquisition_dates": acquisition_dates,
    }

    progress(f"Fetching {len(tile_urls)} 3DEP tile(s)...", 20)
    dem_da = _fetch_and_mosaic_tiles(roi, tile_urls, resolution, progress, fetch_buffer)

    # Validate that we got actual data, not just nodata fill
    _validate_dem_has_data(dem_da, resolution)

    variables: dict[str, DataArray] = {}

    if "elevation" in bands:
        if needs_derivatives:
            # Strip the gradient overhead so the elevation output matches the
            # user-requested extent_buffer_cells.
            variables["elevation"] = _clip_to_roi(dem_da, roi, extent_buffer_cells)
        else:
            variables["elevation"] = dem_da

    if needs_derivatives:
        progress("Computing slope and aspect...", 75)
        cell_size = abs(float(dem_da.rio.transform().a))
        slope_da, aspect_da = _compute_slope_aspect(dem_da, cell_size)
        slope_da = _clip_to_roi(slope_da, roi, extent_buffer_cells)
        aspect_da = _clip_to_roi(aspect_da, roi, extent_buffer_cells)

        if "slope" in bands:
            variables["slope"] = slope_da
        if "aspect" in bands:
            variables["aspect"] = aspect_da

    progress("Aligning output...", 85)
    variables = _apply_alignment_to_vars(variables, alignment, roi, target_grid_doc)

    progress("Building dataset...", 90)
    ds = _to_dataset(variables)
    return ds, tile_metadata


def _apply_alignment_to_vars(
    variables: dict[str, DataArray],
    alignment: dict,
    roi: gpd.GeoDataFrame,
    target_grid_doc: dict | None,
) -> dict[str, DataArray]:
    """Reproject each variable to the alignment destination.

    Called once at the end of 3DEP processing. Slope/aspect have already
    been computed at native resolution; this is the single reprojection
    that aligns the output to the chosen lattice. All variables are
    continuous (categorical default does not apply here).
    """
    sample = next(iter(variables.values()))
    native_resolution = abs(float(sample.rio.transform().a))
    dest = resolve_alignment_destination(
        alignment, roi, target_grid_doc, native_resolution
    )
    if not dest:
        return variables  # target="native" with no resolution change

    method_name = alignment.get("method") or "bilinear"
    resampling = Resampling[method_name]

    aligned: dict[str, DataArray] = {}
    for name, da in variables.items():
        if "destination_transform" in dest and "destination_shape" in dest:
            aligned[name] = da.rio.reproject(
                dest["destination_crs"],
                transform=dest["destination_transform"],
                shape=dest["destination_shape"],
                resampling=resampling,
            )
        elif alignment.get("resolution") is not None:
            aligned[name] = da.rio.reproject(
                dest["destination_crs"],
                resolution=alignment["resolution"],
                resampling=resampling,
            )
        else:
            aligned[name] = da.rio.reproject(
                dest["destination_crs"], resampling=resampling
            )
    return aligned


def _discover_tiles_1m(
    roi: gpd.GeoDataFrame,
) -> tuple[list[str], str, str, list[str] | None]:
    """Discover 1m tiles via S1M (Seamless 1-Meter) coordinate-based lookup.

    Returns:
        Tuple of (tile_urls, tile_source, native_crs, acquisition_dates)
    """
    s1m_urls, s1m_dates = discover_s1m_tiles(roi)
    if s1m_urls:
        return s1m_urls, "s1m", "EPSG:6350", s1m_dates

    return [], "none", "none", None


def _fetch_and_mosaic_tiles(
    roi: gpd.GeoDataFrame,
    tile_urls: list[str],
    resolution: int,
    progress: Callable[[str, int | None], None],
    extent_buffer_cells: int,
) -> DataArray:
    """Fetch tiles and mosaic into a single DataArray.

    Args:
        roi: Region of interest
        tile_urls: S3/HTTPS URLs to COG tiles
        resolution: Target resolution in meters
        progress: Progress callback
        extent_buffer_cells: Output DEM cells of buffer around the ROI

    Returns:
        DataArray with dims (y, x) in ROI's CRS at target resolution
    """
    n_tiles = len(tile_urls)
    tile_arrays = []

    with cog_env(AWS_NO_SIGN_REQUEST="YES"):
        for i, url in enumerate(tile_urls):
            pct = 20 + int(50 * (i + 1) / n_tiles)
            progress(f"Fetching 3DEP tile {i + 1}/{n_tiles}...", pct)

            raster = RasterConnection(url, connection_type="rioxarray", cache=False)

            data = raster.extract_window(
                roi=roi,
                interpolation_padding_cells=extent_buffer_cells,
            )
            tile_arrays.append(data.squeeze("band", drop=True))

    if len(tile_arrays) == 1:
        return tile_arrays[0]

    return merge_arrays(tile_arrays)


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
        Tuple of (slope_da, aspect_da) in degrees, same coords/CRS as input
    """
    values = dem_da.values.astype(np.float64)

    # numpy.gradient computes central differences
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

    slope_da = dem_da.copy(data=slope_deg)
    slope_da.name = "slope"

    aspect_da = dem_da.copy(data=aspect_deg)
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


def _clip_to_roi(
    da: DataArray, roi: gpd.GeoDataFrame, extent_buffer_cells: int = 0
) -> DataArray:
    """Clip a DataArray to the ROI extent (plus optional cell buffer).

    Args:
        da: DataArray to clip. Must have a CRS and affine transform.
        roi: Region of interest (any CRS).
        extent_buffer_cells: Cells at the DataArray's current resolution to
            keep around the ROI. 0 strips all padding back to the ROI extent.
    """
    roi_projected = roi.to_crs(da.rio.crs)
    minx, miny, maxx, maxy = roi_projected.total_bounds
    if extent_buffer_cells > 0:
        cell_size = abs(float(da.rio.transform().a))
        pad = extent_buffer_cells * cell_size
        minx -= pad
        miny -= pad
        maxx += pad
        maxy += pad
    return da.rio.clip_box(minx=minx, miny=miny, maxx=maxx, maxy=maxy)


def _to_dataset(variables: dict[str, DataArray]) -> xr.Dataset:
    """Build a Dataset from named DataArrays, propagating spatial metadata."""
    first = next(iter(variables.values()))
    ds = xr.Dataset(variables)
    ds = ds.rio.write_crs(first.rio.crs)
    ds = ds.rio.write_transform(first.rio.transform())
    return ds
