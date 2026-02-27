"""
3DEP (3D Elevation Program) source handlers.

Fetches elevation data from USGS 3DEP via AWS S3 COGs. Supports 1m, 10m,
and 30m resolutions. Slope and aspect are computed locally from the DEM
using Horn's method via numpy.gradient.

All handlers return xr.Dataset where each variable name is a band name.
"""

import logging
import math
from collections.abc import Callable

import geopandas as gpd
import numpy as np
import rasterio
import rioxarray  # noqa: F401
import xarray as xr
from rioxarray.merge import merge_arrays
from xarray import DataArray

from griddle.errors import ProcessingError
from griddle.handlers.tiles import TileMetadata

logger = logging.getLogger(__name__)

S3_BASE = "https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation"


def fetch_topography(
    roi: gpd.GeoDataFrame,
    resolution: int,
    bands: list[str],
    progress: Callable[[str, int | None], None],
) -> tuple[xr.Dataset, TileMetadata]:
    """Fetch 3DEP topographic data for a domain extent.

    Args:
        roi: GeoDataFrame defining the region of interest
        resolution: Resolution in meters (1, 10, or 30)
        bands: List of band names ("elevation", "slope", "aspect")
        progress: Progress callback

    Returns:
        Tuple of (Dataset with named variables, tile metadata dict)

    Raises:
        ProcessingError: If no tiles found or fetch fails
    """
    needs_derivatives = "slope" in bands or "aspect" in bands

    progress(f"Discovering 3DEP {resolution}m tiles...", 10)

    if resolution in (10, 30):
        tile_urls = _discover_tiles_arc_second(roi, resolution)
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

    # Extra cells around the ROI so slope/aspect don't have edge artifacts.
    # Derivatives need more padding than elevation-only because the gradient
    # computation consumes border cells.
    pad_cells = 10 if needs_derivatives else 4

    progress(f"Fetching {len(tile_urls)} 3DEP tile(s)...", 20)
    dem_da = _fetch_and_mosaic_tiles(roi, tile_urls, resolution, pad_cells, progress)

    # Validate that we got actual data, not just nodata fill
    _validate_dem_has_data(dem_da, resolution)

    variables: dict[str, DataArray] = {}

    if "elevation" in bands:
        if needs_derivatives:
            # Clip elevation to actual domain (remove padding)
            variables["elevation"] = _clip_to_roi(dem_da, roi)
        else:
            variables["elevation"] = dem_da

    if needs_derivatives:
        progress("Computing slope and aspect...", 75)
        cell_size = abs(float(dem_da.rio.transform().a))
        slope_da, aspect_da = _compute_slope_aspect(dem_da, cell_size)
        slope_da = _clip_to_roi(slope_da, roi)
        aspect_da = _clip_to_roi(aspect_da, roi)

        if "slope" in bands:
            variables["slope"] = slope_da
        if "aspect" in bands:
            variables["aspect"] = aspect_da

    progress("Building dataset...", 85)
    ds = _to_dataset(variables)
    return ds, tile_metadata


def _discover_tiles_arc_second(
    roi: gpd.GeoDataFrame,
    resolution: int,
) -> list[str]:
    """Discover 10m or 30m tile URLs by coordinate-based construction.

    10m and 30m 3DEP tiles are 1x1 degree in EPSG:4326.
    """
    code = "13" if resolution == 10 else "1"

    roi_4326 = roi.to_crs("EPSG:4326")
    bounds = roi_4326.total_bounds  # (minx, miny, maxx, maxy)
    min_lon, min_lat, max_lon, max_lat = bounds

    # Compute tile indices
    # Tiles are named n{ceil_lat}w{ceil_abs_lon} for the western hemisphere
    lat_min_tile = math.ceil(min_lat)
    lat_max_tile = math.ceil(max_lat)
    lon_min_tile = math.ceil(abs(max_lon))  # westernmost abs lon = smallest tile
    lon_max_tile = math.ceil(abs(min_lon))  # easternmost abs lon = largest tile

    urls = []
    for lat in range(lat_min_tile, lat_max_tile + 1):
        for lon in range(lon_min_tile, lon_max_tile + 1):
            tile_name = f"n{lat:02d}w{lon:03d}"
            url = (
                f"{S3_BASE}/{code}/TIFF/current/{tile_name}/USGS_{code}_{tile_name}.tif"
            )
            urls.append(url)

    return urls


def _discover_tiles_1m(
    roi: gpd.GeoDataFrame,
) -> tuple[list[str], str, str, list[str] | None]:
    """Discover 1m tiles via S1M (Seamless 1-Meter) coordinate-based lookup.

    Returns:
        Tuple of (tile_urls, tile_source, native_crs, acquisition_dates)
    """
    s1m_urls, s1m_dates = _discover_s1m_tiles(roi)
    if s1m_urls:
        return s1m_urls, "s1m", "EPSG:6350", s1m_dates

    return [], "none", "none", None


def _s1m_tile_path(ty: int, tx: int) -> tuple[str, str]:
    """Convert 10km tile indices to S1M S3 zone and tile directory names.

    S1M tiles are named by their top-left corner. Northing uses the top
    edge (ty + 1), easting uses the left edge (tx). Negative eastings use
    'w' prefix instead of 'e'. Names are multiplied by 10 and zero-padded.

    Returns:
        Tuple of (zone_name, tile_dir_name)
    """
    # 100km zone grouping (zones use floor of absolute coordinate / 100km)
    top_n = ty + 1
    zone_n = int(math.floor(abs(top_n) * 10000 / 100000))
    zone_e = int(math.floor(abs(tx) * 10000 / 100000))
    if tx >= 0:
        zone = f"n{zone_n:02d}e{zone_e:02d}"
    else:
        zone = f"n{zone_n:02d}w{zone_e:02d}"

    # Tile directory: named by top-left corner
    n_label = f"n{(ty + 1) * 10:04d}"
    if tx >= 0:
        e_label = f"e{tx * 10:04d}"
    else:
        e_label = f"w{abs(tx) * 10:04d}"
    tile_dir = f"{n_label}{e_label}"

    return zone, tile_dir


def _discover_s1m_tiles(
    roi: gpd.GeoDataFrame,
) -> tuple[list[str], list[str]]:
    """Discover S1M (Seamless 1-Meter) tiles via coordinate-based S3 listing.

    S1M tiles are 10km x 10km in EPSG:6350 (Albers Equal Area).

    Returns:
        Tuple of (tile_urls, acquisition_dates). Empty lists if tiles not found.
    """
    import s3fs

    # Transform ROI bbox to EPSG:6350
    roi_albers = roi.to_crs("EPSG:6350")
    bounds = roi_albers.total_bounds
    min_x, min_y, max_x, max_y = bounds

    # Compute 10km tile grid cells containing the ROI
    tile_min_x = int(math.floor(min_x / 10000))
    tile_max_x = int(math.floor(max_x / 10000))
    tile_min_y = int(math.floor(min_y / 10000))
    tile_max_y = int(math.floor(max_y / 10000))

    fs = s3fs.S3FileSystem(anon=True)
    urls = []
    dates = []

    for ty in range(tile_min_y, tile_max_y + 1):
        for tx in range(tile_min_x, tile_max_x + 1):
            zone, tile_dir = _s1m_tile_path(ty, tx)

            s3_dir = f"prd-tnm/StagedProducts/Elevation/S1M/{zone}/{tile_dir}/"

            try:
                files = fs.ls(s3_dir)
                tif_files = [f for f in files if f.endswith(".tif")]
                if not tif_files:
                    return [], []
                tif_file = tif_files[0]
                url = (
                    f"https://prd-tnm.s3.amazonaws.com/{tif_file.split('prd-tnm/')[-1]}"
                )
                urls.append(url)

                # Extract date from filename (e.g., S1M_n123e456_20230515.tif)
                filename = tif_file.split("/")[-1]
                parts = filename.replace(".tif", "").split("_")
                if len(parts) >= 3:
                    dates.append(parts[-1])
            except Exception as e:
                logger.warning(f"S1M tile listing failed for {s3_dir}: {e}")
                return [], []

    return urls, dates


def _meters_to_degrees(meters: float, roi: gpd.GeoDataFrame) -> float:
    """Convert a meter padding value to approximate degrees.

    Uses the midpoint latitude of the ROI bounds for the longitude
    scaling factor. Returns the larger of the lat/lon conversions so
    the padding is sufficient in both directions.
    """
    bounds_4326 = roi.to_crs("EPSG:4326").total_bounds
    mid_lat = (bounds_4326[1] + bounds_4326[3]) / 2
    lat_rad = np.radians(mid_lat)
    deg_lat = meters / 111_320
    deg_lon = meters / (111_320 * np.cos(lat_rad))
    return max(deg_lat, deg_lon)


def _fetch_and_mosaic_tiles(
    roi: gpd.GeoDataFrame,
    tile_urls: list[str],
    resolution: int,
    pad_cells: int,
    progress: Callable[[str, int | None], None],
) -> DataArray:
    """Fetch tiles and mosaic into a single DataArray.

    Args:
        roi: Region of interest
        tile_urls: S3/HTTPS URLs to COG tiles
        resolution: Target resolution in meters
        pad_cells: Extra cells of padding (for slope/aspect edge effects)
        progress: Progress callback

    Returns:
        DataArray with dims (y, x) in ROI's CRS at target resolution
    """
    from lib.raster import RasterConnection

    n_tiles = len(tile_urls)
    tile_arrays = []

    # Padding applied in the raster's native CRS before reprojection.
    # Three competing minimums ensure enough context for:
    #   resolution * (pad_cells + 8): derivative border + reprojection warp
    #   resolution * 15:              minimum 15-cell buffer at any resolution
    #   500:                          absolute floor for 1m tiles (avoids
    #                                 sub-tile clips that miss data)
    padding_meters = max(resolution * (pad_cells + 8), resolution * 15, 500)

    with rasterio.Env(AWS_NO_SIGN_REQUEST="YES"):
        for i, url in enumerate(tile_urls):
            pct = 20 + int(50 * (i + 1) / n_tiles)
            progress(f"Fetching 3DEP tile {i + 1}/{n_tiles}...", pct)

            raster = RasterConnection(url, connection_type="rioxarray", cache=False)

            # RasterConnection.extract_window applies padding in the raster's
            # CRS units. For geographic CRS (EPSG:4269 for 10m/30m tiles),
            # convert meters → degrees so we clip a tight window instead of
            # the entire tile.
            if raster.raster_crs.is_geographic:
                padding_crs = _meters_to_degrees(padding_meters, roi)
            else:
                padding_crs = padding_meters

            data = raster.extract_window(
                roi=roi,
                projection_padding_meters=padding_crs,
                interpolation_padding_cells=pad_cells,
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


def _clip_to_roi(da: DataArray, roi: gpd.GeoDataFrame) -> DataArray:
    """Clip a DataArray to the ROI extent (removing padding)."""
    # Use rioxarray clip_box with the ROI bounds in its native CRS
    roi_projected = roi.to_crs(da.rio.crs)
    bounds = roi_projected.total_bounds
    return da.rio.clip_box(
        minx=bounds[0], miny=bounds[1], maxx=bounds[2], maxy=bounds[3]
    )


def _to_dataset(variables: dict[str, DataArray]) -> xr.Dataset:
    """Build a Dataset from named DataArrays, propagating spatial metadata."""
    first = next(iter(variables.values()))
    ds = xr.Dataset(variables)
    ds = ds.rio.write_crs(first.rio.crs)
    ds = ds.rio.write_transform(first.rio.transform())
    return ds
