"""
3DEP tile discovery utilities.

Shared logic for discovering USGS 3DEP tile URLs from AWS S3. Used by both
the API (coverage pre-flight check) and Griddle (grid processing).

All functions are synchronous. The API wraps blocking I/O calls with
asyncio.to_thread() to avoid blocking the event loop.
"""

import logging
import math

import geopandas as gpd

logger = logging.getLogger(__name__)

S3_BASE = "https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation"


def s1m_tile_path(ty: int, tx: int) -> tuple[str, str]:
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


def discover_s1m_tiles(
    roi: gpd.GeoDataFrame,
) -> tuple[list[str], list[str]]:
    """Discover S1M (Seamless 1-Meter) tiles via coordinate-based S3 listing.

    S1M tiles are 10km x 10km in EPSG:6350 (Albers Equal Area).

    This function performs blocking S3 I/O via s3fs. Callers in async
    contexts should wrap with asyncio.to_thread().

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
            zone, tile_dir = s1m_tile_path(ty, tx)

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

    return urls, sorted(set(dates))


def discover_tiles_arc_second(
    roi: gpd.GeoDataFrame,
    resolution: int,
) -> list[str]:
    """Discover 10m or 30m tile URLs by coordinate-based construction.

    10m and 30m 3DEP tiles are 1x1 degree in EPSG:4326. This is pure math
    with no I/O — safe to call directly in async handlers.

    NOTE: Tile naming uses the 'w' (west) prefix, which assumes all ROIs
    are in the western hemisphere. This is valid for CONUS but would need
    updating if the API ever supports non-CONUS regions (e.g. Guam, USVI).
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
