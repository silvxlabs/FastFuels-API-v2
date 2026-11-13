"""
3DEP tile discovery utilities.

Shared logic for discovering USGS 3DEP tile URLs on AWS S3. Used by both
the API (coverage pre-flight check) and Griddle (grid processing).

Products, by ``source_resolution``:

- 1: Seamless 1-Meter (S1M), falling back to the 1 m lidar project tiles
- 2: the 1 m lidar project tiles, read at their 2 m overview
- 10: 1/3 arc-second seamless DEM
- 30: 1 arc-second seamless DEM

All functions are synchronous. The API wraps blocking I/O calls with
asyncio.to_thread() to avoid blocking the event loop. Nothing here may
import GDAL: the API imports this module.
"""

import logging
import math
import time

import geopandas as gpd
import requests

logger = logging.getLogger(__name__)

S3_BASE = "https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation"

# The National Map (TNM) Access product search. Lidar project tile names
# carry the project name, so unlike S1M and arc-second tiles they cannot be
# built from coordinates and have to be looked up.
TNM_PRODUCTS_URL = "https://tnmaccess.nationalmap.gov/api/v1/products"
TNM_1M_DATASET = "Digital Elevation Model (DEM) 1 meter"
_TNM_PAGE_SIZE = 100


class TnmSearchError(Exception):
    """The TNM Access product search failed or returned an unusable response."""


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


def discover_lidar_project_tiles(
    roi: gpd.GeoDataFrame,
    timeout: float = 30.0,
    attempts: int = 1,
) -> tuple[list[str], list[str]]:
    """Discover 1 m lidar project DEM tiles via the TNM Access product search.

    Each tile is a COG in its own UTM zone. Areas flown more than once, and
    ground near a zone boundary, return overlapping tiles. Tiles are ordered
    newest publication date first, ties broken by URL, so a first-wins mosaic
    is reproducible.

    Args:
        roi: Region of interest. Its bounding box is searched.
        timeout: Seconds allowed for each HTTP request.
        attempts: Tries per page. Timeouts, connection errors and 5xx
            responses are retried with a short backoff.

    Returns:
        Tuple of (tile_urls, publication_dates). Dates are unique
        ``YYYYMMDD`` strings, sorted. Empty lists if no tiles intersect.

    Raises:
        TnmSearchError: If the search cannot be completed.
    """
    min_lon, min_lat, max_lon, max_lat = roi.to_crs("EPSG:4326").total_bounds
    params = {
        "datasets": TNM_1M_DATASET,
        "bbox": f"{min_lon},{min_lat},{max_lon},{max_lat}",
        "prodFormats": "GeoTIFF",
        "max": _TNM_PAGE_SIZE,
    }

    dates_by_url: dict[str, str] = {}
    offset = 0
    while True:
        page = _search_tnm({**params, "offset": offset}, timeout, attempts)
        items = page["items"]
        for item in items:
            url = item.get("downloadURL") or ""
            if url.endswith(".tif"):
                dates_by_url[url] = item.get("publicationDate") or ""
        offset += len(items)
        if not items or offset >= page["total"]:
            break

    urls = sorted(dates_by_url)
    urls.sort(key=dates_by_url.__getitem__, reverse=True)
    dates = sorted({d.replace("-", "") for d in dates_by_url.values() if d})
    return urls, dates


def _search_tnm(params: dict, timeout: float, attempts: int) -> dict:
    """Fetch one page of TNM product search results."""
    error: Exception | None = None
    for attempt in range(attempts):
        if attempt:
            time.sleep(2**attempt)
        try:
            response = requests.get(TNM_PRODUCTS_URL, params=params, timeout=timeout)
        except (requests.Timeout, requests.ConnectionError) as e:
            error = e
            logger.warning(f"TNM product search attempt {attempt + 1} failed: {e}")
            continue
        if response.status_code >= 500:
            error = requests.HTTPError(f"HTTP {response.status_code}")
            logger.warning(
                f"TNM product search attempt {attempt + 1} failed: "
                f"HTTP {response.status_code}"
            )
            continue
        try:
            response.raise_for_status()
            body = response.json()
        except (requests.HTTPError, ValueError) as e:
            raise TnmSearchError(f"TNM product search failed: {e}") from e
        if body.get("errors") or not isinstance(body.get("items"), list):
            raise TnmSearchError(
                f"TNM product search returned an error: {body.get('errors')}"
            )
        body.setdefault("total", 0)
        return body
    raise TnmSearchError(
        f"TNM product search failed after {attempts} attempt(s): {error}"
    ) from error
