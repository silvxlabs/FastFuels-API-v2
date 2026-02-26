"""
CHM (Canopy Height Model) source handlers.

Pure functions that fetch CHM data for a domain extent.
All handlers return xr.Dataset where each variable name is a band name.
"""

import os
from collections.abc import Callable

import geopandas as gpd
import rasterio
import xarray as xr
from rioxarray.merge import merge_arrays

from griddle.errors import ProcessingError
from lib.config import TABLES_BUCKET
from lib.raster import RasterConnection

S3_BASE = "s3://dataforgood-fb-data/forests/v1/alsgedi_global_v6_float/chm"


def fetch_meta_chm(
    roi: gpd.GeoDataFrame,
    version: str,
    progress: Callable[[str, int | None], None],
) -> xr.Dataset:
    """Fetch Meta global canopy height model data.

    Downloads tile-mapping GeoJSON from GCS, finds intersecting S3 tiles,
    and extracts the canopy height raster for the region of interest.
    When the domain spans multiple tiles, the tiles are mosaicked together.

    Args:
        roi: GeoDataFrame defining the region of interest
        version: Data version year (e.g., "2024")
        progress: Progress callback

    Returns:
        Dataset with a single "chm" variable (canopy height in meters)

    Raises:
        ValueError: If no tiles intersect the ROI.
    """
    progress("Loading tile mapping...", 10)

    # Load tile-mapping GeoJSON to find which S3 tiles cover the ROI
    tile_map_url = (
        f"gs://{TABLES_BUCKET}/Meta{version}_chm_map_from_polygon_to_geotiff.geojson"
    )
    tile_polygons = gpd.read_file(tile_map_url)

    # Reproject ROI to EPSG:4326 for tile intersection
    roi_4326 = roi.to_crs("EPSG:4326")
    intersecting = tile_polygons[tile_polygons.intersects(roi_4326.union_all())]

    if intersecting.empty:
        raise ValueError(
            "No Meta CHM tiles found for the given domain extent. "
            "The domain may be outside the dataset coverage area."
        )

    tile_names = intersecting["tile"].tolist()
    n_tiles = len(tile_names)

    # Fetch each tile and extract the window covering the ROI
    # Use rasterio.Env to force GDAL unsigned S3 requests for this operation.
    tile_arrays = []
    with rasterio.Env(AWS_NO_SIGN_REQUEST="YES"):
        for i, tile_name in enumerate(tile_names):
            progress(
                f"Fetching CHM tile {i + 1}/{n_tiles}...",
                10 + int(60 * (i + 1) / n_tiles),
            )
            s3_url = f"{S3_BASE}/{tile_name}.tif"
            raster = RasterConnection(s3_url, connection_type="rioxarray", cache=True)
            data = raster.extract_window(
                roi=roi,
                projection_padding_meters=1200,
                interpolation_padding_cells=240,
            )
            tile_arrays.append(data.squeeze("band", drop=True))

    progress("Building dataset...", 80)

    # Merge tiles if the domain spans more than one
    if len(tile_arrays) == 1:
        chm_da = tile_arrays[0]
    else:
        chm_da = merge_arrays(tile_arrays)

    ds = xr.Dataset({"chm": chm_da})
    ds = ds.rio.write_crs(chm_da.rio.crs)
    ds = ds.rio.write_transform(chm_da.rio.transform())
    return ds


# Point directly to our FastFuels bucket for the fixed index
NAIP_INDEX_URL = f"gs://{TABLES_BUCKET}/naip_chm_index.parquet"


def fetch_naip_chm(
    roi: gpd.GeoDataFrame,
    version: str,
    progress: Callable[[str, int | None], None],
) -> xr.Dataset:
    """Fetch NAIP high-resolution canopy height model data (Local Index Test)."""
    progress("Loading NAIP CHM tile index (Local Parquet)...", 10)

    # Project ROI to standard 4326 to match our Parquet file
    roi_4326 = roi.to_crs("EPSG:4326")
    bounds = tuple(roi_4326.total_bounds)

    try:
        # 1. The Pushdown Filter: Only read row groups intersecting our bounding box
        intersecting = gpd.read_parquet(NAIP_INDEX_URL, bbox=bounds)
    except Exception as e:
        raise ProcessingError(
            code="INDEX_FETCH_FAILED",
            message="Failed to load local NAIP CHM parquet index.",
            traceback=str(e),
        )

    if intersecting.empty:
        raise ProcessingError(
            code="COVERAGE_ERROR",
            message="No NAIP CHM tiles found for the given domain extent.",
            suggestion="The domain may be outside the CONUS coverage area.",
        )

    # 2. The Exact Filter: Narrow down the chunks to the precise polygons we touch
    intersecting = intersecting[intersecting.intersects(roi_4326.union_all())]

    n_tiles = len(intersecting)
    tile_arrays = []

    # Get the project ID from our standard V2 environment variables
    gcp_project = os.environ.get("GCP_PROJECT")

    if not gcp_project:
        raise ProcessingError(
            code="MISSING_CONFIG",
            message="GCP_PROJECT environment variable is missing, required for NAIP Requester Pays access.",
        )

    # Configure GDAL to pass our project ID for billing on the Requester Pays bucket
    with rasterio.Env(GS_USER_PROJECT=gcp_project):
        for i, (_, row) in enumerate(intersecting.iterrows()):
            progress(
                f"Fetching NAIP tile {i + 1}/{n_tiles}...",
                10 + int(60 * (i + 1) / n_tiles),
            )

            http_url = row["chm_url"]
            scale_factor = float(row.get("scale_factor", 100.0))

            raster = RasterConnection(http_url, connection_type="rioxarray", cache=True)
            data = raster.extract_window(
                roi=roi,
                projection_padding_meters=1200,
                interpolation_padding_cells=240,
            )

            data_2d = data.squeeze("band", drop=True)
            data_2d = data_2d / scale_factor
            tile_arrays.append(data_2d)

    progress("Building dataset...", 80)

    # Mosaic if the ROI spanned multiple tiles
    if len(tile_arrays) == 1:
        chm_da = tile_arrays[0]
    else:
        chm_da = merge_arrays(tile_arrays)

    # Wrap in xr.Dataset with strict variable naming to satisfy Griddle output rules
    ds = xr.Dataset({"chm": chm_da})
    ds = ds.rio.write_crs(chm_da.rio.crs)
    ds = ds.rio.write_transform(chm_da.rio.transform())

    return ds
