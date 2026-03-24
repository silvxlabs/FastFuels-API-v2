"""
CHM (Canopy Height Model) source handlers.

Pure functions that fetch CHM data for a domain extent.
All handlers return xr.Dataset where each variable name is a band name.
"""

from collections.abc import Callable

import geopandas as gpd
import rasterio
import xarray as xr
from rioxarray.merge import merge_arrays

from griddle.errors import ProcessingError
from griddle.handlers.tiles import TileMetadata
from lib.config import TABLES_BUCKET
from lib.raster import RasterConnection

S3_BASE = "s3://dataforgood-fb-data/forests/v1/alsgedi_global_v6_float/chm"
NAIP_INDEX_URL = f"gs://{TABLES_BUCKET}/naip_chm_index.parquet"


def _process_intersecting_tiles(
    roi: gpd.GeoDataFrame,
    fetch_urls: list[str],
    scale_factors: list[float],
    gdal_env: dict[str, str],
    progress: Callable[[str, int | None], None],
) -> tuple[xr.Dataset, TileMetadata]:
    """Shared core processing logic for CHM extractions."""
    n_tiles = len(fetch_urls)
    tile_arrays = []

    # Apply the specific GDAL environment variables (e.g., AWS auth bypass)
    with rasterio.Env(**gdal_env):
        for i, (url, scale) in enumerate(zip(fetch_urls, scale_factors)):
            progress(
                f"Fetching CHM tile {i + 1}/{n_tiles}...",
                10 + int(60 * (i + 1) / n_tiles),
            )

            raster = RasterConnection(url, connection_type="rioxarray", cache=True)
            data = raster.extract_window(
                roi=roi,
                projection_padding_meters=64,
                interpolation_padding_cells=4,
            )

            # Squeeze band dimension to satisfy strict 2D requirements
            data_2d = data.squeeze("band", drop=True)

            # Apply scale factor if needed (e.g., NAIP UInt16 -> Float meters)
            if scale != 1.0:
                data_2d = data_2d / scale

            tile_arrays.append(data_2d)

    progress("Building dataset...", 80)

    # Mosaic if the ROI spanned multiple tiles
    if len(tile_arrays) == 1:
        chm_da = tile_arrays[0]
    else:
        chm_da = merge_arrays(tile_arrays)

    # Ensure float32 to avoid unnecessary float64 memory usage
    chm_da = chm_da.astype("float32")

    # Wrap in xr.Dataset with strict variable naming
    ds = xr.Dataset({"chm": chm_da})
    ds = ds.rio.write_crs(chm_da.rio.crs)
    ds = ds.rio.write_transform(chm_da.rio.transform())

    tile_metadata: TileMetadata = {
        "tiles": fetch_urls,
        "tile_source": None,
        "tile_count": len(fetch_urls),
        "native_crs": str(chm_da.rio.crs) if chm_da.rio.crs else None,
        "acquisition_dates": None,
    }

    return ds, tile_metadata


def fetch_meta_chm(
    roi: gpd.GeoDataFrame,
    version: str,
    progress: Callable[[str, int | None], None],
) -> tuple[xr.Dataset, TileMetadata]:
    """Fetch Meta global canopy height model data."""
    progress("Loading Meta CHM parquet index...", 10)

    meta_index_url = f"gs://{TABLES_BUCKET}/Meta{version}_chm_index.parquet"
    roi_4326 = roi.to_crs("EPSG:4326")
    bounds = tuple(roi_4326.total_bounds)

    try:
        # Spatial pushdown filter
        intersecting = gpd.read_parquet(meta_index_url, bbox=bounds)
    except Exception as e:
        raise ProcessingError(
            code="INDEX_FETCH_FAILED",
            message="Failed to load Meta CHM parquet index.",
            traceback=str(e),
        )

    intersecting = intersecting[intersecting.intersects(roi_4326.union_all())]

    if intersecting.empty:
        raise ProcessingError(
            code="COVERAGE_ERROR",
            message="No Meta CHM tiles found for the given domain extent.",
            suggestion="The domain may be outside the dataset coverage area.",
        )

    # Extract exactly what the helper needs into explicit Python lists
    fetch_urls = [f"{S3_BASE}/{tile}.tif" for tile in intersecting["tile"]]
    scale_factors = [1.0] * len(fetch_urls)

    return _process_intersecting_tiles(
        roi=roi,
        fetch_urls=fetch_urls,
        scale_factors=scale_factors,
        gdal_env={"AWS_NO_SIGN_REQUEST": "YES"},
        progress=progress,
    )


def fetch_naip_chm(
    roi: gpd.GeoDataFrame,
    version: str,
    progress: Callable[[str, int | None], None],
) -> tuple[xr.Dataset, TileMetadata]:
    """Fetch NAIP high-resolution canopy height model data."""
    progress("Loading NAIP CHM parquet index...", 10)

    roi_4326 = roi.to_crs("EPSG:4326")
    bounds = tuple(roi_4326.total_bounds)

    try:
        # Spatial pushdown filter
        intersecting = gpd.read_parquet(NAIP_INDEX_URL, bbox=bounds)
    except Exception as e:
        raise ProcessingError(
            code="INDEX_FETCH_FAILED",
            message="Failed to load NAIP CHM parquet index.",
            traceback=str(e),
        )

    intersecting = intersecting[intersecting.intersects(roi_4326.union_all())]

    if intersecting.empty:
        raise ProcessingError(
            code="COVERAGE_ERROR",
            message="No NAIP CHM tiles found for the given domain extent.",
            suggestion="The domain may be outside the CONUS coverage area.",
        )

    # Extract exactly what the helper needs into explicit Python lists
    fetch_urls = intersecting["chm_url"].tolist()
    if "scale_factor" in intersecting.columns:
        scale_factors = intersecting["scale_factor"].tolist()
    else:
        scale_factors = [100.0] * len(fetch_urls)

    return _process_intersecting_tiles(
        roi=roi,
        fetch_urls=fetch_urls,
        scale_factors=scale_factors,
        gdal_env={},  # Standard HTTP requests require no auth config
        progress=progress,
    )
