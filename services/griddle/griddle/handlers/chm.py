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
from lib.config import TABLES_BUCKET
from lib.raster import RasterConnection

S3_BASE = "s3://dataforgood-fb-data/forests/v1/alsgedi_global_v6_float/chm"
NAIP_INDEX_URL = f"gs://{TABLES_BUCKET}/naip_chm_index.parquet"


def _process_intersecting_tiles(
    roi: gpd.GeoDataFrame,
    intersecting_tiles: gpd.GeoDataFrame,
    gdal_env: dict[str, str],
    progress: Callable[[str, int | None], None],
) -> xr.Dataset:
    """Shared core processing logic for CHM extractions.

    Expects intersecting_tiles to contain 'fetch_url' and 'scale_factor' columns.
    """
    n_tiles = len(intersecting_tiles)
    tile_arrays = []

    # Apply the specific GDAL environment variables (e.g., AWS auth bypass)
    with rasterio.Env(**gdal_env):
        for i, (_, row) in enumerate(intersecting_tiles.iterrows()):
            progress(
                f"Fetching CHM tile {i + 1}/{n_tiles}...",
                10 + int(60 * (i + 1) / n_tiles),
            )

            raster = RasterConnection(
                row["fetch_url"], connection_type="rioxarray", cache=True
            )
            data = raster.extract_window(
                roi=roi,
                projection_padding_meters=1200,
                interpolation_padding_cells=240,
            )

            # Squeeze band dimension to satisfy strict 2D requirements
            data_2d = data.squeeze("band", drop=True)

            # Apply scale factor if needed (e.g., NAIP UInt16 -> Float meters)
            scale_factor = float(row.get("scale_factor", 1.0))
            if scale_factor != 1.0:
                data_2d = data_2d / scale_factor

            tile_arrays.append(data_2d)

    progress("Building dataset...", 80)

    # Mosaic if the ROI spanned multiple tiles
    if len(tile_arrays) == 1:
        chm_da = tile_arrays[0]
    else:
        chm_da = merge_arrays(tile_arrays)

    # Wrap in xr.Dataset with strict variable naming
    ds = xr.Dataset({"chm": chm_da})
    ds = ds.rio.write_crs(chm_da.rio.crs)
    ds = ds.rio.write_transform(chm_da.rio.transform())

    return ds


def fetch_meta_chm(
    roi: gpd.GeoDataFrame,
    version: str,
    progress: Callable[[str, int | None], None],
) -> xr.Dataset:
    """Fetch Meta global canopy height model data."""
    progress("Loading Meta CHM tile mapping...", 10)

    tile_map_url = (
        f"gs://{TABLES_BUCKET}/Meta{version}_chm_map_from_polygon_to_geotiff.geojson"
    )
    tile_polygons = gpd.read_file(tile_map_url)

    roi_4326 = roi.to_crs("EPSG:4326")
    intersecting = tile_polygons[tile_polygons.intersects(roi_4326.union_all())]

    if intersecting.empty:
        raise ProcessingError(
            code="COVERAGE_ERROR",
            message="No Meta CHM tiles found for the given domain extent.",
            suggestion="The domain may be outside the dataset coverage area.",
        )

    # Standardize columns for the shared processor
    # Meta requires building the S3 URL dynamically and uses no scale factor
    intersecting["fetch_url"] = f"{S3_BASE}/" + intersecting["tile"] + ".tif"
    intersecting["scale_factor"] = 1.0

    return _process_intersecting_tiles(
        roi=roi,
        intersecting_tiles=intersecting,
        gdal_env={"AWS_NO_SIGN_REQUEST": "YES"},
        progress=progress,
    )


def fetch_naip_chm(
    roi: gpd.GeoDataFrame,
    version: str,
    progress: Callable[[str, int | None], None],
) -> xr.Dataset:
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

    # Standardize columns for the shared processor
    # NAIP provides HTTP URLs directly and uses a 100x scale factor
    intersecting["fetch_url"] = intersecting["chm_url"]
    if "scale_factor" not in intersecting.columns:
        intersecting["scale_factor"] = 100.0

    return _process_intersecting_tiles(
        roi=roi,
        intersecting_tiles=intersecting,
        gdal_env={},  # Standard HTTP requests require no auth config
        progress=progress,
    )
