"""
CHM (Canopy Height Model) source handlers.

Pure functions that fetch CHM data for a domain extent.
All handlers return xr.Dataset where each variable name is a band name.
"""

import os
from collections.abc import Callable
from contextlib import contextmanager

import geopandas as gpd
import xarray as xr
from rioxarray.merge import merge_arrays

from lib.config import TABLES_BUCKET
from lib.raster import RasterConnection

S3_BASE = "s3://dataforgood-fb-data/forests/v1/alsgedi_global_v6_float/chm"


@contextmanager
def _anonymous_s3():
    """Temporarily set AWS_NO_SIGN_REQUEST for anonymous S3 access."""
    prev = os.environ.get("AWS_NO_SIGN_REQUEST")
    os.environ["AWS_NO_SIGN_REQUEST"] = "YES"
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop("AWS_NO_SIGN_REQUEST", None)
        else:
            os.environ["AWS_NO_SIGN_REQUEST"] = prev


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
    tile_arrays = []
    with _anonymous_s3():
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
