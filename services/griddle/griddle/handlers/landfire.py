"""
LANDFIRE source handlers.

Pure functions that fetch LANDFIRE data for a domain extent.
All handlers return xr.Dataset where each variable name is a band name.
"""

from collections.abc import Callable

import geopandas as gpd
import xarray as xr
from xarray import DataArray

from lib.config import RASTERS_BUCKET
from lib.raster import RasterConnection


def _fetch_landfire_raster(
    roi: gpd.GeoDataFrame,
    product: str,
    version: str,
) -> DataArray:
    """Fetch a single LANDFIRE raster product.

    Args:
        roi: GeoDataFrame defining the region of interest
        product: Product name as it appears in the GCS filename
        version: LANDFIRE version year

    Returns:
        DataArray with dims (y, x)
    """
    url = f"gs://{RASTERS_BUCKET}/LF{version}_{product}_CONUS.tif"
    raster = RasterConnection(url, connection_type="rioxarray", cache=True)
    data = raster.extract_window(
        roi=roi,
        projection_padding_meters=15 * raster.raster_resolution,
        interpolation_padding_cells=8,
    )
    return data.squeeze("band", drop=True)


def _to_dataset(variables: dict[str, DataArray]) -> xr.Dataset:
    """Build a Dataset from named DataArrays, propagating spatial metadata.

    Args:
        variables: Mapping of band name to DataArray (all must share the
            same CRS and transform)

    Returns:
        Dataset with CRS and transform written via rioxarray
    """
    first = next(iter(variables.values()))
    ds = xr.Dataset(variables)
    ds = ds.rio.write_crs(first.rio.crs)
    ds = ds.rio.write_transform(first.rio.transform())
    return ds


def fetch_fbfm40(roi: gpd.GeoDataFrame, version: str = "2022") -> xr.Dataset:
    """Fetch LANDFIRE FBFM40 fuel model codes.

    Args:
        roi: GeoDataFrame defining the region of interest
        version: LANDFIRE version year (default "2022")

    Returns:
        Dataset with a single "fbfm" variable (int16 categorical codes)
    """
    data = _fetch_landfire_raster(roi, "FBFM40", version)
    return _to_dataset({"fbfm": data})


def fetch_topography(
    roi: gpd.GeoDataFrame,
    version: str,
    bands: list[str],
    progress: Callable[[str, int | None], None],
) -> xr.Dataset:
    """Fetch LANDFIRE topographic data.

    Args:
        roi: GeoDataFrame defining the region of interest
        version: LANDFIRE version year (default "2020")
        bands: List of band names to fetch ("elevation", "slope", "aspect")
        progress: Progress callback

    Returns:
        Dataset with one named variable per requested band, each with
        dims (y, x). Variable names match band keys so they appear as
        correct band descriptions in GeoTIFF exports.
    """
    variables = {}
    for i, band in enumerate(bands):
        pct = 10 + int(70 * i / len(bands))
        progress(f"Fetching LANDFIRE {band}...", pct)
        variables[band] = _fetch_landfire_raster(roi, band, version)

    return _to_dataset(variables)
