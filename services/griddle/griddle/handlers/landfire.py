"""
LANDFIRE source handlers.

Pure functions that fetch LANDFIRE data for a domain extent.
All handlers return xr.Dataset where each variable name is a band name.
"""

from collections.abc import Callable

import geopandas as gpd
import numpy as np
import xarray as xr
from numpy import ndarray
from scipy.ndimage import generic_filter
from xarray import DataArray

from lib.config import RASTERS_BUCKET
from lib.raster import RasterConnection, cog_env

NB_CODE_MAP: dict[str, int] = {
    "NB1": 91,
    "NB2": 92,
    "NB3": 93,
    "NB8": 98,
    "NB9": 99,
}


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
    with cog_env():
        raster = RasterConnection(url, connection_type="rioxarray", cache=True)
        data = raster.extract_window(
            roi=roi,
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


def fetch_fbfm40(
    roi: gpd.GeoDataFrame,
    version: str = "2024",
    remove_non_burnable: list[str] | None = None,
) -> xr.Dataset:
    """Fetch LANDFIRE FBFM40 fuel model codes.

    Args:
        roi: GeoDataFrame defining the region of interest
        version: LANDFIRE version year (default "2024")
        remove_non_burnable: List of non-burnable fuel model names to remove
            (e.g., ["NB1", "NB3", "NB9"]). Removed codes are replaced by the
            most frequent neighboring burnable fuel model via majority filter.

    Returns:
        Dataset with a single "fbfm" variable (int16 categorical codes)
    """
    data = _fetch_landfire_raster(roi, "FBFM40", version)

    if remove_non_burnable:
        non_burnable_keys = [NB_CODE_MAP[code] for code in remove_non_burnable]
        filtered = _remove_non_burnable_blocks(data.values, non_burnable_keys)
        data = data.copy(data=filtered)

    return _to_dataset({"fbfm": data})


def fetch_fccs(
    roi: gpd.GeoDataFrame,
    version: str = "2023",
) -> xr.Dataset:
    """Fetch LANDFIRE FCCS fuel model codes.

    Args:
        roi: GeoDataFrame defining the region of interest
        version: LANDFIRE version year (default "2023")

    Returns:
        Dataset with a single "fccs" variable (int32 categorical codes)
    """
    data = _fetch_landfire_raster(roi, "FCCS", version)

    return _to_dataset({"fccs": data})


def _remove_non_burnable_blocks(grid: ndarray, non_burnable_keys: list[int]) -> ndarray:
    """Replace non-burnable fuel model codes with neighboring burnable codes.

    Uses a 5x5 majority filter to replace each targeted non-burnable cell
    with the most frequent burnable fuel model in its neighborhood. The
    filter is applied iteratively until no targeted codes remain.

    Args:
        grid: 2D array of FBFM40 fuel model codes
        non_burnable_keys: Numeric codes to replace (e.g., [91, 93, 99])

    Returns:
        Copy of grid with targeted non-burnable codes replaced
    """
    nb_mask = np.isin(grid, non_burnable_keys)
    if not np.any(nb_mask):
        return grid.copy()

    filtered = generic_filter(
        grid,
        function=_most_frequent,
        size=(5, 5),
        mode="nearest",
        extra_arguments=(non_burnable_keys,),
    )

    # Re-apply until no targeted non-burnable codes remain in the filtered result
    remaining = np.isin(filtered, non_burnable_keys)
    iterations = 0
    while np.any(remaining):
        if iterations > 1_000_000:
            break
        filtered = generic_filter(
            filtered,
            function=_most_frequent,
            size=(5, 5),
            mode="nearest",
            extra_arguments=(non_burnable_keys,),
        )
        remaining = np.isin(filtered, non_burnable_keys)
        iterations += 1

    output = grid.copy()
    output[nb_mask] = filtered[nb_mask]
    return output


def _most_frequent(x: ndarray, non_burnable_keys: list[int]) -> float:
    """Return the most frequent burnable value in a flattened window.

    Prefers the central pixel when it is burnable and tied for most frequent.
    Falls back to the central pixel if no burnable values exist in the window.
    """
    central = x[x.size // 2]
    values, counts = np.unique(x, return_counts=True)
    max_freq = counts.max()
    modes = values[counts == max_freq]
    if central in modes and central not in non_burnable_keys:
        return central
    sorted_values = values[np.argsort(counts)[::-1]]
    for val in sorted_values:
        if val not in non_burnable_keys:
            return val
    return central


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
