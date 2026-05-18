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

from lib.alignment import RESAMPLING_METHOD_MAP, resolve_alignment_destination
from lib.config import RASTERS_BUCKET
from lib.raster import RasterConnection, cog_env

NB_CODE_MAP: dict[str, int] = {
    "NB1": 91,
    "NB2": 92,
    "NB3": 93,
    "NB8": 98,
    "NB9": 99,
}

CATEGORICAL_DEFAULT = "nearest"
CONTINUOUS_DEFAULT = "bilinear"

LANDFIRE_CANOPY_PRODUCT_MAP: dict[str, str] = {
    "chm": "CH",
    "cbd": "CBD",
    "cbh": "CBH",
    "cc": "CC",
}

LANDFIRE_CANOPY_SCALE_FACTORS: dict[str, float] = {
    "chm": 10.0,
    "cbd": 100.0,
    "cbh": 10.0,
    "cc": 1.0,
}

# LANDFIRE canopy rasters carry two coexisting nodata sentinels: 32767 is
# declared in the TIFF nodata tag; -9999 also appears in pixel data without
# being declared anywhere.
LANDFIRE_CANOPY_EXTRA_NODATA: int = -9999


def _fetch_landfire_raster(
    roi: gpd.GeoDataFrame,
    product: str,
    version: str,
    extent_buffer_cells: int,
    alignment: dict,
    target_grid_doc: dict | None,
    is_categorical: bool,
) -> DataArray:
    """Fetch a single LANDFIRE raster product.

    Args:
        roi: GeoDataFrame defining the region of interest
        product: Product name as it appears in the GCS filename
        version: LANDFIRE version year
        extent_buffer_cells: Result-grid cells of buffer around the ROI
        alignment: Alignment specification dict (see ``GridAlignmentSpecification``).
            Threaded into the single ``rio.reproject`` performed by
            ``extract_window`` — no second reprojection is layered on top.
        target_grid_doc: Loaded grid document used as the alignment target
            when ``alignment["target"] == "grid"``. Required in that case.
        is_categorical: Drives the role-aware default for the resampling
            method when ``alignment.method`` is unset (categorical →
            ``nearest``; continuous → ``bilinear``).

    Returns:
        DataArray with dims (y, x)
    """
    url = f"gs://{RASTERS_BUCKET}/LF{version}_{product}_CONUS.tif"
    with cog_env():
        raster = RasterConnection(url, connection_type="rioxarray", cache=True)
        method_name = alignment.get("method") or (
            CATEGORICAL_DEFAULT if is_categorical else CONTINUOUS_DEFAULT
        )
        dest = resolve_alignment_destination(
            alignment,
            roi,
            target_grid_doc,
            raster.target_native_resolution(roi)[0],
            extent_buffer_cells=extent_buffer_cells,
        )
        data = raster.extract_window(
            roi=roi,
            interpolation_padding_cells=extent_buffer_cells,
            resampling=RESAMPLING_METHOD_MAP[method_name],
            destination_resolution=alignment.get("resolution")
            if alignment["target"] == "native"
            else None,
            **dest,
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
    extent_buffer_cells: int = 0,
    alignment: dict | None = None,
    target_grid_doc: dict | None = None,
) -> xr.Dataset:
    """Fetch LANDFIRE FBFM40 fuel model codes.

    Args:
        roi: GeoDataFrame defining the region of interest
        version: LANDFIRE version year (default "2024")
        remove_non_burnable: List of non-burnable fuel model names to remove
            (e.g., ["NB1", "NB3", "NB9"]). Removed codes are replaced by the
            most frequent neighboring burnable fuel model via majority filter.
        extent_buffer_cells: Result-grid cells of buffer around the ROI
        alignment: Alignment specification dict. Defaults to
            ``{"target": "domain"}`` when omitted.
        target_grid_doc: Loaded grid document used when
            ``alignment["target"] == "grid"``.

    Returns:
        Dataset with a single "fbfm" variable (int16 categorical codes)
    """
    alignment = alignment or {"target": "domain"}
    data = _fetch_landfire_raster(
        roi,
        "FBFM40",
        version,
        extent_buffer_cells,
        alignment,
        target_grid_doc,
        is_categorical=True,
    )

    if remove_non_burnable:
        non_burnable_keys = [NB_CODE_MAP[code] for code in remove_non_burnable]
        filtered = _remove_non_burnable_blocks(data.values, non_burnable_keys)
        data = data.copy(data=filtered)

    return _to_dataset({"fbfm": data})


def fetch_fccs(
    roi: gpd.GeoDataFrame,
    version: str = "2023",
    remove_bare_ground: bool = False,
    extent_buffer_cells: int = 0,
    alignment: dict | None = None,
    target_grid_doc: dict | None = None,
) -> xr.Dataset:
    """Fetch LANDFIRE FCCS fuel model codes.

    Args:
        roi: GeoDataFrame defining the region of interest
        version: LANDFIRE version year (default "2023")
        remove_bare_ground: If True, removes FCCS fuelbed ID 0 (bare ground).
            Removed cells are replaced by the most frequent neighboring
            non-bare-ground fuelbed via majority filter.
        extent_buffer_cells: Result-grid cells of buffer around the ROI
        alignment: Alignment specification dict. Defaults to
            ``{"target": "domain"}`` when omitted.
        target_grid_doc: Loaded grid document used when
            ``alignment["target"] == "grid"``.

    Returns:
        Dataset with a single "fccs" variable (int32 categorical codes)
    """
    alignment = alignment or {"target": "domain"}
    data = _fetch_landfire_raster(
        roi,
        "FCCS",
        version,
        extent_buffer_cells,
        alignment,
        target_grid_doc,
        is_categorical=True,
    )

    if remove_bare_ground:
        filtered = _remove_non_burnable_blocks(data.values, [0])
        data = data.copy(data=filtered)

    return _to_dataset({"fccs": data})


def _remove_non_burnable_blocks(grid: ndarray, non_burnable_keys: list[int]) -> ndarray:
    """Replace non-burnable fuel model codes with neighboring burnable codes.

    Uses a 5x5 majority filter to replace each targeted non-burnable cell
    with the most frequent burnable fuel model in its neighborhood. The
    filter is applied iteratively until no targeted codes remain.

    Args:
        grid: 2D array of LANDFIRE (FBFM40 or FCCS) fuel model codes
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
    extent_buffer_cells: int = 0,
    alignment: dict | None = None,
    target_grid_doc: dict | None = None,
) -> xr.Dataset:
    """Fetch LANDFIRE topographic data.

    Args:
        roi: GeoDataFrame defining the region of interest
        version: LANDFIRE version year (default "2020")
        bands: List of band names to fetch ("elevation", "slope", "aspect")
        progress: Progress callback
        extent_buffer_cells: Result-grid cells of buffer around the ROI
        alignment: Alignment specification dict. Defaults to
            ``{"target": "domain"}`` when omitted.
        target_grid_doc: Loaded grid document used when
            ``alignment["target"] == "grid"``.

    Returns:
        Dataset with one named variable per requested band, each with
        dims (y, x). Variable names match band keys so they appear as
        correct band descriptions in GeoTIFF exports.
    """
    alignment = alignment or {"target": "domain"}
    variables = {}
    for i, band in enumerate(bands):
        pct = 10 + int(70 * i / len(bands))
        progress(f"Fetching LANDFIRE {band}...", pct)
        variables[band] = _fetch_landfire_raster(
            roi,
            band,
            version,
            extent_buffer_cells,
            alignment,
            target_grid_doc,
            is_categorical=False,
        )

    return _to_dataset(variables)


def fetch_canopy_landfire(
    roi: gpd.GeoDataFrame,
    version: str,
    bands: list[str],
    progress: Callable[[str, int | None], None],
    extent_buffer_cells: int = 0,
    alignment: dict | None = None,
    target_grid_doc: dict | None = None,
) -> xr.Dataset:
    """Fetch LANDFIRE canopy fuel data for one or more bands.

    Args:
        roi: GeoDataFrame defining the region of interest
        version: LANDFIRE version year (e.g. "2024")
        bands: Requested API band names; subset of {"chm", "cbd", "cbh", "cc"}
        progress: Progress callback
        extent_buffer_cells: Result-grid cells of buffer around the ROI
        alignment: Alignment specification dict. Defaults to
            ``{"target": "domain"}`` when omitted.
        target_grid_doc: Loaded grid document used when
            ``alignment["target"] == "grid"``.

    Returns:
        Dataset with one named variable per requested band, decoded from
        the int16 storage representation into physical units (m for chm/cbh,
        kg/m**3 for cbd, % for cc) with NaN at both LANDFIRE nodata sentinels.
    """
    alignment = alignment or {"target": "domain"}
    variables: dict[str, DataArray] = {}
    for i, band in enumerate(bands):
        pct = 10 + int(70 * i / len(bands))
        progress(f"Fetching LANDFIRE canopy {band}...", pct)
        product_code = LANDFIRE_CANOPY_PRODUCT_MAP[band]
        raw = _fetch_landfire_raster(
            roi,
            product_code,
            version,
            extent_buffer_cells,
            alignment,
            target_grid_doc,
            is_categorical=False,
        )
        variables[band] = _scale_canopy_band(raw, LANDFIRE_CANOPY_SCALE_FACTORS[band])

    return _to_dataset(variables)


def _scale_canopy_band(data: DataArray, scale: float) -> DataArray:
    """Mask LANDFIRE canopy nodata sentinels and decode to physical units.

    LANDFIRE distributes canopy products as int16 with two coexisting nodata
    sentinels: 32767 (declared in the TIFF nodata tag) and -9999 (present in
    pixel data without being declared anywhere). Naively dividing would
    inject ``sentinel / scale`` into the valid value range, so we mask both
    before any arithmetic.
    """
    declared_nodata = data.rio.nodata
    sentinel_mask = data == LANDFIRE_CANOPY_EXTRA_NODATA
    if declared_nodata is not None:
        sentinel_mask = sentinel_mask | (data == declared_nodata)
    out = data.astype("float32").where(~sentinel_mask)
    if scale != 1.0:
        out = out / scale
    return out.rio.write_nodata(np.nan, encoded=True)
