"""
LANDFIRE source handlers.

Pure functions that fetch LANDFIRE data for a domain extent.
All handlers return xr.Dataset where each variable name is a band name.
"""

from collections.abc import Callable
from contextlib import nullcontext

import geopandas as gpd
import numpy as np
import xarray as xr
from numpy import ndarray
from scipy.ndimage import generic_filter
from xarray import DataArray

from griddle.handlers import landfire_lfps
from lib.alignment import RESAMPLING_METHOD_MAP, resolve_alignment_destination
from lib.config import RASTERS_BUCKET
from lib.landfire import LANDFIRE_VERSIONS, NB_CODE_MAP, validate_landfire_version
from lib.raster import RasterConnection, cog_env

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

# LANDFIRE rasters can carry an undeclared -9999 sentinel that the TIFF
# nodata tag doesn't account for. Before 2024, the declared value was
# simply wrong and -9999 was the real sentinel throughout. From 2024 on,
# the declared value is correct but -9999 still shows up alongside it.
# Either way, both are unified onto the declared value at fetch time so
# downstream code can trust rio.nodata.
LANDFIRE_EXTRA_NODATA: int = -9999


def _landfire_cog_url(product: str, version: str) -> str:
    """Build the gs:// path to a staged LANDFIRE COG."""
    if product.lower() in LANDFIRE_VERSIONS:
        product = product.upper()
    return f"gs://{RASTERS_BUCKET}/LF{version}_{product}_CONUS.tif"


def _needs_lfps(product: str, version: str) -> bool:
    """Whether `version` is only served on-demand via LFPS, not as a staged COG."""
    return version in LANDFIRE_VERSIONS[product].get("lfps_available", ())


def _fetch_landfire_raster(
    roi: gpd.GeoDataFrame,
    url: str,
    extent_buffer_cells: int,
    alignment: dict,
    target_grid_doc: dict | None,
    is_categorical: bool,
) -> DataArray:
    """Fetch a single LANDFIRE raster product.

    Args:
        roi: GeoDataFrame defining the region of interest
        url: Location of the raster to open -- either a `gs://` path to a
             staged COG, or a local filesystem path to a downloaded and
             unzipped LFPS job's output.
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
        data = data.squeeze("band", drop=True)

        return _consolidate_landfire_nodata(data)


def _consolidate_landfire_nodata(data: DataArray) -> DataArray:
    """Fold LANDFIRE's undeclared -9999 sentinel onto the declared nodata value.

    Shared by every LANDFIRE-derived fetch path -- staged COGs and
    on-demand LFPS output alike -- so neither can drift from this rule.
    """
    if data.rio.nodata is None:
        data = data.rio.write_nodata(LANDFIRE_EXTRA_NODATA)
    declared = data.rio.nodata

    return data.where(data != LANDFIRE_EXTRA_NODATA, declared)


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


def fetch_annual_disturbance(
    roi: gpd.GeoDataFrame,
    progress: Callable[[str, int | None], None],
    version: str = LANDFIRE_VERSIONS["annual_disturbance"]["default"],
    extent_buffer_cells: int = 0,
    alignment: dict | None = None,
    target_grid_doc: dict | None = None,
) -> xr.Dataset:
    """Fetch LANDFIRE Limited Annual Disturbance codes.

    Always fetched on demand via LFPS.

    Args:
        roi: GeoDataFrame defining the region of interest
        version: LANDFIRE version year (default from LANDFIRE_VERSIONS)
        extent_buffer_cells: Result-grid cells of buffer around the ROI
        alignment: Alignment specification dict. Defaults to
            ``{"target": "domain"}`` when omitted.
        target_grid_doc: Loaded grid document used when
            ``alignment["target"] == "grid"``.
        progress: Progress callback (submit/wait/download reports through it).

    Returns:
        Dataset with a single "annual_disturbance" variable (raw VALUE codes)
    """
    alignment = alignment or {"target": "domain"}
    with landfire_lfps.fetch_lfps(
        roi,
        "annual_disturbance",
        version,
        alignment,
        target_grid_doc,
        extent_buffer_cells,
        progress,
    ) as url:
        data = _fetch_landfire_raster(
            roi,
            url,
            extent_buffer_cells,
            alignment,
            target_grid_doc,
            is_categorical=True,
        )

    return _to_dataset({"annual_disturbance": data})


FBFM_NON_BURNABLE = set(range(91, 100))
FCCS_BARE_GROUND = {0}


def scatter_categorical_boundaries(
    grid: np.ndarray,
    depth: int,
    seed: int,
    protected: set[int],
) -> np.ndarray:
    """Stochastic boundary scattering for categorical raster data.

    For each cardinal direction, rolls the grid 1..depth cells. At each
    step, cells where the rolled value differs from the previous roll get a
    swap probability of 0.5/distance. Protected values never participate:
    cells holding a protected value won't change, and a protected value
    won't overwrite another cell.

    Args:
        grid: 2D integer array of categorical codes.
        depth: How many cells deep the scattering can reach.
        seed: Random seed for reproducibility.
        protected: Values that must not participate in scattering
            (non-burnable codes, nodata sentinels, etc.).

    Returns:
        Scattered copy of grid, same shape and dtype.
    """
    src = grid.astype(np.float64)
    target = src.copy()
    rng = np.random.default_rng(seed)
    rand_grid = rng.random(src.shape)

    # Cells starting as protected can't accept a swap
    rand_grid[np.isin(src, list(protected))] = 1.0

    for direction, axis, shift in [
        ("up", 0, -1),
        ("right", 1, 1),
        ("down", 0, 1),
        ("left", 1, -1),
    ]:
        prob_grid = np.zeros_like(src)
        current_roll = src.copy()

        for i in range(1, depth + 1):
            next_roll = np.roll(current_roll, shift, axis=axis)
            diff = current_roll - next_roll
            prob_grid[diff != 0] = 0.5 / i
            current_roll = next_roll

        # Don't let protected values overwrite other cells
        rand_grid[np.isin(current_roll, list(protected))] = 1.0

        # Zero out probabilities near cyclic boundaries
        if direction == "up":
            prob_grid[-depth:, :] = 0
        elif direction == "down":
            prob_grid[:depth, :] = 0
        elif direction == "left":
            prob_grid[:, -depth:] = 0
        else:
            prob_grid[:, :depth] = 0

        target[rand_grid < prob_grid] = current_roll[rand_grid < prob_grid]

    return target.astype(grid.dtype)


def _apply_boundary_scatter(
    data: DataArray,
    boundary_scatter: dict,
    protected: set[int],
) -> DataArray:
    """Apply boundary scatter to a categorical DataArray if configured."""
    nodata = data.rio.nodata
    if nodata is not None:
        protected = protected | {int(nodata)}

    scattered = scatter_categorical_boundaries(
        data.values,
        depth=boundary_scatter.get("depth", 10),
        seed=boundary_scatter.get("seed", 42),
        protected=protected,
    )
    return data.copy(data=scattered)


def fetch_fbfm13(
    roi: gpd.GeoDataFrame,
    progress: Callable[[str, int | None], None],
    version: str = LANDFIRE_VERSIONS["fbfm13"]["default"],
    remove_non_burnable: list[str] | None = None,
    boundary_scatter: dict | None = None,
    extent_buffer_cells: int = 0,
    alignment: dict | None = None,
    target_grid_doc: dict | None = None,
) -> xr.Dataset:
    """Fetch LANDFIRE FBFM13 fuel model codes.

    Args:
        roi: GeoDataFrame defining the region of interest
        version: LANDFIRE version year (default from LANDFIRE_VERSIONS)
        remove_non_burnable: List of non-burnable fuel model names to remove
            (e.g., ["NB1", "NB3", "NB9"]). Removed codes are replaced by the
            most frequent neighboring burnable fuel model via majority filter.
        extent_buffer_cells: Result-grid cells of buffer around the ROI
        alignment: Alignment specification dict. Defaults to
            ``{"target": "domain"}`` when omitted.
        target_grid_doc: Loaded grid document used when
            ``alignment["target"] == "grid"``.
        progress: Progress callback, only used when `version` is fetched via
            LFPS (submit/wait/download reports through it).

    Returns:
        Dataset with a single "fbfm13" variable (int16 categorical codes,
        1-13 plus non-burnable 91/92/93/98/99)
    """
    alignment = alignment or {"target": "domain"}
    product = "fbfm13"
    if _needs_lfps(product, version):
        source = landfire_lfps.fetch_lfps(
            roi,
            product,
            version,
            alignment,
            target_grid_doc,
            extent_buffer_cells,
            progress,
        )
    else:
        validate_landfire_version(product, version)
        source = nullcontext(_landfire_cog_url(product, version))

    with source as url:
        data = _fetch_landfire_raster(
            roi,
            url,
            extent_buffer_cells,
            alignment,
            target_grid_doc,
            is_categorical=True,
        )

    if remove_non_burnable:
        non_burnable_keys = [NB_CODE_MAP[code] for code in remove_non_burnable]
        filtered = _remove_non_burnable_blocks(data.values, non_burnable_keys)
        data = data.copy(data=filtered)

    if boundary_scatter:
        data = _apply_boundary_scatter(data, boundary_scatter, FBFM_NON_BURNABLE)

    return _to_dataset({"fbfm13": data})


def fetch_fbfm40(
    roi: gpd.GeoDataFrame,
    progress: Callable[[str, int | None], None],
    version: str = LANDFIRE_VERSIONS["fbfm40"]["default"],
    remove_non_burnable: list[str] | None = None,
    boundary_scatter: dict | None = None,
    extent_buffer_cells: int = 0,
    alignment: dict | None = None,
    target_grid_doc: dict | None = None,
    season: str | None = None,
) -> xr.Dataset:
    """Fetch LANDFIRE FBFM40 fuel model codes.

    Args:
        roi: GeoDataFrame defining the region of interest
        version: LANDFIRE version year (default from LANDFIRE_VERSIONS).
        remove_non_burnable: List of non-burnable fuel model names to remove
            (e.g., ["NB1", "NB3", "NB9"]). Removed codes are replaced by the
            most frequent neighboring burnable fuel model via majority filter.
        extent_buffer_cells: Result-grid cells of buffer around the ROI
        alignment: Alignment specification dict. Defaults to
            ``{"target": "domain"}`` when omitted.
        target_grid_doc: Loaded grid document used when
            ``alignment["target"] == "grid"``.
        progress: Progress callback, only used when `season` is set (LFPS
             submit/wait/download reports through it).
        season: LANDFIRE season code (e.g. "SP"). When set, fetches an
             on-demand seasonal variant via LFPS.

    Returns:
        Dataset with a single "fbfm" variable (int16 categorical codes)
    """
    alignment = alignment or {"target": "domain"}
    product = "fbfm40"
    if season is not None or _needs_lfps(product, version):
        source = landfire_lfps.fetch_lfps(
            roi,
            product,
            version,
            alignment,
            target_grid_doc,
            extent_buffer_cells,
            progress,
            season,
        )
    else:
        validate_landfire_version(product, version)
        source = nullcontext(_landfire_cog_url(product, version))

    with source as url:
        # `url` is the gs:// or local path `_fetch_landfire_raster` opens
        data = _fetch_landfire_raster(
            roi,
            url,
            extent_buffer_cells,
            alignment,
            target_grid_doc,
            is_categorical=True,
        )

    if remove_non_burnable:
        non_burnable_keys = [NB_CODE_MAP[code] for code in remove_non_burnable]
        filtered = _remove_non_burnable_blocks(data.values, non_burnable_keys)
        data = data.copy(data=filtered)

    if boundary_scatter:
        data = _apply_boundary_scatter(data, boundary_scatter, FBFM_NON_BURNABLE)

    return _to_dataset({"fbfm": data})


def fetch_fccs(
    roi: gpd.GeoDataFrame,
    progress: Callable[[str, int | None], None],
    version: str = LANDFIRE_VERSIONS["fccs"]["default"],
    remove_bare_ground: bool = False,
    boundary_scatter: dict | None = None,
    extent_buffer_cells: int = 0,
    alignment: dict | None = None,
    target_grid_doc: dict | None = None,
) -> xr.Dataset:
    """Fetch LANDFIRE FCCS fuel model codes.

    Args:
        roi: GeoDataFrame defining the region of interest
        version: version: LANDFIRE version year (default from LANDFIRE_VERSIONS)
        remove_bare_ground: If True, removes FCCS fuelbed ID 0 (bare ground).
            Removed cells are replaced by the most frequent neighboring
            non-bare-ground fuelbed via majority filter.
        extent_buffer_cells: Result-grid cells of buffer around the ROI
        alignment: Alignment specification dict. Defaults to
            ``{"target": "domain"}`` when omitted.
        target_grid_doc: Loaded grid document used when
            ``alignment["target"] == "grid"``.
        progress: Progress callback, only used when `version` is fetched via
            LFPS (submit/wait/download reports through it).

    Returns:
        Dataset with a single "fccs" variable (int32 categorical codes)
    """
    alignment = alignment or {"target": "domain"}
    product = "fccs"
    if _needs_lfps(product, version):
        source = landfire_lfps.fetch_lfps(
            roi,
            product,
            version,
            alignment,
            target_grid_doc,
            extent_buffer_cells,
            progress,
        )
    else:
        validate_landfire_version(product, version)
        source = nullcontext(_landfire_cog_url(product, version))

    with source as url:
        data = _fetch_landfire_raster(
            roi,
            url,
            extent_buffer_cells,
            alignment,
            target_grid_doc,
            is_categorical=True,
        )

    if remove_bare_ground:
        filtered = _remove_non_burnable_blocks(data.values, [0])
        data = data.copy(data=filtered)

    if boundary_scatter:
        data = _apply_boundary_scatter(data, boundary_scatter, FCCS_BARE_GROUND)

    return _to_dataset({"fccs": data})


def _remove_non_burnable_blocks(grid: ndarray, non_burnable_keys: list[int]) -> ndarray:
    """Replace non-burnable fuel model codes with neighboring burnable codes.

    Uses a 5x5 majority filter to replace each targeted non-burnable cell
    with the most frequent burnable fuel model in its neighborhood. The
    filter is applied iteratively until no targeted codes remain.

    Args:
        grid: 2D array of LANDFIRE (FBFM or FCCS) fuel model codes
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
        url = _landfire_cog_url(band, version)
        variables[band] = _fetch_landfire_raster(
            roi,
            url,
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
        url = _landfire_cog_url(product_code, version)
        raw = _fetch_landfire_raster(
            roi,
            url,
            extent_buffer_cells,
            alignment,
            target_grid_doc,
            is_categorical=False,
        )
        variables[band] = _scale_canopy_band(raw, LANDFIRE_CANOPY_SCALE_FACTORS[band])

    return _to_dataset(variables)


def _scale_canopy_band(data: DataArray, scale: float) -> DataArray:
    """Mask nodata and decode a LANDFIRE canopy band to physical units.

    Assumes `data` has already gone through `_fetch_landfire_raster`, so
    its declared nodata is trustworthy and any undeclared -9999 sentinel
    has already been folded into it. Masks that value, then divides by
    `scale` to convert from the int16 storage representation into
    physical units (m for chm/cbh, kg/m**3 for cbd, % for cc).
    """
    declared_nodata = data.rio.nodata
    out = data.astype("float32").where(data != declared_nodata)
    if scale != 1.0:
        out = out / scale
    return out.rio.write_nodata(np.nan, encoded=True)
