"""
LANDFIRE source handlers.

Pure functions that fetch LANDFIRE data for a domain extent.
All handlers return xr.Dataset where each variable name is a band name.
"""

import logging
from collections.abc import Callable
from contextlib import nullcontext
from functools import cache, lru_cache

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from affine import Affine
from fastfuels_core.fuel_models import (
    build_fdist_raster,
    update_fuel_models,
)
from numpy import ndarray
from rasterio.features import rasterize
from rasterio.transform import array_bounds
from scipy.ndimage import generic_filter
from shapely.geometry import box
from xarray import DataArray

from griddle.handlers import landfire_lfps
from griddle.storage import load_band
from lib.alignment import RESAMPLING_METHOD_MAP, resolve_alignment_destination
from lib.config import (
    FEATURES_BUCKET,
    FEATURES_COLLECTION,
    RASTERS_BUCKET,
    TABLES_BUCKET,
)
from lib.errors import ProcessingError
from lib.firestore import DocumentNotFoundError, get_document
from lib.landfire import LANDFIRE_VERSIONS, NB_CODE_MAP, validate_landfire_version
from lib.raster import RasterConnection, cog_env

logger = logging.getLogger(__name__)

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
# downstream code can trust rio.nodata. The fold must happen before
# reprojection, or resampling blends -9999 into valid neighbours.
LANDFIRE_EXTRA_NODATA: int = -9999

# product: (column in the rules table, name of the output band)
_DISTURBED_FUEL_MODELS = {
    "fbfm13": ("FBFM13", "fbfm13"),
    "fbfm40": ("FBFM40_code", "fbfm"),
    "fccs": ("FCCS", "fccs"),
}

# FDist codes are 100 * D_TYPE + 10 * D_SEVERITY + D_TIME.
_VALID_FDIST_CODES = np.array(
    [
        100 * d_type + 10 * severity + time
        for d_type in range(1, 8)
        for severity in range(1, 4)
        for time in range(1, 4)
    ]
)

# LANDFIRE aspect marks flat cells (slope 0-2 deg) with -1. Topography grids
# store flat as 0, matching 3DEP.
LANDFIRE_FLAT_ASPECT: int = -1


def _landfire_cog_url(product: str, version: str) -> str:
    """Build the gs:// path to a staged LANDFIRE COG."""
    if product.lower() in LANDFIRE_VERSIONS:
        product = product.upper()
    return f"gs://{RASTERS_BUCKET}/LF{version}_{product}_CONUS.tif"


def _needs_lfps(product: str, version: str) -> bool:
    """Whether `version` is only served on-demand via LFPS, not as a staged COG."""
    return version in LANDFIRE_VERSIONS[product].get("lfps_available", ())


@lru_cache(maxsize=1)
def _lf_map_zones() -> gpd.GeoDataFrame:
    """LANDFIRE CONUS map zone polygons, with the zone number in ``ZONE_NUM``.

    Read from the rasters bucket on first use and cached. Not loaded at import:
    only disturbance update requests need it.
    """
    return gpd.read_parquet(f"gs://{RASTERS_BUCKET}/LF_MapZones.parquet")


def _lookup_lf_zones(
    zones: gpd.GeoDataFrame,
    transform: Affine,
    shape: tuple[int, int],
    crs,
    nodata: int = -9999,
) -> np.ndarray:
    """LANDFIRE map zone number for every cell of a grid.

    Args:
        zones: Map zone polygons with the zone number in ``ZONE_NUM``, in
            any CRS.
        transform: The grid's transform, e.g. ``da.rio.transform()``.
        shape: Grid shape, ``(ny, nx)``.
        crs: The grid's CRS, e.g. ``da.rio.crs``.
        nodata: Value for cells outside every map zone.

    Returns:
        int16 zone numbers, shape ``shape``.

    Most grids fall inside a single zone, which is filled directly. Only a
    grid that crosses a zone boundary or the CONUS edge is rasterized. Each
    cell gets the zone containing its center.

    Only the grid's outline is reprojected to find the zones it touches. Its
    edges can bend slightly after reprojection, but over typical grid sizes
    the error is well under one 30 m cell.
    """
    outline = (
        gpd.GeoSeries([box(*array_bounds(*shape, transform))], crs=crs)
        .to_crs(zones.crs)
        .iloc[0]
    )
    touching = zones.iloc[zones.sindex.query(outline, predicate="intersects")]

    if touching.empty:
        return np.full(shape, nodata, dtype=np.int16)
    if len(touching) == 1 and touching.geometry.iloc[0].contains(outline):
        return np.full(shape, int(touching["ZONE_NUM"].iloc[0]), dtype=np.int16)

    touching = touching.to_crs(crs)
    return rasterize(
        zip(touching.geometry, touching["ZONE_NUM"].astype(int)),
        out_shape=shape,
        transform=transform,
        fill=nodata,
        dtype="int16",
    )


@lru_cache(maxsize=1)
def _fuel_rulesets() -> pd.DataFrame:
    """LANDFIRE's Master_Rulesets table, prepped for fastfuels-core.

    Read from the tables bucket on first use and cached. The raw export
    is used as is except for:

    - Whitespace stripped from ``BPSRF``, ``Wildcard`` and ``OnOff``. Core
      compares them to "any" and "On" exactly, so padding would silently
      select the wrong rules.
    - ``FBFM40``, stored as ``"SH2 / 142"``, split into ``FBFM40_label``
      and an ``Int64`` ``FBFM40_code``. ``FBFM13`` and ``FCCS`` are already
      numeric. Core needs the numeric code column.
    """
    rules = pd.read_parquet(f"gs://{TABLES_BUCKET}/LF_fuel_rulesets.parquet")

    for column in ("BPSRF", "Wildcard", "OnOff"):
        rules[column] = rules[column].str.strip()

    parts = rules["FBFM40"].str.split("/", n=1, expand=True)
    rules["FBFM40_label"] = parts[0].str.strip()
    rules["FBFM40_code"] = pd.to_numeric(parts[1].str.strip()).astype("Int64")
    return rules.drop(columns="FBFM40")


@cache
def _ldist_attributes(version: str) -> pd.DataFrame:
    """LANDFIRE's LDist attribute table for a release, cached per version.

    Read from the tables bucket. Core uses its ``VALUE``, ``DIST_TYPE`` and
    ``SEVERITY`` columns to convert LDist codes to FDist codes.

    Raises:
        ProcessingError: The bucket has no table for ``version``.
    """
    try:
        return pd.read_csv(f"gs://{TABLES_BUCKET}/LF{version}_LDist_attributes.csv")
    except FileNotFoundError:
        raise ProcessingError(
            code="LDIST_ATTRIBUTES_NOT_AVAILABLE",
            message=f"No LDist attribute table for version {version} is available.",
            suggestion=f"Upload LF{version}_LDist_attributes.csv to the tables bucket.",
        )


def _check_fdist_codes(fdist: np.ndarray, nodata: float | None = None) -> None:
    """Fail unless every value of a user's FDist codes is valid.

    The codes come from a user's FDist grid (one per cell) or from their
    disturbance polygons (one per polygon).

    Zero, NaN and ``nodata`` mean undisturbed. Any other value must be
    ``100 * D_TYPE + 10 * D_SEVERITY + D_TIME`` with D_TYPE 1-7, D_SEVERITY 1-3
    and D_TIME 1-3, so a negative value other than ``nodata`` is invalid. Core
    silently ignores a positive value that matches no rule, so an invalid code
    would otherwise do nothing.

    Args:
        fdist: The FDist codes: a raster, or the codes of the polygons.
        nodata: The band's declared nodata value, if any. Polygons have none.

    Raises:
        ProcessingError: ``INVALID_FDIST_CODES``, listing the offending codes.
    """
    checked = np.isfinite(fdist) & (fdist != 0)
    if nodata is not None:
        checked &= fdist != nodata

    codes = np.unique(fdist[checked])
    invalid = codes[~np.isin(codes, _VALID_FDIST_CODES)]
    if invalid.size == 0:
        return

    listed = ", ".join(f"{code:g}" for code in invalid)
    raise ProcessingError(
        code="INVALID_FDIST_CODES",
        message=f"The disturbance data has values that aren't valid FDist codes: {listed}.",
        suggestion=(
            "FDist codes are 100 * D_TYPE + 10 * D_SEVERITY + D_TIME, with "
            "D_TYPE 1-7, D_SEVERITY 1-3 and D_TIME 1-3. Use 0 for undisturbed cells."
        ),
    )


def _load_disturbance_polygons(feature_id: str) -> gpd.GeoDataFrame:
    """A user's disturbance polygons, one row per polygon with its code in ``fdist``.

    The feature's record gives the domain its Parquet is stored under.

    Raises:
        ProcessingError: ``DISTURBANCE_FEATURE_NOT_FOUND`` when the feature or
            its Parquet is missing;  ``INVALID_FDIST_CODES`` when a polygon has
            no code.
    """
    try:
        _, snapshot = get_document(FEATURES_COLLECTION, feature_id)
        domain_id = snapshot.to_dict()["domain_id"]
        polygons = gpd.read_parquet(
            f"gs://{FEATURES_BUCKET}/{domain_id}/{feature_id}.parquet"
        )
    except (DocumentNotFoundError, FileNotFoundError) as e:
        raise ProcessingError(
            code="DISTURBANCE_FEATURE_NOT_FOUND",
            message=f"Disturbance feature {feature_id} not found.",
            suggestion="Confirm the feature_id refers to an uploaded disturbance feature.",
        ) from e

    if "fdist" not in polygons or polygons["fdist"].isna().any():
        raise ProcessingError(
            code="INVALID_FDIST_CODES",
            message=f"Disturbance feature {feature_id} has polygons with no fdist code.",
            suggestion="Give every polygon an integer fdist property.",
        )
    return polygons


def _rasterize_fdist(polygons: gpd.GeoDataFrame, like: DataArray) -> np.ndarray:
    """FDist codes of ``polygons`` burned onto ``like``'s grid.

    A cell takes the code of the polygon that covers its center. Where polygons
    overlap, the more recent disturbance wins (the lower last digit, D_TIME);
    if they are equally recent, the later row wins. Every other cell is 0.
    """
    polygons = polygons.to_crs(like.rio.crs)
    polygons = polygons[~(polygons.geometry.isna() | polygons.geometry.is_empty)]
    if polygons.empty:
        return np.zeros(like.shape, dtype=np.int32)

    # Burn the oldest first so the most recent ends up on top. The sort is
    # stable, so equally recent polygons keep their order in the file.
    d_time = polygons["fdist"].astype(int).to_numpy() % 10
    polygons = polygons.iloc[np.argsort(-d_time, kind="stable")]

    return rasterize(
        zip(polygons.geometry, polygons["fdist"].astype(int)),
        out_shape=like.shape,
        transform=like.rio.transform(),
        fill=0,
        dtype="int32",
    )


def _fetch_landfire_raster(
    roi: gpd.GeoDataFrame,
    url: str,
    extent_buffer_cells: int,
    alignment: dict,
    target_grid_doc: dict | None,
    is_categorical: bool,
    band: int = 1,
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
        band: Which band to read (1-indexed). The default, 1, is the only
            band a single-product source has; a multi-band source (a batched
            LFPS download) passes the band its caller wants.

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
            extra_nodata=LANDFIRE_EXTRA_NODATA,
            destination_resolution=alignment.get("resolution")
            if alignment["target"] == "native"
            else None,
            **dest,
        )
        data = data.sel(band=band, drop=True)

        return _consolidate_landfire_nodata(data)


def _fetch_fuel_model(
    roi: gpd.GeoDataFrame,
    product: str,
    version: str,
    progress: Callable[[str, int | None], None],
    extent_buffer_cells: int,
    alignment: dict,
    target_grid_doc: dict | None,
    season: str | None = None,
) -> DataArray:
    """Fetch one fuel model product's raw raster, from a staged COG or on
    demand via LFPS.

    Shared by fetch_fbfm13/40/fccs, which each apply their own
    post-processing (remove_non_burnable, boundary_scatter) on top.

    Args:
        roi: GeoDataFrame defining the region of interest.
        product: "fbfm13", "fbfm40", or "fccs".
        version: LANDFIRE version year.
        progress: Progress callback, only used when this version is
            fetched via LFPS.
        extent_buffer_cells: Result-grid cells of buffer around the ROI.
        alignment: Alignment specification dict.
        target_grid_doc: Loaded grid document used when
            ``alignment["target"] == "grid"``.
        season: LANDFIRE Seasonal Fuels code (fbfm40 only). Forces an LFPS
            fetch regardless of `version`.

    Returns:
        DataArray with dims (y, x), categorical fuel model codes.
    """
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
        return _fetch_landfire_raster(
            roi,
            url,
            extent_buffer_cells,
            alignment,
            target_grid_doc,
            is_categorical=True,
        )


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
    data = _fetch_fuel_model(
        roi,
        "fbfm13",
        version,
        progress,
        extent_buffer_cells,
        alignment,
        target_grid_doc,
    )

    if remove_non_burnable:
        non_burnable_keys = [NB_CODE_MAP[code] for code in remove_non_burnable]
        filtered = _remove_non_burnable_blocks(
            data.values, non_burnable_keys, data.rio.nodata
        )
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
    data = _fetch_fuel_model(
        roi,
        "fbfm40",
        version,
        progress,
        extent_buffer_cells,
        alignment,
        target_grid_doc,
        season,
    )

    if remove_non_burnable:
        non_burnable_keys = [NB_CODE_MAP[code] for code in remove_non_burnable]
        filtered = _remove_non_burnable_blocks(
            data.values, non_burnable_keys, data.rio.nodata
        )
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
    data = _fetch_fuel_model(
        roi,
        "fccs",
        version,
        progress,
        extent_buffer_cells,
        alignment,
        target_grid_doc,
    )

    if remove_bare_ground:
        filtered = _remove_non_burnable_blocks(data.values, [0], data.rio.nodata)
        data = data.copy(data=filtered)

    if boundary_scatter:
        data = _apply_boundary_scatter(data, boundary_scatter, FCCS_BARE_GROUND)

    return _to_dataset({"fccs": data})


def _remove_non_burnable_blocks(
    grid: ndarray, non_burnable_keys: list[int], nodata: float | None = None
) -> ndarray:
    """Replace non-burnable fuel model codes with neighboring burnable codes.

    Uses a 5x5 majority filter to replace each targeted non-burnable cell
    with the most frequent burnable fuel model in its neighborhood. The
    filter is applied iteratively until no targeted codes remain or a pass
    fills nothing; cells with no reachable burnable neighbour keep their code.

    Args:
        grid: 2D array of LANDFIRE (FBFM or FCCS) fuel model codes
        non_burnable_keys: Numeric codes to replace (e.g., [91, 93, 99])
        nodata: Nodata value; never used as a fill value and left untouched.

    Returns:
        Copy of grid with targeted non-burnable codes replaced
    """
    nb_mask = np.isin(grid, non_burnable_keys)
    if not np.any(nb_mask):
        return grid.copy()

    excluded = list(non_burnable_keys)
    if nodata is not None:
        excluded.append(nodata)

    filtered = grid
    remaining = np.count_nonzero(nb_mask)
    while remaining:
        filtered = generic_filter(
            filtered,
            function=_most_frequent,
            size=(5, 5),
            mode="nearest",
            extra_arguments=(excluded,),
        )
        still = np.count_nonzero(np.isin(filtered, non_burnable_keys))
        if still == remaining:
            break
        remaining = still

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


def fetch_disturbed_fuel(
    roi: gpd.GeoDataFrame,
    fuel_model: str,
    version: str,
    disturbance: dict,
    progress: Callable[[str, int | None], None],
    remove_non_burnable: list[str] | None = None,
    remove_bare_ground: bool = False,
    boundary_scatter: dict | None = None,
    extent_buffer_cells: int = 0,
    alignment: dict | None = None,
    target_grid_doc: dict | None = None,
) -> xr.Dataset:
    """Fetch a LANDFIRE fuel model grid updated for disturbance.

    Each disturbed pixel is matched against LANDFIRE's Master_Rulesets and
    takes the matched rule's fuel model; every other pixel keeps its value
    from the `version` release of `fuel_model`.

    Args:
        roi: GeoDataFrame defining the region of interest.
        fuel_model: "fbfm13", "fbfm40", or "fccs" -- the output product.
        version: LANDFIRE version of `fuel_model` whose values undisturbed
            pixels keep.
        disturbance: The plan the API stored on the grid's source. Its
            ``source`` is "landfire" (LANDFIRE's own disturbance) or "grid"
            (a user's grid, with ``grid_id``, ``band`` and, for an FDist band,
            ``include_landfire``) or "feature" (a user's disturbance polygons,
            with ``feature_id`` and ``include_landfire``).
        progress: Progress callback.
        remove_non_burnable: fbfm13/fbfm40 only -- as in fetch_fbfm13/40,
            applied to the updated grid.
        remove_bare_ground: fccs only -- as in fetch_fccs, applied to the
            updated grid.
        boundary_scatter: As in fetch_fbfm13/40/fccs, applied to the updated grid.
        extent_buffer_cells: Result-grid cells of buffer around the ROI.
        alignment: Alignment specification dict. Defaults to
            ``{"target": "domain"}``.
        target_grid_doc: Loaded grid document used when
            ``alignment["target"] == "grid"``.

    Returns:
        Dataset with a single band, named as `fuel_model`'s plain fetch names
        it (e.g. "fbfm" for fbfm40).
    """
    alignment = alignment or {"target": "domain"}
    rules_column, band_key = _DISTURBED_FUEL_MODELS[fuel_model]
    from_grid = disturbance["source"] == "grid"
    from_feature = disturbance["source"] == "feature"
    user_ldist = from_grid and disturbance["band"] == "annual_disturbance"
    user_fdist = from_feature or (from_grid and disturbance["band"] == "fdist")
    user_only = user_fdist and not disturbance["include_landfire"]
    combined = not user_only and disturbance["mode"] != "fdist"

    # The user's disturbance is checked now, before the slow LFPS job.
    if from_grid:
        user_disturbance = load_band(disturbance["grid_id"], disturbance["band"])
        if user_fdist:
            _check_fdist_codes(user_disturbance.values, user_disturbance.rio.nodata)
    elif from_feature:
        polygons = _load_disturbance_polygons(disturbance["feature_id"])
        _check_fdist_codes(polygons["fdist"].to_numpy())

    # One LFPS job for every layer it serves. `previous` joins it only when
    # its version is LFPS-only; a staged one is fetched on its own below.
    layers = {}
    if not user_only:
        layers["fdist"] = disturbance["fdist_version"]
    if combined and not user_ldist:
        layers["annual_disturbance"] = disturbance["ldist_version"]
    for product in ("fvt", "fvc", "fvh", "bps"):
        layers[product] = LANDFIRE_VERSIONS[product]["default"]
    previous_needs_lfps = _needs_lfps(fuel_model, version)
    if previous_needs_lfps:
        layers[fuel_model] = version

    with landfire_lfps.fetch_lfps_multi(
        roi, layers, alignment, target_grid_doc, extent_buffer_cells, progress
    ) as path:
        bands = {
            name: _fetch_landfire_raster(
                roi,
                path,
                extent_buffer_cells,
                alignment,
                target_grid_doc,
                is_categorical=True,
                band=i + 1,
            )
            for i, name in enumerate(layers)
        }

    if previous_needs_lfps:
        previous = bands[fuel_model]
    else:
        progress(f"Fetching LANDFIRE {fuel_model} v{version}...", 75)
        previous = _fetch_fuel_model(
            roi,
            fuel_model,
            version,
            progress,
            extent_buffer_cells,
            alignment,
            target_grid_doc,
        )

    if from_grid:
        # Nearest-neighbor whatever alignment.method says: codes must never blend.
        user_disturbance = user_disturbance.rio.reproject_match(
            previous, resampling=RESAMPLING_METHOD_MAP["nearest"]
        )
        user_codes = user_disturbance.values
        if user_disturbance.rio.nodata is not None:
            user_codes = np.where(
                user_codes == user_disturbance.rio.nodata, 0, user_codes
            )
    elif from_feature:
        user_codes = _rasterize_fdist(polygons, previous)

    progress("Matching disturbance rulesets...", 85)
    if user_only:
        dist, unconverted = build_fdist_raster("fdist", fdist=user_codes)
    else:
        fdist_inputs = {"fdist": bands["fdist"].values}
        if combined:
            fdist_inputs["ldist"] = (
                user_codes if user_ldist else bands["annual_disturbance"].values
            )
            fdist_inputs["ldist_attribute_table"] = _ldist_attributes(
                disturbance["ldist_version"]
            )
        dist, unconverted = build_fdist_raster(disturbance["mode"], **fdist_inputs)
        if user_fdist:
            dist = np.where(user_codes > 0, user_codes, dist)

    zone = _lookup_lf_zones(
        _lf_map_zones(), previous.rio.transform(), previous.shape, previous.rio.crs
    )
    output, updated = update_fuel_models(
        previous.values,
        fuel_model=rules_column,
        dist=dist,
        zone=zone,
        fvt=bands["fvt"].values,
        fvc=bands["fvc"].values,
        fvh=bands["fvh"].values,
        bps=bands["bps"].values,
        rules=_fuel_rulesets(),
    )

    disturbed = int((dist > 0).sum())
    mapped = int(updated.sum())
    if unconverted:
        logger.info(
            "LDist codes with no FDist equivalent, treated as undisturbed: %s",
            unconverted,
        )
    logger.info(
        "%d of %d disturbed pixels were mapped to a new fuel model", mapped, disturbed
    )

    data = previous.copy(data=output)

    if remove_non_burnable:
        non_burnable_keys = [NB_CODE_MAP[code] for code in remove_non_burnable]
        data = data.copy(
            data=_remove_non_burnable_blocks(
                data.values, non_burnable_keys, data.rio.nodata
            )
        )

    if remove_bare_ground:
        data = data.copy(
            data=_remove_non_burnable_blocks(data.values, [0], data.rio.nodata)
        )

    if boundary_scatter:
        protected = FCCS_BARE_GROUND if fuel_model == "fccs" else FBFM_NON_BURNABLE
        data = _apply_boundary_scatter(data, boundary_scatter, protected)

    return _to_dataset({band_key: data})


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
        data = _fetch_landfire_raster(
            roi,
            url,
            extent_buffer_cells,
            alignment,
            target_grid_doc,
            # Aspect is an angle, with -1 on flat cells; interpolating it
            # invents directions (350 and 10 average to 180).
            is_categorical=band == "aspect",
        )
        if band == "aspect":
            data = data.where(data != LANDFIRE_FLAT_ASPECT, 0)
        variables[band] = data

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
