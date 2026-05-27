"""
3DEP (3D Elevation Program) source handlers.

Fetches elevation data from USGS 3DEP via AWS S3 COGs. Supports 1m, 10m,
and 30m resolutions. Slope and aspect are computed locally from the DEM
using Horn's method via numpy.gradient.

All handlers return xr.Dataset where each variable name is a band name.
"""

import logging
from collections.abc import Callable

import geopandas as gpd
import numpy as np
import rioxarray  # noqa: F401
import xarray as xr
from rioxarray.merge import merge_arrays
from xarray import DataArray

from griddle.handlers.tiles import TileMetadata
from griddle.utils import infer_nodata, to_dataset
from lib.alignment import RESAMPLING_METHOD_MAP, resolve_alignment_destination
from lib.errors import ProcessingError
from lib.raster import RasterConnection, cog_env
from lib.threedep import discover_s1m_tiles, discover_tiles_arc_second

logger = logging.getLogger(__name__)

# Extra DEM cells fetched beyond the user's requested extent_buffer_cells when
# slope/aspect are requested. numpy.gradient falls back to one-sided
# differences at the boundary, so we fetch a couple extra cells, compute
# derivatives, then clip back to the user-requested extent. Internal only.
_DERIVATIVE_GRADIENT_OVERHEAD_CELLS = 3


def fetch_topography(
    roi: gpd.GeoDataFrame,
    resolution: int,
    bands: list[str],
    progress: Callable[[str, int | None], None],
    extent_buffer_cells: int = 0,
    alignment: dict | None = None,
    target_grid_doc: dict | None = None,
) -> tuple[xr.Dataset, TileMetadata]:
    """Fetch 3DEP topographic data for a domain extent.

    Each tile's ``extract_window`` reprojects directly into the alignment
    destination lattice — that is the single reprojection. Slope/aspect
    are then computed via ``numpy.gradient`` on the aligned mosaic at the
    destination cell size. When derivatives are requested the destination
    lattice is grown by ``_DERIVATIVE_GRADIENT_OVERHEAD_CELLS`` cells per
    side so ``numpy.gradient`` has central differences at the user-requested
    boundary; those extra cells are stripped from every band before return.

    Args:
        roi: GeoDataFrame defining the region of interest
        resolution: Source product resolution in meters (1, 10, or 30)
        bands: List of band names ("elevation", "slope", "aspect")
        progress: Progress callback
        extent_buffer_cells: Result-grid cells of buffer around the ROI in the
            returned dataset.
        alignment: Alignment specification dict. Defaults to
            ``{"target": "domain"}`` when omitted.
        target_grid_doc: Loaded grid document used when
            ``alignment["target"] == "grid"``.

    Returns:
        Tuple of (Dataset with named variables, tile metadata dict)

    Raises:
        ProcessingError: If no tiles found or fetch fails
    """
    alignment = alignment or {"target": "domain"}
    needs_derivatives = "slope" in bands or "aspect" in bands
    fetch_buffer = (
        extent_buffer_cells + _DERIVATIVE_GRADIENT_OVERHEAD_CELLS
        if needs_derivatives
        else extent_buffer_cells
    )

    progress(f"Discovering 3DEP {resolution}m tiles...", 10)

    if resolution in (10, 30):
        tile_urls = discover_tiles_arc_second(roi, resolution)
        tile_source = None
        native_crs = "EPSG:4326"
        acquisition_dates = None
    elif resolution == 1:
        tile_urls, tile_source, native_crs, acquisition_dates = _discover_tiles_1m(roi)
    else:
        raise ProcessingError(
            code="INVALID_RESOLUTION",
            message=f"Unsupported 3DEP resolution: {resolution}m",
            suggestion="Supported resolutions: 1, 10, 30",
        )

    if not tile_urls:
        raise ProcessingError(
            code="COVERAGE_ERROR",
            message=(f"No 3DEP {resolution}m tiles found covering the domain extent."),
            suggestion=(
                "The domain may be outside 3DEP coverage. "
                "Try a different resolution or verify the domain location."
            ),
        )

    tile_metadata: TileMetadata = {
        "tiles": tile_urls,
        "tile_source": tile_source,
        "tile_count": len(tile_urls),
        "native_crs": native_crs,
        "acquisition_dates": acquisition_dates,
    }

    progress(f"Fetching {len(tile_urls)} 3DEP tile(s)...", 20)
    dem_da = _fetch_and_mosaic_tiles(
        roi,
        tile_urls,
        resolution,
        progress,
        fetch_buffer,
        alignment,
        target_grid_doc,
    )
    dem_da = dem_da.rio.write_nodata(infer_nodata(dem_da.dtype, dem_da))

    # Validate that we got actual data, not just nodata fill
    _validate_dem_has_data(dem_da, resolution)

    variables: dict[str, DataArray] = {}

    if "elevation" in bands:
        if needs_derivatives:
            variables["elevation"] = _trim_derivative_overhead(
                dem_da, alignment, roi, extent_buffer_cells
            )
        else:
            variables["elevation"] = dem_da

    if needs_derivatives:
        progress("Computing slope and aspect...", 80)
        cell_size = abs(float(dem_da.rio.transform().a))
        slope_da, aspect_da = _compute_slope_aspect(dem_da, cell_size)
        slope_da = _trim_derivative_overhead(
            slope_da, alignment, roi, extent_buffer_cells
        )
        aspect_da = _trim_derivative_overhead(
            aspect_da, alignment, roi, extent_buffer_cells
        )

        if "slope" in bands:
            variables["slope"] = slope_da
        if "aspect" in bands:
            variables["aspect"] = aspect_da

    progress("Building dataset...", 90)
    ds = to_dataset(variables)
    return ds, tile_metadata


def _discover_tiles_1m(
    roi: gpd.GeoDataFrame,
) -> tuple[list[str], str, str, list[str] | None]:
    """Discover 1m tiles via S1M (Seamless 1-Meter) coordinate-based lookup.

    Returns:
        Tuple of (tile_urls, tile_source, native_crs, acquisition_dates)
    """
    s1m_urls, s1m_dates = discover_s1m_tiles(roi)
    if s1m_urls:
        return s1m_urls, "s1m", "EPSG:6350", s1m_dates

    return [], "none", "none", None


def _fetch_and_mosaic_tiles(
    roi: gpd.GeoDataFrame,
    tile_urls: list[str],
    resolution: int,
    progress: Callable[[str, int | None], None],
    extent_buffer_cells: int,
    alignment: dict,
    target_grid_doc: dict | None,
) -> DataArray:
    """Fetch tiles and mosaic them onto the alignment destination lattice.

    Each tile's ``extract_window`` reprojects directly into the destination
    transform/shape resolved from ``alignment`` — all tiles land on the same
    output grid, so the mosaic is a nodata-aware composite of identically
    shaped arrays with no second interpolation pass.

    Args:
        roi: Region of interest
        tile_urls: S3/HTTPS URLs to COG tiles
        resolution: Source product resolution in meters (informational only;
            the actual output cell size comes from ``alignment``)
        progress: Progress callback
        extent_buffer_cells: Output DEM cells of buffer around the ROI in the
            destination CRS, baked into ``destination_transform``/``shape``
            for ``target='domain'`` and ``target='grid'``
        alignment: Alignment specification dict (see ``GridAlignmentSpecification``)
        target_grid_doc: Loaded grid document used as the alignment target
            when ``alignment["target"] == "grid"``. Required in that case.

    Returns:
        DataArray with dims (y, x) in the alignment destination CRS at the
        alignment destination resolution
    """
    n_tiles = len(tile_urls)
    tile_arrays = []
    method_name = alignment.get("method") or "bilinear"

    with cog_env(AWS_NO_SIGN_REQUEST="YES"):
        for i, url in enumerate(tile_urls):
            pct = 20 + int(50 * (i + 1) / n_tiles)
            progress(f"Fetching 3DEP tile {i + 1}/{n_tiles}...", pct)

            raster = RasterConnection(url, connection_type="rioxarray", cache=False)
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
            tile_arrays.append(data.squeeze("band", drop=True))

    if len(tile_arrays) == 1:
        return tile_arrays[0]

    return merge_arrays(tile_arrays)


def _compute_slope_aspect(
    dem_da: DataArray,
    cell_size: float,
) -> tuple[DataArray, DataArray]:
    """Compute slope and aspect from a DEM using Horn's method.

    Uses numpy.gradient which implements a central difference approximation
    (equivalent to a simplified Horn's method).

    Args:
        dem_da: DEM DataArray with dims (y, x)
        cell_size: Cell size in meters

    Returns:
        Tuple of (slope_da, aspect_da) in degrees, same coords/CRS as input
    """
    values = dem_da.values.astype(np.float64)

    # numpy.gradient computes central differences
    dy, dx = np.gradient(values, cell_size)

    # Slope: arctan of the magnitude of the gradient vector
    slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
    slope_deg = np.degrees(slope_rad).astype(np.float32)

    # Aspect: direction of steepest descent, clockwise from north
    # -dx because aspect convention is CW from north, and np.arctan2 is CCW from east
    aspect_rad = np.arctan2(-dx, dy)
    aspect_deg = np.degrees(aspect_rad).astype(np.float32)
    # Normalize to [0, 360)
    aspect_deg = np.where(aspect_deg < 0, aspect_deg + 360, aspect_deg)

    slope_da = dem_da.copy(data=slope_deg)
    slope_da.name = "slope"

    aspect_da = dem_da.copy(data=aspect_deg)
    aspect_da.name = "aspect"

    return slope_da, aspect_da


def _validate_dem_has_data(dem_da: DataArray, resolution: int) -> None:
    """Raise ProcessingError if the DEM is entirely nodata.

    Tiles may exist on S3 but contain no actual elevation data for a given
    area. Detect this early rather than returning a dataset full of nodata.
    """
    nodata = dem_da.rio.nodata
    values = dem_da.values

    if nodata is not None:
        valid_mask = ~np.isnan(values) & (values != nodata)
    else:
        valid_mask = ~np.isnan(values)

    if not valid_mask.any():
        raise ProcessingError(
            code="COVERAGE_ERROR",
            message=(
                f"3DEP {resolution}m tiles were found but contain no valid "
                f"elevation data for this domain."
            ),
            suggestion=(
                "The tiles may be placeholders without survey data for this "
                "area. Try a different resolution (10m or 30m have broader "
                "coverage)."
            ),
        )


def _trim_derivative_overhead(
    da: DataArray,
    alignment: dict,
    roi: gpd.GeoDataFrame,
    extent_buffer_cells: int,
) -> DataArray:
    """Remove the gradient-overhead ring fetched for numpy.gradient.

    For ``target="domain"`` / ``target="grid"`` the destination lattice
    was sized exactly via ``lattice_from_bounds``, so the ring is exactly
    ``_DERIVATIVE_GRADIENT_OVERHEAD_CELLS`` rows/columns per side — we
    index-slice. For ``target="native"`` the lattice depends on where the
    pixel anchor falls relative to the buffered bounds, so the ring count
    can vary per edge; we clip-box back to the same bounds
    ``extract_window`` would use for an elevation-only fetch.
    """
    if alignment["target"] in ("domain", "grid"):
        n = _DERIVATIVE_GRADIENT_OVERHEAD_CELLS
        return da.isel(y=slice(n, -n), x=slice(n, -n))

    cell_size = abs(float(da.rio.transform().a))
    pad = extent_buffer_cells * cell_size
    minx, miny, maxx, maxy = roi.to_crs(da.rio.crs).total_bounds
    return da.rio.clip_box(
        minx=minx - pad,
        miny=miny - pad,
        maxx=maxx + pad,
        maxy=maxy + pad,
    )
