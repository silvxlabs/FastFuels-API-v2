"""Fosberg 1-hour dead fuel moisture content handler for Griddle.

Looks up 1-hour dead fuel moisture from a topography grid (slope + aspect,
degrees) and a leaflux surface irradiance grid, using the Fosberg & Deeming
(1971) model in ``fastfuels_core``. Per-cell shading is ``1 - surface_relative``.
The output is a single 2-D ``fuel_moisture.dead.1hr`` band on the topography
grid's lattice.

The request is validated by the API (ownership, band presence, source grid
status). This handler only loads the referenced data and computes the lookup.
"""

import math
from collections.abc import Callable

import numpy as np
import rioxarray  # noqa: F401 — registers the `.rio` accessor
import xarray as xr
from fastfuels_core.fuel_moisture.fosberg import calculate_1hr_fuel_moisture
from rasterio.enums import Resampling

from griddle.storage import load_zarr
from lib.crs import crs_equal
from lib.errors import ProcessingError

SLOPE_KEY = "slope"
ASPECT_KEY = "aspect"
SURFACE_IRRADIANCE_KEY = "irradiance.surface.relative"
OUTPUT_KEY = "fuel_moisture.dead.1hr"

# The core's `elevation` is a relative-to-weather-station category index, not an
# actual elevation: 0=below, 1=near (no correction), 2=above.
_ELEVATION_INDEX = {"below": 0, "near": 1, "above": 2}

# Fosberg splits slope into two correction classes at a 30% grade (the tables'
# "31%" row). Our slope band is in degrees; because tan() is monotonic the same
# split is atan(0.30) ~= 16.7 degrees, so we compare against that breakpoint
# rather than tan-converting every cell.
_SLOPE_CLASS_BREAK_DEG = math.degrees(math.atan(0.30))
_GENTLE_SLOPE = 0.0
_STEEP_SLOPE = 31.0


def fosberg_grid(
    grid: dict,
    source: dict,
    progress: Callable[[str, int | None], None],
) -> xr.Dataset:
    """Compute a Fosberg 1-hr dead fuel moisture grid from two source grids."""
    progress("Loading source grids...", 20)
    topo = _load(source["source_topography_grid_id"])
    irr = _load(source["source_irradiance_grid_id"])

    slope = topo[SLOPE_KEY]
    aspect = topo[ASPECT_KEY]
    surface = irr[SURFACE_IRRADIANCE_KEY]
    if "z" in surface.dims:  # a combined canopy grid stores the surface at z=0
        surface = surface.isel(z=0, drop=True)
    if not _same_grid(surface, slope):
        surface = surface.rio.reproject_match(slope, resampling=Resampling.bilinear)

    progress("Computing 1-hr dead fuel moisture...", 50)
    valid = _defined(slope) & _defined(aspect) & _defined(surface)
    fmc = calculate_1hr_fuel_moisture(
        dry_bulb_temp=source["dry_bulb_temp"],
        relative_humidity=source["relative_humidity"],
        aspect=np.where(valid, aspect.values, 0.0),
        slope=np.where(
            slope.values > _SLOPE_CLASS_BREAK_DEG, _STEEP_SLOPE, _GENTLE_SLOPE
        ),
        time=source["time"],
        month=source["month"],
        elevation=_ELEVATION_INDEX[source["elevation"]],
        shading=np.where(valid, 1.0 - surface.values, 0.0),
    )
    fmc = np.where(valid, fmc, np.nan).astype(np.float32)

    band = xr.DataArray(fmc, dims=slope.dims, coords=slope.coords).rio.write_nodata(
        np.nan
    )
    out = xr.Dataset({OUTPUT_KEY: band})
    return out.rio.write_crs(slope.rio.crs).rio.write_transform(slope.rio.transform())


def _load(grid_id: str) -> xr.Dataset:
    """Load a source grid's zarr, surfacing an unreadable store as a terminal error."""
    try:
        return load_zarr(grid_id)
    except Exception as e:
        raise ProcessingError(
            code="FOSBERG_SOURCE_UNAVAILABLE",
            message=f"Could not load Fosberg source grid '{grid_id}': {e}",
            suggestion="Ensure the source grids exist and have been processed.",
        )


def _same_grid(a: xr.DataArray, b: xr.DataArray) -> bool:
    """True when two arrays already share a CRS, shape, and transform."""
    return (
        crs_equal(str(a.rio.crs), str(b.rio.crs))
        and a.shape == b.shape
        and a.rio.transform() == b.rio.transform()
    )


def _defined(da: xr.DataArray) -> np.ndarray:
    """Boolean mask of cells holding real data (not nodata)."""
    arr = da.values
    if np.issubdtype(arr.dtype, np.floating):
        # NaN/inf are never real data; also honor a non-NaN numeric sentinel.
        # (rio.nodata can be a numpy float, for which isinstance(..., float)
        # is False, so test it with np.isnan rather than a Python-type check.)
        mask = np.isfinite(arr)
        nodata = da.rio.nodata
        if nodata is not None and np.isfinite(nodata):
            mask &= arr != nodata
        return mask
    nodata = da.rio.nodata
    if nodata is None:
        return np.ones(arr.shape, dtype=bool)
    return arr != nodata
