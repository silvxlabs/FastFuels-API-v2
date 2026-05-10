"""
Resample handler for Griddle.

Resamples a source grid to a different spatial resolution using rioxarray,
with support for per-variable resampling method overrides.
"""

from collections.abc import Callable

import rioxarray  # noqa: F401
import xarray as xr
from rasterio.enums import Resampling

from griddle.errors import ProcessingError
from griddle.storage import load_zarr

RESAMPLING_METHODS = {
    "nearest": Resampling.nearest,
    "bilinear": Resampling.bilinear,
    "cubic": Resampling.cubic,
    "cubic_spline": Resampling.cubic_spline,
    "lanczos": Resampling.lanczos,
    "average": Resampling.average,
    "mode": Resampling.mode,
    "max": Resampling.max,
    "min": Resampling.min,
    "median": Resampling.med,
    "first_quartile": Resampling.q1,
    "third_quartile": Resampling.q3,
    "sum": Resampling.sum,
    "root_mean_square": Resampling.rms,
}


def resample_grid(
    source_grid_id: str,
    target_resolution: float,
    method: str,
    method_overrides: dict[str, str],
    progress: Callable[[str, int | None], None],
) -> xr.Dataset:
    """Resample a grid to a new spatial resolution.

    Iterates over ALL variables in the source Dataset, resamples each
    independently, and returns a Dataset with named 2D (y, x) variables.

    Args:
        source_grid_id: ID of the source grid to resample
        target_resolution: Target pixel size in CRS units (meters)
        method: Default resampling method name
        method_overrides: Per-variable method overrides {var_name: method_name}
        progress: Callback for progress reporting

    Returns:
        Dataset resampled to target resolution
    """
    progress("Loading source grid...", 20)

    try:
        source_ds = load_zarr(source_grid_id)
    except Exception as e:
        raise ProcessingError(
            code="SOURCE_GRID_NOT_FOUND",
            message=f"Could not load source grid {source_grid_id}: {e}",
            suggestion="Ensure the source grid exists and has been processed.",
        )

    var_names = list(source_ds.data_vars)
    if not var_names:
        raise ProcessingError(
            code="SOURCE_GRID_READ_ERROR",
            message="Source grid dataset has no data variables.",
            suggestion="Ensure the source grid contains valid data.",
        )

    progress("Resampling grid...", 40)

    crs = source_ds.rio.crs
    resampled_vars = {}
    for var_name in var_names:
        da = source_ds[var_name]
        method_name = method_overrides.get(var_name, method)
        resampling = RESAMPLING_METHODS[method_name]
        resampled = da.rio.reproject(
            crs,
            resolution=target_resolution,
            resampling=resampling,
        )
        # rio.reproject preserves the source's zarr encoding (including
        # chunks=(src_ny, src_nx)). Save_zarr re-chunks the dask graph to
        # the target chunk shape, which then mismatches the stale
        # encoding and trips xarray's safe-chunks alignment check on
        # write. Clear the encoding so save_zarr can re-encode cleanly.
        resampled.encoding.pop("chunks", None)
        resampled_vars[var_name] = resampled

    progress("Resample complete.", 80)

    ds = xr.Dataset(resampled_vars)
    ds = ds.rio.write_crs(crs)
    ds = ds.rio.write_transform()
    return ds
