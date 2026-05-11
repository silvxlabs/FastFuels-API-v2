"""
Resample handler for Griddle.

Resamples a source grid to a different spatial resolution and/or anchor
using rioxarray, with support for per-variable resampling-method overrides
and the shared ``alignment`` discriminated union.
"""

from collections.abc import Callable

import geopandas as gpd
import rioxarray  # noqa: F401
import xarray as xr

from griddle.errors import ProcessingError
from griddle.storage import load_zarr
from lib.alignment import RESAMPLING_METHOD_MAP, resolve_alignment_destination


def resample_grid(
    source_grid_id: str,
    alignment: dict,
    method_overrides: dict[str, str],
    domain_gdf: gpd.GeoDataFrame,
    target_grid_doc: dict | None,
    band_types: dict[str, str],
    progress: Callable[[str, int | None], None],
) -> xr.Dataset:
    """Resample a grid to a new spatial resolution and/or anchor.

    Iterates over ALL variables in the source Dataset, resamples each
    independently, and returns a Dataset with named 2D (y, x) variables.
    The single ``rio.reproject`` call per variable is the alignment step;
    no second reprojection is layered on top.

    Args:
        source_grid_id: ID of the source grid to resample
        alignment: Alignment specification dict (see ``GridAlignmentSpecification``).
            For ``target="domain"`` and ``target="native"``, ``resolution``
            is required.
        method_overrides: Per-variable method overrides {var_name: method_name}.
            Wins over ``alignment.method``.
        domain_gdf: Domain geometry as a GeoDataFrame (used when
            ``alignment["target"] == "domain"`` to compute the output lattice).
        target_grid_doc: Loaded grid document used when
            ``alignment["target"] == "grid"``.
        band_types: Mapping of variable name to band type string
            ("categorical" or "continuous"). Drives the role-aware default
            when ``alignment.method`` is unset and the band has no override.
        progress: Callback for progress reporting

    Returns:
        Dataset resampled to the alignment destination.
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
    source_native_resolution = abs(float(source_ds.rio.transform().a))
    dest = resolve_alignment_destination(
        alignment, domain_gdf, target_grid_doc, source_native_resolution
    )

    resampled_vars = {}
    for var_name in var_names:
        da = source_ds[var_name]

        # method_overrides win per band, then alignment.method, then role default.
        if var_name in method_overrides:
            method_name = method_overrides[var_name]
        elif alignment.get("method") is not None:
            method_name = alignment["method"]
        else:
            method_name = (
                "nearest" if band_types.get(var_name) == "categorical" else "bilinear"
            )
        resampling = RESAMPLING_METHOD_MAP[method_name]

        if "destination_transform" in dest and "destination_shape" in dest:
            resampled = da.rio.reproject(
                dest["destination_crs"],
                transform=dest["destination_transform"],
                shape=dest["destination_shape"],
                resampling=resampling,
            )
        elif alignment.get("resolution") is not None:
            resampled = da.rio.reproject(
                dest.get("destination_crs", crs),
                resolution=alignment["resolution"],
                resampling=resampling,
            )
        else:
            resampled = da.rio.reproject(
                dest.get("destination_crs", crs), resampling=resampling
            )
        resampled_vars[var_name] = resampled

    progress("Resample complete.", 80)

    ds = xr.Dataset(resampled_vars)
    ds = ds.rio.write_crs(crs)
    ds = ds.rio.write_transform()
    return ds
