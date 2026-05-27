"""
Layerset rasterization handler.

Loads a flat layerset GeoParquet from GCS into a ``GeoDataFrame`` whose columns
already match ``fastfuels_core.rasterize_layerset``'s input contract (one row
per fuelbed, ``fuel_type`` / ``fuel_loading`` / ``fuel_height`` / etc.),
rasterizes it via ``fastfuels_core.layersets.rasterize_layerset``,
and reprojects the result to the request's alignment destination when one
is asked for.

Output shape (one DataArray per unique ``fuel_type``):
    dims:       (band, y, x)
    band coord: ["loading", "height", "live_fuel_moisture",
                 "dead_fuel_moisture", "heat_of_combustion"]
    dtype:      float32

The Grid resource's ``bands`` field is derived from the output Dataset by
``build_layerset_bands`` and written back to Firestore by ``handle_layerset``
in ``griddle.dispatch``.
"""

from collections.abc import Callable

import geopandas as gpd
import numpy as np
import xarray as xr
from fastfuels_core.layersets import rasterize_layerset

from griddle.utils import infer_nodata, to_dataset
from lib.alignment import RESAMPLING_METHOD_MAP, resolve_alignment_destination
from lib.config import FEATURES_BUCKET
from lib.errors import ProcessingError

# Maps the API-surface OverlapMethod string values to numpy callables.
# Keep keys in sync with api.resources.grids.rasterize.layerset.schema.OverlapMethod.
# Limited to mean/min/max because fastfuels_core.rasterize_layerset raises
# ValueError for anything else.
OVERLAP_METHODS: dict[str, Callable] = {
    "mean": np.mean,
    "max": np.max,
    "min": np.min,
}

# Per-band units reported by fastfuels_core.layersets.rasterize_layerset.
# Keep keys in sync with the published 5-band output coord. Values must be
# in canonical ASCII UDUNITS-2 form (see docs/units.md). The drift guard
# is in tests/handlers/test_layerset.py::TestLayersetBandUnits — matches
# the repo-wide pattern used by uniform/topography/lookup/canopy/voxelize.
_LAYERSET_BAND_UNITS: dict[str, str] = {
    "loading": "kg/m**2",
    "height": "m",
    "live_fuel_moisture": "%",
    "dead_fuel_moisture": "%",
    "heat_of_combustion": "kJ/kg",
}

# Default reprojection method when caller does not override; all layerset
# output bands are continuous physical quantities, so bilinear is the natural
# default (matches the resample handler's continuous-band default).
_DEFAULT_RESAMPLING_METHOD = "bilinear"

# Default rasterization resolution when the request omits alignment.resolution.
# Matches fastfuels_core.rasterize_layerset's own default.
_DEFAULT_NATIVE_RESOLUTION_M = 2.0


def fetch_layerset(
    domain_gdf: gpd.GeoDataFrame,
    layerset_id: str,
    domain_id: str,
    overlap_method: str,
    progress: Callable[[str, int | None], None],
    extent_buffer_cells: int = 0,
    alignment: dict | None = None,
    target_grid_doc: dict | None = None,
) -> xr.Dataset:
    """Load a layerset GeoParquet from GCS and rasterize it.

    Args:
        domain_gdf: Domain geometry used to resolve the alignment destination.
        layerset_id: Feature ID of the layerset (stored at
            ``gs://{FEATURES_BUCKET}/{domain_id}/{layerset_id}.parquet``).
        domain_id: Domain the layerset belongs to. Used to build the GCS path.
        overlap_method: One of ``OVERLAP_METHODS``' keys.
        progress: Progress callback ``(message, percent)``.
        extent_buffer_cells: Result-grid cells of buffer around the domain
            extent. See ``resolve_alignment_destination``.
        alignment: Alignment specification dict. Defaults to
            ``{"target": "domain"}``.
        target_grid_doc: Loaded grid document for ``alignment["target"] == "grid"``.
            Required in that case.

    Returns:
        ``xr.Dataset`` with one ``float32`` variable per unique ``fuel_type``
        in the input layerset, each with dims ``(band, y, x)``. Reprojected
        to the alignment destination when the request asks for one.

    Raises:
        ProcessingError: ``UNKNOWN_OVERLAP_METHOD`` for an unrecognized
            overlap method; ``LAYERSET_NOT_FOUND`` when the Parquet is
            missing from GCS.
    """
    if overlap_method not in OVERLAP_METHODS:
        raise ProcessingError(
            code="UNKNOWN_OVERLAP_METHOD",
            message=f"Unknown overlap_method: {overlap_method!r}",
            suggestion=f"Use one of: {sorted(OVERLAP_METHODS)}.",
        )
    overlap_callable = OVERLAP_METHODS[overlap_method]

    progress("Loading layerset Parquet...", 10)
    gcs_path = f"gs://{FEATURES_BUCKET}/{domain_id}/{layerset_id}.parquet"
    try:
        # The Parquet is a flat layerset where each row carries one fuelbed's
        # input columns for ``fastfuels_core.rasterize_layerset``. The CRS is
        # stored on the GeoParquet metadata and round-trips through
        # ``gpd.read_parquet`` (the team's payloads use EPSG:32612).
        gdf = gpd.read_parquet(gcs_path)
    except FileNotFoundError as e:
        raise ProcessingError(
            code="LAYERSET_NOT_FOUND",
            message=f"Layerset {layerset_id} not found at {gcs_path}.",
            suggestion=(
                "Confirm the layerset_id refers to an existing layerset "
                "uploaded via POST /domains/{domain_id}/features/layerset "
                "and that it belongs to this domain."
            ),
        ) from e

    progress("Rasterizing layerset...", 40)
    alignment = alignment or {"target": "domain"}
    resolution = alignment.get("resolution") or _DEFAULT_NATIVE_RESOLUTION_M
    ds = rasterize_layerset(
        gdf,
        resolution=resolution,
        overlap_method=overlap_callable,
    )

    if _needs_post_reproject(alignment):
        progress("Aligning rasterized layerset...", 75)
        dest = resolve_alignment_destination(
            alignment=alignment,
            domain_gdf=domain_gdf,
            target_grid_doc=target_grid_doc,
            source_native_resolution=resolution,
            extent_buffer_cells=extent_buffer_cells,
        )
        ds = _reproject_dataset(ds, dest, alignment.get("method"))

    variables = {
        name: da.rio.write_nodata(infer_nodata(da.dtype, da))
        for name, da in ds.data_vars.items()
    }
    progress("Layerset rasterized.", 100)
    return to_dataset(variables)


def _needs_post_reproject(alignment: dict) -> bool:
    """True when alignment asks for anything other than the native rasterization.

    ``target="native"`` with no ``resolution`` override means "use whatever
    ``rasterize_layerset`` produced" — skip reprojection. All other cases
    (domain/grid alignment, or native with a custom resolution) need a
    ``rio.reproject`` pass to honour the request.
    """
    if alignment["target"] != "native":
        return True
    return alignment.get("resolution") is not None


def _reproject_dataset(ds: xr.Dataset, dest: dict, method: str | None) -> xr.Dataset:
    """Reproject each variable of ``ds`` to the alignment destination.

    Mirrors ``griddle.handlers.resample.resample_grid``'s per-variable loop
    so the alignment semantics stay consistent across handlers. All layerset
    output bands are continuous physical quantities, so the default
    resampling method is ``bilinear``; the caller can override via
    ``alignment.method``.
    """
    resampling = RESAMPLING_METHOD_MAP[method or _DEFAULT_RESAMPLING_METHOD]
    reprojected: dict[str, xr.DataArray] = {}
    for var_name in ds.data_vars:
        da = ds[var_name]
        if "destination_transform" in dest and "destination_shape" in dest:
            reprojected[var_name] = da.rio.reproject(
                dest["destination_crs"],
                transform=dest["destination_transform"],
                shape=dest["destination_shape"],
                resampling=resampling,
            )
        elif "destination_crs" in dest:
            reprojected[var_name] = da.rio.reproject(
                dest["destination_crs"], resampling=resampling
            )
        else:
            reprojected[var_name] = da

    out = xr.Dataset(reprojected)
    if "destination_crs" in dest:
        out = out.rio.write_crs(dest["destination_crs"])
        out = out.rio.write_transform()
    return out


def build_layerset_bands(ds: xr.Dataset) -> list[dict]:
    """Derive the Grid's ``bands`` list from a rasterized layerset Dataset.

    Returns one entry per (variable, band) pair, in the order produced by
    ``xr.Dataset.data_vars`` × the variable's ``band`` coord. Each entry is
    a plain dict matching ``api.resources.grids.schema.Band``'s serialized
    shape so the worker can write it directly to Firestore without crossing
    the API↔griddle package boundary.

    Each layerset band carries continuous physical units (kg/m**2, m, %, kJ/kg).
    """
    bands: list[dict] = []
    idx = 0
    for var_name in ds.data_vars:
        band_coord = ds[var_name].coords.get("band")
        if band_coord is None:
            continue
        for band_name in band_coord.values:
            bands.append(
                {
                    "key": f"{var_name}.{band_name}",
                    "type": "continuous",
                    "unit": _LAYERSET_BAND_UNITS.get(str(band_name)),
                    "index": idx,
                }
            )
            idx += 1
    return bands
