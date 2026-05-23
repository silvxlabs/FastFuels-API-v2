"""
Grid modification pipeline.

Applies the ``modifications`` field of a grid document to the in-memory
``xr.Dataset`` produced by ``dispatch_handler``. Each modification rule is
``{"conditions": [...], "actions": [...]}``: every condition (attribute,
inline geometry, or feature) is ANDed into a single ``(y, x)`` bool mask, then
each action is applied to its target band under that mask.

Memory: we hold the full result Dataset in memory (matching the rest of the
griddle pipeline) plus at most one bool mask per rule. Bands are mutated in
place via ``arr[mask] = ...``-style writes so we never allocate a second
band-sized array.
"""

import operator
from collections.abc import Callable

import geopandas as gpd
import numpy as np
import pyproj
import rasterio.features
import xarray as xr
from shapely.geometry import mapping, shape

from lib.config import FEATURES_BUCKET, FEATURES_COLLECTION
from lib.domain_utils import buffer_gdf
from lib.errors import ProcessingError
from lib.firestore import DocumentNotFoundError, get_document

OPERATOR_MAP: dict[str, Callable] = {
    "eq": operator.eq,
    "ne": operator.ne,
    "gt": operator.gt,
    "lt": operator.lt,
    "ge": operator.ge,
    "le": operator.le,
}


def apply_modifications(
    ds: xr.Dataset,
    modifications: list[dict],
    domain_id: str,
) -> xr.Dataset:
    """Apply every modification rule to ``ds`` and return it.

    The Dataset is mutated in place — the return value is the same object,
    returned for caller convenience.

    Args:
        ds: Result Dataset from ``dispatch_handler`` (rio-extended).
        modifications: List of GridModification dicts as stored in Firestore.
        domain_id: Grid's parent domain. Used to enforce that every
            feature-reference condition resolves to a Feature in the same
            domain.
    """
    feature_cache: dict[tuple[str, float], object] = {}
    for mod in modifications:
        _apply_single_modification(ds, mod, domain_id, feature_cache)
    return ds


def _apply_single_modification(
    ds: xr.Dataset,
    mod: dict,
    domain_id: str,
    feature_cache: dict[tuple[str, float], object],
) -> None:
    conditions = mod.get("conditions", [])
    actions = mod.get("actions", [])
    if not conditions or not actions:
        return

    mask = _build_condition_mask(ds, conditions, domain_id, feature_cache)
    if not mask.any():
        return

    for action in actions:
        _apply_action(ds, action, mask)


def _build_condition_mask(
    ds: xr.Dataset,
    conditions: list[dict],
    domain_id: str,
    feature_cache: dict[tuple[str, float], object],
) -> np.ndarray:
    height, width = ds.rio.height, ds.rio.width
    mask = np.ones((height, width), dtype=bool)
    for cond in conditions:
        if "source" in cond:
            cond_mask = _evaluate_spatial_condition(ds, cond, domain_id, feature_cache)
        else:
            cond_mask = _evaluate_attribute_condition(ds, cond)
        mask &= cond_mask
    return mask


def _evaluate_attribute_condition(ds: xr.Dataset, cond: dict) -> np.ndarray:
    band_key = cond["band"]
    op_name = cond["operator"]
    value = cond["value"]

    var_name, band_coord_val = _resolve_band(ds, band_key)
    da = ds[var_name]
    if band_coord_val is not None:
        da = da.sel(band=band_coord_val)
    arr = da.values

    if isinstance(value, list):
        if op_name == "eq":
            return np.isin(arr, value)
        if op_name == "ne":
            return ~np.isin(arr, value)
        raise ProcessingError(
            code="INVALID_OPERATOR",
            message=(
                f"Operator '{op_name}' does not support list values "
                f"(band '{band_key}')."
            ),
            suggestion="Use 'eq' or 'ne' with list values.",
        )

    return OPERATOR_MAP[op_name](arr, value)


def _evaluate_spatial_condition(
    ds: xr.Dataset,
    cond: dict,
    domain_id: str,
    feature_cache: dict[tuple[str, float], object],
) -> np.ndarray:
    source = cond["source"]
    op_name = cond["operator"]
    target = cond.get("target") or "centroid"
    buffer_m = float(cond.get("buffer_m") or 0)
    target_crs = ds.rio.crs

    if source == "feature":
        geom = _resolve_feature_geometry(
            domain_id,
            cond["feature_id"],
            buffer_m,
            feature_cache,
            target_crs,
        )
    else:
        geom = _resolve_inline_geometry(cond, buffer_m, target_crs)

    out_shape = (ds.rio.height, ds.rio.width)
    transform = ds.rio.transform()

    # target=cell + op=within is the only combo that needs the strict
    # "cell fully inside the geometry" semantics. Every other combo reduces
    # to a single geometry_mask call (with all_touched / invert toggles).
    if target == "cell" and op_name == "within":
        return _strict_within_mask(geom, out_shape, transform)

    all_touched = target == "cell"
    mask = rasterio.features.geometry_mask(
        [mapping(geom)],
        out_shape=out_shape,
        transform=transform,
        all_touched=all_touched,
        invert=True,  # True where the geometry covers a pixel.
    )
    if op_name == "outside":
        mask = ~mask
    return mask


def _strict_within_mask(geom, out_shape, transform) -> np.ndarray:
    """Cells whose entire footprint is inside ``geom``.

    A cell is fully inside iff its centroid is inside the geometry AND the
    geometry's boundary does not cross the cell. Computed as
    ``centroid_inside & ~boundary_touched`` — two ``geometry_mask`` calls.

    For 1D / 0D geometries (lines, points) no cell can be "fully inside" a
    zero-area shape, so we short-circuit to an all-False mask.
    """
    if geom.is_empty or geom.area == 0:
        return np.zeros(out_shape, dtype=bool)

    centroid_inside = rasterio.features.geometry_mask(
        [mapping(geom)],
        out_shape=out_shape,
        transform=transform,
        all_touched=False,
        invert=True,
    )
    boundary_touched = rasterio.features.geometry_mask(
        [mapping(geom.boundary)],
        out_shape=out_shape,
        transform=transform,
        all_touched=True,
        invert=True,
    )
    return centroid_inside & ~boundary_touched


def _resolve_feature_geometry(
    domain_id: str,
    feature_id: str,
    buffer_m: float,
    cache: dict[tuple[str, float], object],
    target_crs,
) -> object:
    key = (feature_id, buffer_m)
    if key in cache:
        return cache[key]

    try:
        _, snapshot = get_document(FEATURES_COLLECTION, feature_id)
    except DocumentNotFoundError:
        raise ProcessingError(
            code="FEATURE_NOT_FOUND",
            message=f"Feature '{feature_id}' not found.",
            suggestion=("Ensure the feature exists in the same domain as the grid."),
        )
    feature_doc = snapshot.to_dict()

    if feature_doc.get("domain_id") != domain_id:
        raise ProcessingError(
            code="FEATURE_DOMAIN_MISMATCH",
            message=(
                f"Feature '{feature_id}' belongs to domain "
                f"'{feature_doc.get('domain_id')}', not '{domain_id}'."
            ),
            suggestion=(
                "Feature references must point to a feature in the same "
                "domain as the grid."
            ),
        )
    if feature_doc.get("status") != "completed":
        raise ProcessingError(
            code="FEATURE_NOT_READY",
            message=(
                f"Feature '{feature_id}' has status "
                f"'{feature_doc.get('status')}'; expected 'completed'."
            ),
            suggestion="Wait for the feature to finish processing.",
        )

    parquet_path = f"gs://{FEATURES_BUCKET}/{domain_id}/{feature_id}.parquet"
    gdf = gpd.read_parquet(parquet_path)
    if gdf.empty:
        raise ProcessingError(
            code="FEATURE_EMPTY",
            message=f"Feature '{feature_id}' has no geometry rows.",
            suggestion="Re-create the feature so it contains geometries.",
        )

    # Vector reproject is a safety net — etcher writes features in the domain
    # CRS by construction, and the grid is in the domain CRS by default.
    # When both already agree, to_crs is a fast identity transform.
    if pyproj.CRS(gdf.crs) != pyproj.CRS(target_crs):
        gdf = gdf.to_crs(target_crs)

    if buffer_m > 0:
        gdf = buffer_gdf(gdf, buffer_m)

    geom = gdf.geometry.union_all()
    cache[key] = geom
    return geom


def _resolve_inline_geometry(cond: dict, buffer_m: float, target_crs) -> object:
    geom = shape(cond["geometry"])

    crs_field = cond.get("crs")
    if crs_field is not None:
        source_crs = crs_field["properties"]["name"]
        if pyproj.CRS(source_crs) != pyproj.CRS(target_crs):
            geom = gpd.GeoSeries([geom], crs=source_crs).to_crs(target_crs).iloc[0]

    if buffer_m > 0:
        buffered = buffer_gdf(
            gpd.GeoDataFrame(geometry=[geom], crs=target_crs), buffer_m
        )
        geom = buffered.geometry.iloc[0]

    return geom


def _apply_action(ds: xr.Dataset, action: dict, mask: np.ndarray) -> None:
    band_key = action["band"]
    modifier = action["modifier"]
    value = action["value"]

    var_name, band_coord_val = _resolve_band(ds, band_key)
    da = ds[var_name]
    arr = da.values

    if band_coord_val is None:
        target = arr
    else:
        band_idx = list(da.coords["band"].values).index(band_coord_val)
        target = arr[band_idx]

    if modifier == "replace":
        target[mask] = value
    elif modifier == "add":
        target[mask] += value
    elif modifier == "subtract":
        target[mask] -= value
    elif modifier == "multiply":
        target[mask] *= value
    elif modifier == "divide":
        target[mask] /= value
    else:
        raise ProcessingError(
            code="UNKNOWN_MODIFIER",
            message=f"Unknown modifier '{modifier}' for band '{band_key}'.",
            suggestion=(
                "Supported modifiers: replace, add, subtract, multiply, divide."
            ),
        )

    # Grid bands today are physical quantities (loads, depths, moistures,
    # heights, savr) that can't be negative. Mirrors v1 surfer's
    # np.maximum(data, 0). Skipped for `replace` since the user is setting
    # the value explicitly.
    if modifier != "replace":
        target[mask] = np.maximum(target[mask], 0)


def _resolve_band(ds: xr.Dataset, band_key: str) -> tuple[str, str | None]:
    """Resolve a dot-notation band key to a Dataset variable name and
    optional band-coord value.

    Two shapes are supported:

    - Flat: ``ds[band_key]`` is a 2D DataArray (uniform, lookup, fbfm).
    - Var + band coord: ``ds[prefix]`` has dims ``(band, y, x)`` and a
      ``band`` coord containing ``suffix`` (layerset).

    Returns ``(var_name, band_coord_value_or_None)``.
    """
    if band_key in ds.data_vars:
        return band_key, None

    if "." in band_key:
        prefix, suffix = band_key.split(".", 1)
        if prefix in ds.data_vars:
            band_coord = ds[prefix].coords.get("band")
            if band_coord is not None and suffix in band_coord.values.tolist():
                return prefix, suffix

    raise ProcessingError(
        code="UNKNOWN_BAND",
        message=f"Band '{band_key}' not found in the grid.",
        suggestion=(
            "Modification band keys must match a Dataset variable or a "
            "'<var>.<band-coord-value>' pair."
        ),
    )
