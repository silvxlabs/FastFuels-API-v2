"""
standgen/modifications.py

Tree inventory modification processing.

Applies condition-based modifications to tree DataFrames. Supports
attribute conditions, expression conditions, and actions with optional
pint unit conversion.
"""

import ast
import copy
import json
import logging
import operator

import geopandas as gpd
import pandas as pd
import pint
import pyproj
from shapely.geometry import shape

from lib.config import FEATURES_BUCKET, FEATURES_COLLECTION
from lib.domain_utils import buffer_gdf
from lib.errors import ProcessingError
from lib.firestore import DocumentNotFoundError, get_document

logger = logging.getLogger(__name__)

ureg = pint.UnitRegistry()
Q_ = ureg.Quantity

OPERATOR_MAP = {
    "eq": operator.eq,
    "ne": operator.ne,
    "gt": operator.gt,
    "lt": operator.lt,
    "ge": operator.ge,
    "le": operator.le,
}

MODIFIER_MAP = {
    "multiply": operator.mul,
    "divide": operator.truediv,
    "add": operator.add,
    "subtract": operator.sub,
}

NATIVE_UNITS = {
    "dbh": "cm",
    "height": "m",
    "crown_ratio": "dimensionless",
}

CLAMP_RULES = {
    "dbh": (0, None),
    "height": (0, None),
    "crown_ratio": (0, 1),
}


def convert_value(attribute: str, value, unit: str | None):
    """Convert a value to the attribute's native unit if a unit is specified.

    If no unit is provided, returns the value unchanged.
    """
    if unit is None:
        return value

    native = NATIVE_UNITS.get(attribute)
    if native is None:
        return value

    if native == "dimensionless":
        return float(value)

    converted = Q_(value, unit).to(native).magnitude
    return converted


def referenced_columns(modifications: list[dict]) -> set[str]:
    """Return the inventory column names a list of modification dicts references.

    Mirrors the API's ``modification_referenced_columns`` for the Firestore-dict
    shape standgen consumes: attribute conditions/actions contribute their
    ``attribute``; expression conditions contribute every name used in the
    expression. Spatial conditions and remove actions reference no measurement
    column. Used by the handler to fail fast (with an actionable error) before a
    rule touches a column the inventory's Parquet doesn't have.
    """
    columns: set[str] = set()
    for mod in modifications:
        for cond in mod.get("conditions", []):
            if cond.get("source") in ("geometry", "feature"):
                continue
            if "expression" in cond:
                tree = ast.parse(cond["expression"], mode="eval")
                columns.update(
                    node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
                )
            elif "attribute" in cond:
                columns.add(cond["attribute"])
        for action in mod.get("actions", []):
            attribute = action.get("attribute")
            if attribute is not None:
                columns.add(attribute)
    return columns


def apply_modifications(df: pd.DataFrame, modifications: list[dict]) -> pd.DataFrame:
    """Apply a list of modifications to a pandas DataFrame.

    This is the entry point for dask map_partitions.
    """
    for mod in modifications:
        df = apply_single_modification(df, mod)
    return df


def apply_single_modification(df: pd.DataFrame, mod: dict) -> pd.DataFrame:
    """Apply one modification (conditions + actions) to the DataFrame."""
    if df.empty:
        return df

    conditions = mod.get("conditions", [])
    actions = mod.get("actions", [])

    mask = build_condition_mask(df, conditions)

    # Check for remove action
    is_remove = any(a.get("modifier") == "remove" for a in actions)
    if is_remove:
        return df[~mask].reset_index(drop=True)

    # Apply non-remove actions
    for action in actions:
        df = apply_action(df, action, mask)

    return df


def build_condition_mask(df: pd.DataFrame, conditions: list[dict]) -> pd.Series:
    """Build a boolean mask by ANDing all conditions together."""
    mask = pd.Series(True, index=df.index)

    for cond in conditions:
        if cond.get("source") in ("geometry", "feature"):
            cond_mask = evaluate_spatial_condition(df, cond)
        elif "expression" in cond:
            cond_mask = evaluate_expression(df, cond["expression"])
        else:
            cond_mask = evaluate_attribute_condition(df, cond)
        mask = mask & cond_mask

    return mask


def evaluate_attribute_condition(df: pd.DataFrame, condition: dict) -> pd.Series:
    """Evaluate a single attribute condition, with optional unit conversion."""
    attribute = condition["attribute"]
    op_name = condition["operator"]
    value = condition["value"]
    unit = condition.get("unit")

    op_func = OPERATOR_MAP[op_name]
    col = df[attribute]

    # Handle list values for eq/ne (isin / ~isin)
    if isinstance(value, list):
        converted = [convert_value(attribute, v, unit) for v in value]
        if op_name == "eq":
            return col.isin(converted)
        elif op_name == "ne":
            return ~col.isin(converted)

    converted = convert_value(attribute, value, unit)
    return op_func(col, converted)


def evaluate_expression(df: pd.DataFrame, expression: str) -> pd.Series:
    """Evaluate a boolean expression against the DataFrame using pd.eval."""
    return df.eval(expression, engine="numexpr")


def evaluate_spatial_condition(df: pd.DataFrame, cond: dict) -> pd.Series:
    """Evaluate a spatial condition against each tree's (x, y) point.

    Runs per-partition under ``map_partitions``. The geometry was resolved
    once in the handler by ``resolve_spatial_conditions`` and injected under
    ``"_resolved_geometry"`` — already in the domain CRS, which is the CRS of
    the tree ``x``/``y`` columns, so no reprojection happens here.

    Returns a boolean Series indexed by ``df.index`` so it ANDs cleanly with
    the running condition mask.
    """
    geom = cond.get("_resolved_geometry")
    if geom is None:
        raise ProcessingError(
            code="SPATIAL_CONDITION_UNRESOLVED",
            message=(
                "Spatial condition was not resolved before map_partitions "
                "(missing '_resolved_geometry')."
            ),
            suggestion=(
                "Call resolve_spatial_conditions() in the handler before "
                "applying modifications."
            ),
        )

    op_name = cond["operator"]
    points = gpd.GeoSeries(gpd.points_from_xy(df["x"], df["y"]), index=df.index)

    if op_name == "within":
        return points.within(geom)
    if op_name == "outside":
        return ~points.within(geom)
    if op_name == "intersects":
        return points.intersects(geom)

    raise ProcessingError(
        code="INVALID_SPATIAL_OPERATOR",
        message=f"Unknown spatial operator '{op_name}'.",
        suggestion="Use 'within', 'outside', or 'intersects'.",
    )


def apply_action(df: pd.DataFrame, action: dict, mask: pd.Series) -> pd.DataFrame:
    """Apply a single action to masked rows, with optional unit conversion and clamping."""
    attribute = action["attribute"]
    modifier = action["modifier"]
    value = action["value"]
    unit = action.get("unit")

    converted = convert_value(attribute, value, unit)

    if modifier == "replace":
        df.loc[mask, attribute] = converted
    else:
        op_func = MODIFIER_MAP[modifier]
        df.loc[mask, attribute] = op_func(df.loc[mask, attribute], converted)

    # Clamp values for masked rows only
    clamp = CLAMP_RULES.get(attribute)
    if clamp and mask.any():
        lo, hi = clamp
        if lo is not None:
            df.loc[mask, attribute] = df.loc[mask, attribute].clip(lower=lo)
        if hi is not None:
            df.loc[mask, attribute] = df.loc[mask, attribute].clip(upper=hi)

    return df


# Spatial-condition resolution.
#
# These functions run ONCE in the handler (before map_partitions), where
# Firestore/GCS access and the domain CRS are available. Resolving feature
# geometries here — rather than inside the per-partition apply_modifications —
# keeps network I/O off the dask worker path and out of every partition.


def _has_spatial_condition(modifications: list[dict]) -> bool:
    """True if any modification has a spatial (geometry/feature) condition."""
    return any(
        cond.get("source") in ("geometry", "feature")
        for mod in modifications
        for cond in mod.get("conditions", [])
    )


def resolve_spatial_conditions(
    modifications: list[dict],
    domain_id: str,
    target_crs,
) -> list[dict]:
    """Resolve every spatial condition's geometry into the domain CRS.

    Deep-copies ``modifications`` (so we never write a non-serializable shapely
    object back into the Firestore-sourced document) and injects the resolved
    geometry under ``"_resolved_geometry"`` on each geometry/feature condition.
    Feature geometries are cached by ``(feature_id, buffer_m)`` so a feature
    referenced by multiple conditions is read from GCS only once.

    Args:
        modifications: List of InventoryModification dicts from Firestore.
        domain_id: Inventory's parent domain. Feature references must resolve
            to a Feature in this same domain.
        target_crs: Domain CRS — the CRS of the tree x/y columns.

    Returns:
        A deep-copied modifications list with resolved geometries injected.
    """
    resolved = copy.deepcopy(modifications)
    feature_cache: dict[tuple[str, float], object] = {}

    for mod in resolved:
        for cond in mod.get("conditions", []):
            source = cond.get("source")
            if source not in ("geometry", "feature"):
                continue
            buffer_m = float(cond.get("buffer_m") or 0)
            if source == "feature":
                cond["_resolved_geometry"] = _resolve_feature_geometry(
                    domain_id,
                    cond["feature_id"],
                    buffer_m,
                    feature_cache,
                    target_crs,
                )
            else:
                cond["_resolved_geometry"] = _resolve_inline_geometry(
                    cond, buffer_m, target_crs
                )

    return resolved


def _resolve_feature_geometry(
    domain_id: str,
    feature_id: str,
    buffer_m: float,
    cache: dict[tuple[str, float], object],
    target_crs,
) -> object:
    """Load a Feature's geometry from GCS, reproject, buffer, and union.

    Scoped to ``domain_id`` — cross-domain references are rejected. Results are
    cached by ``(feature_id, buffer_m)``.
    """
    key = (feature_id, buffer_m)
    if key in cache:
        return cache[key]

    try:
        _, snapshot = get_document(FEATURES_COLLECTION, feature_id)
    except DocumentNotFoundError:
        raise ProcessingError(
            code="FEATURE_NOT_FOUND",
            message=f"Feature '{feature_id}' not found.",
            suggestion="Ensure the feature exists in the same domain as the inventory.",
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
                "domain as the inventory."
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

    # Reproject is a safety net — features are written in the domain CRS by
    # construction, and trees are in the domain CRS too. When both agree,
    # to_crs is a fast identity transform.
    if pyproj.CRS(gdf.crs) != pyproj.CRS(target_crs):
        gdf = gdf.to_crs(target_crs)

    if buffer_m > 0:
        gdf = buffer_gdf(gdf, buffer_m)

    geom = gdf.geometry.union_all()
    cache[key] = geom
    return geom


def _resolve_inline_geometry(cond: dict, buffer_m: float, target_crs) -> object:
    """Parse an inline GeoJSON geometry, reproject to the domain CRS, buffer."""
    geometry = cond["geometry"]
    # The API stringifies geometry coordinates before the Firestore write
    # (Firestore rejects nested arrays). Decode them back to nested lists.
    # Accept both shapes so the read is forward- and backward-compatible.
    if isinstance(geometry.get("coordinates"), str):
        geometry = {**geometry, "coordinates": json.loads(geometry["coordinates"])}
    geom = shape(geometry)

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
