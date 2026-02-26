"""
standgen/modifications.py

Tree inventory modification processing.

Applies condition-based modifications to tree DataFrames. Supports
attribute conditions, expression conditions, and actions with optional
pint unit conversion.
"""

import logging
import operator

import pandas as pd
import pint

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
        if "expression" in cond:
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
