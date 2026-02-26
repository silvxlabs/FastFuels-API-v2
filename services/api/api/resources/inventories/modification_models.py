"""
api/v2/resources/inventories/modifications.py

Modification schemas for tree inventories.

Provides condition/action models for filtering and modifying tree attributes
(dbh, height, crown_ratio, fia_species_code). Supports optional unit
conversion via pint for conditions and actions.
"""

import ast
from enum import StrEnum
from typing import Literal

import pint
from pydantic import BaseModel, Field, field_validator, model_validator

from api.resources.modifications import Modifier, Operator

ureg = pint.UnitRegistry()


class InventoryAttribute(StrEnum):
    """Attributes available for inventory modifications."""

    dbh = "dbh"
    height = "height"
    crown_ratio = "crown_ratio"
    fia_species_code = "fia_species_code"


# Native units for each attribute (used for pint dimensional compatibility checks)
ATTRIBUTE_UNITS = {
    "dbh": "cm",
    "height": "m",
    "crown_ratio": "dimensionless",
    "fia_species_code": None,
}

# Attributes allowed in expressions (excludes categorical fia_species_code)
EXPRESSION_ALLOWED_NAMES = {"dbh", "height", "crown_ratio"}


class InventoryModificationCondition(BaseModel):
    """Condition that checks a tree attribute against a value.

    Optionally specify a unit (e.g., "in", "ft") to convert the value
    to the attribute's native unit before comparison.
    """

    attribute: InventoryAttribute
    operator: Operator
    value: int | float | str | list[int | float | str] = Field(
        ..., description="The value(s) to compare against"
    )
    unit: str | None = Field(
        default=None,
        description="Optional pint-compatible unit for the value (e.g., 'in', 'ft', 'mm'). "
        "Converted to the attribute's native unit before comparison.",
    )

    @field_validator("operator")
    @classmethod
    def validate_species_operator(cls, v, info):
        """fia_species_code only supports eq/ne operators."""
        attribute = info.data.get("attribute")
        if attribute == InventoryAttribute.fia_species_code and v not in (
            Operator.eq,
            Operator.ne,
        ):
            raise ValueError(
                f"fia_species_code only supports 'eq' and 'ne' operators, got '{v}'"
            )
        return v

    @model_validator(mode="after")
    def validate_list_value_operators(self):
        """List values are only allowed with eq/ne operators."""
        if isinstance(self.value, list) and self.operator not in (
            Operator.eq,
            Operator.ne,
        ):
            raise ValueError(
                f"List values are only supported with 'eq' and 'ne' operators, "
                f"got '{self.operator}'"
            )
        return self

    @model_validator(mode="after")
    def validate_unit_compatibility(self):
        """If unit is provided, verify pint can parse it and it's dimensionally
        compatible with the attribute's native unit."""
        if self.unit is None:
            return self

        native = ATTRIBUTE_UNITS.get(self.attribute)
        if native is None:
            raise ValueError(f"Unit conversion is not supported for '{self.attribute}'")

        try:
            user_unit = ureg.parse_expression(self.unit)
        except pint.UndefinedUnitError:
            raise ValueError(f"Unknown unit: '{self.unit}'")

        if native == "dimensionless":
            if not user_unit.dimensionless:
                raise ValueError(
                    f"Unit '{self.unit}' is not dimensionless, "
                    f"but '{self.attribute}' is dimensionless"
                )
        else:
            native_unit = ureg.parse_expression(native)
            if not user_unit.is_compatible_with(native_unit):
                raise ValueError(
                    f"Unit '{self.unit}' is not compatible with "
                    f"native unit '{native}' for attribute '{self.attribute}'"
                )

        return self


class InventoryExpressionCondition(BaseModel):
    """Boolean expression condition evaluated against tree attributes.

    Expressions use native units (cm, m, 0-1 fraction). No unit field
    is provided — convert values in the expression yourself.

    Example: "dbh < 5 and height < 2"
    """

    expression: str = Field(
        ..., description="Boolean expression using dbh, height, crown_ratio"
    )

    @field_validator("expression")
    @classmethod
    def validate_expression_ast(cls, v):
        """Validate expression only uses allowed names, no function calls,
        and no attribute access."""
        try:
            tree = ast.parse(v, mode="eval")
        except SyntaxError:
            raise ValueError(f"Invalid expression syntax: '{v}'")

        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                if node.id not in EXPRESSION_ALLOWED_NAMES:
                    raise ValueError(
                        f"Expression contains disallowed name '{node.id}'. "
                        f"Allowed: {sorted(EXPRESSION_ALLOWED_NAMES)}"
                    )
            if isinstance(node, ast.Call):
                raise ValueError("Function calls are not allowed in expressions")
            if isinstance(node, ast.Attribute):
                raise ValueError("Attribute access is not allowed in expressions")

        return v


class InventoryModificationAction(BaseModel):
    """Action that modifies a tree attribute value.

    Optionally specify a unit to convert the value to the attribute's
    native unit before applying the modifier.
    """

    attribute: InventoryAttribute
    modifier: Modifier
    value: int | float | str = Field(
        ..., description="The value to use with the modifier"
    )
    unit: str | None = Field(
        default=None,
        description="Optional pint-compatible unit for the value.",
    )

    @model_validator(mode="after")
    def validate_divide_by_zero(self):
        """Divide modifier cannot have a value of zero."""
        if self.modifier == Modifier.divide and self.value == 0:
            raise ValueError("Cannot divide by zero")
        return self

    @model_validator(mode="after")
    def validate_unit_compatibility(self):
        """If unit is provided, verify pint compatibility."""
        if self.unit is None:
            return self

        native = ATTRIBUTE_UNITS.get(self.attribute)
        if native is None:
            raise ValueError(f"Unit conversion is not supported for '{self.attribute}'")

        try:
            user_unit = ureg.parse_expression(self.unit)
        except pint.UndefinedUnitError:
            raise ValueError(f"Unknown unit: '{self.unit}'")

        if native == "dimensionless":
            if not user_unit.dimensionless:
                raise ValueError(
                    f"Unit '{self.unit}' is not dimensionless, "
                    f"but '{self.attribute}' is dimensionless"
                )
        else:
            native_unit = ureg.parse_expression(native)
            if not user_unit.is_compatible_with(native_unit):
                raise ValueError(
                    f"Unit '{self.unit}' is not compatible with "
                    f"native unit '{native}' for attribute '{self.attribute}'"
                )

        return self


class RemoveAction(BaseModel):
    """Action that removes matching trees from the inventory."""

    modifier: Literal["remove"] = "remove"


class InventoryModification(BaseModel):
    """A modification rule: when all conditions match, apply actions.

    If a RemoveAction is present, it must be the only action.
    """

    conditions: list[InventoryModificationCondition | InventoryExpressionCondition] = (
        Field(..., min_length=1)
    )
    actions: list[InventoryModificationAction | RemoveAction] = Field(..., min_length=1)

    @field_validator("conditions", "actions", mode="before")
    @classmethod
    def convert_to_list(cls, value):
        """Convert single condition/action to list for convenience."""
        if not isinstance(value, list):
            return [value]
        return value

    @model_validator(mode="after")
    def validate_remove_is_sole_action(self):
        """RemoveAction must be the only action if present."""
        has_remove = any(isinstance(a, RemoveAction) for a in self.actions)
        if has_remove and len(self.actions) > 1:
            raise ValueError("RemoveAction must be the sole action if present")
        return self
