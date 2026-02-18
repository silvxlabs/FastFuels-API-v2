"""
api/v2/resources/modifications.py

Base classes and enums for modification schemas used across resources.

This module provides generic modification primitives. Resource-specific
modifications (e.g., grids) should extend these base classes with
constrained attribute names and value types.
"""

from enum import StrEnum

from pydantic import BaseModel, Field, field_validator


class Operator(StrEnum):
    """Comparison operators for attribute-based conditions."""

    eq = "eq"
    ne = "ne"
    gt = "gt"
    lt = "lt"
    ge = "ge"
    le = "le"


class SpatialOperator(StrEnum):
    """
    Spatial relationship operators for geometry-based conditions.

    - within: Select items whose target (centroid or cell) is inside the geometry
    - outside: Select items whose target is outside the geometry (inverse of within)
    - intersects: Select items whose target overlaps with the geometry
    """

    within = "within"
    outside = "outside"
    intersects = "intersects"


class SpatialTarget(StrEnum):
    """
    Specifies which part of the cell/item is tested against the geometry.

    - centroid: Test only the cell's centroid point against the geometry
    - cell: Test the entire cell bounds against the geometry
    """

    centroid = "centroid"
    cell = "cell"


class Modifier(StrEnum):
    """Modifiers for modification actions."""

    multiply = "multiply"
    divide = "divide"
    add = "add"
    subtract = "subtract"
    replace = "replace"


class BaseModificationCondition(BaseModel):
    """
    Base condition for attribute-based modifications.

    Checks if an attribute's value satisfies a comparison operator.
    Resource-specific subclasses should constrain the `attribute` field
    to valid attribute names for that resource.
    """

    attribute: str = Field(..., description="The attribute to check")
    operator: Operator = Field(..., description="The comparison operator")
    value: int | float | str | list[int | float | str] = Field(
        ..., description="The value(s) to compare against"
    )


class BaseSpatialCondition(BaseModel):
    """
    Base condition for spatial/geometry-based modifications.

    Allows modifications to be applied based on the spatial relationship
    between items (cells, trees, etc.) and a GeoJSON geometry. This enables
    targeting specific geographic areas for modifications.

    The geometry and crs fields are defined in resource-specific subclasses
    to allow for appropriate geometry type constraints.
    """

    operator: SpatialOperator = Field(
        ...,
        description="The spatial relationship to test: 'within' (inside geometry), "
        "'outside' (not inside geometry), or 'intersects' (overlaps geometry)",
    )
    target: SpatialTarget = Field(
        default=SpatialTarget.centroid,
        description="Which part of the cell to test against the geometry: "
        "'centroid' (cell center point) or 'cell' (entire cell bounds)",
    )


class BaseModificationAction(BaseModel):
    """
    Base action for modifications.

    Specifies how to modify an attribute's value when conditions are met.
    Resource-specific subclasses should constrain the `attribute` field
    to valid attribute names for that resource.
    """

    attribute: str = Field(..., description="The attribute to modify")
    modifier: Modifier = Field(
        ...,
        description="How to modify the value: multiply, divide, add, subtract, or replace",
    )
    value: int | float | str = Field(
        ..., description="The value to use with the modifier"
    )


class BaseModification(BaseModel):
    """
    A modification rule with conditions and actions.

    When all conditions are met, the specified actions are applied.
    Conditions can be attribute-based (checking values) or spatial
    (checking geographic location).

    Resource-specific subclasses should override the conditions and actions
    fields with appropriately typed lists.
    """

    conditions: list[BaseModificationCondition]
    actions: list[BaseModificationAction]

    @field_validator("conditions", "actions", mode="before")
    @classmethod
    def convert_to_list(cls, value):
        """Convert single condition/action to list for convenience."""
        if not isinstance(value, list):
            return [value]
        return value
