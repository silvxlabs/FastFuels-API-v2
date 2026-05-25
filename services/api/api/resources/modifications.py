"""
api/v2/resources/modifications.py

Base classes and enums for modification schemas used across resources.

This module provides generic modification primitives. Resource-specific
modifications (e.g., grids) should extend these base classes with
constrained attribute names and value types.
"""

import json
from enum import StrEnum

from pydantic import BaseModel, Field, field_validator


def stringify_modification_coordinates(modifications: list[dict]) -> list[dict]:
    """JSON-encode inline-geometry coordinates for Firestore storage.

    Firestore does not support nested arrays natively. Inline-geometry spatial
    conditions (``source == "geometry"``) carry a GeoJSON Polygon/MultiPolygon
    whose ``coordinates`` field is a deeply nested array. This walks each
    modification's conditions and JSON-encodes the coordinates of every
    geometry-source condition in place, mirroring the domains pattern
    (``api.resources.domains.schema._stringify_coordinates``).

    Feature-source conditions (which store only a ``feature_id``) and
    attribute-based conditions are left untouched. Idempotent: already
    stringified coordinates are skipped.

    Args:
        modifications: A list of modification dicts (as produced by
            ``model_dump()``), each shaped ``{"conditions": [...], ...}``.

    Returns:
        The same list, mutated in place, ready for the Firestore write.
    """
    for modification in modifications:
        for condition in modification.get("conditions", []):
            if condition.get("source") == "geometry":
                geometry = condition.get("geometry")
                if isinstance(geometry, dict) and not isinstance(
                    geometry.get("coordinates"), str
                ):
                    geometry["coordinates"] = json.dumps(geometry["coordinates"])
    return modifications


def parse_modification_coordinates(modifications: list[dict]) -> list[dict]:
    """Decode stringified inline-geometry coordinates back to nested lists.

    Inverse of :func:`stringify_modification_coordinates`. Detects when a
    geometry-source condition's ``coordinates`` field is a JSON string (from
    Firestore) and parses it back into the nested list structure expected by
    GeoJSON consumers. Idempotent: already-parsed (list) coordinates pass
    through unchanged.

    Args:
        modifications: A list of modification dicts loaded from Firestore.

    Returns:
        The same list, mutated in place, with geometry coordinates as proper
        nested lists.
    """
    for modification in modifications:
        for condition in modification.get("conditions", []):
            if condition.get("source") == "geometry":
                geometry = condition.get("geometry")
                if isinstance(geometry, dict) and isinstance(
                    geometry.get("coordinates"), str
                ):
                    geometry["coordinates"] = json.loads(geometry["coordinates"])
    return modifications


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
