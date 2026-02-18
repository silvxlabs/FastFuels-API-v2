"""
api/v2/resources/grids/modifications.py

Grid-specific modification schemas.

Grids use dot-notation band keys (e.g., "fuel_load.1hr", "fbfm") instead of
generic attributes. This module provides grid-specific condition and action
classes that use "band" terminology.
"""

from pydantic import BaseModel, Field, field_validator

from api.resources.modifications import (
    Modifier,
    Operator,
    SpatialOperator,
    SpatialTarget,
)


class GridModificationCondition(BaseModel):
    """
    Attribute-based condition for grid modifications.

    Uses dot-notation band keys (e.g., "fuel_load.1hr", "fbfm").
    """

    band: str = Field(..., description="The band to check (dot-notation key)")
    operator: Operator = Field(..., description="The comparison operator")
    value: int | float | str | list[int | float | str] = Field(
        ..., description="The value(s) to compare against"
    )


class GridSpatialCondition(BaseModel):
    """
    Spatial condition for grid modifications.

    Selects cells based on their spatial relationship to a GeoJSON geometry.
    """

    operator: SpatialOperator = Field(
        ...,
        description="The spatial relationship to test: 'within', 'outside', or 'intersects'",
    )
    geometry: dict = Field(
        ..., description="GeoJSON geometry (Polygon or MultiPolygon)"
    )
    crs: dict | None = Field(
        default=None,
        description="CRS of geometry in GeoJSON format. Defaults to domain CRS if not specified",
    )
    target: SpatialTarget = Field(
        default=SpatialTarget.centroid,
        description="Which part of the cell to test: 'centroid' or 'cell'",
    )


class GridModificationAction(BaseModel):
    """
    Action to perform when grid modification conditions are met.

    Uses dot-notation band keys (e.g., "fuel_load.1hr", "fbfm").
    """

    band: str = Field(..., description="The band to modify (dot-notation key)")
    modifier: Modifier = Field(..., description="How to modify the value")
    value: int | float | str = Field(
        ..., description="The value to use with the modifier"
    )


class GridModification(BaseModel):
    """
    A grid modification rule with conditions and actions.

    All conditions are ANDed together. When all match, actions are applied.
    Conditions can be band-based (checking values) or spatial (checking location).
    """

    conditions: list[GridModificationCondition | GridSpatialCondition] = Field(
        ..., description="Conditions that must all be true"
    )
    actions: list[GridModificationAction] = Field(
        ..., description="Actions to apply when conditions match"
    )

    @field_validator("conditions", "actions", mode="before")
    @classmethod
    def convert_to_list(cls, value):
        """Convert single condition/action to list for convenience."""
        if not isinstance(value, list):
            return [value]
        return value
