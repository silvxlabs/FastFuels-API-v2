"""
api/v2/resources/grids/modifications.py

Grid-specific modification schemas.

Grids use dot-notation band keys (e.g., "fuel_load.1hr", "fbfm") instead of
generic attributes. This module provides grid-specific condition and action
classes that use "band" terminology.
"""

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from api.resources.modifications import (
    Modifier,
    Operator,
    SpatialOperator,
)


class GridSpatialTarget(StrEnum):
    """
    Specifies which part of a grid cell is tested against the geometry.

    - centroid: Test only the cell's centroid point against the geometry
    - cell: Test the entire cell bounds against the geometry
    """

    centroid = "centroid"
    cell = "cell"


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


class GridGeometrySpatialCondition(BaseModel):
    """
    Spatial condition that tests cells against an inline GeoJSON geometry.

    Use this variant when the geometry is supplied directly in the request
    (e.g. a hand-drawn polygon). For a persisted geometry hosted as a
    Feature resource, use ``GridFeatureSpatialCondition`` instead.
    """

    source: Literal["geometry"] = Field(
        ...,
        description=(
            "Discriminator selecting this variant. Must be the literal "
            'string `"geometry"`. Use `"feature"` instead to reference '
            "a persisted Feature resource by id."
        ),
    )
    operator: SpatialOperator = Field(
        ...,
        description=(
            "The spatial relationship to test between each cell and the "
            "geometry: `within` (cell is inside the geometry), `outside` "
            "(cell is not inside the geometry), or `intersects` (cell "
            "overlaps the geometry)."
        ),
    )
    geometry: dict = Field(
        ...,
        description=(
            "Inline GeoJSON geometry. Polygon and MultiPolygon are the "
            "common shapes; LineString works in combination with "
            '`target="cell"` since centroid-mode rarely matches a '
            "bare line."
        ),
    )
    crs: dict | None = Field(
        default=None,
        description=(
            "CRS of `geometry`, expressed as a GeoJSON CRS object "
            '(`{"type": "name", "properties": {"name": "EPSG:..."}}`). '
            "Defaults to the domain CRS when null."
        ),
    )
    buffer_m: float | None = Field(
        default=None,
        ge=0,
        description=(
            "Optional buffer distance in meters applied to the geometry "
            "(in the domain's projected CRS) before testing. Use a non-zero "
            "buffer to widen the masked region beyond the literal geometry "
            "(e.g. shoreline tolerance around a water polygon)."
        ),
    )
    target: GridSpatialTarget = Field(
        default=GridSpatialTarget.centroid,
        description=(
            "Which part of each grid cell is tested against the geometry. "
            "`centroid` tests only the cell center point; `cell` tests the "
            "full cell bounds (any overlap matches). Use `cell` to catch "
            "every cell a linestring crosses without needing a buffer."
        ),
    )


class GridFeatureSpatialCondition(BaseModel):
    """
    Spatial condition that tests cells against a persisted Feature resource.

    The referenced Feature must belong to the same domain as the grid being
    modified. The processing service loads the feature's geometry, reprojects
    it into the domain CRS, optionally buffers it, and then evaluates the
    spatial operator against grid cells.
    """

    source: Literal["feature"] = Field(
        ...,
        description=(
            "Discriminator selecting this variant. Must be the literal "
            'string `"feature"`. Use `"geometry"` instead to supply '
            "inline GeoJSON."
        ),
    )
    operator: SpatialOperator = Field(
        ...,
        description=(
            "The spatial relationship to test between each cell and the "
            "feature's geometry: `within`, `outside`, or `intersects`."
        ),
    )
    feature_id: str = Field(
        ...,
        description=(
            "ID of a Feature resource (road, water, or layerset) hosted in "
            "the same domain as the grid. Cross-domain references are "
            "rejected; the feature must be in `completed` status."
        ),
    )
    buffer_m: float | None = Field(
        default=None,
        ge=0,
        description=(
            "Optional buffer distance in meters applied to the feature "
            "geometry (in the domain's projected CRS) before testing. "
            'With `target="centroid"` (the default), a buffer is '
            "typically needed for linestring features such as roads "
            "since a bare linestring rarely passes through a cell "
            'centroid; `target="cell"` catches every cell the line '
            "crosses without a buffer."
        ),
    )
    target: GridSpatialTarget = Field(
        default=GridSpatialTarget.centroid,
        description=(
            "Which part of each grid cell is tested against the geometry. "
            "`centroid` tests only the cell center point; `cell` tests the "
            "full cell bounds (any overlap matches)."
        ),
    )


GridSpatialCondition = Annotated[
    GridGeometrySpatialCondition | GridFeatureSpatialCondition,
    Field(discriminator="source"),
]


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
    A grid modification rule: a list of conditions and a list of actions.

    Conditions can be band-based (checking values) or spatial (checking
    location). All conditions in a rule are **ANDed** — the actions apply only
    to cells that satisfy *every* condition, so adding a condition narrows the
    selection (the intersection).

    There is no OR within a rule. To act on a **union** of selections (e.g.
    roads *or* water bodies), supply multiple rules: each rule is applied
    independently, so adding a rule widens the overall selection. Putting two
    mutually exclusive conditions (a road feature AND a water feature) in one
    rule selects cells that are both at once — usually none.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                # Wipe all surface fuel inside a road Feature (linestring +
                # target=cell catches every cell the road crosses)
                {
                    "conditions": [
                        {
                            "source": "feature",
                            "operator": "intersects",
                            "feature_id": "feat_road_abc",
                            "target": "cell",
                        }
                    ],
                    "actions": [
                        {"band": "fuel_load.1hr", "modifier": "replace", "value": 0},
                        {"band": "fuel_load.10hr", "modifier": "replace", "value": 0},
                        {"band": "fuel_load.100hr", "modifier": "replace", "value": 0},
                    ],
                },
                # Wipe surface fuel inside a water Feature (polygon)
                {
                    "conditions": [
                        {
                            "source": "feature",
                            "operator": "intersects",
                            "feature_id": "feat_water_xyz",
                        }
                    ],
                    "actions": [
                        {"band": "fuel_load.1hr", "modifier": "replace", "value": 0},
                        {"band": "fuel_load.10hr", "modifier": "replace", "value": 0},
                        {"band": "fuel_load.100hr", "modifier": "replace", "value": 0},
                    ],
                },
                # Remove 90% of surface fuel along a 4 m road buffer (multiply 0.1)
                {
                    "conditions": [
                        {
                            "source": "feature",
                            "operator": "intersects",
                            "feature_id": "feat_road_abc",
                            "buffer_m": 4,
                        }
                    ],
                    "actions": [
                        {"band": "fuel_load.1hr", "modifier": "multiply", "value": 0.1},
                        {
                            "band": "fuel_load.10hr",
                            "modifier": "multiply",
                            "value": 0.1,
                        },
                        {
                            "band": "fuel_load.100hr",
                            "modifier": "multiply",
                            "value": 0.1,
                        },
                    ],
                },
                # Inline geometry variant with buffer
                {
                    "conditions": [
                        {
                            "source": "geometry",
                            "operator": "within",
                            "geometry": {
                                "type": "Polygon",
                                "coordinates": [
                                    [
                                        [-120.0, 38.0],
                                        [-119.5, 38.0],
                                        [-119.5, 38.5],
                                        [-120.0, 38.5],
                                        [-120.0, 38.0],
                                    ]
                                ],
                            },
                            "buffer_m": 5,
                        }
                    ],
                    "actions": [
                        {"band": "fuel_load.1hr", "modifier": "replace", "value": 0}
                    ],
                },
            ]
        }
    )

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
