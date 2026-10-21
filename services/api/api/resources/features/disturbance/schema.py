"""
api/v2/resources/features/disturbance/schema.py

Schemas for the Disturbance feature type.

The upload payload is a flat GeoJSON FeatureCollection of disturbance polygons.
Every Feature carries the LANDFIRE FDist code of one disturbance on
``properties.fdist``. Where polygons overlap, the more recent disturbance (the
lower last digit, D_TIME) wins when the feature is used; equally recent
polygons are resolved by file order, the later one winning.
"""

from typing import Any

from geojson_pydantic import Feature, FeatureCollection
from geojson_pydantic.geometries import MultiPolygon, Polygon
from pydantic import BaseModel, Field, StrictInt, field_validator, model_validator

from lib.landfire import VALID_FDIST_CODES


class DisturbanceSource(BaseModel):
    """Source tracking for user-uploaded disturbance polygons."""

    product: str = "Upload"
    description: str = "User-uploaded disturbance polygons"
    mapping_year: int


class DisturbanceProperties(BaseModel):
    """Per-feature properties: the FDist code of one disturbance polygon, and
    optional labels describing it.

    The labels are stored as given and are never checked against the code;
    only ``fdist`` drives the fuel model update. Other properties on an
    uploaded Feature are ignored.
    """

    fdist: StrictInt = Field(
        ...,
        description=(
            "LANDFIRE FDist code, 100 * D_TYPE + 10 * D_SEVERITY + D_TIME. "
            "The endpoint description explains in further detail."
        ),
    )
    name: str | None = Field(
        None, max_length=255, description="Name of the disturbance, e.g. a fire name."
    )
    type: str | None = Field(
        None, max_length=255, description="Kind of disturbance, e.g. wildfire."
    )
    year: StrictInt | None = Field(None, description="Year the disturbance happened.")
    severity: str | None = Field(
        None, max_length=255, description="Severity of the disturbance, e.g. high."
    )

    @field_validator("fdist")
    @classmethod
    def fdist_is_a_valid_code(cls, value: int) -> int:
        if value not in VALID_FDIST_CODES:
            raise ValueError(
                f"{value} isn't a valid FDist code. Codes are "
                "100 * D_TYPE + 10 * D_SEVERITY + D_TIME with D_TYPE 1-7, "
                "D_SEVERITY 1-3 and D_TIME 1-3; 0 isn't accepted. The FDist codes are "
                "defined in the Attribute Data Dictionary at https://landfire.gov/fuel/fdist."
            )
        return value


class DisturbanceFeature(Feature[Polygon | MultiPolygon, DisturbanceProperties]):
    """One Feature in the disturbance FeatureCollection.

    ``Polygon`` and ``MultiPolygon`` are both accepted, as for layerset
    uploads. ``properties`` is non-Optional: every polygon needs a code.
    """

    properties: DisturbanceProperties


class DisturbanceCrs(BaseModel):
    """Optional GeoJSON crs block (deprecated by RFC 7946, but widely emitted)."""

    type: str = "name"
    properties: dict[str, Any] = Field(default_factory=dict)


class DisturbanceFeatureCollection(FeatureCollection[DisturbanceFeature]):
    """GeoJSON FeatureCollection of disturbance polygons."""

    name: str | None = None
    crs: DisturbanceCrs | None = None

    @model_validator(mode="after")
    def at_least_one_feature(self) -> "DisturbanceFeatureCollection":
        if not self.features:
            raise ValueError(
                "Disturbance upload must contain at least one Feature. "
                "An empty FeatureCollection has nothing to apply."
            )
        return self


class CreateDisturbanceRequestBody(DisturbanceFeatureCollection):
    """Request body for uploading disturbance polygons.

    The body **is** the GeoJSON FeatureCollection, as for layerset and
    ``POST /domains``, extended with the resource-metadata fields. ``name``
    overrides the optional GeoJSON ``name`` member and doubles as the
    resource name.
    """

    mapping_year: StrictInt = Field(
        ...,
        ge=1900,
        le=2100,
        description=(
            "The year you are mapping. The last digit of each `fdist` code is "
            "counted back from it; see the endpoint description."
        ),
    )
    name: str = Field("", max_length=255)
    description: str = Field("", max_length=2000)
    tags: list[str] = Field(default_factory=list, max_length=50)
