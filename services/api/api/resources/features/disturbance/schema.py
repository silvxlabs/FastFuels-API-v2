"""
api/v2/resources/features/disturbance/schema.py

Schemas for the Disturbance feature type.

A disturbance feature is uploaded as GeoJSON like road, water and building
features (see ``features/geojson.py``). Every polygon carries the LANDFIRE
FDist code of one disturbance on ``properties.fdist``. Where polygons overlap,
the more recent disturbance (the lower last digit, D_TIME) wins when the
feature is used; equally recent polygons are resolved by file order, the later
one winning.
"""

from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    field_serializer,
    field_validator,
)

from api.resources.features.geojson import (
    CreateGeojsonFeatureRequest,
    GeojsonPolygonFeature,
)
from lib.landfire.config import VALID_FDIST_CODES


class DisturbanceProperties(BaseModel):
    """Per-feature properties: the FDist code of one disturbance polygon, and
    optional labels describing it.

    The labels are stored as given and are never checked against the code;
    only ``fdist`` drives the fuel model update. Other properties are stored as
    columns, as for any GeoJSON upload.
    """

    model_config = ConfigDict(extra="allow")

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
    year: StrictInt | None = Field(
        None, ge=1900, le=2100, description="Year the disturbance happened."
    )
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


class DisturbanceFeature(GeojsonPolygonFeature):
    """One uploaded disturbance polygon. ``properties`` is required: every
    polygon needs a code."""

    properties: DisturbanceProperties

    @field_serializer("properties")
    def properties_as_given(self, value: DisturbanceProperties) -> dict[str, Any]:
        # Store only the properties the upload set, as for other GeoJSON uploads.
        return value.model_dump(exclude_unset=True)


class CreateDisturbanceFeatureRequest(CreateGeojsonFeatureRequest):
    """Request body for uploading a disturbance feature."""

    features: list[DisturbanceFeature]
