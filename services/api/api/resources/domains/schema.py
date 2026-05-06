"""
api/v2/resources/domains/schema.py
"""

# Core imports
import json
from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from geojson_pydantic import FeatureCollection

# External imports
from pydantic import (
    BaseModel,
    Field,
    SerializationInfo,
    SerializerFunctionWrapHandler,
    model_serializer,
    model_validator,
)

# Internal imports
from api.schema import PaginatedResponse


class GeoJsonCRSProperties(BaseModel):
    name: str = "EPSG:4326"


class GeoJsonCRS(BaseModel):
    type: Literal["name"] = "name"
    properties: GeoJsonCRSProperties


def default_crs_factory():
    return GeoJsonCRS(properties=GeoJsonCRSProperties(name="EPSG:4326"))


class DomainStyle(BaseModel):
    """Optional visual style for rendering a domain on a map.

    All fields are optional. On PATCH only the provided fields are merged
    into the stored style; unspecified fields preserve their current values.
    """

    stroke_color: str | None = Field(
        None,
        max_length=64,
        description="Stroke color in any renderer-supported format.",
    )
    stroke_opacity: float | None = Field(
        None, ge=0, le=1, description="Stroke opacity in [0, 1]."
    )
    stroke_width: float | None = Field(
        None, ge=0, description="Stroke width in pixels (non-negative)."
    )
    fill_color: str | None = Field(
        None,
        max_length=64,
        description="Fill color in any renderer-supported format.",
    )
    fill_opacity: float | None = Field(
        None, ge=0, le=1, description="Fill opacity in [0, 1]."
    )


def _stringify_coordinates(data: dict) -> dict:
    """Convert nested coordinate arrays to JSON strings for Firestore storage.

    Firestore does not support nested arrays natively. This function serializes
    GeoJSON coordinate arrays (which are deeply nested for Polygon/MultiPolygon
    geometries) into JSON strings that Firestore can store.

    Args:
        data: A dictionary representation of a GeoJSON FeatureCollection.

    Returns:
        The same dictionary with geometry coordinates converted to JSON strings.
    """
    if "features" in data:
        for feature in data["features"]:
            if "geometry" in feature and "coordinates" in feature["geometry"]:
                feature["geometry"]["coordinates"] = json.dumps(
                    feature["geometry"]["coordinates"]
                )
    return data


def _parse_coordinates(data: dict) -> dict:
    """Convert JSON string coordinates back to nested arrays.

    This is the inverse of _stringify_coordinates. It detects when coordinate
    data is stored as a JSON string (from Firestore) and parses it back into
    the nested list structure expected by GeoJSON.

    Args:
        data: A dictionary representation of a GeoJSON FeatureCollection,
            potentially with stringified coordinates from Firestore.

    Returns:
        The same dictionary with geometry coordinates as proper nested lists.
    """
    if "features" in data:
        for feature in data["features"]:
            coords = feature.get("geometry", {}).get("coordinates")
            if isinstance(coords, str):
                feature["geometry"]["coordinates"] = json.loads(coords)
    return data


class CreateDomainRequestBody(FeatureCollection):
    name: str = Field("", max_length=255, description="The name of the domain.")
    description: str = Field(
        "", max_length=2000, description="A description of the domain."
    )
    crs: GeoJsonCRS = Field(
        default_factory=default_crs_factory,
        description="The GeoJSON specification formatted coordinate reference system (CRS) of the domain.",
    )
    tags: list[str] | None = Field(
        [], max_length=50, description="A list of tags associated with the domain."
    )
    pad_to_resolution: float | None = Field(
        None,
        gt=0,
        description=(
            "Optional resolution in meters to snap the domain bounding box to. "
            "When set, the bounding box (the 'domain' feature) is snapped outward "
            "to the nearest multiple of this value. Grids whose resolutions divide "
            "evenly into this value will produce identical, aligned footprints on "
            "this domain."
        ),
    )
    style: DomainStyle | None = Field(
        None,
        description="Optional visual style for rendering the domain on a map.",
    )


class Domain(CreateDomainRequestBody):
    """
    Represents a domain resource.
    """

    id: str = Field(None, description="A unique identifier for the domain.")
    created_on: datetime | None = Field(
        None, description="The date and time the domain was created."
    )
    modified_on: datetime | None = Field(
        None, description="The date and time the domain was last modified."
    )

    @model_validator(mode="before")
    @classmethod
    def _deserialize_from_firestore(cls, data: Any) -> Any:
        """Automatically deserialize stringified coordinates from Firestore.

        This validator runs before Pydantic validation and detects when coordinate
        data has been stored as JSON strings in Firestore (due to Firestore's
        limitation with nested arrays). It automatically parses these strings back
        into proper nested list structures so that geojson-pydantic validation
        succeeds.

        The detection is automatic: if coordinates are strings, they get parsed;
        if they're already lists (e.g., from an API request), they pass through
        unchanged.

        Example:
            # Data from Firestore with stringified coordinates
            firestore_data = {
                "features": [{
                    "geometry": {
                        "coordinates": "[[[...], [...], ...]]"  # string
                    }
                }]
            }
            # Automatically parsed when creating Domain instance
            domain = Domain(**firestore_data)
            # domain.features[0].geometry.coordinates is now a nested list
        """
        if isinstance(data, dict) and "features" in data:
            return _parse_coordinates(data)
        return data

    @model_serializer(mode="wrap")
    def _serialize_for_context(
        self, handler: SerializerFunctionWrapHandler, info: SerializationInfo
    ) -> dict:
        """Serialize with optional Firestore-compatible coordinate stringification.

        This serializer supports two output formats controlled by the serialization
        context:

        1. Default (API response): Coordinates remain as nested lists, which is
           valid GeoJSON that clients expect.

        2. Firestore storage: When context={'for_firestore': True} is passed,
           coordinates are converted to JSON strings to work around Firestore's
           limitation with nested arrays.

        Example:
            domain = Domain(...)

            # For API response (standard GeoJSON)
            api_data = domain.model_dump()

            # For Firestore storage (stringified coordinates)
            firestore_data = domain.model_dump(context={'for_firestore': True})
            await doc_ref.set(firestore_data)
        """
        data = handler(self)
        if info.context and info.context.get("for_firestore"):
            return _stringify_coordinates(data)
        return data


class UpdateDomainRequestBody(BaseModel):
    """Request body for updating a domain's metadata.

    All fields are optional. Only provided fields will be updated.
    Geometry (features) and CRS cannot be modified after creation.
    """

    name: str | None = Field(
        None, max_length=255, description="The name of the domain."
    )
    description: str | None = Field(
        None, max_length=2000, description="A description of the domain."
    )
    tags: list[str] | None = Field(
        None, max_length=50, description="A list of tags associated with the domain."
    )
    style: DomainStyle | None = Field(
        None,
        description=(
            "Update visual style fields. Only provided sub-fields are merged "
            "into the existing style; unspecified sub-fields preserve their "
            "current values."
        ),
    )


class DomainSortField(StrEnum):
    """Fields available for sorting domain list results."""

    created_on = "created_on"
    modified_on = "modified_on"
    name = "name"


class DomainSortOrder(StrEnum):
    """Sort order for domain list results."""

    ascending = "ascending"
    descending = "descending"


class ListDomainsResponse(PaginatedResponse):
    """Paginated response for listing domain resources."""

    domains: list[Domain] = Field(
        ..., description="The list of domain resources for the current page."
    )
