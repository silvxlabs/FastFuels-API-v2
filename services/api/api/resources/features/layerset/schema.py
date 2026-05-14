"""
api/v2/resources/features/layerset/schema.py

Schemas for the Layerset feature type.
"""

from typing import Any

from pydantic import BaseModel, Field

from api.resources.features.schema import CreateFeatureRequestBase


class LayersetSource(BaseModel):
    """Source tracking for custom uploaded layersets."""

    product: str = "Upload"
    description: str = "User-uploaded layerset"


# --- Custom Hierarchical GeoJSON Models ---


class LayersetMultiPolygon(BaseModel):
    """Nested geometry for fuelbed polygons."""

    type: str = "MultiPolygon"
    coordinates: list[Any] = Field(default_factory=list)


class Fuelbed(BaseModel):
    """Individual fuelbed mapping."""

    number: Any
    settings: dict[str, Any] = Field(
        default_factory=dict,
        description="Generic dictionary to hold strata-specific variables.",
    )
    polygons: LayersetMultiPolygon


class LayersetProperties(BaseModel):
    """Strata properties containing the nested fuelbeds."""

    strata: str
    fuelbeds: list[Fuelbed] = Field(default_factory=list)


class LayersetFeature(BaseModel):
    """Individual feature representing a single stratum."""

    type: str = "Feature"
    properties: LayersetProperties


class LayersetFeatureCollection(BaseModel):
    """Custom FeatureCollection for hierarchical fuelbeds."""

    type: str = "FeatureCollection"
    metadata: dict[str, Any] = Field(default_factory=dict)
    features: list[LayersetFeature] = Field(default_factory=list)


# --- Request Body ---


class CreateLayersetRequestBody(CreateFeatureRequestBase):
    """Request body for uploading a custom hierarchical GeoJSON layerset."""

    geojson: LayersetFeatureCollection = Field(
        ...,
        description="The hierarchical GeoJSON FeatureCollection data.",
    )
