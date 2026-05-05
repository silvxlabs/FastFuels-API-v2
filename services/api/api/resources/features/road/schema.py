"""
api/v2/resources/features/road/schema.py

Schema models for Road feature creation and metadata.
"""

from typing import Literal

from pydantic import BaseModel

from api.resources.features.schema import CreateFeatureRequestBase, FeatureType


class OsmRoadSource(BaseModel):
    """Source metadata stored on the road feature document.

    Records that this feature was generated from OpenStreetMap data.
    """

    product: Literal["osm"] = "osm"
    description: Literal["OpenStreetMap road network"] = "OpenStreetMap road network"


class CreateOsmRoadFeatureRequest(CreateFeatureRequestBase):
    """Request body for creating a road feature via OpenStreetMap."""

    type: Literal[FeatureType.road] = FeatureType.road
