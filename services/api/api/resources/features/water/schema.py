"""
api/v2/resources/features/water/schema.py

Schema models for Water feature creation and metadata.
"""

from typing import Literal

from pydantic import BaseModel

from api.resources.features.schema import CreateFeatureRequestBase, FeatureType


class OsmWaterSource(BaseModel):
    """Source metadata stored on the water feature document.

    Records that this feature was generated from OpenStreetMap data.
    """

    product: Literal["osm"] = "osm"
    description: Literal["OpenStreetMap water features"] = (
        "OpenStreetMap water features"
    )


class CreateOsmWaterFeatureRequest(CreateFeatureRequestBase):
    """Request body for creating a water feature via OpenStreetMap."""

    type: Literal[FeatureType.water] = FeatureType.water
