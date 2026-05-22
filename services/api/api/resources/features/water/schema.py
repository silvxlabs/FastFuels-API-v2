"""
api/v2/resources/features/water/schema.py

Schema models for Water feature creation and metadata.
"""

from typing import Literal

from pydantic import BaseModel, Field

from api.resources.features.schema import CreateFeatureRequestBase, FeatureType


class OsmWaterSource(BaseModel):
    """Source metadata stored on the water feature document.

    Records that this feature was generated from OpenStreetMap data.
    """

    product: Literal["osm"] = "osm"
    description: Literal["OpenStreetMap water features"] = (
        "OpenStreetMap water features"
    )
    extent_buffer_m: float = Field(0, ge=0, le=100)


class CreateOsmWaterFeatureRequest(CreateFeatureRequestBase):
    """Request body for creating a water feature via OpenStreetMap."""

    type: Literal[FeatureType.water] = FeatureType.water
    extent_buffer_m: float = Field(
        0,
        ge=0,
        le=100,
        description=(
            "Distance in meters to expand the domain extent outward before "
            "clipping fetched features. Lets streams and rivers that exit "
            "the domain at the boundary extend slightly past the edge, "
            "providing context for visualization and downstream operations "
            "(fuel breaks, perimeter analysis). Applied in the domain's "
            "projected CRS (reprojected to UTM if the domain CRS is "
            "geographic). Default 0 clips exactly to the domain boundary. "
            "Maximum: 100 meters."
        ),
    )
