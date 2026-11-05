"""
api/v2/resources/features/building/schema.py

Schema models for Building feature creation and metadata.
"""

from typing import Literal

from pydantic import BaseModel, Field

from api.resources.features.schema import CreateFeatureRequestBase, FeatureType


class OvertureBuildingSource(BaseModel):
    """Source metadata stored on the building feature document.

    Records that this feature was generated from Overture Maps data. ``release``
    is null until the feature is processed, then holds the Overture release
    that was read.
    """

    product: Literal["overture"] = "overture"
    description: Literal["Overture Maps building footprints"] = (
        "Overture Maps building footprints"
    )
    extent_buffer_m: float = Field(0, ge=0, le=100)
    release: str | None = None


class CreateOvertureBuildingFeatureRequest(CreateFeatureRequestBase):
    """Request body for creating a building feature via Overture Maps."""

    type: Literal[FeatureType.building] = FeatureType.building
    extent_buffer_m: float = Field(
        0,
        ge=0,
        le=100,
        description=(
            "Distance in meters to expand the domain extent outward before "
            "clipping buildings. Lets buildings that cross the domain "
            "boundary extend past the edge instead of being cut at it. "
            "Applied in the domain's projected CRS. Default 0 clips exactly "
            "to the domain boundary. Maximum: 100 meters."
        ),
    )
