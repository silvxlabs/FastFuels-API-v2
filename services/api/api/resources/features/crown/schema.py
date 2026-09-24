"""
api/v2/resources/features/crown/schema.py

Schema models for Crown feature creation from a CHM.
"""

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from api.resources.crown_segmentation import CrownSegmentationBase
from api.resources.features.schema import CreateFeatureRequestBase, FeatureType


class ChmCrownSegmentation(CrownSegmentationBase):
    """Crown segmentation settings for a crown Feature.

    When set, ``max_height`` must be greater than ``min_height``.
    """

    min_height: float = Field(
        default=2.0,
        ge=0,
        description=(
            "A cell joins a crown only if its CHM height is at least this (m)."
        ),
    )
    max_height: float | None = Field(
        default=120.0,
        description=(
            "A cell joins a crown only if its CHM height is at most this (m). "
            "Set to null for no ceiling."
        ),
    )

    @model_validator(mode="after")
    def validate_max_height_above_min(self):
        if self.max_height is not None and self.max_height <= self.min_height:
            raise ValueError("'max_height' must be greater than 'min_height'.")
        return self


class ChmCrownSource(BaseModel):
    """Source metadata stored on the crown feature document."""

    product: Literal["chm"] = "chm"
    source_inventory_id: str
    source_inventory_checksum: str | None = Field(
        default=None,
        description=(
            "The source inventory's `checksum` at the time this feature was "
            "created from it."
        ),
    )
    source_chm_grid_id: str
    source_chm_grid_checksum: str | None = Field(
        default=None,
        description=(
            "The source CHM grid's `checksum` at the time this feature was "
            "created from it."
        ),
    )
    crown_segmentation: ChmCrownSegmentation


class CreateChmCrownFeatureRequest(CreateFeatureRequestBase):
    """Request body for creating a crown feature from a CHM."""

    type: Literal[FeatureType.crown] = FeatureType.crown
    source_inventory_id: str = Field(
        description=(
            "ID of a completed tree inventory in this domain. Each tree seeds "
            "one crown at its `x`, `y`."
        ),
    )
    source_chm_grid_id: str = Field(
        description=(
            "ID of a completed grid in this domain with a `chm` band in metres, "
            "at 2 m resolution or finer."
        ),
    )
    crown_segmentation: ChmCrownSegmentation = Field(
        default_factory=ChmCrownSegmentation,
        description="Crown segmentation settings. Omit to use the defaults.",
    )
