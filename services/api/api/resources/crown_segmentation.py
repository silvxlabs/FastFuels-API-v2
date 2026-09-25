"""
api/v2/resources/crown_segmentation.py

Crown segmentation settings shared by CHM tree detection and crown Features.
"""

from enum import StrEnum

from pydantic import BaseModel, Field

MAX_CROWN_SEGMENTATION_RESOLUTION_M = 2.0


class CrownSegmentationMethod(StrEnum):
    dalponte2016 = "dalponte2016"


class CrownSegmentationBase(BaseModel):
    """Seeded region growing after Dalponte & Coomes (2016)."""

    method: CrownSegmentationMethod = Field(
        default=CrownSegmentationMethod.dalponte2016,
        description=(
            "Segmentation algorithm. `dalponte2016` grows each crown outward "
            "from its treetop, one ring of cells at a time, and stops at the "
            "crown's edge."
        ),
    )
    min_relative_height: float = Field(
        default=0.45,
        ge=0,
        lt=1,
        description=(
            "A cell joins a crown only if its CHM height is at least this "
            "fraction of the treetop cell's height. Provisional default."
        ),
    )
    min_relative_crown_height: float = Field(
        default=0.55,
        ge=0,
        lt=1,
        description=(
            "A cell joins a crown only if its CHM height is at least this "
            "fraction of the crown's current mean height. Provisional default."
        ),
    )
    max_crown_radius: float = Field(
        default=10.0,
        gt=0,
        le=30,
        description=(
            "A cell joins a crown only if its center lies within this distance "
            "(m) of the treetop cell's center. Provisional default."
        ),
    )
