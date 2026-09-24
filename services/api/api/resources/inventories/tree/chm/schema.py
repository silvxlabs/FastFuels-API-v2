"""
api/v2/resources/inventories/tree/chm/schema.py

Schema models for CHM extraction inventory creation.
"""

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from api.resources.crown_segmentation import CrownSegmentationBase
from api.resources.inventories.modification_models import InventoryModification
from api.resources.inventories.schema import CreateInventoryRequestBase
from api.resources.inventories.treatment_models import InventoryTreatment


class StemIsolationLmf(BaseModel):
    """Parameters for Local Maximum Filter (LMF) stem isolation.

    When set, ``max_height`` must be greater than ``min_height``.
    """

    name: Literal["lmf"] = "lmf"
    min_height: float = Field(
        default=2.0,
        description="Minimum height threshold (in meters) for a treetop.",
    )
    max_height: float | None = Field(
        default=120.0,
        description=(
            "Maximum height threshold (in meters) for a treetop. CHM returns "
            "taller than this are treated as artifacts (e.g. LiDAR noise spikes) "
            "and excluded before detection. Defaults to 120, above the tallest "
            "known tree; set to null to disable the ceiling."
        ),
    )
    footprint_size: int = Field(
        default=3,
        description="Diameter of the circular footprint in pixels. Must be an odd integer.",
    )

    @field_validator("footprint_size")
    @classmethod
    def validate_footprint_size_is_odd(cls, v: int) -> int:
        if v % 2 == 0:
            raise ValueError(
                "The LMF 'footprint_size' parameter must be an odd integer."
            )
        return v

    @model_validator(mode="after")
    def validate_max_height_above_min(self):
        if self.max_height is not None and self.max_height <= self.min_height:
            raise ValueError("'max_height' must be greater than 'min_height'.")
        return self


class StemIsolationVwf(BaseModel):
    """Parameters for Variable Window Filter (VWF) stem isolation.

    When set, ``max_height`` must be greater than ``min_height``.
    """

    name: Literal["vwf"] = "vwf"
    min_height: float = Field(
        default=2.0,
        description="Minimum height threshold (in meters) for a treetop.",
    )
    max_height: float | None = Field(
        default=120.0,
        description=(
            "Maximum height threshold (in meters) for a treetop. CHM returns "
            "taller than this are treated as artifacts (e.g. LiDAR noise spikes) "
            "and excluded before detection. Defaults to 120, above the tallest "
            "known tree; set to null to disable the ceiling."
        ),
    )
    spatial_resolution: float | None = Field(
        default=None,
        description="Spatial resolution of the CHM. If omitted, it will be automatically inferred from the source grid metadata.",
    )
    crown_ratio: float = Field(
        default=0.10,
        description="Multiplier used to dynamically scale the search window based on pixel height.",
    )
    crown_offset: float = Field(
        default=1.0,
        description="Constant offset (in meters) added to the dynamic search window.",
    )

    @model_validator(mode="after")
    def validate_max_height_above_min(self):
        if self.max_height is not None and self.max_height <= self.min_height:
            raise ValueError("'max_height' must be greater than 'min_height'.")
        return self


# FastAPI will automatically route validation to the correct model based on the "name" field.
StemIsolationAlgorithm = Annotated[
    StemIsolationLmf | StemIsolationVwf, Field(discriminator="name")
]


class CrownRadiusEstimator(StrEnum):
    area_equivalent = "area_equivalent"


class ChmCrownSegmentation(CrownSegmentationBase):
    """Crown segmentation run after stem isolation. Each tree gets a
    `crown_radius` column measured from its segmented crown."""

    radius_estimator: CrownRadiusEstimator = Field(
        default=CrownRadiusEstimator.area_equivalent,
        description=(
            "How a crown becomes one radius. `area_equivalent` is the radius of "
            "a circle with the crown's area, sqrt(area / pi)."
        ),
    )


class ChmInventorySource(BaseModel):
    """Source metadata stored on the inventory document."""

    name: Literal["chm"] = "chm"
    source_chm_grid_id: str
    source_chm_grid_checksum: str | None = Field(
        default=None,
        description=(
            "The source CHM grid's `checksum` at the time this inventory was "
            "created from it. Compare it against the source grid's current "
            "`checksum` to tell whether the source has changed since."
        ),
    )
    algorithm: StemIsolationAlgorithm
    crown_segmentation: ChmCrownSegmentation | None = None


class CreateChmInventoryRequest(CreateInventoryRequestBase):
    """Request body for creating an inventory via CHM extraction."""

    source_chm_grid_id: str = Field(
        description="ID of a completed CHM grid to use as the source.",
    )
    algorithm: StemIsolationAlgorithm = Field(
        default_factory=StemIsolationLmf,
        description="Stem isolation algorithm and its parameters.",
    )
    crown_segmentation: ChmCrownSegmentation | None = Field(
        default=None,
        description=(
            "Segment each detected tree's crown on the CHM and add a "
            "`crown_radius` column (m). Crowns grow outward from each treetop "
            "over CHM cells within the detection height range and stop at the "
            "crown's edge; each cell belongs to at most one tree. Every tree "
            "gets a radius of at least one cell's area-equivalent radius. "
            "Requires a CHM cell size of 2 m or finer. Omit to skip "
            "segmentation."
        ),
    )
    modifications: list[InventoryModification] = Field(
        default_factory=list,
        description="Modifications to apply after stem extraction.",
    )
    treatments: list[InventoryTreatment] = Field(
        default_factory=list,
        description=(
            "Silvicultural treatments thin against tree diameter, so they "
            "require a diameter (`dbh`) column. CHM stem isolation produces only "
            "height and position (`x`, `y`, `height`), so treatments are not "
            "supported here and this must be empty."
        ),
    )

    @model_validator(mode="after")
    def reject_treatments(self):
        """Treatments thin against diameter; CHM extraction produces none."""
        if self.treatments:
            raise ValueError(
                "Silvicultural treatments require a tree diameter (`dbh`) to thin "
                "against. CHM stem isolation produces only height and position "
                "(`x`, `y`, `height`), so treatments cannot be applied to a "
                "CHM-derived inventory."
            )
        return self
