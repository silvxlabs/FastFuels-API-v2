"""
api/v2/resources/inventories/tree/chm/schema.py

Schema models for CHM extraction inventory creation.
"""

from typing import Annotated, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from api.resources.inventories.modification_models import InventoryModification
from api.resources.inventories.schema import CreateInventoryRequestBase
from api.resources.inventories.treatment_models import InventoryTreatment


class StemIsolationLmf(BaseModel):
    """Parameters for Local Maximum Filter (LMF) stem isolation."""

    name: Literal["lmf"] = "lmf"
    min_height: float = Field(
        default=2.0,
        description="Minimum height threshold (in CHM units) for a treetop.",
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


class StemIsolationVwf(BaseModel):
    """Parameters for Variable Window Filter (VWF) stem isolation."""

    name: Literal["vwf"] = "vwf"
    min_height: float = Field(
        default=2.0,
        description="Minimum height threshold (in CHM units) for a treetop.",
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


# FastAPI will automatically route validation to the correct model based on the "name" field.
StemIsolationAlgorithm = Annotated[
    StemIsolationLmf | StemIsolationVwf, Field(discriminator="name")
]


class ChmInventorySource(BaseModel):
    """Source metadata stored on the inventory document."""

    name: Literal["chm"] = "chm"
    source_chm_grid_id: str
    algorithm: StemIsolationAlgorithm


class CreateChmInventoryRequest(CreateInventoryRequestBase):
    """Request body for creating an inventory via CHM extraction."""

    source_chm_grid_id: str = Field(
        description="ID of a completed CHM grid to use as the source.",
    )
    algorithm: StemIsolationAlgorithm = Field(
        default_factory=StemIsolationLmf,
        description="Stem isolation algorithm and its parameters.",
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
