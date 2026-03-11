"""
api/v2/resources/inventories/chm/schema.py

Schema models for CHM extraction inventory creation.
"""

from typing import Annotated, Literal

from pydantic import BaseModel, Field, field_validator

from api.resources.inventories.modification_models import InventoryModification
from api.resources.inventories.schema import CreateInventoryRequestBase


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
            raise ValueError("The LMF 'footprint_size' parameter must be an odd integer.")
        return v


# Use an Annotated Union so FastAPI can discriminate between algorithms automatically
# if/when we add more (e.g., watershed) in the future.
StemIsolationAlgorithm = Annotated[StemIsolationLmf, Field(discriminator="name")]


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
