"""
api/v2/resources/grids/tree/inventory/schema.py

Schema models for the tree-inventory voxelization grid product.

A tree inventory is voxelized onto a 3D grid using species-specific crown
profile models and biomass models to produce per-voxel fuel properties.
"""

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from api.resources.grids.schema import validate_no_duplicates
from api.resources.grids.tree.schema import (
    BiomassModel,
    CrownProfileModel,
    MoistureModel,
    TreeBand,
    UniformMoistureModel,
)


class TreeInventorySource(BaseModel):
    """Source metadata stored on the Grid document for reproducibility.

    Records the inventory that was voxelized and every resolved model choice
    so the grid can be exactly reproduced.
    """

    name: Literal["inventory"] = "inventory"
    product: Literal["tree"] = "tree"
    description: Literal["3D tree fuel grid from tree inventory voxelization"] = (
        "3D tree fuel grid from tree inventory voxelization"
    )

    source_inventory_id: str
    resolution: tuple[float, float, float] = Field(
        description="Voxel resolution (x, y, z) in meters.",
    )
    bands: list[TreeBand]
    crown_profile_model: CrownProfileModel
    biomass_model: BiomassModel
    biomass_column: str | None = Field(
        default=None,
        description=(
            "Inventory column to read foliage biomass from. Only stored "
            "when biomass_model == 'inventory'."
        ),
    )
    moisture_model: MoistureModel | None = None


class CreateTreeInventoryRequest(BaseModel):
    """Request body for creating a tree fuel grid from a tree inventory.

    Does not extend CreateGridRequestBase because 3D grids do not support
    modifications — modifications must be applied to the inventory before
    voxelization, not to the resulting voxel grid.
    """

    name: str = Field("", max_length=255)
    description: str = Field("", max_length=2000)
    tags: list[str] = Field(default_factory=list, max_length=50)

    source_inventory_id: str = Field(
        description="ID of a completed tree inventory to voxelize.",
    )
    resolution: tuple[float, float, float] = Field(
        description="Voxel resolution (x, y, z) in meters.",
    )
    bands: list[TreeBand] = Field(
        min_length=1,
        description="Which output bands to produce.",
    )
    crown_profile_model: CrownProfileModel = Field(
        default=CrownProfileModel.purves,
        description="Crown geometry model. Default: purves.",
    )
    biomass_model: BiomassModel = Field(
        default=BiomassModel.nsvb,
        description="Foliage biomass model. Default: nsvb.",
    )
    biomass_column: str | None = Field(
        default=None,
        description=(
            "Inventory column name when biomass_model == 'inventory'. "
            "Ignored for other biomass models. Defaults to 'crown_fuel_load' "
            "when biomass_model is 'inventory' and this field is omitted."
        ),
    )
    moisture_model: MoistureModel | None = Field(
        default=None,
        description=(
            "Live fuel moisture model. Required only when fuel_moisture.live "
            "is in bands. Defaults to uniform live=100.0 when fuel_moisture.live "
            "is requested without an explicit model. Silently ignored otherwise."
        ),
    )

    @field_validator("resolution")
    @classmethod
    def validate_resolution_positive(
        cls, v: tuple[float, float, float]
    ) -> tuple[float, float, float]:
        if any(component <= 0 for component in v):
            raise ValueError(
                "All resolution components must be positive (x, y, z > 0)."
            )
        return v

    @field_validator("bands")
    @classmethod
    def no_duplicate_bands(cls, v: list[TreeBand]) -> list[TreeBand]:
        return validate_no_duplicates(v)

    @model_validator(mode="after")
    def resolve_conditional_defaults(self) -> "CreateTreeInventoryRequest":
        # Auto-populate moisture_model with the uniform default when
        # fuel_moisture.live is requested without an explicit model.
        if TreeBand.fuel_moisture_live in self.bands and self.moisture_model is None:
            self.moisture_model = UniformMoistureModel()

        # biomass_column is only meaningful when biomass_model == "inventory".
        # Silently drop it for other biomass models so it isn't stored in the
        # source model and cause confusion during reproducibility checks.
        if self.biomass_model != BiomassModel.inventory:
            self.biomass_column = None

        return self
