"""
api/v2/resources/grids/tree/inventory/schema.py

Schema models for the tree-inventory voxelization grid product.

A tree inventory is voxelized onto a 3D grid using species-specific crown
profile models and biomass models to produce per-voxel fuel properties.
"""

from random import randint
from typing import Literal, Self

from pydantic import BaseModel, Field, PositiveFloat, field_validator, model_validator

from api.resources.grids.schema import validate_no_duplicates
from api.resources.grids.tree.schema import (
    BiomassModel,
    CrownProfileModel,
    MoistureModel,
    TreeBand,
    UniformMoistureModel,
)

Resolution3D = tuple[PositiveFloat, PositiveFloat, PositiveFloat]


def _generate_random_seed() -> int:
    return randint(1, 1_000_000_000)


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
    resolution: Resolution3D = Field(
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
    seed: int = Field(
        description=(
            "Random seed that drove stochastic sampling during voxelization. "
            "Persisted so the grid can be exactly reproduced."
        ),
    )


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
    resolution: Resolution3D = Field(
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
            "Live fuel moisture model. Applied only when fuel_moisture.live "
            "is in bands — defaults to uniform live=100.0 in that case. "
            "Silently dropped if supplied when fuel_moisture.live is absent."
        ),
    )
    seed: int = Field(
        default_factory=_generate_random_seed,
        description=(
            "Random seed for reproducibility. Controls stochastic tree "
            "voxel sampling and biomass distribution. Generated randomly "
            "if omitted; persisted on the grid document either way so "
            "re-running a grid always yields bit-identical output."
        ),
    )

    @field_validator("bands")
    @classmethod
    def no_duplicate_bands(cls, v: list[TreeBand]) -> list[TreeBand]:
        return validate_no_duplicates(v)

    @model_validator(mode="after")
    def resolve_conditional_defaults(self) -> Self:
        # moisture_model is only meaningful when fuel_moisture.live is in
        # bands. Populate the uniform default when the band is requested
        # without a model; drop any supplied model when the band is absent
        # so the stored source reflects only what was actually applied.
        if TreeBand.fuel_moisture_live in self.bands:
            if self.moisture_model is None:
                self.moisture_model = UniformMoistureModel()
        else:
            self.moisture_model = None

        # Same pattern for biomass_column: only meaningful when biomass
        # is read from an inventory column, dropped otherwise.
        if self.biomass_model != BiomassModel.inventory:
            self.biomass_column = None

        return self
