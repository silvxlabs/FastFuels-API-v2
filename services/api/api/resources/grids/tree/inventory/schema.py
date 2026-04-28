"""
api/v2/resources/grids/tree/inventory/schema.py

Schema models for the tree-inventory voxelization grid product.

A tree inventory is voxelized onto a 3D grid using species-specific crown
profile models and biomass configuration to produce per-voxel fuel properties.
"""

from random import randint
from typing import Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PositiveFloat,
    field_validator,
    model_validator,
)

from api.resources.grids.schema import validate_no_duplicates
from api.resources.grids.tree.schema import (
    AllometryBiomassSource,
    BiomassSource,
    CrownProfileModel,
    MoistureModel,
    TreeBand,
    UniformMoistureValue,
    validate_bulk_density_bands_have_components,
)

Resolution3D = tuple[PositiveFloat, PositiveFloat, PositiveFloat]


def _generate_random_seed() -> int:
    return randint(1, 1_000_000_000)


class TreeInventorySource(BaseModel):
    """Source metadata stored on the Grid document for reproducibility.

    Records the inventory that was voxelized and every resolved model choice
    so the grid can be exactly reproduced.
    """

    model_config = ConfigDict(extra="forbid")

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
    biomass_source: BiomassSource
    moisture_model: MoistureModel | None = None
    seed: int = Field(
        description=(
            "Random seed that drove stochastic sampling during voxelization. "
            "Persisted so the grid can be exactly reproduced."
        ),
    )

    @model_validator(mode="after")
    def validate_band_components_configured(self) -> Self:
        validate_bulk_density_bands_have_components(self.bands, self.biomass_source)
        return self


class CreateTreeInventoryRequest(BaseModel):
    """Request body for creating a tree fuel grid from a tree inventory.

    Does not extend CreateGridRequestBase because 3D grids do not support
    modifications — modifications must be applied to the inventory before
    voxelization, not to the resulting voxel grid.
    """

    model_config = ConfigDict(extra="forbid")

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
    biomass_source: BiomassSource = Field(
        default_factory=AllometryBiomassSource,
        description="Biomass source and requested biomass components.",
    )
    moisture_model: MoistureModel | None = Field(
        default=None,
        description=(
            "Live/dead fuel moisture model. Applied only when matching "
            "fuel_moisture bands are requested. Live defaults to uniform "
            "100.0; dead defaults to uniform 10.0."
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
        # moisture_model is only meaningful for requested moisture bands.
        # Populate state-specific uniform defaults when requested without a
        # model, and drop unrequested states so stored source reflects what
        # was actually applied.
        live_requested = TreeBand.fuel_moisture_live in self.bands
        dead_requested = TreeBand.fuel_moisture_dead in self.bands
        if live_requested or dead_requested:
            model = self.moisture_model or MoistureModel()
            self.moisture_model = MoistureModel(
                live=(
                    model.live or UniformMoistureValue(value=100.0)
                    if live_requested
                    else None
                ),
                dead=(
                    model.dead or UniformMoistureValue(value=10.0)
                    if dead_requested
                    else None
                ),
            )
        else:
            self.moisture_model = None

        return self

    @model_validator(mode="after")
    def validate_band_components_configured(self) -> Self:
        validate_bulk_density_bands_have_components(self.bands, self.biomass_source)
        return self
