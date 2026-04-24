"""
api/v2/resources/grids/tree/schema.py

Shared schema primitives for 3D tree fuel grid products.

Tree fuel grids are voxelized crown representations of tree inventories.
Different source types (inventory, point-cloud, etc.) share the same band
vocabulary and model configuration types, which live here. Source-specific
request schemas live under the source sub-packages (e.g. `tree/inventory/`).
"""

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from api.resources.grids.schema import Band, BandType


class TreeBand(StrEnum):
    """Available output bands for tree fuel grids.

    Each value is the band's dot-notation key as it appears in the stored
    Grid document. Band definitions (type, unit) live in TREE_BAND_DEFS.
    """

    bulk_density_foliage = "bulk_density.foliage"
    bulk_density_branchwood = "bulk_density.branchwood"
    bulk_density_fine = "bulk_density.fine"
    fuel_moisture_live = "fuel_moisture.live"
    savr_foliage = "savr.foliage"
    spcd = "spcd"
    tree_id = "tree_id"
    volume_fraction = "volume_fraction"


TREE_BAND_DEFS: dict[TreeBand, dict] = {
    TreeBand.bulk_density_foliage: {
        "key": "bulk_density.foliage",
        "type": BandType.continuous,
        "unit": "kg/m³",
    },
    TreeBand.bulk_density_branchwood: {
        "key": "bulk_density.branchwood",
        "type": BandType.continuous,
        "unit": "kg/m³",
    },
    TreeBand.bulk_density_fine: {
        "key": "bulk_density.fine",
        "type": BandType.continuous,
        "unit": "kg/m³",
    },
    TreeBand.fuel_moisture_live: {
        "key": "fuel_moisture.live",
        "type": BandType.continuous,
        "unit": "%",
    },
    TreeBand.savr_foliage: {
        "key": "savr.foliage",
        "type": BandType.continuous,
        "unit": "m⁻¹",
    },
    TreeBand.spcd: {
        "key": "spcd",
        "type": BandType.categorical,
        "unit": None,
    },
    TreeBand.tree_id: {
        "key": "tree_id",
        "type": BandType.categorical,
        "unit": None,
    },
    TreeBand.volume_fraction: {
        "key": "volume_fraction",
        "type": BandType.continuous,
        "unit": None,
    },
}


class CrownProfileModel(StrEnum):
    """Crown geometry models — which voxels a tree's crown occupies."""

    purves = "purves"
    beta = "beta"


class BiomassEquations(StrEnum):
    """Allometric equation families for estimating biomass components."""

    nsvb = "nsvb"
    jenkins = "jenkins"


class BiomassComponent(StrEnum):
    """Biomass components that can be requested or supplied."""

    foliage = "foliage"
    branchwood = "branchwood"
    fine = "fine"


class BiomassUnit(StrEnum):
    """Accepted inventory biomass units."""

    kg = "kg"


class InventoryBiomassColumn(BaseModel):
    """Inventory column containing per-tree biomass for one component."""

    model_config = ConfigDict(extra="forbid")

    column: str
    unit: BiomassUnit = BiomassUnit.kg


class FineBiomassConfig(BaseModel):
    """Configuration for derived fine biomass."""

    model_config = ConfigDict(extra="forbid")

    recipe: Literal["foliage_plus_branchwood_fraction"]
    branchwood_fraction: float = Field(gt=0, le=1)


class BiomassSourceBase(BaseModel):
    """Common biomass component request behavior."""

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def validate_components(self):
        if not self.components:
            raise ValueError("At least one biomass component is required.")

        seen: set[BiomassComponent] = set()
        duplicates: list[str] = []
        for component in self.components:
            if component in seen:
                duplicates.append(component.value)
            seen.add(component)
        if duplicates:
            raise ValueError(
                "Duplicate biomass components are not allowed: "
                f"{', '.join(duplicates)}."
            )

        if self.fine is not None and BiomassComponent.fine not in self.components:
            raise ValueError("fine configuration requires 'fine' in components.")

        return self


class AllometryBiomassSource(BiomassSourceBase):
    """Estimate biomass from allometric equations."""

    type: Literal["allometry"] = "allometry"
    equations: BiomassEquations = BiomassEquations.nsvb
    components: list[BiomassComponent] = Field(
        default_factory=lambda: [BiomassComponent.foliage],
        min_length=1,
    )
    fine: FineBiomassConfig | None = None

    @model_validator(mode="after")
    def validate_fine_definition(self):
        if BiomassComponent.fine in self.components and self.fine is None:
            raise ValueError("Allometry fine biomass requires a fine configuration.")
        return self


class InventoryColumnsBiomassSource(BiomassSourceBase):
    """Read per-tree component biomass from inventory columns."""

    type: Literal["inventory_columns"] = "inventory_columns"
    columns: dict[BiomassComponent, InventoryBiomassColumn] = Field(
        min_length=1,
        description="Per-component inventory columns. Values must be per-tree kg.",
    )
    components: list[BiomassComponent] = Field(
        default_factory=lambda: [BiomassComponent.foliage],
        min_length=1,
    )
    fine: FineBiomassConfig | None = None

    @model_validator(mode="after")
    def validate_source_supports_components(self):
        for component in self.components:
            if component == BiomassComponent.fine and self.fine is not None:
                missing = [
                    required
                    for required in (
                        BiomassComponent.foliage,
                        BiomassComponent.branchwood,
                    )
                    if required not in self.columns
                ]
                if missing:
                    missing_names = ", ".join(m.value for m in missing)
                    raise ValueError(
                        "Fine biomass recipe requires inventory columns for: "
                        f"{missing_names}."
                    )
            elif component not in self.columns:
                raise ValueError(
                    f"Inventory biomass source is missing a {component.value!r} column."
                )

        return self


BiomassSource = Annotated[
    AllometryBiomassSource | InventoryColumnsBiomassSource,
    Field(discriminator="type"),
]


class UniformMoistureModel(BaseModel):
    """Uniform live fuel moisture — every canopy voxel gets the same value."""

    method: Literal["uniform"] = "uniform"
    live: float = Field(
        default=100.0,
        description="Live fuel moisture content (%), applied uniformly.",
    )


MoistureModel = UniformMoistureModel


def build_tree_bands(requested: list[TreeBand]) -> list[Band]:
    """Build Band objects for requested tree bands with indices in request order."""
    return [Band(index=i, **TREE_BAND_DEFS[band]) for i, band in enumerate(requested)]
