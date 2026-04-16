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

from pydantic import BaseModel, Field

from api.resources.grids.schema import Band, BandType


class TreeBand(StrEnum):
    """Available output bands for tree fuel grids.

    Each value is the band's dot-notation key as it appears in the stored
    Grid document. Band definitions (type, unit) live in TREE_BAND_DEFS.
    """

    bulk_density_foliage = "bulk_density.foliage"
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


class BiomassModel(StrEnum):
    """Foliage biomass computation models."""

    nsvb = "nsvb"
    jenkins = "jenkins"
    inventory = "inventory"


class UniformMoistureModel(BaseModel):
    """Uniform live fuel moisture — every canopy voxel gets the same value."""

    method: Literal["uniform"] = "uniform"
    live: float = Field(
        default=100.0,
        description="Live fuel moisture content (%), applied uniformly.",
    )


MoistureModel = Annotated[UniformMoistureModel, Field(discriminator="method")]


def build_tree_bands(requested: list[TreeBand]) -> list[Band]:
    """Build Band objects for requested tree bands with indices in request order."""
    return [Band(index=i, **TREE_BAND_DEFS[band]) for i, band in enumerate(requested)]
