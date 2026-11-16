"""
api/v2/resources/crown_radius.py

Sources of a tree's maximum crown radius, shared by tree voxelization, inventory
canopy grids, and crown-spacing inventory treatments.
"""

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class MaxCrownRadiusUnit(StrEnum):
    """Accepted inventory max crown radius units."""

    m = "m"


class AllometryMaxCrownRadiusSource(BaseModel):
    """Use the crown profile model's allometric max crown radius (default)."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["allometry"] = "allometry"


class InventoryColumnMaxCrownRadiusSource(BaseModel):
    """Read per-tree max crown radius from an inventory column.

    The crown profile model still drives the crown shape — the supplied
    radius rescales it so the maximum radius matches the per-tree value.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["inventory_column"] = "inventory_column"
    column: str
    unit: MaxCrownRadiusUnit = MaxCrownRadiusUnit.m


MaxCrownRadiusSource = Annotated[
    AllometryMaxCrownRadiusSource | InventoryColumnMaxCrownRadiusSource,
    Field(discriminator="type"),
]


class CanopyCrownWidthEquations(StrEnum):
    """Allometric equation families for a tree's maximum crown radius.

    ``purves`` (Purves et al. 2007) is the national default and the one
    the tree voxelization endpoint uses; it varies with height and
    crown base height as well as diameter. ``crookston_stage`` (Crookston &
    Stage 1999, RMRS-GTR-24, reached through FVS) is the crown width
    behind FuelCalc's canopy cover — diameter alone above breast
    height, with regionally fitted coefficients — and is offered for
    compatibility studies.
    """

    purves = "purves"
    crookston_stage = "crookston_stage"


class CanopyAllometryMaxCrownRadiusSource(BaseModel):
    """Compute each tree's maximum crown radius from allometric equations.

    Supersets the voxelize allometry source with an `equations` choice,
    because canopy cover and crown biomass are separate axes here: which
    allometry supplies the radius is independent of how a cover method
    treats crowns that overlap, so a run can vary either without
    confounding the other.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["allometry"] = "allometry"
    equations: CanopyCrownWidthEquations = Field(
        default=CanopyCrownWidthEquations.purves,
        description=(
            "Allometric equation family for maximum crown radius. "
            "`purves` is the national default, consistent with the tree "
            "voxelization endpoint. `crookston_stage` reproduces the "
            "crown widths FuelCalc uses for canopy cover."
        ),
    )


CanopyMaxCrownRadiusSource = Annotated[
    CanopyAllometryMaxCrownRadiusSource | InventoryColumnMaxCrownRadiusSource,
    Field(discriminator="type"),
]
