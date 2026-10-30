"""
api/v2/resources/inventories/tree/pim/schema.py

Schema models for PIM expansion inventory creation.
"""

from random import randint
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from api.resources.input_requirements import grid, input_requirements
from api.resources.inventories.modification_models import InventoryModification
from api.resources.inventories.schema import CreateInventoryRequestBase, PointProcess
from api.resources.inventories.treatment_models import InventoryTreatment
from api.resources.inventories.tree.allometry.schema import (
    CrownBaseHeightModel,
    default_crown_base_height_model,
)


def _generate_random_seed() -> int:
    return randint(1, 1_000_000_000)


class PimInventorySource(BaseModel):
    """Source metadata stored on the inventory document.

    Records which PIM grid was used, the point process, and the seed
    so the inventory can be exactly reproduced.
    """

    name: Literal["pim"] = "pim"
    source_pim_grid_id: str
    source_pim_grid_checksum: str | None = Field(
        default=None,
        description=(
            "The source PIM grid's `checksum` at the time this inventory was "
            "created from it. Compare it against the source grid's current "
            "`checksum` to tell whether the source has changed since."
        ),
    )
    point_process: PointProcess
    seed: int
    crown_base_height_model: CrownBaseHeightModel = Field(
        default_factory=default_crown_base_height_model,
        description="The model that computed `crown_base_height`.",
    )


class CreatePimInventoryRequest(CreateInventoryRequestBase):
    """Request body for creating an inventory via PIM expansion."""

    model_config = ConfigDict(
        json_schema_extra=input_requirements(
            grid("source_pim_grid_id", bands=["tm_id"], source="pim"),
        ),
    )

    source_pim_grid_id: str = Field(
        description="ID of a completed PIM grid to use as the source.",
    )
    seed: int = Field(
        default_factory=_generate_random_seed,
        description="Random seed for reproducibility. Generated randomly if omitted.",
    )
    point_process: PointProcess = Field(
        default=PointProcess.inhomogeneous_poisson,
        description="Spatial point process for tree coordinate assignment.",
    )
    modifications: list[InventoryModification] = Field(
        default_factory=list,
        description="Modifications to apply after point process expansion.",
    )
    treatments: list[InventoryTreatment] = Field(
        default_factory=list,
        description="Silvicultural treatments to apply after modifications.",
    )
