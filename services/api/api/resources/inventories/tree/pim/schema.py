"""
api/v2/resources/inventories/tree/pim/schema.py

Schema models for PIM expansion inventory creation.
"""

from random import randint
from typing import Literal

from pydantic import BaseModel, Field

from api.resources.inventories.modification_models import InventoryModification
from api.resources.inventories.schema import CreateInventoryRequestBase, PointProcess


def _generate_random_seed() -> int:
    return randint(1, 1_000_000_000)


class PimInventorySource(BaseModel):
    """Source metadata stored on the inventory document.

    Records which PIM grid was used, the point process, and the seed
    so the inventory can be exactly reproduced.
    """

    name: Literal["pim"] = "pim"
    source_pim_grid_id: str
    point_process: PointProcess
    seed: int


class CreatePimInventoryRequest(CreateInventoryRequestBase):
    """Request body for creating an inventory via PIM expansion."""

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
