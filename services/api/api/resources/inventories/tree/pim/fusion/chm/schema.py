"""
api/v2/resources/inventories/tree/pim/fusion/chm/schema.py

Schema models for PIM-CHM fusion inventory creation.

The path names the fused sources (a PIM refined by a CHM); the ``method``
object names the algorithm, discriminated on ``name`` the way ``algorithm``
selects LMF or VWF on ``tree/chm``. A new algorithm for the same two sources is
a new ``method`` member here and never a new URL.
"""

from random import randint
from typing import Annotated, Literal

from pydantic import BaseModel, Field

from api.resources.inventories.modification_models import InventoryModification
from api.resources.inventories.schema import CreateInventoryRequestBase, PointProcess
from api.resources.inventories.treatment_models import InventoryTreatment


def _generate_random_seed() -> int:
    return randint(1, 1_000_000_000)


class ReimputationMethod(BaseModel):
    """The v1 fusion algorithm (``fastfuels_core.onramps.hag_pim``).

    Resample the PIM to ``resolution``, keep a cell's plot only where the CHM's
    canopy cover — the fraction of CHM cells taller than ``min_height`` — exceeds
    ``cover_threshold``, then expand the surviving plots as ``tree/pim``. Defaults
    are the v1 production values; ``fastfuels-core``'s own defaults are 1.0 m and
    0.25.
    """

    name: Literal["reimputation"] = "reimputation"
    resolution: float = Field(
        default=7.5,
        gt=0,
        description=(
            "Resolution (meters) the PIM is resampled to before conditioning. "
            "Must be no finer than the CHM cell and no coarser than the PIM cell."
        ),
    )
    min_height: float = Field(
        default=2.0,
        ge=0,
        description="CHM height (meters) above which a cell counts as canopy.",
    )
    cover_threshold: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum canopy cover fraction (0-1) a resampled cell needs to keep "
            "its plot. Cells at or below this become gaps (no trees)."
        ),
    )


# Discriminated on `name` so a second algorithm for these two sources — e.g.
# `surface_matching` once the solver lands in silvxlabs/fastfuels-core#103 — is a
# new member here and never a new URL. One member today.
FusionMethod = Annotated[ReimputationMethod, Field(discriminator="name")]


class PimChmFusionInventorySource(BaseModel):
    """Source metadata stored on the inventory document.

    ``name`` stays ``"pim"`` so the ``?source=pim`` list filter still finds
    fusion inventories; ``fusion`` records which other sources refined it.
    """

    name: Literal["pim"] = "pim"
    fusion: list[str] = Field(
        default_factory=lambda: ["chm"],
        description="Non-primary sources fused into this inventory.",
    )
    source_pim_grid_id: str
    source_pim_grid_checksum: str | None = Field(
        default=None,
        description=(
            "The source PIM grid's `checksum` at the time this inventory was "
            "created from it. Compare it against the source grid's current "
            "`checksum` to tell whether the source has changed since."
        ),
    )
    source_chm_grid_id: str
    source_chm_grid_checksum: str | None = Field(
        default=None,
        description=(
            "The source CHM grid's `checksum` at the time this inventory was "
            "created from it. Compare it against the source grid's current "
            "`checksum` to tell whether the source has changed since."
        ),
    )
    method: FusionMethod
    point_process: PointProcess
    seed: int


class CreatePimChmFusionInventoryRequest(CreateInventoryRequestBase):
    """Request body for creating an inventory by fusing a PIM with a CHM."""

    source_pim_grid_id: str = Field(
        description="ID of a completed PIM grid to use as the primary source.",
    )
    source_chm_grid_id: str = Field(
        description="ID of a completed CHM grid used to condition the PIM.",
    )
    method: FusionMethod = Field(
        default_factory=ReimputationMethod,
        description="Fusion algorithm and its parameters. Defaults to reimputation.",
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
