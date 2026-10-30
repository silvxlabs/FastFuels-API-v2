"""
api/v2/resources/inventories/tree/allometry/crown_base_height/schema.py

Schema models for crown base height inventory creation.

A crown base height model computes each tree's crown base height from the
species, dbh, crown ratio and height already on a tree inventory. No GDAM call
is made.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from api.resources.input_requirements import input_requirements, inventory
from api.resources.inventories.schema import (
    BASE_INVENTORY_COLUMNS,
    CreateInventoryRequestBase,
)
from api.resources.inventories.tree.allometry.schema import (
    CrownBaseHeightModel,
    default_crown_base_height_model,
)

# Columns every tree needs a value in; none of them can be imputed.
CROWN_BASE_HEIGHT_REQUIRED_SOURCE_COLUMNS = ("x", "y", "height")

# Model inputs GDAM can fill, in the order GDAM's impute_columns lists them.
CROWN_BASE_HEIGHT_GDAM_INPUTS = ("dbh", "crown_ratio", "fia_species_code")


class CrownBaseHeightInventorySource(BaseModel):
    """Source metadata stored on the inventory document.

    Records which tree inventory the crown base heights were computed for, its
    checksum at the time this inventory was created, the model and whether
    existing crown base heights were replaced.
    """

    name: Literal["crown_base_height"] = "crown_base_height"
    source_tree_inventory_id: str
    source_tree_inventory_checksum: str | None = Field(
        default=None,
        description=(
            "The source tree inventory's `checksum` at the time this inventory "
            "was created from it. Compare it against the source inventory's "
            "current `checksum` to tell whether the source has changed since."
        ),
    )
    model: CrownBaseHeightModel = Field(
        default_factory=default_crown_base_height_model,
        description="The model that computed `crown_base_height`.",
    )
    overwrite: bool = Field(
        default=False,
        description=(
            "Whether every tree's crown base height was recomputed (`true`) or "
            "only empty cells were filled (`false`)."
        ),
    )


class CreateCrownBaseHeightInventoryRequest(CreateInventoryRequestBase):
    """Request body for creating an inventory with model-computed crown base
    heights."""

    model_config = ConfigDict(
        json_schema_extra=input_requirements(
            inventory(
                "source_tree_inventory_id",
                columns=(
                    *CROWN_BASE_HEIGHT_REQUIRED_SOURCE_COLUMNS,
                    *CROWN_BASE_HEIGHT_GDAM_INPUTS,
                ),
            ),
        ),
    )

    source_tree_inventory_id: str = Field(
        description=(
            "ID of a completed tree inventory with species, dbh, crown ratio and "
            "height for every tree."
        ),
    )
    model: CrownBaseHeightModel = Field(
        default_factory=default_crown_base_height_model,
        description=(
            "The model that computes each tree's crown base height from its "
            "species, dbh, crown ratio and height, selected by `name`. Species "
            "the model does not cover get no crown base height."
        ),
    )
    overwrite: bool = Field(
        default=False,
        description=(
            "`false` keeps existing crown base heights and fills only empty "
            "cells. `true` recomputes every tree's crown base height with "
            "`model`."
        ),
    )


def resolve_crown_base_height_columns(source_columns: list[dict]) -> list[dict]:
    """Columns the new inventory will carry: the source's columns plus
    `crown_base_height` when the source lacks it."""
    resolved = [dict(c) for c in source_columns]
    if not any(c["key"] == "crown_base_height" for c in resolved):
        resolved.append(
            next(
                c.model_dump()
                for c in BASE_INVENTORY_COLUMNS
                if c.key == "crown_base_height"
            )
        )
    return resolved
