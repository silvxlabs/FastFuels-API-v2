"""
api/v2/resources/inventories/tree/allometry/gdam/schema.py

Schema models for GDAM allometry imputation inventory creation.

GDAM fills the missing morphology columns (dbh, crown ratio, species) on an
existing position+height tree inventory. The endpoint is zero-config: the only
input is the source inventory to fill in.
"""

from typing import Literal

from pydantic import BaseModel, Field

from api.resources.inventories.schema import CreateInventoryRequestBase


class GdamInventorySource(BaseModel):
    """Source metadata stored on the inventory document.

    Records which tree inventory GDAM filled in, and its checksum at the time
    this inventory was created so the source can be detected as stale later.
    """

    name: Literal["gdam"] = "gdam"
    source_tree_inventory_id: str
    source_tree_inventory_checksum: str | None = Field(
        default=None,
        description=(
            "The source tree inventory's `checksum` at the time this inventory "
            "was created from it. Compare it against the source inventory's "
            "current `checksum` to tell whether the source has changed since."
        ),
    )


class CreateGdamInventoryRequest(CreateInventoryRequestBase):
    """Request body for creating an inventory via GDAM allometry imputation."""

    source_tree_inventory_id: str = Field(
        description=(
            "ID of a completed tree inventory whose missing morphology columns "
            "(dbh, crown ratio, species) GDAM will fill in. Existing values are "
            "preserved; only missing cells are imputed."
        ),
    )
