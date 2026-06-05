"""
api/v2/resources/inventories/modifications/schema.py

Request and source schemas for the standalone modifications endpoint.
"""

from typing import Literal

from pydantic import BaseModel, Field

from api.resources.inventories.modification_models import InventoryModification
from api.resources.inventories.schema import CreateInventoryRequestBase


class ModificationsInventorySource(BaseModel):
    """Source metadata for an inventory created via modifications."""

    name: Literal["modifications"] = "modifications"
    source_inventory_id: str
    source_inventory_checksum: str | None = Field(
        default=None,
        description=(
            "The source inventory's `checksum` at the time this inventory was "
            "created from it. Compare it against the source inventory's current "
            "`checksum` to tell whether the source has changed since."
        ),
    )
    modifications: list[dict]  # serialized InventoryModification dicts


class ApplyModificationsRequest(CreateInventoryRequestBase):
    """Request body for creating a new inventory by applying modifications
    to an existing one."""

    modifications: list[InventoryModification] = Field(
        ...,
        min_length=1,
        description="Modifications to apply to the source inventory.",
    )
