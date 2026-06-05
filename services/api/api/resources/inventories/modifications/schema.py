"""
api/v2/resources/inventories/modifications/schema.py

Request schema for the in-place inventory modifications endpoint.
"""

from pydantic import BaseModel, Field

from api.resources.inventories.modification_models import InventoryModification


class ApplyModificationsRequest(BaseModel):
    """Request body for applying modifications to an inventory in place.

    Metadata (name, description, tags) is not accepted here — the inventory
    keeps its identity; use PATCH to edit metadata.
    """

    modifications: list[InventoryModification] = Field(
        ...,
        min_length=1,
        description="Modifications to append to this inventory and apply to its data.",
    )
