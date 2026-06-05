"""
api/v2/resources/inventories/treatments/schema.py

Request schema for the in-place inventory treatments endpoint.
"""

from pydantic import BaseModel, Field

from api.resources.inventories.treatment_models import InventoryTreatment


class ApplyTreatmentsRequest(BaseModel):
    """Request body for applying treatments to an inventory in place.

    Metadata (name, description, tags) is not accepted here — the inventory
    keeps its identity; use PATCH to edit metadata.
    """

    treatments: list[InventoryTreatment] = Field(
        ...,
        min_length=1,
        description="Treatments to append to this inventory and apply to its data.",
    )
