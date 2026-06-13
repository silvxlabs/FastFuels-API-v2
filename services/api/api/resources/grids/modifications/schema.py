"""
api/v2/resources/grids/modifications/schema.py

Request schema for the in-place grid modifications endpoint.
"""

from pydantic import BaseModel, Field

from api.resources.grids.modification_models import GridModification


class ApplyGridModificationsRequest(BaseModel):
    """Request body for applying modifications to a grid in place.

    Metadata (name, description, tags) is not accepted here — the grid keeps
    its identity; use PATCH to edit metadata.
    """

    modifications: list[GridModification] = Field(
        ...,
        min_length=1,
        description="Modifications to append to this grid and apply to its data.",
    )
