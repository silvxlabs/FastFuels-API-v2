"""
api/v2/resources/grids/duplicate/schema.py

Request schema for the grid duplicate endpoint.
"""

from pydantic import BaseModel, Field


class DuplicateGridRequest(BaseModel):
    """Optional metadata overrides for a duplicated grid.

    Every field is optional. Any field omitted is carried over verbatim from
    the source grid.
    """

    name: str | None = Field(
        None,
        max_length=255,
        description="Name for the copy. Omit to reuse the source grid's name.",
    )
    description: str | None = Field(
        None,
        max_length=2000,
        description=(
            "Description for the copy. Omit to reuse the source grid's description."
        ),
    )
    tags: list[str] | None = Field(
        None,
        max_length=50,
        description="Tags for the copy. Omit to reuse the source grid's tags.",
    )
