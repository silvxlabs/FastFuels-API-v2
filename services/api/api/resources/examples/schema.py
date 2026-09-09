"""
api/resources/examples/schema.py

Response models for GET /examples — the discovery surface for the shared,
prebuilt example (#582). The client reads ids from here instead of hardcoding
them, then deep-links to the domain and grid through the normal read endpoints.
"""

from pydantic import BaseModel, Field


class ExampleGrid(BaseModel):
    """A completed, publicly-readable grid belonging to an example domain."""

    id: str = Field(..., description="The grid's unique identifier.")
    name: str = Field("", description="The grid's display name.")
    domain_id: str = Field(..., description="The example domain the grid belongs to.")


class Example(BaseModel):
    """One example domain and its publicly-readable grids."""

    domain_id: str = Field(..., description="The example domain's unique identifier.")
    name: str = Field("", description="The example domain's display name.")
    grids: list[ExampleGrid] = Field(
        default_factory=list,
        description="Completed grids on this domain, readable without ownership.",
    )


class ListExamplesResponse(BaseModel):
    """The set of shared examples any authenticated caller may read."""

    examples: list[Example] = Field(
        default_factory=list,
        description="Shared example domains and their readable grids.",
    )
