"""
api/v2/resources/grids/lookup/schema.py

Base schema for fuel model lookup operations.

Lookup endpoints convert categorical fuel model codes to continuous fuel
parameters using standard lookup tables. Each product (FBFM40, FBFM13,
FCCS, ...) defines its own source, request, and band schemas in its own
subpackage; this module holds only the shared base that every product
lookup source inherits from.
"""

from typing import Literal

from pydantic import BaseModel, Field


class LookupSource(BaseModel):
    """Base source for grids created via lookup table transformation.

    Converts categorical codes to continuous fuel parameters using
    standard lookup tables.
    """

    name: Literal["lookup"] = "lookup"
    table: str = Field(..., description="Lookup table identifier")
    source_grid_id: str = Field(..., description="Grid containing codes to look up")
    source_grid_checksum: str | None = Field(
        default=None,
        description=(
            "The source grid's `checksum` at the time this grid was created from "
            "it. Compare it against the source grid's current `checksum` to tell "
            "whether the source has changed since."
        ),
    )
    source_band: str = Field(..., description="Band in source grid containing codes")
