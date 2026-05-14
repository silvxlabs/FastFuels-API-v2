"""
api/v2/resources/grids/providers/landfire.py

Shared base model for LANDFIRE data products.

LANDFIRE provides raster products at 30m resolution. Product-specific
subclasses (FBFM40, Topography, etc.) live in their respective product
directories and inherit from LandfireSource.
"""

from typing import Literal

from pydantic import BaseModel, Field

from api.resources.grids.alignment import (
    GridAlignmentDomainTarget,
    GridAlignmentSpecification,
)


class LandfireSource(BaseModel):
    """Base source specification for LANDFIRE data products."""

    name: Literal["landfire"] = "landfire"
    product: str
    version: str
    description: str = ""
    extent_buffer_cells: int = Field(0, ge=0, le=10)
    alignment: GridAlignmentSpecification = Field(
        default_factory=GridAlignmentDomainTarget
    )
