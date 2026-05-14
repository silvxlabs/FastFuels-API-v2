"""
api/v2/resources/grids/providers/canopy.py

Shared base model for 2D canopy raster data products.

Canopy products provide rasters with one or more 2D canopy bands —
``chm`` (canopy height in meters), ``cbd`` (canopy bulk density),
``cbh`` (canopy base height), ``cc`` (canopy cover). A given source
exposes some subset of these bands. Product-specific subclasses (Meta,
NAIP, LANDFIRE, ...) live in their respective product directories and
inherit from CanopySource.
"""

from typing import Literal

from pydantic import BaseModel, Field

from api.resources.grids.alignment import (
    GridAlignmentDomainTarget,
    GridAlignmentSpecification,
)


class CanopySource(BaseModel):
    """Base source specification for canopy data products."""

    name: Literal["canopy"] = "canopy"
    product: str
    description: str = ""
    extent_buffer_cells: int = Field(0, ge=0, le=10)
    alignment: GridAlignmentSpecification = Field(
        default_factory=GridAlignmentDomainTarget
    )
