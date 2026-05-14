"""
api/v2/resources/grids/providers/threedep.py

Shared base model for 3DEP (3D Elevation Program) data products.

3DEP provides elevation data at 1m, 10m, and 30m resolution from the USGS.
Product-specific subclasses (Topography) live in their respective product
directories and inherit from ThreeDepSource.
"""

from typing import Literal

from pydantic import BaseModel, Field

from api.resources.grids.alignment import (
    GridAlignmentDomainTarget,
    GridAlignmentSpecification,
)


class ThreeDepSource(BaseModel):
    """Base source specification for 3DEP data products.

    `source_resolution` selects the source product (1, 10, or 30 m). The
    output cell size is controlled by ``alignment.resolution``.

    `extent_buffer_cells` records the user-requested output buffer. Griddle may
    fetch additional internal cells for derivative calculations, but those
    cells are clipped away before the grid is stored.
    """

    name: Literal["3dep"] = "3dep"
    product: str
    source_resolution: int
    description: str = ""
    extent_buffer_cells: int = Field(0, ge=0, le=10)
    alignment: GridAlignmentSpecification = Field(
        default_factory=GridAlignmentDomainTarget
    )
