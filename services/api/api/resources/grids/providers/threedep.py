"""
api/v2/resources/grids/providers/threedep.py

Shared base model for 3DEP (3D Elevation Program) data products.

3DEP provides elevation data at 1m, 10m, and 30m resolution from the USGS.
Product-specific subclasses (Topography) live in their respective product
directories and inherit from ThreeDepSource.
"""

from typing import Literal

from pydantic import BaseModel


class ThreeDepSource(BaseModel):
    """Base source specification for 3DEP data products."""

    name: Literal["3dep"] = "3dep"
    product: str
    resolution: int
    description: str = ""
