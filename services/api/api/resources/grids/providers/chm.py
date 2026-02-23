"""
api/v2/resources/grids/providers/chm.py

Shared base model for CHM (Canopy Height Model) data products.

CHM products provide rasters where each pixel contains canopy height in meters.
Product-specific subclasses (Meta, NAIP-CHM, GEDI) live in their respective
product directories and inherit from ChmSource.
"""

from typing import Literal

from pydantic import BaseModel


class ChmSource(BaseModel):
    """Base source specification for CHM data products."""

    name: Literal["chm"] = "chm"
    product: str
    version: str
    description: str = ""
