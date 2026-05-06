"""
api/v2/resources/grids/providers/pim.py

Shared base model for PIM (Plot Imputation Map) data products.

PIM products provide rasters where each pixel contains a plot ID that maps
to FIA tree records. Product-specific subclasses (TreeMap, etc.) live in
their respective product directories and inherit from PimSource.
"""

from typing import Literal

from pydantic import BaseModel, Field


class PimSource(BaseModel):
    """Base source specification for PIM data products."""

    name: Literal["pim"] = "pim"
    product: str
    version: str
    description: str = ""
    extent_buffer_cells: int = Field(0, ge=0, le=10)
