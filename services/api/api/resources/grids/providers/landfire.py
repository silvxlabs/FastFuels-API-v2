"""
api/v2/resources/grids/providers/landfire.py

Shared base model for LANDFIRE data products.

LANDFIRE provides raster products at 30m resolution. Product-specific
subclasses (FBFM40, Topography, etc.) live in their respective product
directories and inherit from LandfireSource.
"""

from enum import StrEnum
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


class NonBurnableFuelModel(StrEnum):
    """Non-burnable LANDFIRE fuel model codes, shared across FBFM40/FBFM13."""

    NB1 = "NB1"  # Urban/developed (91)
    NB2 = "NB2"  # Snow/ice (92)
    NB3 = "NB3"  # Agriculture (93)
    NB8 = "NB8"  # Water (98)
    NB9 = "NB9"  # Bare ground (99)


NB_CODE_MAP: dict[str, int] = {
    "NB1": 91,
    "NB2": 92,
    "NB3": 93,
    "NB8": 98,
    "NB9": 99,
}


def check_no_duplicate_non_burnable(v):
    """Shared validator for remove_non_burnable fields."""
    if v is not None and len(v) != len(set(v)):
        raise ValueError("Duplicate non-burnable fuel model codes are not allowed")
    return v
