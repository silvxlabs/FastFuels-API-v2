"""
api/v2/resources/grids/chm/schema.py

Schema models for the Meta CHM grid product.

Meta provides a global canopy height model at ~1m resolution from satellite
imagery. Single band: canopy height in meters.
"""

from typing import Literal

from pydantic import BaseModel

from api.resources.grids.providers.chm import ChmSource
from api.resources.grids.schema import (
    Band,
    BandType,
    CreateGridRequestBase,
    TileMetadata,
)


CHM_BAND = Band(key="chm", type=BandType.continuous, unit="m", index=0)


class Attribution(BaseModel):
    """License and citation metadata for data compliance."""

    license_name: str
    license_url: str
    citation: str
    access_url: str
    accessed_on: str


class MetaChmSource(ChmSource):
    """Source for Meta global canopy height data.

    Returns a continuous canopy height raster at ~1m resolution. Each pixel
    contains the estimated canopy height in meters.
    """

    product: Literal["meta"] = "meta"
    description: Literal["Meta global canopy height model at ~1m resolution"] = (
        "Meta global canopy height model at ~1m resolution"
    )

    # Post-processing metadata populated by Griddle after processing
    tile_metadata: TileMetadata | None = None
    attribution: Attribution | None = None


class CreateMetaChmRequest(CreateGridRequestBase):
    """Request to create a grid from Meta CHM.

    Returns a grid with a single continuous band:
    - chm: Canopy height in meters
    """


def build_chm_bands() -> list[Band]:
    """Build Band objects for CHM. Always returns a single band."""
    return [CHM_BAND]


class NaipChmSource(ChmSource):
    """Source for NAIP high-resolution canopy height data.

    Returns a continuous canopy height raster at ~0.6m resolution (CONUS).
    """

    product: Literal["naip"] = "naip"
    description: Literal[
        "NAIP high-resolution canopy height model at ~0.6m resolution (CONUS)"
    ] = "NAIP high-resolution canopy height model at ~0.6m resolution (CONUS)"

    # Post-processing metadata populated by Griddle after processing
    tile_metadata: TileMetadata | None = None


class CreateNaipChmRequest(CreateGridRequestBase):
    """Request to create a grid from NAIP CHM.

    Returns a grid with a single continuous band:
    - chm: Canopy height in meters
    """
