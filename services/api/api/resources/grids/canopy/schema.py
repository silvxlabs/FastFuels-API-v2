"""
api/v2/resources/grids/canopy/schema.py

Schema models for Meta and NAIP canopy grid products.

Both sources produce a single ``chm`` (canopy height in meters) band under
the canopy product family. Future canopy sources (e.g. LANDFIRE) will add
``cbd``/``cbh``/``cc`` bands alongside ``chm``.
"""

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel

from api.resources.grids.providers.canopy import CanopySource
from api.resources.grids.schema import (
    Band,
    BandType,
    CreateSourceGridRequestBase,
    TileMetadata,
)

CHM_BAND = Band(key="chm", type=BandType.continuous, unit="m", index=0)


class MetaCHMVersion(StrEnum):
    """Available Meta CHM data versions."""

    v1 = "1"
    v2 = "2"


class Attribution(BaseModel):
    """License and citation metadata for data compliance."""

    license_name: str
    license_url: str
    citation: str
    access_url: str
    accessed_on: str


class MetaChmSource(CanopySource):
    """Source for Meta global canopy height data.

    Returns a continuous canopy height raster at ~1m resolution. Each pixel
    contains the estimated canopy height in meters.
    """

    product: Literal["meta"] = "meta"
    description: Literal["Meta global canopy height model at ~1m resolution"] = (
        "Meta global canopy height model at ~1m resolution"
    )
    version: MetaCHMVersion

    # Post-processing metadata populated by Griddle after processing
    tile_metadata: TileMetadata | None = None
    attribution: Attribution | None = None


class CreateMetaChmRequest(CreateSourceGridRequestBase):
    """Request to create a grid from Meta CHM.

    Returns a grid with a single continuous band:
    - chm: Canopy height in meters
    """

    version: MetaCHMVersion = MetaCHMVersion.v2


def build_chm_bands() -> list[Band]:
    """Build Band objects for CHM. Always returns a single band."""
    return [CHM_BAND]


class NaipChmSource(CanopySource):
    """Source for NAIP high-resolution canopy height data.

    Returns a continuous canopy height raster at ~0.6m resolution (CONUS).
    """

    product: Literal["naip"] = "naip"
    description: Literal[
        "NAIP high-resolution canopy height model at ~0.6m resolution (CONUS)"
    ] = "NAIP high-resolution canopy height model at ~0.6m resolution (CONUS)"

    # Post-processing metadata populated by Griddle after processing
    tile_metadata: TileMetadata | None = None


class CreateNaipChmRequest(CreateSourceGridRequestBase):
    """Request to create a grid from NAIP CHM.

    Returns a grid with a single continuous band:
    - chm: Canopy height in meters
    """
