"""
api/v2/resources/grids/chm/schema.py

Schema models for the Meta CHM grid product.

Meta provides a global canopy height model at ~1m resolution from satellite
imagery. Single band: canopy height in meters.
"""

from enum import StrEnum
from typing import Literal

from api.resources.grids.providers.chm import ChmSource
from api.resources.grids.schema import Band, BandType, CreateGridRequestBase


class MetaChmVersion(StrEnum):
    """Available Meta CHM data versions."""

    v2024 = "2024"


CHM_BAND = Band(key="chm", type=BandType.continuous, unit="m", index=0)


class MetaChmSource(ChmSource):
    """Source for Meta global canopy height data.

    Returns a continuous canopy height raster at ~1m resolution. Each pixel
    contains the estimated canopy height in meters.
    """

    product: Literal["meta"] = "meta"
    version: MetaChmVersion
    description: Literal["Meta global canopy height model at ~1m resolution"] = (
        "Meta global canopy height model at ~1m resolution"
    )


class CreateMetaChmRequest(CreateGridRequestBase):
    """Request to create a grid from Meta CHM.

    Returns a grid with a single continuous band:
    - chm: Canopy height in meters
    """

    version: MetaChmVersion = MetaChmVersion.v2024


def build_chm_bands() -> list[Band]:
    """Build Band objects for CHM. Always returns a single band."""
    return [CHM_BAND]
