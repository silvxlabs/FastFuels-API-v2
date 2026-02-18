"""
api/v2/resources/grids/threedep/schema.py

Schema models for USGS 3DEP grid source.
"""

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel

from api.resources.grids.schema import CreateGridRequestBase


class ThreeDepResolution(StrEnum):
    """Native resolution options for 3DEP data."""

    one_meter = "1m"
    ten_meter = "10m"
    thirty_meter = "30m"


class ThreeDepSource(BaseModel):
    """Source specification for USGS 3DEP elevation data."""

    name: Literal["3dep"] = "3dep"
    resolution: ThreeDepResolution
    version: str


class ThreeDepQuantity(StrEnum):
    """Quantities available from 3DEP."""

    elevation = "elevation"
    slope = "slope"
    aspect = "aspect"


class CreateThreeDepRequest(CreateGridRequestBase):
    """Request to create a grid from USGS 3DEP."""

    quantities: list[ThreeDepQuantity]
    native_resolution: ThreeDepResolution = ThreeDepResolution.ten_meter
    version: str = "2023"
