"""
api/v2/resources/grids/fbfm13/schema.py

Schema models for the FBFM13 grid product.

FBFM13 returns categorical Anderson 13 fuel model codes at 30m
resolution from LANDFIRE. To convert codes to fuel parameters (fuel
loads, SAV, depth, moisture of extinction), use the
/grids/lookup/fbfm13 endpoint.
"""

from enum import StrEnum
from typing import Literal

from pydantic import field_validator

from api.resources.grids.providers.landfire import (
    LandfireSource,
    NonBurnableFuelModel,
    check_no_duplicate_non_burnable,
)
from api.resources.grids.schema import Band, BandType, CreateSourceGridRequestBase


class LandfireFbfm13Version(StrEnum):
    """Available LANDFIRE FBFM13 data versions."""

    v2023 = "2023"
    v2024 = "2024"


class LandfireFbfm13Source(LandfireSource):
    """Source for LANDFIRE FBFM13 (Anderson 13 Fire Behavior Fuel Model).

    Returns categorical fuel model codes at 30m resolution. The codes
    correspond to Anderson 13 fuel model classifications.
    """

    product: Literal["fbfm13"] = "fbfm13"
    description: Literal[
        "LANDFIRE FBFM13 fuel model codes (Anderson 13 classification)"
    ] = "LANDFIRE FBFM13 fuel model codes (Anderson 13 classification)"
    remove_non_burnable: list[str] | None = None


class CreateLandfireFbfm13Request(CreateSourceGridRequestBase):
    """Request to create a grid from LANDFIRE FBFM13.

    Returns a single-band grid with categorical fuel model codes.
    To convert codes to fuel parameters, use /grids/lookup/fbfm13.
    """

    version: LandfireFbfm13Version = LandfireFbfm13Version.v2024
    remove_non_burnable: list[NonBurnableFuelModel] | None = None

    @field_validator("remove_non_burnable")
    @classmethod
    def check_no_duplicates(cls, v):
        return check_no_duplicate_non_burnable(v)


FBFM13_BAND = Band(
    key="fbfm13",
    name="Anderson 13 Fuel Model",
    description=(
        "Anderson 13 fire behavior fuel model code (1-13). "
        "Convert to fuel parameters via /grids/lookup/fbfm13."
    ),
    type=BandType.categorical,
    unit=None,
    index=0,
)
