"""
api/v2/resources/grids/fbfm40/schema.py

Schema models for the FBFM40 grid product.

FBFM40 returns categorical Scott-Burgan 40 fuel model codes at 30m
resolution from LANDFIRE. To convert codes to fuel parameters (fuel loads,
SAV, depth), use the /grids/lookup/fbfm40 endpoint.
"""

from enum import StrEnum
from typing import Literal

from pydantic import field_validator

from api.resources.grids.providers.landfire import LandfireSource
from api.resources.grids.schema import Band, BandType, CreateGridRequestBase


class LandfireFbfm40Version(StrEnum):
    """Available LANDFIRE FBFM40 data versions."""

    v2019 = "2019"
    v2020 = "2020"
    v2022 = "2022"
    v2023 = "2023"


class NonBurnableFuelModel(StrEnum):
    """Non-burnable Scott-Burgan 40 fuel model codes."""

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


class LandfireFbfm40Source(LandfireSource):
    """Source for LANDFIRE FBFM40 (Fire Behavior Fuel Model 40).

    Returns categorical fuel model codes at 30m resolution. The codes
    correspond to Scott-Burgan 40 fuel model classifications.
    """

    product: Literal["fbfm40"] = "fbfm40"
    description: Literal[
        "LANDFIRE FBFM40 fuel model codes (Scott-Burgan 40 classification)"
    ] = "LANDFIRE FBFM40 fuel model codes (Scott-Burgan 40 classification)"
    remove_non_burnable: list[str] | None = None


class CreateLandfireFbfm40Request(CreateGridRequestBase):
    """Request to create a grid from LANDFIRE FBFM40.

    Returns a single-band grid with categorical fuel model codes.
    To convert codes to fuel parameters, use /grids/lookup/fbfm40.
    """

    version: LandfireFbfm40Version = LandfireFbfm40Version.v2022
    remove_non_burnable: list[NonBurnableFuelModel] | None = None

    @field_validator("remove_non_burnable")
    @classmethod
    def check_no_duplicates(cls, v):
        if v is not None and len(v) != len(set(v)):
            raise ValueError("Duplicate non-burnable fuel model codes are not allowed")
        return v


FBFM40_BAND = Band(key="fbfm", type=BandType.categorical, unit=None, index=0)
