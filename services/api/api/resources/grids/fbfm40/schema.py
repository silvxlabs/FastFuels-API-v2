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

from api.resources.grids.providers.landfire import (
    LandfireSource,
    NonBurnableFuelModel,
    check_no_duplicate_non_burnable,
)
from api.resources.grids.schema import Band, BandType, CreateSourceGridRequestBase


class LandfireFbfm40Version(StrEnum):
    """Available LANDFIRE FBFM40 data versions."""

    v2019 = "2019"
    v2020 = "2020"
    v2022 = "2022"
    v2023 = "2023"
    v2024 = "2024"


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


class CreateLandfireFbfm40Request(CreateSourceGridRequestBase):
    """Request to create a grid from LANDFIRE FBFM40.

    Returns a single-band grid with categorical fuel model codes.
    To convert codes to fuel parameters, use /grids/lookup/fbfm40.
    """

    version: LandfireFbfm40Version = LandfireFbfm40Version.v2024
    remove_non_burnable: list[NonBurnableFuelModel] | None = None

    @field_validator("remove_non_burnable")
    @classmethod
    def check_no_duplicates(cls, v):
        return check_no_duplicate_non_burnable(v)


FBFM40_BAND = Band(
    key="fbfm",
    name="Scott & Burgan 40 Fuel Model",
    description=(
        "Scott & Burgan 40 fire behavior fuel model code (e.g. GR1, TL3, SH5). "
        "Convert to fuel parameters via /grids/lookup/fbfm40."
    ),
    type=BandType.categorical,
    unit=None,
    index=0,
)
