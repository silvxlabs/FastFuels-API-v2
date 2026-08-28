"""
api/v2/resources/grids/fccs/schema.py

Schema models for the FCCS grid product.

FCCS returns categorical fuel classification system fuelbed IDs at 30m
resolution from LANDFIRE. To convert IDs to fuel parameters (fuel loads,
SAV, depth), use the /grids/lookup/fccs endpoint.
"""

from enum import StrEnum
from typing import Literal

from pydantic import Field

from api.resources.grids.providers.landfire import LandfireSource
from api.resources.grids.schema import Band, BandType, CreateSourceGridRequestBase
from lib.landfire import LANDFIRE_VERSIONS

# Build the version enum class from LANDFIRE_VERSIONS, combining both
# "available" (staged annual) and "lfps_available" (current-year, fetched
# on demand from LANDFIRE Product Service) versions.
_FCCS_ALL_VERSIONS = list(
    dict.fromkeys(
        LANDFIRE_VERSIONS["fccs"]["available"]
        + LANDFIRE_VERSIONS["fccs"]["lfps_available"]
    )
)
LandfireFccsVersion = StrEnum(
    "LandfireFccsVersion",
    {f"v{version}": version for version in _FCCS_ALL_VERSIONS},
)


class LandfireFccsSource(LandfireSource):
    """Source for LANDFIRE FCCS (Fuel Characteristic Classification System).

    Returns categorical fuelbed IDs at 30m resolution. The integer IDs
    correspond to Fuel Classification System fuelbeds.
    """

    product: Literal["fccs"] = "fccs"
    description: Literal["LANDFIRE FCCS fuelbed IDs"] = "LANDFIRE FCCS fuelbed IDs"
    remove_bare_ground: bool = False


class CreateLandfireFccsRequest(CreateSourceGridRequestBase):
    """Request to create a grid from LANDFIRE FCCS.

    Returns a single-band grid with categorical fuelbed IDs.
    To convert IDs to fuel parameters, use /grids/lookup/fccs.
    """

    version: LandfireFccsVersion = Field(
        default=LandfireFccsVersion(LANDFIRE_VERSIONS["fccs"]["default"]),
        description=(
            "LANDFIRE version. Most years are served from staged LANDFIRE "
            f"data; {', '.join(LANDFIRE_VERSIONS['fccs']['lfps_available'])} "
            "is instead fetched on demand from LANDFIRE Product Service."
        ),
    )
    remove_bare_ground: bool = False


FCCS_BAND = Band(
    key="fccs",
    name="FCCS Fuelbed ID",
    description=(
        "Fuel Characteristic Classification System (FCCS) fuelbed identifier. "
        "Convert to fuel parameters via /grids/lookup/fccs."
    ),
    type=BandType.categorical,
    unit=None,
    index=0,
)
