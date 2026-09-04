"""
api/v2/resources/grids/disturbance/annual/schema.py

Schema models for the LANDFIRE Limited Annual Disturbance grid product.

Limited Annual Disturbance (LDist) returns categorical codes describing
recent vegetation disturbances (e.g. fire, insects, disease) at 30m resolution.
Always fetched on demand from LANDFIRE Product Service.
"""

from enum import StrEnum
from typing import Literal

from pydantic import Field

from api.resources.grids.providers.landfire import LandfireSource
from api.resources.grids.schema import Band, BandType, CreateSourceGridRequestBase
from lib.landfire import LANDFIRE_VERSIONS

LandfireDisturbanceVersion = StrEnum(
    "LandfireDisturbanceVersion",
    {
        f"v{version}": version
        for version in LANDFIRE_VERSIONS["annual_disturbance"]["lfps_available"]
    },
)


class LandfireDisturbanceSource(LandfireSource):
    """Source for LANDFIRE Limited Annual Disturbance (LDist).

    Returns categorical disturbance codes at 30m resolution, always fetched
    on demand from LANDFIRE Product Service.
    """

    product: Literal["annual_disturbance"] = "annual_disturbance"
    description: Literal["LANDFIRE Limited Annual Disturbance codes"] = (
        "LANDFIRE Limited Annual Disturbance codes"
    )


class CreateLandfireDisturbanceRequest(CreateSourceGridRequestBase):
    """Request to create a grid from LANDFIRE Limited Annual Disturbance.

    Returns a single-band grid with categorical disturbance codes. Always
    fetched on demand from LANDFIRE Product Service.
    """

    version: LandfireDisturbanceVersion = Field(
        default=LandfireDisturbanceVersion(
            LANDFIRE_VERSIONS["annual_disturbance"]["default"]
        ),
        description=(
            "LANDFIRE version, fetched on demand from LANDFIRE Product "
            "Service -- this product has no staged national release."
        ),
    )


DISTURBANCE_BAND = Band(
    key="annual_disturbance",
    name="Limited Annual Disturbance Code",
    description=(
        "LANDFIRE Limited Annual Disturbance (LDist) code, describing the "
        "type of vegetation disturbance detected within the past year "
        "(e.g. fire, insects, disease)."
    ),
    type=BandType.categorical,
    unit=None,
    index=0,
)
