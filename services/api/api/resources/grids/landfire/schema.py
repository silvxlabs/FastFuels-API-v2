"""
api/v2/resources/grids/landfire/schema.py

Schema models for LANDFIRE grid sources.

LANDFIRE provides raster products at 30m resolution. Currently supports:
- FBFM40: Fire Behavior Fuel Model codes (categorical)
- Topography: Elevation, slope, and aspect (continuous)

Note: FBFM40 returns only the fuel model codes. To convert codes to fuel
parameters (fuel loads, SAV, depth), use the /grids/lookup/fbfm40 endpoint.
"""

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field

from api.resources.grids.schema import Band, BandType, CreateGridRequestBase


class LandfireSource(BaseModel):
    """Base source specification for LANDFIRE data products."""

    name: Literal["landfire"] = "landfire"
    product: str
    version: str
    description: str = ""


class LandfireFbfm40Source(LandfireSource):
    """Source for LANDFIRE FBFM40 (Fire Behavior Fuel Model 40).

    Returns categorical fuel model codes at 30m resolution. The codes
    correspond to Scott-Burgan 40 fuel model classifications.
    """

    product: Literal["fbfm40"] = "fbfm40"
    description: Literal[
        "LANDFIRE FBFM40 fuel model codes (Scott-Burgan 40 classification)"
    ] = "LANDFIRE FBFM40 fuel model codes (Scott-Burgan 40 classification)"


class CreateLandfireFbfm40Request(CreateGridRequestBase):
    """Request to create a grid from LANDFIRE FBFM40.

    Returns a single-band grid with categorical fuel model codes.
    To convert codes to fuel parameters, use /grids/lookup/fbfm40.
    """

    version: str = "2022"


FBFM40_BAND = Band(key="fbfm", type=BandType.categorical, unit=None, index=0)


class TopographyBand(StrEnum):
    """Available bands for LANDFIRE topographic data."""

    elevation = "elevation"
    slope = "slope"
    aspect = "aspect"


TOPOGRAPHY_BAND_DEFS = {
    TopographyBand.elevation: {
        "key": "elevation",
        "type": BandType.continuous,
        "unit": "m",
    },
    TopographyBand.slope: {
        "key": "slope",
        "type": BandType.continuous,
        "unit": "degrees",
    },
    TopographyBand.aspect: {
        "key": "aspect",
        "type": BandType.continuous,
        "unit": "degrees",
    },
}


class LandfireTopographySource(LandfireSource):
    """Source for LANDFIRE topographic data.

    Returns continuous elevation (meters), slope (degrees), and/or aspect
    (degrees) data at 30m resolution from the LANDFIRE 3DEP-derived products.
    """

    product: Literal["topography"] = "topography"
    bands: list[TopographyBand]
    description: Literal["LANDFIRE topographic data (elevation, slope, aspect)"] = (
        "LANDFIRE topographic data (elevation, slope, aspect)"
    )


class CreateLandfireTopographyRequest(CreateGridRequestBase):
    """Request to create a grid from LANDFIRE topographic data.

    Returns a grid with one or more continuous bands: elevation (m),
    slope (degrees), and/or aspect (degrees).
    """

    version: str = "2020"
    bands: list[TopographyBand] = Field(
        default=[
            TopographyBand.elevation,
            TopographyBand.slope,
            TopographyBand.aspect,
        ],
        min_length=1,
    )


def build_topography_bands(requested: list[TopographyBand]) -> list[Band]:
    """Build Band objects for requested topography bands with correct indices."""
    return [
        Band(index=i, **TOPOGRAPHY_BAND_DEFS[band]) for i, band in enumerate(requested)
    ]
