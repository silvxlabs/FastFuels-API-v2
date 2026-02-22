"""
api/v2/resources/grids/topography/schema.py

Schema models for the Topography grid product.

Topography data includes elevation, slope, and aspect. Currently sourced
from LANDFIRE at 30m resolution. Future sources include 3DEP (1m/10m/30m).
"""

from enum import StrEnum
from typing import Literal

from pydantic import Field

from api.resources.grids.providers.landfire import LandfireSource
from api.resources.grids.schema import Band, BandType, CreateGridRequestBase


class TopographyBand(StrEnum):
    """Available bands for topographic data."""

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
