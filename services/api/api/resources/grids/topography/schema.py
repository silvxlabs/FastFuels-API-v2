"""
api/v2/resources/grids/topography/schema.py

Schema models for the Topography grid product.

Topography data includes elevation, slope, and aspect. Sourced from
LANDFIRE at 30m resolution or 3DEP at 1m/10m/30m resolution.
"""

from enum import IntEnum, StrEnum
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from api.resources.grids.providers.landfire import LandfireSource
from api.resources.grids.providers.threedep import ThreeDepSource
from api.resources.grids.schema import (
    Band,
    BandType,
    CreateGridRequestBase,
    TileMetadata,
    validate_no_duplicates,
)


class LandfireTopographyVersion(StrEnum):
    """Available LANDFIRE topography data versions."""

    v2020 = "2020"


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

    version: LandfireTopographyVersion = LandfireTopographyVersion.v2020
    bands: list[TopographyBand] = Field(
        default=[
            TopographyBand.elevation,
            TopographyBand.slope,
            TopographyBand.aspect,
        ],
        min_length=1,
    )

    @field_validator("bands")
    @classmethod
    def no_duplicate_bands(cls, v: list[TopographyBand]) -> list[TopographyBand]:
        return validate_no_duplicates(v)


class ThreeDepResolution(IntEnum):
    """Available resolutions for 3DEP data (meters)."""

    one_meter = 1
    ten_meter = 10
    thirty_meter = 30


class ThreeDepTopographySource(ThreeDepSource):
    """Source for 3DEP topographic data.

    Returns continuous elevation (meters), slope (degrees), and/or aspect
    (degrees) data at user-selected resolution (1m, 10m, or 30m) from USGS
    3DEP products accessed via AWS S3 COGs.
    """

    product: Literal["topography"] = "topography"
    bands: list[TopographyBand]
    description: Literal["3DEP topographic data (elevation, slope, aspect)"] = (
        "3DEP topographic data (elevation, slope, aspect)"
    )

    # Post-processing metadata populated by Griddle after processing
    tile_metadata: TileMetadata | None = None


class CreateThreeDepTopographyRequest(CreateGridRequestBase):
    """Request to create a grid from 3DEP topographic data.

    Returns a grid with one or more continuous bands: elevation (m),
    slope (degrees), and/or aspect (degrees). Resolution is user-selectable:
    1m, 10m (default), or 30m.
    """

    resolution: ThreeDepResolution = ThreeDepResolution.ten_meter
    bands: list[TopographyBand] = Field(
        default=[TopographyBand.elevation],
        min_length=1,
    )

    @field_validator("bands")
    @classmethod
    def no_duplicate_bands(cls, v: list[TopographyBand]) -> list[TopographyBand]:
        return validate_no_duplicates(v)


class ThreeDepCoverageResponse(BaseModel):
    """Response model for 3DEP tile coverage pre-flight check."""

    resolution: ThreeDepResolution
    available: bool
    tile_count: int
    tiles: list[str]
    acquisition_dates: list[str] | None = None


def build_topography_bands(requested: list[TopographyBand]) -> list[Band]:
    """Build Band objects for requested topography bands with correct indices."""
    return [
        Band(index=i, **TOPOGRAPHY_BAND_DEFS[band]) for i, band in enumerate(requested)
    ]
