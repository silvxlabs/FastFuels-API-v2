"""
api/v2/resources/grids/topography/schema.py

Schema models for the Topography grid product.

Topography data includes elevation, slope, and aspect. Sourced from
LANDFIRE at 30m resolution or 3DEP at 1m/2m/10m/30m resolution.
"""

from enum import IntEnum, StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from api.resources.band_metadata import BandEnum, publish_band_defs
from api.resources.grids.providers.landfire import LandfireSource
from api.resources.grids.providers.threedep import ThreeDepSource
from api.resources.grids.schema import (
    Band,
    BandType,
    CreateSourceGridRequestBase,
    TileMetadata,
    validate_no_duplicates,
)


class LandfireTopographyVersion(StrEnum):
    """Available LANDFIRE topography data versions."""

    v2020 = "2020"


class TopographyBand(BandEnum):
    """Available bands for topographic data."""

    elevation = "elevation"
    slope = "slope"
    aspect = "aspect"


TOPOGRAPHY_BAND_DEFS = {
    TopographyBand.elevation: {
        "key": "elevation",
        "name": "Elevation",
        "description": "Height above sea level.",
        "type": BandType.continuous,
        "unit": "m",
    },
    TopographyBand.slope: {
        "key": "slope",
        "name": "Slope",
        "description": "Terrain slope (0-90 degrees).",
        "type": BandType.continuous,
        "unit": "deg",
    },
    TopographyBand.aspect: {
        "key": "aspect",
        "name": "Aspect",
        "description": "Downslope compass direction, degrees clockwise from north (0-360). Flat cells are 0.",
        "type": BandType.continuous,
        "unit": "deg",
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


class CreateLandfireTopographyRequest(CreateSourceGridRequestBase):
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
    """3DEP source products, by resolution in meters.

    - 1: Seamless 1-Meter DEM (S1M); where S1M has no tiles, the 1 m lidar
      project DEM tiles
    - 2: the 1 m lidar project DEM tiles, read at their 2 m overview
    - 10: 1/3 arc-second seamless DEM
    - 30: 1 arc-second seamless DEM
    """

    one_meter = 1
    two_meter = 2
    ten_meter = 10
    thirty_meter = 30


THREEDEP_RESOLUTION_DESCRIPTION = (
    "3DEP source product, in meters. 1: Seamless 1-Meter DEM (S1M), falling "
    "back to the 1 m lidar project DEM tiles where S1M has no tiles. 2: the "
    "1 m lidar project DEM tiles, read at their 2 m overview. 10: 1/3 "
    "arc-second seamless DEM. 30: 1 arc-second seamless DEM."
)


class ThreeDepTopographySource(ThreeDepSource):
    """Source for 3DEP topographic data.

    Returns continuous elevation (meters), slope (degrees), and/or aspect
    (degrees) data at user-selected resolution (1m, 2m, 10m, or 30m) from
    USGS 3DEP products accessed via AWS S3 COGs.
    """

    product: Literal["topography"] = "topography"
    bands: list[TopographyBand]
    description: Literal["3DEP topographic data (elevation, slope, aspect)"] = (
        "3DEP topographic data (elevation, slope, aspect)"
    )

    # Post-processing metadata populated by Griddle after processing
    tile_metadata: TileMetadata | None = None


class CreateThreeDepTopographyRequest(CreateSourceGridRequestBase):
    """Request to create a grid from 3DEP topographic data.

    Returns a grid with one or more continuous bands: elevation (m),
    slope (degrees), and/or aspect (degrees).

    `source_resolution` selects the 3DEP product (1m, 2m, 10m, or 30m).
    To change the *output* cell size, set ``alignment.resolution`` instead.
    """

    source_resolution: ThreeDepResolution = Field(
        default=ThreeDepResolution.ten_meter,
        description=THREEDEP_RESOLUTION_DESCRIPTION,
    )
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

    model_config = ConfigDict(title="TopographyThreeDepCoverageResponse")

    resolution: ThreeDepResolution = Field(description=THREEDEP_RESOLUTION_DESCRIPTION)
    available: bool | None = Field(
        description=(
            "Whether tiles cover the domain. null means coverage could not be "
            "determined because the USGS National Map search did not respond; "
            "try again shortly."
        )
    )
    tile_count: int
    tiles: list[str]
    acquisition_dates: list[str] | None = None
    unavailable_reason: str | None = Field(
        default=None,
        description="Why coverage could not be determined. Set only when "
        "`available` is null.",
    )


publish_band_defs(TopographyBand, TOPOGRAPHY_BAND_DEFS)


def build_topography_bands(requested: list[TopographyBand]) -> list[Band]:
    """Build Band objects for requested topography bands with correct indices."""
    return [
        Band(index=i, **TOPOGRAPHY_BAND_DEFS[band]) for i, band in enumerate(requested)
    ]
