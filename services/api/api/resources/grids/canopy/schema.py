"""
api/v2/resources/grids/canopy/schema.py

Schema models for canopy grid products (Meta, NAIP, LANDFIRE).

Sources under the canopy product family produce some subset of the
shared 2D canopy band vocabulary: ``chm`` (canopy height in meters),
``cbd`` (canopy bulk density), ``cbh`` (canopy base height), ``cc``
(canopy cover). Meta and NAIP produce only ``chm``. LANDFIRE can
produce any combination of the four.
"""

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from api.resources.grids.providers.canopy import CanopySource
from api.resources.grids.schema import (
    Band,
    BandType,
    CreateSourceGridRequestBase,
    TileMetadata,
    validate_no_duplicates,
)


class LandfireCanopyFuelBand(StrEnum):
    """Selectable bands on the LANDFIRE canopy fuel source.

    These are the four LANDFIRE canopy fuel products: canopy height
    (``chm``), canopy bulk density (``cbd``), canopy base height
    (``cbh``), and canopy cover (``cc``). The ``chm`` member shares its
    band key with the Meta and NAIP CHM sources so downstream consumers
    that read a single ``chm`` band can use any of the three.
    """

    chm = "chm"
    cbd = "cbd"
    cbh = "cbh"
    cc = "cc"


LANDFIRE_CANOPY_BAND_DEFS = {
    LandfireCanopyFuelBand.chm: {
        "key": "chm",
        "name": "Canopy Height",
        "description": "Height of the canopy top above ground.",
        "type": BandType.continuous,
        "unit": "m",
    },
    LandfireCanopyFuelBand.cbd: {
        "key": "cbd",
        "name": "Canopy Bulk Density",
        "description": "Mass of available canopy fuel per unit canopy volume.",
        "type": BandType.continuous,
        "unit": "kg/m**3",
    },
    LandfireCanopyFuelBand.cbh: {
        "key": "cbh",
        "name": "Canopy Base Height",
        "description": "Height above ground of the base of the live crown.",
        "type": BandType.continuous,
        "unit": "m",
    },
    LandfireCanopyFuelBand.cc: {
        "key": "cc",
        "name": "Canopy Cover",
        "description": "Fraction of ground covered by tree canopy (%).",
        "type": BandType.continuous,
        "unit": "%",
    },
}


def build_landfire_canopy_bands(
    requested: list[LandfireCanopyFuelBand],
) -> list[Band]:
    """Build Band objects for requested LANDFIRE canopy bands with correct indices."""
    return [
        Band(index=i, **LANDFIRE_CANOPY_BAND_DEFS[band])
        for i, band in enumerate(requested)
    ]


CHM_BAND = Band(
    key="chm",
    name="Canopy Height",
    description="Height of the canopy top above ground.",
    type=BandType.continuous,
    unit="m",
    index=0,
)


class MetaCHMVersion(StrEnum):
    """Available Meta CHM data versions."""

    v1 = "1"
    v2 = "2"


class Attribution(BaseModel):
    """License and citation metadata for data compliance."""

    license_name: str
    license_url: str
    citation: str
    access_url: str
    accessed_on: str


class MetaChmSource(CanopySource):
    """Source for Meta global canopy height data.

    Returns a continuous canopy height raster at ~1m resolution. Each pixel
    contains the estimated canopy height in meters.
    """

    product: Literal["meta"] = "meta"
    description: Literal["Meta global canopy height model at ~1m resolution"] = (
        "Meta global canopy height model at ~1m resolution"
    )
    version: MetaCHMVersion

    # Post-processing metadata populated by Griddle after processing
    tile_metadata: TileMetadata | None = None
    attribution: Attribution | None = None


class CreateMetaChmRequest(CreateSourceGridRequestBase):
    """Request to create a grid from Meta CHM.

    Returns a grid with a single continuous band:
    - chm: Canopy height in meters
    """

    version: MetaCHMVersion = MetaCHMVersion.v2


def build_chm_bands() -> list[Band]:
    """Build Band objects for CHM. Always returns a single band."""
    return [CHM_BAND]


class NaipChmSource(CanopySource):
    """Source for NAIP high-resolution canopy height data.

    Returns a continuous canopy height raster at ~0.6m resolution (CONUS).
    """

    product: Literal["naip"] = "naip"
    description: Literal[
        "NAIP high-resolution canopy height model at ~0.6m resolution (CONUS)"
    ] = "NAIP high-resolution canopy height model at ~0.6m resolution (CONUS)"

    # Post-processing metadata populated by Griddle after processing
    tile_metadata: TileMetadata | None = None


class CreateNaipChmRequest(CreateSourceGridRequestBase):
    """Request to create a grid from NAIP CHM.

    Returns a grid with a single continuous band:
    - chm: Canopy height in meters
    """


class PointCloudGroundSource(StrEnum):
    """Where the ground surface under the canopy came from."""

    classification = "classification"
    derived = "derived"


class PointCloudGround(BaseModel):
    """How ground was established, and how well the data constrained it.

    Written by the worker. Canopy height is height *above ground*, so a CHM is
    only as good as the ground surface under it — and where a cloud carries no
    ground classification that surface is inferred. These fields are what make a
    weakly-constrained result visible instead of silently confident.
    """

    ground_source: PointCloudGroundSource = Field(
        description=(
            "`classification` when the point cloud's own ASPRS class 2 returns "
            "were used, `derived` when no ground classification was present and "
            "the surface was inferred from the data."
        )
    )
    ground_coverage: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Fraction of cells containing at least one ground return. Canopy "
            "heights over cells far from any ground return rest on an "
            "interpolated surface."
        ),
    )
    max_ground_distance_m: float = Field(
        ge=0.0,
        description=(
            "Distance from the cell furthest from any ground return to the "
            "nearest one, in meters. Large values indicate a wide area — a "
            "building footprint or closed canopy — where the ground surface is "
            "inferred rather than measured."
        ),
    )


class ChmSpikeFilter(BaseModel):
    """Removal of lone spurious returns from a point-cloud canopy height model.

    A cell takes the tallest return that falls in it, so one bad return — a
    bird, haze, a multiple-time-around artifact — becomes the cell's height
    unless the cloud classified it as noise. Many do not.

    Such a return leaves a shape real canopy cannot: a single cell towering
    over everything around it. Both fields are the two halves of that shape, in
    meters, so they mean the same thing at any `alignment.resolution`.
    """

    min_canopy_footprint_m: float = Field(
        default=3.0,
        gt=0,
        description=(
            "Narrowest ground footprint real canopy can occupy, in meters. A "
            "cell is judged against everything within this distance, and only "
            "a cell narrower than it can be rejected — so the filter does not "
            "run at all once `alignment.resolution` reaches this value, where "
            "one cell holds a stand rather than a crown."
        ),
    )
    min_prominence_m: float = Field(
        default=25.0,
        gt=0,
        description=(
            "How far above every neighbour a cell must rise to be rejected, in "
            "meters. Measured noise returns stood 40-80 m above their "
            "surroundings; a real crown's peak is within a few meters of the "
            "cells beside it."
        ),
    )


class PointCloudChmSource(CanopySource):
    """Source for a canopy height model rasterized from a point cloud.

    Produces a continuous canopy height raster by taking the greatest height
    above ground of any return in each cell.
    """

    product: Literal["point_cloud"] = "point_cloud"
    description: Literal["Canopy height model rasterized from a point cloud"] = (
        "Canopy height model rasterized from a point cloud"
    )
    source_point_cloud_id: str = Field(
        description="The point cloud this canopy height model was rasterized from."
    )
    source_point_cloud_checksum: str | None = Field(
        default=None,
        description=(
            "The source point cloud's `checksum` at the time this grid was "
            "created from it. Compare it against the point cloud's current "
            "`checksum` to tell whether the source has changed since."
        ),
    )

    ground: PointCloudGround | None = Field(
        default=None,
        description=(
            "How the ground surface beneath the canopy was established, and "
            "how well the point cloud constrained it. Recorded when the grid "
            "finishes processing; `null` before then."
        ),
    )
    spike_filter: ChmSpikeFilter | None = Field(
        default=None,
        description=(
            "Spurious-return removal this grid was built with, resolved from "
            "the request. `null` means the grid was built without it."
        ),
    )


class CreatePointCloudChmRequest(CreateSourceGridRequestBase):
    """Request to create a canopy height model grid from a point cloud.

    Returns a grid with a single continuous band:
    - chm: Canopy height in meters

    The point cloud must be airborne (`type: als`) and `completed`. Cell size
    comes from `alignment.resolution`, defaulting to 1 m — unlike the
    raster-backed canopy sources there is no source pixel size to inherit.
    """

    source_point_cloud_id: str = Field(
        description="ID of the point cloud to rasterize. Must be an ALS cloud in this domain."
    )
    spike_filter: ChmSpikeFilter | None = Field(
        default_factory=ChmSpikeFilter,
        description=(
            "Removal of lone spurious returns. Omit for the defaults; send "
            "`null` to keep every return, leaving unclassified noise in the "
            "grid."
        ),
    )


class LandfireCanopyVersion(StrEnum):
    """Available LANDFIRE canopy data versions."""

    v2024 = "2024"


class LandfireCanopySource(CanopySource):
    """Source for LANDFIRE canopy fuel data.

    Returns one or more continuous canopy bands at 30m resolution (CONUS)
    from the LANDFIRE program: ``chm`` (canopy height, m), ``cbd`` (canopy
    bulk density, kg/m**3), ``cbh`` (canopy base height, m), and ``cc``
    (canopy cover, %).
    """

    product: Literal["landfire"] = "landfire"
    version: LandfireCanopyVersion
    bands: list[LandfireCanopyFuelBand]
    description: Literal[
        "LANDFIRE canopy fuel data (chm, cbd, cbh, cc) at 30m resolution (CONUS)"
    ] = "LANDFIRE canopy fuel data (chm, cbd, cbh, cc) at 30m resolution (CONUS)"


class CreateLandfireCanopyRequest(CreateSourceGridRequestBase):
    """Request to create a grid from LANDFIRE canopy data.

    Returns a grid with one or more continuous canopy bands at 30m
    resolution (CONUS):
    - chm: Canopy height (m)
    - cbd: Canopy bulk density (kg/m**3)
    - cbh: Canopy base height (m)
    - cc:  Canopy cover (%)

    Bands are validated against the canopy band vocabulary and may not be
    duplicated.
    """

    version: LandfireCanopyVersion = LandfireCanopyVersion.v2024
    bands: list[LandfireCanopyFuelBand] = Field(
        default=[
            LandfireCanopyFuelBand.chm,
            LandfireCanopyFuelBand.cbd,
            LandfireCanopyFuelBand.cbh,
            LandfireCanopyFuelBand.cc,
        ],
        min_length=1,
    )

    @field_validator("bands")
    @classmethod
    def no_duplicate_bands(
        cls, v: list[LandfireCanopyFuelBand]
    ) -> list[LandfireCanopyFuelBand]:
        return validate_no_duplicates(v)
