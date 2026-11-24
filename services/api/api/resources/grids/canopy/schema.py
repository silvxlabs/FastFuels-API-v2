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
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from api.resources.band_metadata import BandEnum, publish_band_defs
from api.resources.grids.alignment import (
    GridLatticeDomainTarget,
    GridLatticeSpecification,
)
from api.resources.grids.providers.canopy import CanopySource
from api.resources.grids.schema import (
    Band,
    BandType,
    CreateSourceGridRequestBase,
    TileMetadata,
    validate_no_duplicates,
)
from api.resources.input_requirements import input_requirements, point_cloud
from api.resources.point_clouds.schema import PointCloudType
from lib.landfire import LANDFIRE_VERSIONS
from lib.naip_chm import NaipChmCoverage as LibNaipChmCoverage


class LandfireCanopyFuelBand(BandEnum):
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
        "description": "Height above ground of the canopy top.",
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
        "description": "Height above ground of the base of the canopy fuel layer.",
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


publish_band_defs(LandfireCanopyFuelBand, LANDFIRE_CANOPY_BAND_DEFS)


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
    description="Height above ground of the canopy top.",
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


NAIP_CHM_FIRST_YEAR = 2007


class NaipChmTileMetadata(TileMetadata):
    """Tiles read for a NAIP CHM grid.

    The ``tile_*`` lists run parallel to ``tiles``: entry *i* describes tile *i*.
    """

    tile_years: list[int] | None = Field(
        default=None, description="NAIP acquisition year of each tile."
    )
    tile_source_gsds: list[float] | None = Field(
        default=None,
        description="Pixel size of the NAIP imagery each tile was derived from, in meters.",
    )
    tile_models: list[str] | None = Field(
        default=None, description="Model that produced each tile's canopy heights."
    )


class NaipChmSource(CanopySource):
    """Source for NAIP high-resolution canopy height data.

    Returns a continuous canopy height raster at ~0.6m resolution (CONUS)
    from the NAIP-CHM v1 dataset.
    """

    product: Literal["naip"] = "naip"
    description: Literal[
        "NAIP high-resolution canopy height model at ~0.6m resolution (CONUS)"
    ] = "NAIP high-resolution canopy height model at ~0.6m resolution (CONUS)"
    year: int | None = Field(
        default=None,
        description=(
            "NAIP year the grid was built from: the requested year. When no "
            "year was requested it is null until the grid finishes "
            "processing, then holds the latest year that fully covers the "
            "domain, or stays null when no single year did and tiles from "
            "several years were combined; `tile_metadata.tile_years` lists "
            "each tile's year."
        ),
    )

    # Post-processing metadata populated by Griddle after processing
    tile_metadata: NaipChmTileMetadata | None = None


class CreateNaipChmRequest(CreateSourceGridRequestBase):
    """Request to create a grid from NAIP CHM.

    Returns a grid with a single continuous band:
    - chm: Canopy height in meters
    """

    year: int | None = Field(
        default=None,
        ge=NAIP_CHM_FIRST_YEAR,
        description=(
            "NAIP year to build the grid from. Omit it to use the latest year "
            "that fully covers the domain; if no single year does, each part "
            "of the domain uses the latest year available there. With a year, "
            "only that year's tiles are used: parts of the domain it doesn't "
            "cover are nodata, and a year that covers none of the domain is "
            "rejected with 422. `GET /domains/{domain_id}/grids/canopy/naip/coverage` "
            "lists each year's coverage of the domain."
        ),
    )


class NaipChmCoverage(StrEnum):
    """How much of a domain a NAIP year covers."""

    full = "full"
    partial = "partial"
    none = "none"


class NaipChmCreateLink(BaseModel):
    """The create request that builds a grid from one NAIP year."""

    href: str = Field(
        description="Path of the create endpoint on the API host, including `/v2`."
    )
    method: Literal["POST"] = "POST"
    body: dict = Field(description="Request body selecting this year.")


class NaipChmYearLinks(BaseModel):
    """Actions available for a NAIP year on this domain."""

    create: NaipChmCreateLink | None = Field(
        description=(
            "Request that creates a grid from this year. Null when the year "
            "doesn't cover the domain, so the create would be rejected."
        )
    )


class NaipChmYearCoverageResponse(BaseModel):
    """One NAIP year and its coverage of the domain."""

    year: int = Field(description="NAIP acquisition year.")
    coverage: NaipChmCoverage = Field(
        description=(
            "`full` or `partial` years can be created for this domain; a "
            "`partial` year leaves the uncovered part of the domain nodata. "
            "`none` years don't cover the domain."
        )
    )
    percent: float = Field(
        ge=0, le=100, description="Share of the domain this year covers, in percent."
    )
    links: NaipChmYearLinks


class NaipChmCoverageResponse(BaseModel):
    """Response model for the NAIP CHM coverage pre-flight check."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "product": "naip",
                    "default": 2023,
                    "years": [
                        {
                            "year": 2025,
                            "coverage": "partial",
                            "percent": 62.76,
                            "links": {
                                "create": {
                                    "href": "/v2/domains/abc123/grids/canopy/naip",
                                    "method": "POST",
                                    "body": {"year": 2025},
                                }
                            },
                        },
                        {
                            "year": 2023,
                            "coverage": "full",
                            "percent": 100.0,
                            "links": {
                                "create": {
                                    "href": "/v2/domains/abc123/grids/canopy/naip",
                                    "method": "POST",
                                    "body": {"year": 2023},
                                }
                            },
                        },
                        {
                            "year": 2022,
                            "coverage": "none",
                            "percent": 0.0,
                            "links": {"create": None},
                        },
                    ],
                }
            ]
        }
    )

    product: Literal["naip"] = "naip"
    default: int | None = Field(
        description=(
            "Year a create request without `year` uses: the latest year that "
            "fully covers the domain. Null when no single year does; such a "
            "request then uses the latest year available in each part of the "
            "domain."
        )
    )
    years: list[NaipChmYearCoverageResponse] = Field(
        description="Every NAIP year in the dataset, newest first."
    )


def build_naip_chm_coverage_response(
    coverage: LibNaipChmCoverage, create_href: str
) -> NaipChmCoverageResponse:
    """Build the coverage response, attaching a create link to each creatable year."""
    years = []
    for item in coverage.years:
        status = NaipChmCoverage(item.coverage)
        create = None
        if status is not NaipChmCoverage.none:
            create = NaipChmCreateLink(href=create_href, body={"year": item.year})
        years.append(
            NaipChmYearCoverageResponse(
                year=item.year,
                coverage=status,
                percent=round(item.percent, 2),
                links=NaipChmYearLinks(create=create),
            )
        )
    return NaipChmCoverageResponse(default=coverage.default_year, years=years)


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

    Under the default `max` aggregation a cell takes the tallest return that
    falls in it, so one bad return — a bird, haze, a multiple-time-around
    artifact — becomes the cell's height unless the cloud classified it as
    noise. Many do not.

    Such a return leaves a shape real canopy cannot: a single cell towering
    over everything around it. Both fields are the two halves of that shape, in
    meters, so they mean the same thing at any `alignment.resolution`. A `mean`,
    `median`, or `percentile` aggregation already resists a lone return, so this
    filter is aimed at the `max` case.
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


class ChmMaxAggregation(BaseModel):
    """Tallest return in the cell.

    The height of whatever the cell's tallest thing is, which is a crown at a
    cell narrower than one and a stand's tallest tree at a cell wider than one.
    """

    model_config = ConfigDict(extra="forbid")

    method: Literal["max"] = "max"


class ChmMeanAggregation(BaseModel):
    """Average height of every return in the cell.

    Weighted by how the returns fall, so a cell that is half canopy and half
    gap reads between the two rather than at the canopy.
    """

    model_config = ConfigDict(extra="forbid")

    method: Literal["mean"] = "mean"


class ChmMedianAggregation(BaseModel):
    """Height half the cell's returns lie below.

    The same statistic as `percentile` at 50, spelled the way it is usually
    asked for.
    """

    model_config = ConfigDict(extra="forbid")

    method: Literal["median"] = "median"


class ChmPercentileAggregation(BaseModel):
    """Height a given fraction of the cell's returns lie below."""

    model_config = ConfigDict(extra="forbid")

    method: Literal["percentile"] = "percentile"
    percentile: float = Field(
        gt=0,
        le=100,
        description=(
            "Rank to take, as a percentage of the cell's returns. 100 is the "
            "tallest return and 50 the median. Between two returns the height "
            "is interpolated linearly."
        ),
    )


ChmAggregation = Annotated[
    ChmMaxAggregation
    | ChmMeanAggregation
    | ChmMedianAggregation
    | ChmPercentileAggregation,
    Field(discriminator="method"),
]


class PointCloudChmSource(CanopySource):
    """Source for a canopy height model rasterized from a point cloud.

    Produces a continuous canopy height raster by reducing the heights above
    ground of the returns in each cell to one value.
    """

    product: Literal["point_cloud"] = "point_cloud"
    description: Literal["Canopy height model rasterized from a point cloud"] = (
        "Canopy height model rasterized from a point cloud"
    )
    alignment: GridLatticeSpecification = Field(default_factory=GridLatticeDomainTarget)
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
    aggregation: ChmAggregation = Field(
        default_factory=ChmMaxAggregation,
        description=(
            "Statistic this grid's cells reduce their returns with, resolved "
            "from the request."
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

    model_config = ConfigDict(
        json_schema_extra=input_requirements(
            point_cloud("source_point_cloud_id", type=PointCloudType.als.value)
        ),
    )

    source_point_cloud_id: str = Field(
        description="ID of the point cloud to rasterize. Must be an ALS cloud in this domain."
    )
    alignment: GridLatticeSpecification = Field(
        default_factory=GridLatticeDomainTarget,
        description=(
            "Output lattice. Against the domain (`target: 'domain'`, the "
            "default) `resolution` defaults to 1 m — a point cloud has no "
            "native cell size to inherit. Against another grid "
            "(`target: 'grid'`) omitting `resolution` matches that grid "
            "cell-for-cell; the target grid must be in this domain's CRS."
        ),
    )
    spike_filter: ChmSpikeFilter | None = Field(
        default_factory=ChmSpikeFilter,
        description=(
            "Removal of lone spurious returns. Omit for the defaults; send "
            "`null` to keep every return, leaving unclassified noise in the "
            "grid."
        ),
    )
    aggregation: ChmAggregation = Field(
        default_factory=ChmMaxAggregation,
        description=(
            "Statistic each cell reduces the heights above ground of its "
            "returns with. Carries `percentile` only on `method: percentile`."
        ),
    )


# Staged annual versions plus on-demand LFPS versions, both accepted here.
_CANOPY_ALL_VERSIONS = list(
    dict.fromkeys(
        LANDFIRE_VERSIONS["canopy"]["available"]
        + LANDFIRE_VERSIONS["canopy"]["lfps_available"]
    )
)
LandfireCanopyVersion = StrEnum(
    "LandfireCanopyVersion",
    {f"v{version}": version for version in _CANOPY_ALL_VERSIONS},
)


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

    version: LandfireCanopyVersion = Field(
        default=LandfireCanopyVersion(LANDFIRE_VERSIONS["canopy"]["default"]),
        description=(
            "LANDFIRE landscape vintage year (e.g. '2024'). Most years are "
            "served from staged LANDFIRE data; version "
            f"{', '.join(LANDFIRE_VERSIONS['canopy']['lfps_available'])} "
            "is instead fetched on demand from LANDFIRE Product Service."
        ),
    )
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
