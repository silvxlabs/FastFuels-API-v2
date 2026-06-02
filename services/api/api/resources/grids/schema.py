"""Core schema models for the Grid resource."""

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from api.resources.grids.alignment import (
    GridAlignmentDomainTarget,
    GridAlignmentSpecification,
)
from api.resources.grids.modifications import GridModification
from api.resources.modifications import parse_modification_coordinates
from api.schema import JobError, JobProgress, JobStatus, PaginatedResponse

CHUNK_SHAPE = [512, 512]


def validate_no_duplicates(values: list) -> list:
    """Raise ValueError if a list contains duplicate entries."""
    seen = set()
    duplicates = set()
    for v in values:
        if v in seen:
            duplicates.add(v)
        seen.add(v)
    if duplicates:
        sorted_dupes = sorted(str(d) for d in duplicates)
        raise ValueError(f"Duplicate values are not allowed: {', '.join(sorted_dupes)}")
    return values


class TileMetadata(BaseModel):
    """Post-processing metadata about source tiles fetched during grid creation."""

    tiles: list[str]
    tile_source: str | None = None
    tile_count: int
    native_crs: str | None = None
    acquisition_dates: list[str] | None = None


class BandType(StrEnum):
    """Type of band data."""

    continuous = "continuous"
    categorical = "categorical"


class GridSortField(StrEnum):
    """Fields available for sorting grid list results."""

    created_on = "created_on"
    modified_on = "modified_on"
    name = "name"


class Band(BaseModel):
    """A single band in a grid."""

    key: str = Field(..., description="Dot-notation key (e.g., 'fuel_load.1hr')")
    type: BandType
    unit: str | None = Field(
        None,
        description=(
            "Physical unit of the band's pixel values, in UDUNITS-2-conformant "
            "ASCII form with `**` for exponents (e.g. `kg/m**3`, `1/m`, `%`). "
            "`None` for categorical/identifier bands. See docs/units.md."
        ),
        examples=["kg/m**3", "1/m", "%", "m"],
    )
    index: int
    nodata: int | float | None = Field(
        None,
        description=(
            "Value marking missing pixels in this band; pixels equal to it "
            "carry no data and should be excluded from analysis. `null` when "
            "the band has no missing pixels, or when they are represented as "
            "floating-point NaN."
        ),
        examples=[32767, -9999, None],
    )


class Georeference(BaseModel):
    """Spatial reference for a 2D grid.

    Uses rasterio/GDAL conventions:
    - transform: Affine coefficients [a, b, c, d, e, f] where:
        x = a * col + b * row + c
        y = d * col + e * row + f
      For north-up images: a = pixel_width, e = -pixel_height, b = d = 0
    - shape: (height, width) in pixels
    """

    crs: str = Field(..., description="e.g., 'EPSG:32610'")
    transform: tuple[float, float, float, float, float, float] = Field(
        ..., description="Affine transform [a, b, c, d, e, f]"
    )
    shape: tuple[int, int] = Field(..., description="(height, width)")


class Georeference3D(Georeference):
    """Spatial reference for a 3D grid."""

    shape: tuple[int, int, int] = Field(..., description="(z, height, width)")
    z_resolution: float
    z_origin: float


class Chunks(BaseModel):
    """Chunk layout for a grid."""

    shape: tuple[int, int] | tuple[int, int, int] = Field(
        ...,
        description=(
            "Size of a single chunk. 2D grids: (y, x). 3D grids: (z, y, x). "
            "Edge chunks may be smaller."
        ),
    )
    count: int | None = Field(
        default=None,
        description="Total number of chunks in the grid.",
    )
    count_by_axis: dict[str, int] | None = Field(
        default=None,
        description=(
            "Number of chunks along each axis. Keys are 'y','x' for 2D grids "
            "and 'z','y','x' for 3D grids."
        ),
    )


class CreateGridRequestBase(BaseModel):
    """Base fields for grid creation requests.

    Note: domain_id comes from the URL path parameter, not the request body.
    Grids are always created at native resolution. To change resolution,
    use the explicit /grids/resample endpoint after creation. See grids.md for
    design rationale.
    """

    name: str = Field("", max_length=255)
    description: str = Field("", max_length=2000)
    tags: list[str] = Field(default_factory=list, max_length=50)
    modifications: list[GridModification] = Field(
        default_factory=list,
        description=(
            "Rules applied to the grid after it is built from its source. Each "
            "rule has a list of `conditions` (ANDed together) and a list of "
            "`actions` (applied where the conditions match). Conditions can be "
            "attribute-based (compare a band value) or spatial (test cell "
            "location against a geometry). Spatial conditions come in two "
            "variants discriminated by `source`: `geometry` (inline GeoJSON) "
            "or `feature` (reference a persisted Feature resource — road, "
            "water, layerset — in the same domain by `feature_id`). Both "
            "spatial variants accept `buffer_m` (meters, applied in the "
            "domain's projected CRS) to widen the geometry, and `target` "
            "(`centroid` or `cell`) to choose which part of the cell is "
            "tested. Actions modify band values via `replace`, `multiply`, "
            "`divide`, `add`, or `subtract`. See the `GridModification` "
            "schema for the full field reference and worked examples."
        ),
    )


class CreateSourceGridRequestBase(CreateGridRequestBase):
    """Base for raster-backed source grid creation requests.

    Adds the optional `extent_buffer_cells` field and the `alignment`
    discriminated union shared by every endpoint that fetches data from
    an external raster (LANDFIRE, PIM, CHM, 3DEP).
    """

    extent_buffer_cells: int = Field(
        0,
        ge=0,
        le=10,
        description=(
            "Number of result-grid cells included as a buffer around the "
            "domain extent in the stored grid. The buffer is measured after "
            "the source raster is projected into the domain CRS, so a cell "
            "means one cell in the returned grid rather than one source "
            "raster cell. Provides context for later operations (resample, "
            "reproject, focal filters, derivative calculations) that are "
            "sensitive to edges. Default 0 adds no buffer. Maximum: 10 cells."
        ),
    )

    alignment: GridAlignmentSpecification = Field(
        default_factory=GridAlignmentDomainTarget,
        description=(
            'Per-fetch alignment target. Default `target="domain"` anchors '
            "output cells to the domain origin so cross-source composition "
            'works by construction. `target="native"` preserves the source '
            'pixel anchor. `target="grid"` aligns to an existing grid by id.'
        ),
    )


class UpdateGridRequestBody(BaseModel):
    """Request body for updating grid metadata."""

    name: str | None = Field(None, max_length=255)
    description: str | None = Field(None, max_length=2000)
    tags: list[str] | None = Field(None, max_length=50)


class Grid(BaseModel):
    """The Grid resource.

    When status is "pending" or "running", georeference will be null.
    The backend populates georeference after successfully fetching data,
    at which point status transitions to "completed".

    When status is "failed", the error field contains details about what
    went wrong and suggestions for the user. The full traceback is stored
    in Firestore but not exposed in API responses.
    """

    id: str
    domain_id: str
    name: str = ""
    description: str = ""
    status: JobStatus
    progress: JobProgress | None = Field(
        default=None,
        description="Progress info when status is 'running'. Null otherwise.",
    )
    created_on: datetime | None = None
    modified_on: datetime | None = None

    # Source and lineage (validated by source-specific routers)
    source: dict
    modifications: list[GridModification] = Field(default_factory=list)

    # Data specification - bands known at creation, georeference set by backend
    bands: list[Band]
    georeference: Georeference | Georeference3D | None = Field(
        default=None,
        description="Spatial reference. Null until backend completes data fetch.",
    )

    # Error information (populated when status is "failed")
    error: JobError | None = Field(
        default=None,
        description="Error details if status is 'failed'. Traceback stored but not exposed.",
    )

    # Storage
    chunks: Chunks | None = Field(
        default=None,
        description=(
            "Chunk layout. Null until the grid finishes processing. "
            "Use chunks.count to know how many chunks are available to fetch."
        ),
    )

    # User organization
    tags: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _parse_modification_coordinates(cls, data: Any) -> Any:
        """Decode stringified inline-geometry coordinates loaded from Firestore.

        Inline-geometry modification conditions are stored with their
        ``coordinates`` JSON-encoded (Firestore rejects nested arrays). This
        runs before validation so GET / list / create responses return proper
        nested-list GeoJSON. Idempotent: API-request data (already nested
        lists) passes through unchanged.
        """
        if isinstance(data, dict) and isinstance(data.get("modifications"), list):
            parse_modification_coordinates(data["modifications"])
        return data


class ListGridsResponse(PaginatedResponse):
    """Paginated response for listing grids."""

    grids: list[Grid]


class GridDataArrayFormat(StrEnum):
    dense = "dense"
    sparse = "sparse"


class GridDataOrder(StrEnum):
    C = "C"
    F = "F"


class GridDataChunkMetadata(BaseModel):
    index: int
    shape: tuple[int, int] | tuple[int, int, int]
    offset: tuple[int, int] | tuple[int, int, int]
    transform: tuple[float, float, float, float, float, float]
    z_origin: float | None = None
    z_resolution: float | None = None


class DenseGridData(BaseModel):
    format: Literal["dense"]
    values: list[float | int]


class SparseGridData(BaseModel):
    format: Literal["sparse"]
    fill_value: float | int | None
    indices: list[int]
    values: list[float | int]


class GridDataResponse(BaseModel):
    shape: list[int]
    order: Literal["C", "F"]
    metadata: GridDataChunkMetadata
    data: DenseGridData | SparseGridData = Field(discriminator="format")
