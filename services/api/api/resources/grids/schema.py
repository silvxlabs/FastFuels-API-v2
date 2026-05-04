"""
api/v2/resources/grids/schema.py

Core schema models for the Grid resource.
"""

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field

from api.resources.grids.modifications import GridModification
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
    unit: str | None = None
    index: int


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
    modifications: list[GridModification] = Field(default_factory=list)


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
    chunk_shape: tuple[int, int] | tuple[int, int, int] | None = Field(
        default=None,
        description=(
            "Zarr chunk shape. 2D grids: (height, width). 3D grids: (z, height, width)."
        ),
    )

    # User organization
    tags: list[str] = Field(default_factory=list)


class ListGridsResponse(PaginatedResponse):
    """Paginated response for listing grids."""

    grids: list[Grid]


class GridDataFormat(StrEnum):
    json = "json"
    binary = "binary"


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


class GridDataResponse(BaseModel):
    shape: list[int]
    order: Literal["C", "F"]
    data: list[float | int]
