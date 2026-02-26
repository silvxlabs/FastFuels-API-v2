"""
api/v2/resources/exports/schema.py

Core schema models for the Export resource.
"""

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field

from api.schema import JobError, JobProgress, JobStatus, PaginatedResponse


class ExportSortField(StrEnum):
    """Fields available for sorting export list results."""

    created_on = "created_on"
    modified_on = "modified_on"
    name = "name"


class Export(BaseModel):
    """The Export resource.

    Exports are standalone artifacts that record provenance (domain_id, grid_id)
    but have independent lifecycle — deleting a domain does not delete its exports.

    When status is "completed", signed_url contains a signed URL for
    downloading the exported file.
    """

    id: str
    domain_id: str = Field(
        ...,
        description="Domain the source grids belong to (provenance, not lifecycle dependency).",
    )
    name: str = ""
    description: str = ""
    status: JobStatus
    progress: JobProgress | None = Field(
        default=None,
        description="Progress info when status is 'running'. Null otherwise.",
    )
    created_on: datetime
    modified_on: datetime

    source: dict = Field(
        ...,
        description=(
            "Format-specific export configuration. Contains 'name' (the export format) "
            "plus additional fields depending on the source type."
        ),
    )

    signed_url: str | None = Field(
        default=None,
        description="Signed URL for downloading the exported file. Populated on completion.",
    )
    expires_on: datetime | None = Field(
        default=None,
        description="When the signed URL expires.",
    )

    error: JobError | None = Field(
        default=None,
        description="Error details if status is 'failed'.",
    )

    tags: list[str] = Field(default_factory=list)


class UpdateExportRequestBody(BaseModel):
    """Request body for updating export metadata."""

    name: str | None = Field(None, max_length=255)
    description: str | None = Field(None, max_length=2000)
    tags: list[str] | None = Field(None, max_length=50)


class ListExportsResponse(PaginatedResponse):
    """Paginated response for listing exports."""

    exports: list[Export]


class GridExportFormat(StrEnum):
    """Supported grid export formats."""

    geotiff = "geotiff"
    zarr = "zarr"


class ExportGridRequest(BaseModel):
    """Request body for creating a grid export.

    Used at: POST /domains/{domain_id}/grids/{grid_id}/exports/{format}
    """

    bands: list[str] | None = Field(
        default=None,
        description=(
            "Band keys to include (e.g. 'fuel_load.1hr', 'fbfm'). "
            "Omit to export all bands from the grid."
        ),
    )
    expiration_days: int = Field(
        default=7,
        ge=1,
        le=7,
        description="Number of days until the signed download URL expires (max 7). Default: 7.",
    )
    name: str = Field("", max_length=255)
    description: str = Field("", max_length=2000)
    tags: list[str] = Field(default_factory=list, max_length=50)


class InventoryExportFormat(StrEnum):
    """Supported inventory export formats."""

    parquet = "parquet"
    csv = "csv"
    geojson = "geojson"
    geopackage = "geopackage"


class ExportInventoryRequest(BaseModel):
    """Request body for creating an inventory export.

    Used at: POST /domains/{domain_id}/inventories/{inventory_id}/exports/{format}
    """

    columns: list[str] | None = Field(
        default=None,
        max_length=100,
        description=(
            "Column keys to include in the export (e.g. 'x', 'dbh', 'height'). "
            "Omit to export all columns. Maximum 100 columns."
        ),
    )
    expiration_days: int = Field(
        default=7,
        ge=1,
        le=7,
        description="Number of days until the signed download URL expires (max 7). Default: 7.",
    )
    name: str = Field("", max_length=255)
    description: str = Field("", max_length=2000)
    tags: list[str] = Field(default_factory=list, max_length=50)


class InventoryExportSource(BaseModel):
    """Stored source metadata for inventory exports.

    Recorded in the Export.source field for provenance.
    """

    name: str = Field(..., pattern="^(parquet|csv|geojson|geopackage)$")
    inventory_id: str
    columns: list[str] | None = Field(
        default=None,
        description="Column keys included, or null for all columns.",
    )
    crs: str | None = Field(
        default=None,
        description="CRS from the inventory georeference, included for spatial formats.",
    )


class GridExportSource(BaseModel):
    """Stored source metadata for grid exports.

    Recorded in the Export.source field for provenance.
    """

    name: str = Field(..., pattern="^(geotiff|zarr)$")
    grid_id: str
    bands: list[str] | None = Field(
        default=None,
        description="Band keys included, or null for all bands.",
    )
