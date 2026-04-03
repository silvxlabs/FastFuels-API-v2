"""
api/v2/resources/features/schema.py

Core schema models for the Feature resource.
"""

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field

from api.schema import JobError, JobProgress, JobStatus, PaginatedResponse


class FeatureType(StrEnum):
    """Type of geographic feature."""

    road = "road"
    water = "water"


class FeatureSortField(StrEnum):
    """Fields available for sorting feature list results."""

    created_on = "created_on"
    modified_on = "modified_on"
    name = "name"


class CreateFeatureRequestBase(BaseModel):
    """Base fields for feature creation requests."""

    type: FeatureType
    name: str = Field("", max_length=255)
    description: str = Field("", max_length=2000)
    tags: list[str] = Field(default_factory=list, max_length=50)


class UpdateFeatureRequestBody(BaseModel):
    """Request body for updating feature metadata."""

    name: str | None = Field(None, max_length=255)
    description: str | None = Field(None, max_length=2000)
    tags: list[str] | None = Field(None, max_length=50)


class FeatureGeoreference(BaseModel):
    """Spatial reference for a feature, computed from the domain geometry."""

    crs: str
    bounds: tuple[float, float, float, float]


class Feature(BaseModel):
    """The Feature resource.

    When status is "pending" or "running", georeference will be null.
    The backend worker populates it after successfully generating the GeoJSON
    and uploading it to GCS, at which point status transitions to "completed".
    """

    id: str
    domain_id: str
    type: FeatureType
    name: str = ""
    description: str = ""
    status: JobStatus
    progress: JobProgress | None = Field(
        default=None,
        description="Progress info when status is 'running'. Null otherwise.",
    )
    created_on: datetime | None = None
    modified_on: datetime | None = None
    source: dict
    georeference: FeatureGeoreference | None = Field(
        default=None,
        description="Spatial reference. Null until backend completes processing.",
    )
    error: JobError | None = Field(
        default=None,
        description="Error details if status is 'failed'.",
    )
    tags: list[str] = Field(default_factory=list)


class ListFeaturesResponse(PaginatedResponse):
    """Paginated response for listing features."""

    features: list[Feature]
