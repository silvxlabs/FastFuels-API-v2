"""
api/v2/resources/inventories/schema.py

Core schema models for the Inventory resource.
"""

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field

from api.resources.grids.schema import Georeference
from api.schema import JobError, JobProgress, JobStatus, PaginatedResponse


class InventoryType(StrEnum):
    """Type of entities in the inventory."""

    tree = "tree"


class InventorySortField(StrEnum):
    """Fields available for sorting inventory list results."""

    created_on = "created_on"
    modified_on = "modified_on"
    name = "name"


class PointProcess(StrEnum):
    """Spatial point process for tree coordinate assignment."""

    inhomogeneous_poisson = "inhomogeneous_poisson"


class SummaryStats(BaseModel):
    """Min/max/mean/std summary for a numeric attribute."""

    min: float
    max: float
    mean: float
    std: float


class InventorySummary(BaseModel):
    """Summary statistics populated by the backend on completion."""

    total_entities: int
    entities_per_hectare: float
    area_hectares: float
    species_count: int
    height_stats: SummaryStats
    dbh_stats: SummaryStats
    basal_area: float


class CreateInventoryRequestBase(BaseModel):
    """Base fields for inventory creation requests."""

    type: InventoryType = InventoryType.tree
    name: str = Field("", max_length=255)
    description: str = Field("", max_length=2000)
    tags: list[str] = Field(default_factory=list, max_length=50)


class UpdateInventoryRequestBody(BaseModel):
    """Request body for updating inventory metadata."""

    name: str | None = Field(None, max_length=255)
    description: str | None = Field(None, max_length=2000)
    tags: list[str] | None = Field(None, max_length=50)


class Inventory(BaseModel):
    """The Inventory resource.

    When status is "pending" or "running", summary and georeference will be null.
    The backend populates these after successfully processing data,
    at which point status transitions to "completed".
    """

    id: str
    domain_id: str
    type: InventoryType
    name: str = ""
    description: str = ""
    status: JobStatus
    progress: JobProgress | None = Field(
        default=None,
        description="Progress info when status is 'running'. Null otherwise.",
    )
    created_on: datetime
    modified_on: datetime
    source: dict
    modifications: list = Field(default_factory=list)
    summary: InventorySummary | None = Field(
        default=None,
        description="Summary statistics. Null until backend completes processing.",
    )
    georeference: Georeference | None = Field(
        default=None,
        description="Spatial reference. Null until backend completes processing.",
    )
    error: JobError | None = Field(
        default=None,
        description="Error details if status is 'failed'.",
    )
    tags: list[str] = Field(default_factory=list)


class ListInventoriesResponse(PaginatedResponse):
    """Paginated response for listing inventories."""

    inventories: list[Inventory]
