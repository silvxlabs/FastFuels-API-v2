"""
api/v2/resources/inventories/schema.py

Core schema models for the Inventory resource.
"""

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field

from api.resources.inventories.modification_models import InventoryModification
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


class ColumnType(StrEnum):
    """Type of column data."""

    continuous = "continuous"
    categorical = "categorical"


class Column(BaseModel):
    """A single column in an inventory."""

    key: str = Field(..., description="Column name (e.g., 'dbh', 'fia_species_code')")
    type: ColumnType
    unit: str | None = None


# Base columns always present in a tree inventory
BASE_INVENTORY_COLUMNS = [
    Column(key="x", type=ColumnType.continuous, unit="m"),
    Column(key="y", type=ColumnType.continuous, unit="m"),
    Column(key="fia_species_code", type=ColumnType.categorical),
    Column(key="fia_status_code", type=ColumnType.categorical),
    Column(key="dbh", type=ColumnType.continuous, unit="cm"),
    Column(key="height", type=ColumnType.continuous, unit="m"),
    Column(key="crown_ratio", type=ColumnType.continuous),
]


class InventoryGeoreference(BaseModel):
    """Spatial reference for an inventory, computed from the domain geometry."""

    crs: str
    bounds: tuple[float, float, float, float]


class Inventory(BaseModel):
    """The Inventory resource.

    When status is "pending" or "running", georeference will be null.
    The backend populates it after successfully processing data,
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
    created_on: datetime | None = None
    modified_on: datetime | None = None
    source: dict
    modifications: list[InventoryModification] = Field(default_factory=list)
    columns: list[Column] = Field(default_factory=list)
    georeference: InventoryGeoreference | None = Field(
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
