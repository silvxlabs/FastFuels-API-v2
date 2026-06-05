"""
api/v2/resources/inventories/schema.py

Core schema models for the Inventory resource.
"""

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, model_validator

from api.resources.inventories.modification_models import InventoryModification
from api.resources.inventories.treatment_models import InventoryTreatment
from api.resources.modifications import parse_modification_coordinates
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


class DuplicateInventoryRequest(BaseModel):
    """Optional metadata overrides for a duplicated inventory.

    Every field is optional. Any field omitted is carried over verbatim from
    the source inventory.
    """

    name: str | None = Field(
        None,
        max_length=255,
        description="Name for the copy. Omit to reuse the source inventory's name.",
    )
    description: str | None = Field(
        None,
        max_length=2000,
        description=(
            "Description for the copy. Omit to reuse the source inventory's "
            "description."
        ),
    )
    tags: list[str] | None = Field(
        None,
        max_length=50,
        description="Tags for the copy. Omit to reuse the source inventory's tags.",
    )


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
    checksum: str | None = Field(
        default=None,
        description=(
            "Version marker for this inventory's content. It changes each time the "
            "inventory is rebuilt and is unaffected by metadata-only edits (name, "
            "description, tags). A resource derived from this inventory stores the "
            "checksum it was built from; comparing that stored value against this "
            "field reveals whether this inventory has changed since. May be null "
            "for inventories created before checksums were introduced."
        ),
    )
    source: dict
    modifications: list[InventoryModification] = Field(default_factory=list)
    treatments: list[InventoryTreatment] = Field(default_factory=list)
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

    @model_validator(mode="before")
    @classmethod
    def _parse_modification_coordinates(cls, data: Any) -> Any:
        """Decode stringified inline-geometry coordinates loaded from Firestore.

        Inline-geometry modification/treatment conditions are stored with their
        ``coordinates`` JSON-encoded (Firestore rejects nested arrays). Both
        ``modifications`` and ``treatments`` are create-time fields carrying
        spatial conditions, so decode the coordinates on each. Also decode
        ``source.modifications`` for the (gated) modifications-source inventory.
        Idempotent on already-parsed data.
        """
        if isinstance(data, dict):
            if isinstance(data.get("modifications"), list):
                parse_modification_coordinates(data["modifications"])
            if isinstance(data.get("treatments"), list):
                parse_modification_coordinates(data["treatments"])
            source = data.get("source")
            if isinstance(source, dict) and isinstance(
                source.get("modifications"), list
            ):
                parse_modification_coordinates(source["modifications"])
        return data


class ListInventoriesResponse(PaginatedResponse):
    """Paginated response for listing inventories."""

    inventories: list[Inventory]


class InventoryDataFormat(StrEnum):
    json = "json"
    csv = "csv"


class InventoryJsonOrientation(StrEnum):
    split = "split"
    records = "records"


class InventoryPartitionInfo(BaseModel):
    index: int
    num_rows: int


class InventoryDataMetadata(BaseModel):
    inventory_id: str
    num_partitions: int
    total_rows: int
    columns: list[str]
    partitions: list[InventoryPartitionInfo]


class InventoryDataResponse(BaseModel):
    partition: int
    num_rows: int
    columns: list[str]
    data: list[list] | list[dict]
