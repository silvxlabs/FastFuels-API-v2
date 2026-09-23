"""
api/v2/resources/inventories/tree/upload/schema.py

Schema models for direct inventory file uploads.
"""

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from api.resources.inventories.schema import Inventory


class InventoryUploadFormat(StrEnum):
    csv = "csv"
    geojson = "geojson"
    geopackage = "geopackage"


class InventoryColumnMapping(BaseModel):
    """Maps v2 column names to the corresponding column names in the uploaded file.

    Omit any entry whose column already uses the v2 name. For GeoJSON and
    GeoPackage formats, x and y are extracted from geometry — their mapping
    entries are ignored.
    """

    model_config = ConfigDict(extra="forbid")

    tree_id: str | None = Field(
        None,
        description=(
            "Per-tree identifier to keep as the inventory's `tree_id`. Values "
            "must be non-null numbers in 1 … 2,147,483,647; they are cast to "
            "integers and must then be unique across the file; otherwise the upload fails with "
            "`SCHEMA_VALIDATION_ERROR`. If the file has no `tree_id` column and none "
            "is mapped, IDs 1 … N are generated."
        ),
    )
    x: str | None = Field(
        None, description="Tree position in the domain's projected CRS (m)."
    )
    y: str | None = Field(
        None, description="Tree position in the domain's projected CRS (m)."
    )
    height: str | None = Field(None, description="Total tree height (m).")
    fia_species_code: str | None = Field(None, description="FIA species code (SPCD).")
    fia_status_code: str | None = Field(
        None,
        description="FIA tree status code (STATUSCD): 1 live, 2 dead, 3 removed.",
    )
    fia_crown_class_code: str | None = Field(
        None,
        description=(
            "FIA crown class code (CCLCD): 1 open-grown, 2 dominant, "
            "3 codominant, 4 intermediate, 5 overtopped."
        ),
    )
    dbh: str | None = Field(None, description="Diameter at breast height (cm).")
    crown_ratio: str | None = Field(
        None,
        description="Live crown ratio: fraction of total height with live crown (0-1).",
    )


class CreateInventoryUploadRequest(BaseModel):
    format: InventoryUploadFormat
    columns: InventoryColumnMapping = Field(default_factory=InventoryColumnMapping)
    name: str = Field("", max_length=255)
    description: str = Field("", max_length=2000)
    tags: list[str] = Field(default_factory=list)


class InventoryUploadSpec(BaseModel):
    method: Literal["PUT"] = "PUT"
    url: str
    headers: dict[str, str]
    content_type: str
    expires_at: datetime
    max_size_bytes: int


class InventoryUploadCreatedResponse(BaseModel):
    inventory: Inventory
    upload: InventoryUploadSpec
