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

    x: str | None = None
    y: str | None = None
    height: str | None = None
    fia_species_code: str | None = None
    fia_status_code: str | None = None
    dbh: str | None = None
    crown_ratio: str | None = None


class CreateInventoryUploadRequest(BaseModel):
    format: InventoryUploadFormat
    columns: InventoryColumnMapping = Field(default_factory=InventoryColumnMapping)
    name: str = Field("", max_length=255)
    description: str = Field("", max_length=2000)
    tags: list[str] = Field(default_factory=list)


class InventoryUploadSpec(BaseModel):
    method: Literal["PUT"] = "PUT"
    url: str
    content_type: str
    expires_at: datetime
    max_size_bytes: int


class InventoryUploadCreatedResponse(BaseModel):
    inventory: Inventory
    upload: InventoryUploadSpec
