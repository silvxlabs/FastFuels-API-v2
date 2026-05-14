"""Schema models for the grid upload endpoint."""

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from api.resources.grids.schema import BandType, Grid


class GridUploadFormat(StrEnum):
    geotiff = "geotiff"


class UploadBandDefinition(BaseModel):
    key: str = Field(
        ..., description="Dot-notation variable name, e.g. 'bulk_density.foliage'"
    )
    type: BandType
    unit: str | None = None
    description: str | None = None


class CreateGridUploadRequest(BaseModel):
    format: GridUploadFormat = GridUploadFormat.geotiff
    bands: list[UploadBandDefinition] = Field(..., min_length=1)
    name: str = Field("", max_length=255)
    description: str = Field("", max_length=2000)
    tags: list[str] = Field(default_factory=list)

    @field_validator("bands")
    @classmethod
    def no_duplicate_keys(
        cls, v: list[UploadBandDefinition]
    ) -> list[UploadBandDefinition]:
        keys = [b.key for b in v]
        seen, duplicates = set(), set()
        for k in keys:
            (duplicates if k in seen else seen).add(k)
        if duplicates:
            raise ValueError(
                f"Duplicate band keys are not allowed: {', '.join(sorted(duplicates))}"
            )
        return v


class GridUploadSpec(BaseModel):
    method: Literal["PUT"] = "PUT"
    url: str
    content_type: str
    expires_at: datetime
    max_size_bytes: int


class GridUploadCreatedResponse(BaseModel):
    grid: Grid
    upload: GridUploadSpec
