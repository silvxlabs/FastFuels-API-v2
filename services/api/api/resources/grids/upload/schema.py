"""Schema models for the grid upload endpoint."""

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from api.resources.grids.schema import BandType, Grid
from lib.units import validate_unit


class GridUploadFormat(StrEnum):
    geotiff = "geotiff"


class UploadBandDefinition(BaseModel):
    key: str = Field(
        ..., description="Dot-notation variable name, e.g. 'bulk_density.foliage'"
    )
    type: BandType
    unit: str | None = Field(
        None,
        description=(
            "Physical unit of the band's pixel values, in UDUNITS-2-conformant "
            "ASCII form with `**` for exponents (e.g. `kg/m**3`, `1/m`, `%`). "
            "Optional for categorical/identifier bands. Non-canonical forms "
            "(`kg/m³`, `kg/m^3`, `kg/m3`) are rejected. See docs/units.md."
        ),
        examples=["kg/m**3", "1/m", "%", "m"],
    )

    @field_validator("unit")
    @classmethod
    def _check_canonical_unit(cls, v: str | None) -> str | None:
        validate_unit(v)
        return v


class CreateGridUploadRequest(BaseModel):
    format: GridUploadFormat = GridUploadFormat.geotiff
    bands: list[UploadBandDefinition] = Field(..., min_length=1)
    name: str = Field("", max_length=255)
    description: str = Field("", max_length=2000)
    tags: list[str] = Field(default_factory=list)
    num_buffer_cells: int = Field(
        0,
        ge=0,
        description=(
            "Number of extra native-resolution cells to keep around the domain "
            "extent in the stored grid. The uploaded GeoTIFF must cover the "
            "domain bbox expanded by num_buffer_cells * native_pixel_size on "
            "each side; pixels beyond that expanded extent are clipped away."
        ),
    )

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
