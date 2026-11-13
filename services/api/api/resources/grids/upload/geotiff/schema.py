"""Schema models for the GeoTIFF grid upload endpoint."""

from typing import Self

from pydantic import BaseModel, Field, field_validator, model_validator

from api.resources.grids.schema import BandKey, BandType
from lib.band_quantities import BAND_QUANTITIES, band_mismatch, band_quantity
from lib.units import validate_unit


class UploadBandDefinition(BaseModel):
    key: BandKey
    type: BandType | None = Field(
        None,
        description="Optional; set by the key's quantity. If given, it must match.",
    )
    unit: str | None = Field(
        None,
        description=(
            "Optional; set by the key's quantity. If given, it must match, in "
            "UDUNITS-2-conformant ASCII form with `**` for exponents (e.g. "
            "`kg/m**3`, `1/m`, `%`)."
        ),
        examples=["kg/m**3", "1/m", "%", "m"],
    )

    @field_validator("unit")
    @classmethod
    def _check_canonical_unit(cls, v: str | None) -> str | None:
        validate_unit(v)
        return v

    @model_validator(mode="after")
    def _type_and_unit_from_key(self) -> Self:
        quantity = BAND_QUANTITIES[band_quantity(self.key)]
        given_type = quantity.type if self.type is None else self.type
        given_unit = quantity.unit if self.unit is None else self.unit
        if reason := band_mismatch(self.key, given_type, given_unit):
            raise ValueError(f"Band {self.key!r}: {reason}.")
        self.type = BandType(quantity.type)
        self.unit = quantity.unit
        return self


class CreateGeoTIFFUploadRequest(BaseModel):
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
