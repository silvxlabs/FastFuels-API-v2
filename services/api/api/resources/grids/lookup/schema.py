"""
api/v2/resources/grids/lookup/schema.py

Schema models for fuel model lookup operations.

Lookup endpoints convert categorical fuel model codes to continuous fuel
parameters using standard lookup tables. For example, converting FBFM40
codes (GR1, TL3, etc.) to fuel loads, SAV ratios, and fuel depth using
Scott-Burgan 40 tables.
"""

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from api.resources.grids.modifications import GridModification
from api.resources.grids.schema import Band, BandType, validate_no_duplicates


class LookupSource(BaseModel):
    """Base source for grids created via lookup table transformation.

    Converts categorical codes to continuous fuel parameters using
    standard lookup tables.
    """

    name: Literal["lookup"] = "lookup"
    table: str = Field(..., description="Lookup table identifier")
    source_grid_id: str = Field(..., description="Grid containing codes to look up")
    source_band: str = Field(..., description="Band in source grid containing codes")


class Fbfm40LookupSource(LookupSource):
    """Source for grids created via FBFM40 fuel model lookup.

    Converts categorical FBFM40 codes to continuous fuel parameters
    using Scott-Burgan 40 lookup tables.
    """

    table: Literal["fbfm40"] = "fbfm40"
    source_band: str = Field(
        default="fbfm", description="Band in source grid containing FBFM codes"
    )


class Fbfm40LookupBand(StrEnum):
    """Bands available from FBFM40/SB40 lookup tables."""

    fuel_load_1hr = "fuel_load.1hr"
    fuel_load_10hr = "fuel_load.10hr"
    fuel_load_100hr = "fuel_load.100hr"
    fuel_load_live_herb = "fuel_load.live_herb"
    fuel_load_live_woody = "fuel_load.live_woody"
    savr_1hr = "savr.1hr"
    savr_10hr = "savr.10hr"
    savr_100hr = "savr.100hr"
    savr_live_herb = "savr.live_herb"
    savr_live_woody = "savr.live_woody"
    fuel_depth = "fuel_depth"
    moisture_of_extinction = "moisture_of_extinction"
    heat_content = "heat_content"
    is_dynamic = "is_dynamic"


class CreateFbfm40LookupRequest(BaseModel):
    """Request to create a grid by looking up FBFM40 fuel parameters.

    Unlike entry-point grid creation requests, domain_id is not required
    because derived grids carry the same domain reference as their source.
    """

    source_grid_id: str = Field(..., description="Grid containing FBFM40 codes")
    source_band: str = Field(
        default="fbfm",
        description="Band in source grid containing FBFM codes",
    )
    bands: list[Fbfm40LookupBand] = Field(..., min_length=1)
    name: str = Field("", max_length=255)
    description: str = Field("", max_length=2000)
    tags: list[str] = Field(default_factory=list, max_length=50)
    modifications: list[GridModification] = Field(default_factory=list)

    @field_validator("bands")
    @classmethod
    def no_duplicate_bands(
        cls,
        v: list[Fbfm40LookupBand],
    ) -> list[Fbfm40LookupBand]:
        return validate_no_duplicates(v)


FBFM40_LOOKUP_BAND_METADATA: dict[Fbfm40LookupBand, tuple[BandType, str | None]] = {
    Fbfm40LookupBand.fuel_load_1hr: (BandType.continuous, "kg/m**2"),
    Fbfm40LookupBand.fuel_load_10hr: (BandType.continuous, "kg/m**2"),
    Fbfm40LookupBand.fuel_load_100hr: (BandType.continuous, "kg/m**2"),
    Fbfm40LookupBand.fuel_load_live_herb: (BandType.continuous, "kg/m**2"),
    Fbfm40LookupBand.fuel_load_live_woody: (BandType.continuous, "kg/m**2"),
    Fbfm40LookupBand.savr_1hr: (BandType.continuous, "1/m"),
    Fbfm40LookupBand.savr_10hr: (BandType.continuous, "1/m"),
    Fbfm40LookupBand.savr_100hr: (BandType.continuous, "1/m"),
    Fbfm40LookupBand.savr_live_herb: (BandType.continuous, "1/m"),
    Fbfm40LookupBand.savr_live_woody: (BandType.continuous, "1/m"),
    Fbfm40LookupBand.fuel_depth: (BandType.continuous, "m"),
    Fbfm40LookupBand.moisture_of_extinction: (BandType.continuous, "%"),
    Fbfm40LookupBand.heat_content: (BandType.continuous, "kJ/kg"),
    Fbfm40LookupBand.is_dynamic: (BandType.categorical, None),
}


def get_fbfm40_lookup_band(band: Fbfm40LookupBand, index: int) -> Band:
    """Return Band metadata for an FBFM40 lookup band."""
    band_type, unit = FBFM40_LOOKUP_BAND_METADATA[band]
    return Band(key=band.value, type=band_type, unit=unit, index=index)
