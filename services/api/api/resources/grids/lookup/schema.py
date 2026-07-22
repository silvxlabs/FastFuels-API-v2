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

from api.resources.grids.modification_models import GridModification
from api.resources.grids.schema import Band, BandType, validate_no_duplicates


class LookupSource(BaseModel):
    """Base source for grids created via lookup table transformation.

    Converts categorical codes to continuous fuel parameters using
    standard lookup tables.
    """

    name: Literal["lookup"] = "lookup"
    table: str = Field(..., description="Lookup table identifier")
    source_grid_id: str = Field(..., description="Grid containing codes to look up")
    source_grid_checksum: str | None = Field(
        default=None,
        description=(
            "The source grid's `checksum` at the time this grid was created from "
            "it. Compare it against the source grid's current `checksum` to tell "
            "whether the source has changed since."
        ),
    )
    source_band: str = Field(..., description="Band in source grid containing codes")


class Fbfm13LookupSource(LookupSource):
    """Source for grids created via FBFM13 fuel model lookup.

    Converts categorical Anderson 13 fuel model codes to continuous fuel
    parameters using the Anderson 13 lookup table.
    """

    table: Literal["fbfm13"] = "fbfm13"
    source_band: str = Field(
        default="fbfm13", description="Band in source grid containing FBFM13 codes"
    )


class Fbfm13LookupBand(StrEnum):
    """Bands available from the Anderson 13 lookup table."""

    fuel_load_1hr = "fuel_load.1hr"
    fuel_load_10hr = "fuel_load.10hr"
    fuel_load_100hr = "fuel_load.100hr"
    fuel_load_live_foliage = "fuel_load.live_foliage"
    savr_1hr = "savr.1hr"
    savr_10hr = "savr.10hr"
    savr_100hr = "savr.100hr"
    savr_live_foliage = "savr.live_foliage"
    fuel_depth = "fuel_depth"


class CreateFbfm13LookupRequest(BaseModel):
    """Request to create a grid by looking up FBFM13 fuel parameters.

    Unlike entry-point grid creation requests, domain_id is not required
    because derived grids carry the same domain reference as their source.
    """

    source_grid_id: str = Field(..., description="Grid containing FBFM13 codes")
    source_band: str = Field(
        default="fbfm13",
        description="Band in source grid containing FBFM13 codes",
    )
    bands: list[Fbfm13LookupBand] = Field(..., min_length=1)
    name: str = Field("", max_length=255)
    description: str = Field("", max_length=2000)
    tags: list[str] = Field(default_factory=list, max_length=50)
    modifications: list[GridModification] = Field(default_factory=list)

    @field_validator("bands")
    @classmethod
    def no_duplicate_bands(
        cls,
        v: list[Fbfm13LookupBand],
    ) -> list[Fbfm13LookupBand]:
        return validate_no_duplicates(v)


FBFM13_LOOKUP_BAND_METADATA: dict[Fbfm13LookupBand, dict] = {
    Fbfm13LookupBand.fuel_load_1hr: {
        "name": "1-hour Fuel Load",
        "description": "Oven-dry mass per unit area of 1-hour timelag dead fuels.",
        "type": BandType.continuous,
        "unit": "kg/m**2",
    },
    Fbfm13LookupBand.fuel_load_10hr: {
        "name": "10-hour Fuel Load",
        "description": "Oven-dry mass per unit area of 10-hour timelag dead fuels.",
        "type": BandType.continuous,
        "unit": "kg/m**2",
    },
    Fbfm13LookupBand.fuel_load_100hr: {
        "name": "100-hour Fuel Load",
        "description": "Oven-dry mass per unit area of 100-hour timelag dead fuels.",
        "type": BandType.continuous,
        "unit": "kg/m**2",
    },
    Fbfm13LookupBand.fuel_load_live_foliage: {
        "name": "Live Foliage Fuel Load",
        "description": "Oven-dry mass per unit area of live foliage fuels.",
        "type": BandType.continuous,
        "unit": "kg/m**2",
    },
    Fbfm13LookupBand.savr_1hr: {
        "name": "1-hour Surface-Area-to-Volume Ratio",
        "description": "Surface-area-to-volume ratio of 1-hour timelag dead fuels.",
        "type": BandType.continuous,
        "unit": "1/m",
    },
    Fbfm13LookupBand.savr_10hr: {
        "name": "10-hour Surface-Area-to-Volume Ratio",
        "description": "Surface-area-to-volume ratio of 10-hour timelag dead fuels.",
        "type": BandType.continuous,
        "unit": "1/m",
    },
    Fbfm13LookupBand.savr_100hr: {
        "name": "100-hour Surface-Area-to-Volume Ratio",
        "description": "Surface-area-to-volume ratio of 100-hour timelag dead fuels.",
        "type": BandType.continuous,
        "unit": "1/m",
    },
    Fbfm13LookupBand.savr_live_foliage: {
        "name": "Live Foliage Surface-Area-to-Volume Ratio",
        "description": "Surface-area-to-volume ratio of live foliage fuels.",
        "type": BandType.continuous,
        "unit": "1/m",
    },
    Fbfm13LookupBand.fuel_depth: {
        "name": "Fuel Bed Depth",
        "description": "Vertical depth of the surface fuel bed.",
        "type": BandType.continuous,
        "unit": "m",
    },
}


def get_fbfm13_lookup_band(band: Fbfm13LookupBand, index: int) -> Band:
    """Return Band metadata for an FBFM13 lookup band."""
    return Band(key=band.value, index=index, **FBFM13_LOOKUP_BAND_METADATA[band])


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


FBFM40_LOOKUP_BAND_METADATA: dict[Fbfm40LookupBand, dict] = {
    Fbfm40LookupBand.fuel_load_1hr: {
        "name": "1-hour Fuel Load",
        "description": "Oven-dry mass per unit area of 1-hour timelag dead fuels.",
        "type": BandType.continuous,
        "unit": "kg/m**2",
    },
    Fbfm40LookupBand.fuel_load_10hr: {
        "name": "10-hour Fuel Load",
        "description": "Oven-dry mass per unit area of 10-hour timelag dead fuels.",
        "type": BandType.continuous,
        "unit": "kg/m**2",
    },
    Fbfm40LookupBand.fuel_load_100hr: {
        "name": "100-hour Fuel Load",
        "description": "Oven-dry mass per unit area of 100-hour timelag dead fuels.",
        "type": BandType.continuous,
        "unit": "kg/m**2",
    },
    Fbfm40LookupBand.fuel_load_live_herb: {
        "name": "Live Herbaceous Fuel Load",
        "description": "Oven-dry mass per unit area of live herbaceous fuels.",
        "type": BandType.continuous,
        "unit": "kg/m**2",
    },
    Fbfm40LookupBand.fuel_load_live_woody: {
        "name": "Live Woody Fuel Load",
        "description": "Oven-dry mass per unit area of live woody fuels.",
        "type": BandType.continuous,
        "unit": "kg/m**2",
    },
    Fbfm40LookupBand.savr_1hr: {
        "name": "1-hour Surface-Area-to-Volume Ratio",
        "description": "Surface-area-to-volume ratio of 1-hour timelag dead fuels.",
        "type": BandType.continuous,
        "unit": "1/m",
    },
    Fbfm40LookupBand.savr_10hr: {
        "name": "10-hour Surface-Area-to-Volume Ratio",
        "description": "Surface-area-to-volume ratio of 10-hour timelag dead fuels.",
        "type": BandType.continuous,
        "unit": "1/m",
    },
    Fbfm40LookupBand.savr_100hr: {
        "name": "100-hour Surface-Area-to-Volume Ratio",
        "description": "Surface-area-to-volume ratio of 100-hour timelag dead fuels.",
        "type": BandType.continuous,
        "unit": "1/m",
    },
    Fbfm40LookupBand.savr_live_herb: {
        "name": "Live Herbaceous Surface-Area-to-Volume Ratio",
        "description": "Surface-area-to-volume ratio of live herbaceous fuels.",
        "type": BandType.continuous,
        "unit": "1/m",
    },
    Fbfm40LookupBand.savr_live_woody: {
        "name": "Live Woody Surface-Area-to-Volume Ratio",
        "description": "Surface-area-to-volume ratio of live woody fuels.",
        "type": BandType.continuous,
        "unit": "1/m",
    },
    Fbfm40LookupBand.fuel_depth: {
        "name": "Fuel Bed Depth",
        "description": "Vertical depth of the surface fuel bed.",
        "type": BandType.continuous,
        "unit": "m",
    },
}


def get_fbfm40_lookup_band(band: Fbfm40LookupBand, index: int) -> Band:
    """Return Band metadata for an FBFM40 lookup band."""
    return Band(key=band.value, index=index, **FBFM40_LOOKUP_BAND_METADATA[band])
