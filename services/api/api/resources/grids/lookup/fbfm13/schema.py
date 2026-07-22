"""
api/v2/resources/grids/lookup/fbfm13/schema.py

Schema models for FBFM13 fuel model lookup operations.

Converts categorical Anderson 13 fuel model codes (from
`/grids/fbfm13/landfire`) to continuous fuel parameters using the
Anderson 13 lookup table.
"""

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from api.resources.grids.lookup.schema import LookupSource
from api.resources.grids.modification_models import GridModification
from api.resources.grids.schema import Band, BandType, validate_no_duplicates


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
