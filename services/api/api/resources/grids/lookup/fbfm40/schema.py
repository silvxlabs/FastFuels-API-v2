"""
api/v2/resources/grids/lookup/fbfm40/schema.py

Schema models for FBFM40 fuel model lookup operations.

Converts categorical FBFM40 codes (from `/grids/fbfm40/landfire`) to
continuous fuel parameters using Scott-Burgan 40 lookup tables.
"""

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from api.resources.grids.lookup.schema import LookupSource
from api.resources.grids.modification_models import GridModification
from api.resources.grids.schema import Band, BandType, validate_no_duplicates
from api.resources.input_requirements import band_of


class Fbfm40LookupSource(LookupSource):
    """Source for grids created via FBFM40 fuel model lookup.

    Converts categorical FBFM40 codes to continuous fuel parameters
    using Scott-Burgan 40 lookup tables.
    """

    table: Literal["fbfm40"] = "fbfm40"
    source_band: str = Field(
        default="fbfm", description="Band in source grid containing FBFM codes"
    )
    herbaceous_curing: float | None = Field(
        default=None,
        description=(
            "Fraction of herbaceous load cured in dynamic fuel models. Null for "
            "grids created before curing was supported, whose herbaceous load "
            "stayed live."
        ),
    )


class Fbfm40LookupBand(StrEnum):
    """Bands available from FBFM40/SB40 lookup tables."""

    fuel_load_1hr = "fuel_load.1hr"
    fuel_load_10hr = "fuel_load.10hr"
    fuel_load_100hr = "fuel_load.100hr"
    fuel_load_live_herb = "fuel_load.live_herb"
    fuel_load_live_woody = "fuel_load.live_woody"
    fuel_load_dead_herb = "fuel_load.dead_herb"
    fuel_load_fine_dead = "fuel_load.fine_dead"
    savr_1hr = "savr.1hr"
    savr_10hr = "savr.10hr"
    savr_100hr = "savr.100hr"
    savr_live_herb = "savr.live_herb"
    savr_live_woody = "savr.live_woody"
    savr_dead_herb = "savr.dead_herb"
    savr_fine_dead = "savr.fine_dead"
    fuel_depth = "fuel_depth"


class CreateFbfm40LookupRequest(BaseModel):
    """Request to create a grid by looking up FBFM40 fuel parameters.

    Unlike entry-point grid creation requests, domain_id is not required
    because derived grids carry the same domain reference as their source.
    """

    source_grid_id: str = Field(..., description="Grid containing FBFM40 codes")
    source_band: str = Field(
        default="fbfm",
        description="A `fbfm` band of the source grid holding the FBFM codes.",
        json_schema_extra=band_of("source_grid_id", "fbfm"),
    )
    bands: list[Fbfm40LookupBand] = Field(..., min_length=1)
    herbaceous_curing: float = Field(
        default=0.67,
        ge=0.0,
        le=1.0,
        description=(
            "Fraction of herbaceous load that is cured (0 = green, 1 = fully "
            "cured). In dynamic fuel models, this fraction of the live "
            "herbaceous load moves to dead herbaceous fuel. Defaults to 0.67, "
            "the two-thirds-cured reference scenario of Scott and Burgan (2005)."
        ),
    )
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
        "description": (
            "Oven-dry mass per unit area of live herbaceous fuels: the "
            "uncured part of the herbaceous load."
        ),
        "type": BandType.continuous,
        "unit": "kg/m**2",
    },
    Fbfm40LookupBand.fuel_load_live_woody: {
        "name": "Live Woody Fuel Load",
        "description": "Oven-dry mass per unit area of live woody fuels.",
        "type": BandType.continuous,
        "unit": "kg/m**2",
    },
    Fbfm40LookupBand.fuel_load_dead_herb: {
        "name": "Dead Herbaceous Fuel Load",
        "description": (
            "Oven-dry mass per unit area of cured herbaceous fuels. Takes the "
            "dead 1-hour fuel moisture."
        ),
        "type": BandType.continuous,
        "unit": "kg/m**2",
    },
    Fbfm40LookupBand.fuel_load_fine_dead: {
        "name": "Fine Dead Fuel Load",
        "description": (
            "Oven-dry mass per unit area of fine dead fuels: 1-hour plus dead "
            "herbaceous. Takes the dead 1-hour fuel moisture."
        ),
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
    Fbfm40LookupBand.savr_dead_herb: {
        "name": "Dead Herbaceous Surface-Area-to-Volume Ratio",
        "description": (
            "Surface-area-to-volume ratio of dead herbaceous fuels, the same "
            "as live herbaceous."
        ),
        "type": BandType.continuous,
        "unit": "1/m",
    },
    Fbfm40LookupBand.savr_fine_dead: {
        "name": "Fine Dead Surface-Area-to-Volume Ratio",
        "description": (
            "Load-weighted surface-area-to-volume ratio of 1-hour and dead "
            "herbaceous fuels."
        ),
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
