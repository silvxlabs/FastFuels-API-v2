"""
api/v2/resources/grids/lookup/fccs/schema.py

Schema models for FCCS fuel model lookup operations.

Converts categorical FCCS codes (from `/grids/fccs/landfire`) to continuous
fuel parameters using the FOFEM FCCS fuelbed lookup table.
"""

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from api.resources.grids.lookup.schema import LookupSource
from api.resources.grids.modification_models import GridModification
from api.resources.grids.schema import Band, BandType, validate_no_duplicates


class FccsLookupSource(LookupSource):
    """Source for grids created via FCCS fuel model lookup.

    Converts categorical FCCS codes to continuous fuel parameters using the
    FOFEM FCCS fuelbed lookup table.
    """

    table: Literal["fccs"] = "fccs"
    source_band: str = Field(
        default="fccs", description="Band in source grid containing FCCS codes"
    )


class FccsLookupBand(StrEnum):
    """Bands available from the FOFEM FCCS fuelbed lookup table.

    A starting subset of the parameters FOFEM provides. Region/scenario
    columns, the FOFEM cover-group code, and emission-factor columns are
    not exposed here since they aren't structural fuel parameters.
    """

    fuel_load_litter = "fuel_load.litter"
    fuel_load_duff = "fuel_load.duff"
    duff_depth = "duff_depth"
    fuel_load_live_shrub = "fuel_load.live_shrub"
    fuel_load_live_herb = "fuel_load.live_herb"
    fuel_load_1hr = "fuel_load.1hr"
    fuel_load_10hr = "fuel_load.10hr"
    fuel_load_100hr = "fuel_load.100hr"
    fuel_load_1000hr_sound = "fuel_load.1000hr_sound"
    fuel_load_1000hr_rotten = "fuel_load.1000hr_rotten"
    fuel_load_live_foliage = "fuel_load.live_foliage"
    fuel_load_live_branch = "fuel_load.live_branch"


class CreateFccsLookupRequest(BaseModel):
    """Request to create a grid by looking up FCCS fuel parameters.

    Unlike entry-point grid creation requests, domain_id is not required
    because derived grids carry the same domain reference as their source.
    """

    source_grid_id: str = Field(..., description="Grid containing FCCS codes")
    source_band: str = Field(
        default="fccs",
        description="Band in source grid containing FCCS codes",
    )
    bands: list[FccsLookupBand] = Field(..., min_length=1)
    name: str = Field("", max_length=255)
    description: str = Field("", max_length=2000)
    tags: list[str] = Field(default_factory=list, max_length=50)
    modifications: list[GridModification] = Field(default_factory=list)

    @field_validator("bands")
    @classmethod
    def no_duplicate_bands(
        cls,
        v: list[FccsLookupBand],
    ) -> list[FccsLookupBand]:
        return validate_no_duplicates(v)


FCCS_LOOKUP_BAND_METADATA: dict[FccsLookupBand, dict] = {
    FccsLookupBand.fuel_load_litter: {
        "name": "Litter Fuel Load",
        "description": "Oven-dry mass per unit area of the litter layer.",
        "type": BandType.continuous,
        "unit": "kg/m**2",
    },
    FccsLookupBand.fuel_load_duff: {
        "name": "Duff Fuel Load",
        "description": "Oven-dry mass per unit area of the duff layer.",
        "type": BandType.continuous,
        "unit": "kg/m**2",
    },
    FccsLookupBand.duff_depth: {
        "name": "Duff Layer Depth",
        "description": ("Vertical depth of the duff layer."),
        "type": BandType.continuous,
        "unit": "m",
    },
    FccsLookupBand.fuel_load_live_shrub: {
        "name": "Live Shrub Fuel Load",
        "description": "Oven-dry mass per unit area of live shrub fuels.",
        "type": BandType.continuous,
        "unit": "kg/m**2",
    },
    FccsLookupBand.fuel_load_live_herb: {
        "name": "Live Herbaceous Fuel Load",
        "description": "Oven-dry mass per unit area of live herbaceous fuels.",
        "type": BandType.continuous,
        "unit": "kg/m**2",
    },
    FccsLookupBand.fuel_load_1hr: {
        "name": "1-hour Fuel Load",
        "description": ("Oven-dry mass per unit area of 1-hour timelag dead fuels "),
        "type": BandType.continuous,
        "unit": "kg/m**2",
    },
    FccsLookupBand.fuel_load_10hr: {
        "name": "10-hour Fuel Load",
        "description": ("Oven-dry mass per unit area of 10-hour timelag dead fuels "),
        "type": BandType.continuous,
        "unit": "kg/m**2",
    },
    FccsLookupBand.fuel_load_100hr: {
        "name": "100-hour Fuel Load",
        "description": ("Oven-dry mass per unit area of 100-hour timelag dead fuels "),
        "type": BandType.continuous,
        "unit": "kg/m**2",
    },
    FccsLookupBand.fuel_load_1000hr_sound: {
        "name": "1000-hour Sound Fuel Load",
        "description": (
            "Oven-dry mass per unit area of sound (non-decayed) dead fuels "
            "over 3 in. diameter, summed across FOFEM's 3-9, 9-20, and "
            "20+ in. size classes."
        ),
        "type": BandType.continuous,
        "unit": "kg/m**2",
    },
    FccsLookupBand.fuel_load_1000hr_rotten: {
        "name": "1000-hour Rotten Fuel Load",
        "description": (
            "Oven-dry mass per unit area of rotten (decayed) dead fuels "
            "over 3 in. diameter, summed across FOFEM's 3-9, 9-20, and "
            "20+ in. size classes."
        ),
        "type": BandType.continuous,
        "unit": "kg/m**2",
    },
    FccsLookupBand.fuel_load_live_foliage: {
        "name": "Live Foliage Fuel Load",
        "description": "Oven-dry mass per unit area of live crown foliage.",
        "type": BandType.continuous,
        "unit": "kg/m**2",
    },
    FccsLookupBand.fuel_load_live_branch: {
        "name": "Live Branch Fuel Load",
        "description": "Oven-dry mass per unit area of live crown branches.",
        "type": BandType.continuous,
        "unit": "kg/m**2",
    },
}


def get_fccs_lookup_band(band: FccsLookupBand, index: int) -> Band:
    """Return Band metadata for an FCCS lookup band."""
    return Band(key=band.value, index=index, **FCCS_LOOKUP_BAND_METADATA[band])
