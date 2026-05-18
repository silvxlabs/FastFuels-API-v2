"""
api/v2/resources/grids/uniform/schema.py

Schema models for uniform (constant-value) grid sources.

Uniform grids fill every cell with a user-specified constant value for one
or more bands. Useful for fuel moisture scenarios, constant fuel loads,
and other spatially-uniform inputs.

NOTE: The UniformBand enum currently covers 12 core bands. As new data
sources (FCCS, etc.) are added, this predefined list may need to evolve
into a more scalable system (e.g., free-form keys with a reference endpoint).
"""

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from api.resources.grids.schema import (
    Band,
    BandType,
    CreateGridRequestBase,
    validate_no_duplicates,
)


class UniformBand(StrEnum):
    """Predefined bands available for uniform grids."""

    fuel_moisture_1hr = "fuel_moisture.1hr"
    fuel_moisture_10hr = "fuel_moisture.10hr"
    fuel_moisture_100hr = "fuel_moisture.100hr"
    fuel_moisture_live_herb = "fuel_moisture.live_herb"
    fuel_moisture_live_woody = "fuel_moisture.live_woody"
    curing = "curing"
    fuel_load_1hr = "fuel_load.1hr"
    fuel_load_10hr = "fuel_load.10hr"
    fuel_load_100hr = "fuel_load.100hr"
    fuel_load_live_herb = "fuel_load.live_herb"
    fuel_load_live_woody = "fuel_load.live_woody"
    fuel_depth = "fuel_depth"


UNIFORM_BAND_DEFS: dict[UniformBand, dict] = {
    UniformBand.fuel_moisture_1hr: {
        "key": "fuel_moisture.1hr",
        "type": BandType.continuous,
        "unit": "%",
    },
    UniformBand.fuel_moisture_10hr: {
        "key": "fuel_moisture.10hr",
        "type": BandType.continuous,
        "unit": "%",
    },
    UniformBand.fuel_moisture_100hr: {
        "key": "fuel_moisture.100hr",
        "type": BandType.continuous,
        "unit": "%",
    },
    UniformBand.fuel_moisture_live_herb: {
        "key": "fuel_moisture.live_herb",
        "type": BandType.continuous,
        "unit": "%",
    },
    UniformBand.fuel_moisture_live_woody: {
        "key": "fuel_moisture.live_woody",
        "type": BandType.continuous,
        "unit": "%",
    },
    UniformBand.curing: {
        "key": "curing",
        "type": BandType.continuous,
        "unit": "%",
    },
    UniformBand.fuel_load_1hr: {
        "key": "fuel_load.1hr",
        "type": BandType.continuous,
        "unit": "kg/m**2",
    },
    UniformBand.fuel_load_10hr: {
        "key": "fuel_load.10hr",
        "type": BandType.continuous,
        "unit": "kg/m**2",
    },
    UniformBand.fuel_load_100hr: {
        "key": "fuel_load.100hr",
        "type": BandType.continuous,
        "unit": "kg/m**2",
    },
    UniformBand.fuel_load_live_herb: {
        "key": "fuel_load.live_herb",
        "type": BandType.continuous,
        "unit": "kg/m**2",
    },
    UniformBand.fuel_load_live_woody: {
        "key": "fuel_load.live_woody",
        "type": BandType.continuous,
        "unit": "kg/m**2",
    },
    UniformBand.fuel_depth: {
        "key": "fuel_depth",
        "type": BandType.continuous,
        "unit": "m",
    },
}


class UniformBandInput(BaseModel):
    """A single band specification for a uniform grid.

    Users provide a band key (from the predefined list) and a constant value.
    The API resolves the key to unit and type.
    """

    key: UniformBand
    value: float | int


class UniformSource(BaseModel):
    """Source specification for uniform grids.

    Stored in the grid document for reproducibility — contains the full
    configuration needed to recreate the grid.
    """

    name: Literal["uniform"] = "uniform"
    bands: list[UniformBandInput]
    resolution: float


class CreateUniformRequest(CreateGridRequestBase):
    """Request to create a uniform (constant-value) grid.

    Each band fills the entire domain with a single value at the specified
    resolution. No default resolution — it must be explicitly provided since
    uniform grids have no "native resolution."
    """

    resolution: float = Field(..., ge=1, description="Grid resolution in meters")
    bands: list[UniformBandInput] = Field(..., min_length=1)

    @model_validator(mode="after")
    def validate_unique_bands(self):
        """Ensure no duplicate band keys in bands."""
        validate_no_duplicates([b.key for b in self.bands])
        return self


def build_uniform_bands(inputs: list[UniformBandInput]) -> list[Band]:
    """Build Band objects from uniform band inputs.

    Looks up key, unit, and type from UNIFORM_BAND_DEFS and assigns
    sequential indices.
    """
    return [Band(index=i, **UNIFORM_BAND_DEFS[inp.key]) for i, inp in enumerate(inputs)]
