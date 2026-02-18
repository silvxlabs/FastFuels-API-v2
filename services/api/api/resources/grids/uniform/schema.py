"""
api/v2/resources/grids/uniform/schema.py

Schema models for uniform (constant-value) grid sources.

Uniform grids fill every cell with a user-specified constant value for one
or more fuel quantities. Useful for fuel moisture scenarios, constant fuel
loads, and other spatially-uniform inputs.

NOTE: The UniformQuantity enum currently covers 12 core quantities. As new
data sources (FCCS, etc.) are added, this predefined list may need to evolve
into a more scalable system (e.g., free-form keys with a reference endpoint).
"""

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from api.resources.grids.schema import Band, BandType, CreateGridRequestBase


class UniformQuantity(StrEnum):
    """Predefined quantities available for uniform grids."""

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


UNIFORM_QUANTITY_DEFS: dict[UniformQuantity, dict] = {
    UniformQuantity.fuel_moisture_1hr: {
        "key": "fuel_moisture.1hr",
        "type": BandType.continuous,
        "unit": "%",
    },
    UniformQuantity.fuel_moisture_10hr: {
        "key": "fuel_moisture.10hr",
        "type": BandType.continuous,
        "unit": "%",
    },
    UniformQuantity.fuel_moisture_100hr: {
        "key": "fuel_moisture.100hr",
        "type": BandType.continuous,
        "unit": "%",
    },
    UniformQuantity.fuel_moisture_live_herb: {
        "key": "fuel_moisture.live_herb",
        "type": BandType.continuous,
        "unit": "%",
    },
    UniformQuantity.fuel_moisture_live_woody: {
        "key": "fuel_moisture.live_woody",
        "type": BandType.continuous,
        "unit": "%",
    },
    UniformQuantity.curing: {
        "key": "curing",
        "type": BandType.continuous,
        "unit": "%",
    },
    UniformQuantity.fuel_load_1hr: {
        "key": "fuel_load.1hr",
        "type": BandType.continuous,
        "unit": "kg/m²",
    },
    UniformQuantity.fuel_load_10hr: {
        "key": "fuel_load.10hr",
        "type": BandType.continuous,
        "unit": "kg/m²",
    },
    UniformQuantity.fuel_load_100hr: {
        "key": "fuel_load.100hr",
        "type": BandType.continuous,
        "unit": "kg/m²",
    },
    UniformQuantity.fuel_load_live_herb: {
        "key": "fuel_load.live_herb",
        "type": BandType.continuous,
        "unit": "kg/m²",
    },
    UniformQuantity.fuel_load_live_woody: {
        "key": "fuel_load.live_woody",
        "type": BandType.continuous,
        "unit": "kg/m²",
    },
    UniformQuantity.fuel_depth: {
        "key": "fuel_depth",
        "type": BandType.continuous,
        "unit": "m",
    },
}


class UniformBandInput(BaseModel):
    """A single band specification for a uniform grid.

    Users provide a quantity (from the predefined list) and a constant value.
    The API resolves the quantity to a band key, unit, and type.
    """

    quantity: UniformQuantity
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
    def validate_unique_quantities(self):
        """Ensure no duplicate quantities in bands."""
        quantities = [b.quantity for b in self.bands]
        if len(quantities) != len(set(quantities)):
            raise ValueError("Duplicate quantities are not allowed")
        return self


def build_uniform_bands(inputs: list[UniformBandInput]) -> list[Band]:
    """Build Band objects from uniform band inputs.

    Looks up key, unit, and type from UNIFORM_QUANTITY_DEFS and assigns
    sequential indices.
    """
    return [
        Band(index=i, **UNIFORM_QUANTITY_DEFS[inp.quantity])
        for i, inp in enumerate(inputs)
    ]
