"""
api/v2/resources/grids/fuel_moisture/dead/fosberg/schema.py

Schema models for the Fosberg 1-hour dead fuel moisture content (DFMC) grid.

This is a derived grid: it computes a single 2-D `fuel_moisture.dead.1hr` band
from a topography grid (slope + aspect) and a leaflux irradiance grid
(surface shading) using the Fosberg & Deeming (1971) model in
`fastfuels_core.fuel_moisture.fosberg`.
"""

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from api.resources.grids.schema import Band, BandType


class FuelMoistureMonth(StrEnum):
    """Month of the burn scenario, selecting the Fosberg correction table."""

    january = "January"
    february = "February"
    march = "March"
    april = "April"
    may = "May"
    june = "June"
    july = "July"
    august = "August"
    september = "September"
    october = "October"
    november = "November"
    december = "December"


class RelativeElevation(StrEnum):
    """Site elevation relative to the reference weather station.

    This is a Fosberg correction category, NOT the topography elevation band:
    `below` = 1000-2000 ft below the station, `near` = within 1000 ft (no
    correction), `above` = 1000-2000 ft above the station.
    """

    below = "below"
    near = "near"
    above = "above"


class FosbergFuelMoistureSource(BaseModel):
    """Persisted source metadata for a Fosberg 1-hr DFMC grid.

    Records both input grids (with the checksums they were built from, for
    staleness detection) and every scalar weather/scenario parameter, so the
    grid is exactly reproducible from its own `source` dict.
    """

    model_config = ConfigDict(extra="forbid")

    name: Literal["fosberg"] = "fosberg"

    source_topography_grid_id: str
    source_topography_grid_checksum: str | None = Field(
        default=None,
        description=(
            "The topography grid's `checksum` at the time this grid was built "
            "from it. Compare against the source grid's current `checksum` to "
            "tell whether the source has changed since."
        ),
    )
    source_irradiance_grid_id: str
    source_irradiance_grid_checksum: str | None = Field(
        default=None,
        description=(
            "The irradiance grid's `checksum` at the time this grid was built from it."
        ),
    )

    dry_bulb_temp: float
    relative_humidity: float
    time: int
    month: FuelMoistureMonth
    elevation: RelativeElevation


class CreateFosbergFuelMoistureRequest(BaseModel):
    """Request body for a Fosberg 1-hour dead fuel moisture content grid.

    Does not extend CreateSourceGridRequestBase: this is a grid -> grid
    derivation with no external raster and no alignment input. The output
    inherits the topography grid's domain, CRS, transform, and georeference.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field("", max_length=255)
    description: str = Field("", max_length=2000)
    tags: list[str] = Field(default_factory=list, max_length=50)

    source_topography_grid_id: str = Field(
        description=(
            "ID of a completed 2D topography grid with `slope` and `aspect` "
            "bands (both in degrees)."
        ),
    )
    source_irradiance_grid_id: str = Field(
        description=(
            "ID of a completed leaflux irradiance grid with an "
            "`irradiance.surface.relative` band, on the topography grid's exact "
            "horizontal lattice (equivalent CRS, y/x shape, and affine "
            "transform). Per-cell shading is derived as "
            "1 - irradiance.surface.relative. Resample one grid onto the other "
            "when their lattices differ."
        ),
    )

    dry_bulb_temp: float = Field(
        ge=10,
        description=(
            "Dry-bulb air temperature in degrees Fahrenheit (the Fosberg table "
            "lineage is Fahrenheit). Must be >= 10."
        ),
    )
    relative_humidity: float = Field(
        ge=0,
        le=100,
        description="Relative humidity as a percent (0-100).",
    )
    time: int = Field(
        ge=800,
        le=1959,
        description=(
            "Local time of day in 24-hour HHMM form (e.g. 1200 for noon). "
            "Restricted to 0800-1959; the model has no daytime table outside "
            "that window."
        ),
    )
    month: FuelMoistureMonth = Field(
        description=(
            "Month of the burn scenario, selecting the Fosberg seasonal "
            "correction table."
        ),
    )
    elevation: RelativeElevation = Field(
        default=RelativeElevation.near,
        description=(
            "Site elevation relative to the reference weather station "
            "(`below` | `near` | `above`), a Fosberg correction category. This "
            "is NOT the topography elevation band. Default `near` applies no "
            "correction."
        ),
    )

    @field_validator("time")
    @classmethod
    def _valid_hhmm(cls, value: int) -> int:
        if value % 100 >= 60:
            raise ValueError("time must be a valid HHMM clock value (minutes 00-59)")
        return value


DEAD_1HR_BAND = Band(
    key="fuel_moisture.dead.1hr",
    name="1-hour Dead Fuel Moisture",
    description=(
        "1-hour timelag dead fuel moisture content from the Fosberg & Deeming "
        "(1971) model, as a percent of oven-dry fuel weight."
    ),
    type=BandType.continuous,
    unit="%",
    index=0,
)
