"""
api/v2/resources/grids/duet/schema.py

Schema models for DUET surface fuel grids.

DUET distributes leaf and needle litter from a 3D canopy onto the ground along
wind-driven elliptical fall trajectories, then grows grass as a function of
shade and litter cover. It consumes a 3D tree grid and produces 2D surface
bands.

Includes the band vocabulary, the calibration models, and the request and
persisted-source schemas.
"""

from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from api.resources.grids.schema import Band, BandType, validate_no_duplicates


class DuetBand(StrEnum):
    """Available output bands for DUET surface fuel grids.

    DUET decomposes surface fuels by fuel *type*, not by size class, so this
    vocabulary mirrors duet-tools' fuel types rather than the `fuel_load.1hr`
    size classes used by the FBFM40 lookup bands.
    """

    fuel_load_grass = "fuel_load.grass"
    fuel_load_litter = "fuel_load.litter"
    fuel_load_litter_coniferous = "fuel_load.litter.coniferous"
    fuel_load_litter_deciduous = "fuel_load.litter.deciduous"
    fuel_load_total = "fuel_load.total"

    fuel_depth_grass = "fuel_depth.grass"
    fuel_depth_litter = "fuel_depth.litter"
    fuel_depth_litter_coniferous = "fuel_depth.litter.coniferous"
    fuel_depth_litter_deciduous = "fuel_depth.litter.deciduous"
    fuel_depth_total = "fuel_depth.total"

    fuel_moisture_grass = "fuel_moisture.grass"
    fuel_moisture_litter = "fuel_moisture.litter"
    fuel_moisture_litter_coniferous = "fuel_moisture.litter.coniferous"
    fuel_moisture_litter_deciduous = "fuel_moisture.litter.deciduous"
    fuel_moisture_total = "fuel_moisture.total"


DUET_BAND_DEFS: dict[DuetBand, dict] = {
    DuetBand.fuel_load_grass: {
        "key": "fuel_load.grass",
        "name": "Grass Fuel Load",
        "description": "Oven-dry mass per unit ground area of grass.",
        "type": BandType.continuous,
        "unit": "kg/m**2",
    },
    DuetBand.fuel_load_litter: {
        "key": "fuel_load.litter",
        "name": "Litter Fuel Load",
        "description": "Oven-dry mass per unit ground area of litter.",
        "type": BandType.continuous,
        "unit": "kg/m**2",
    },
    DuetBand.fuel_load_litter_coniferous: {
        "key": "fuel_load.litter.coniferous",
        "name": "Coniferous Litter Fuel Load",
        "description": "Oven-dry mass per unit ground area of coniferous litter.",
        "type": BandType.continuous,
        "unit": "kg/m**2",
    },
    DuetBand.fuel_load_litter_deciduous: {
        "key": "fuel_load.litter.deciduous",
        "name": "Deciduous Litter Fuel Load",
        "description": "Oven-dry mass per unit ground area of deciduous litter.",
        "type": BandType.continuous,
        "unit": "kg/m**2",
    },
    DuetBand.fuel_load_total: {
        "key": "fuel_load.total",
        "name": "Total Surface Fuel Load",
        "description": (
            "Oven-dry mass per unit ground area of all surface fuel "
            "(grass plus litter)."
        ),
        "type": BandType.continuous,
        "unit": "kg/m**2",
    },
    DuetBand.fuel_depth_grass: {
        "key": "fuel_depth.grass",
        "name": "Grass Fuel Depth",
        "description": "Vertical depth of the grass fuel bed.",
        "type": BandType.continuous,
        "unit": "m",
    },
    DuetBand.fuel_depth_litter: {
        "key": "fuel_depth.litter",
        "name": "Litter Fuel Depth",
        "description": "Vertical depth of the litter fuel bed.",
        "type": BandType.continuous,
        "unit": "m",
    },
    DuetBand.fuel_depth_litter_coniferous: {
        "key": "fuel_depth.litter.coniferous",
        "name": "Coniferous Litter Fuel Depth",
        "description": "Vertical depth of the coniferous litter fuel bed.",
        "type": BandType.continuous,
        "unit": "m",
    },
    DuetBand.fuel_depth_litter_deciduous: {
        "key": "fuel_depth.litter.deciduous",
        "name": "Deciduous Litter Fuel Depth",
        "description": "Vertical depth of the deciduous litter fuel bed.",
        "type": BandType.continuous,
        "unit": "m",
    },
    DuetBand.fuel_depth_total: {
        "key": "fuel_depth.total",
        "name": "Total Surface Fuel Depth",
        "description": (
            "Vertical depth of the total surface fuel bed (grass plus litter)."
        ),
        "type": BandType.continuous,
        "unit": "m",
    },
    DuetBand.fuel_moisture_grass: {
        "key": "fuel_moisture.grass",
        "name": "Grass Fuel Moisture",
        "description": "Moisture content of grass (% of oven-dry weight).",
        "type": BandType.continuous,
        "unit": "%",
    },
    DuetBand.fuel_moisture_litter: {
        "key": "fuel_moisture.litter",
        "name": "Litter Fuel Moisture",
        "description": "Moisture content of litter (% of oven-dry weight).",
        "type": BandType.continuous,
        "unit": "%",
    },
    DuetBand.fuel_moisture_litter_coniferous: {
        "key": "fuel_moisture.litter.coniferous",
        "name": "Coniferous Litter Fuel Moisture",
        "description": "Moisture content of coniferous litter (% of oven-dry weight).",
        "type": BandType.continuous,
        "unit": "%",
    },
    DuetBand.fuel_moisture_litter_deciduous: {
        "key": "fuel_moisture.litter.deciduous",
        "name": "Deciduous Litter Fuel Moisture",
        "description": "Moisture content of deciduous litter (% of oven-dry weight).",
        "type": BandType.continuous,
        "unit": "%",
    },
    DuetBand.fuel_moisture_total: {
        "key": "fuel_moisture.total",
        "name": "Total Surface Fuel Moisture",
        "description": "Moisture content of all surface fuel (% of oven-dry weight).",
        "type": BandType.continuous,
        "unit": "%",
    },
}


def build_duet_bands(requested: list[DuetBand]) -> list[Band]:
    """Build Band objects for requested DUET bands with indices in request order."""
    return [Band(index=i, **DUET_BAND_DEFS[band]) for i, band in enumerate(requested)]


class DuetMaxMinCalibrationTarget(BaseModel):
    """Rescale a fuel type to a target maximum and minimum.

    Best when fuel data are limited, or when their distribution does not
    resemble DUET's.
    """

    model_config = ConfigDict(extra="forbid")

    method: Literal["maxmin"] = "maxmin"
    max: float = Field(ge=0, description="Target maximum.")
    min: float = Field(default=0.0, ge=0, description="Target minimum.")

    @model_validator(mode="after")
    def validate_max_ge_min(self) -> Self:
        if self.max < self.min:
            raise ValueError("max must be greater than or equal to min.")
        return self


class DuetMeanSdCalibrationTarget(BaseModel):
    """Rescale a fuel type to a target mean and standard deviation.

    Appropriate only when the targets come from a dataset large enough to
    approximate a normal distribution.
    """

    model_config = ConfigDict(extra="forbid")

    method: Literal["meansd"] = "meansd"
    mean: float = Field(ge=0, description="Target mean.")
    sd: float = Field(ge=0, description="Target standard deviation.")


class DuetConstantCalibrationTarget(BaseModel):
    """Assign a single value to every fuel-bearing cell.

    Reasonable only when that value is the only one available.
    """

    model_config = ConfigDict(extra="forbid")

    method: Literal["constant"] = "constant"
    value: float = Field(ge=0, description="Target value.")


DuetCalibrationTarget = Annotated[
    DuetMaxMinCalibrationTarget
    | DuetMeanSdCalibrationTarget
    | DuetConstantCalibrationTarget,
    Field(discriminator="method"),
]


class DuetParameterCalibration(BaseModel):
    """Per-fuel-type calibration targets for one fuel parameter.

    `all` is exclusive: it calibrates every fuel type together and cannot be
    combined with a per-type target.
    """

    model_config = ConfigDict(extra="forbid")

    grass: DuetCalibrationTarget | None = None
    coniferous: DuetCalibrationTarget | None = None
    deciduous: DuetCalibrationTarget | None = None
    litter: DuetCalibrationTarget | None = None
    all: DuetCalibrationTarget | None = None

    @model_validator(mode="after")
    def validate_targets(self) -> Self:
        named = {
            name: target
            for name in ("grass", "coniferous", "deciduous", "litter")
            if (target := getattr(self, name)) is not None
        }
        if self.all is not None and named:
            raise ValueError(
                "'all' calibrates every fuel type at once and cannot be combined "
                f"with per-type targets: {', '.join(sorted(named))}."
            )
        if self.all is None and not named:
            raise ValueError(
                "At least one fuel type target is required: grass, coniferous, "
                "deciduous, litter, or all."
            )
        if self.litter is not None and (
            self.coniferous is not None or self.deciduous is not None
        ):
            raise ValueError(
                "'litter' already covers coniferous and deciduous; specify either "
                "'litter' or the individual types, not both."
            )
        return self


class DuetCalibration(BaseModel):
    """Calibration targets, keyed by fuel parameter.

    Each parameter is calibrated independently; omitted parameters keep DUET's
    raw values.
    """

    model_config = ConfigDict(extra="forbid")

    fuel_load: DuetParameterCalibration | None = None
    fuel_depth: DuetParameterCalibration | None = None
    fuel_moisture: DuetParameterCalibration | None = None

    @model_validator(mode="after")
    def validate_any_parameter(self) -> Self:
        if not any((self.fuel_load, self.fuel_depth, self.fuel_moisture)):
            raise ValueError(
                "calibration requires at least one of: fuel_load, fuel_depth, "
                "fuel_moisture. Omit `calibration` entirely to store raw DUET "
                "output."
            )
        return self


def _default_bands() -> list[DuetBand]:
    return [DuetBand.fuel_load_grass, DuetBand.fuel_load_litter]


# Bands the source tree grid must carry. DUET derives litter from foliage mass,
# keys its litter parameters off species, and requires a canopy moisture field.
DUET_REQUIRED_SOURCE_BANDS = ("bulk_density.foliage.live", "spcd", "fuel_moisture.live")


class DuetSourceBase(BaseModel):
    """Fields shared by the DUET request and the persisted source."""

    years_since_burn: int = Field(
        ge=1,
        le=100,
        description=(
            "Years of litter accumulation to simulate. DUET begins the year of "
            "the last burn, when standing grass and litter have been consumed, "
            "so this is the stand's time since fire. It is the highest-leverage "
            "parameter in the model and also drives runtime."
        ),
    )
    # Integers, not floats: DUET reads both with a Fortran integer list read, so
    # a fractional bearing aborts the model with "Bad integer for item 1".
    wind_direction: int = Field(
        default=270,
        ge=0,
        lt=360,
        description="Prevailing wind direction in whole degrees clockwise from north.",
    )
    wind_variability: int = Field(
        default=30,
        ge=0,
        le=180,
        description="Angular spread of wind direction, in whole degrees.",
    )


class DuetSource(DuetSourceBase):
    """Source metadata stored on the Grid document for reproducibility.

    Records the tree grid DUET consumed and every resolved parameter, so the
    grid can be exactly reproduced.
    """

    model_config = ConfigDict(extra="forbid")

    operation: Literal["duet"] = "duet"
    input: Literal["grid"] = "grid"
    entity: Literal["tree"] = "tree"

    source_grid_id: str
    source_grid_checksum: str | None = Field(
        default=None,
        description=(
            "The source grid's `checksum` at the time this grid was created from "
            "it. Compare it against the source grid's current `checksum` to tell "
            "whether the source has changed since."
        ),
    )
    bands: list[DuetBand]
    calibration: DuetCalibration | None = None


class CreateDuetRequest(DuetSourceBase):
    """Request body for creating a DUET surface fuel grid from a tree grid.

    Does not extend CreateGridRequestBase: like the 3D grids it derives from,
    DUET grids do not support modifications.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field("", max_length=255)
    description: str = Field("", max_length=2000)
    tags: list[str] = Field(default_factory=list, max_length=50)

    source_grid_id: str = Field(
        description=(
            "ID of a completed 3D tree grid carrying the "
            "`bulk_density.foliage.live`, `spcd`, and `fuel_moisture.live` bands."
        ),
    )
    bands: list[DuetBand] = Field(
        default_factory=_default_bands,
        min_length=1,
        description=(
            "Which output bands to produce. Defaults to `fuel_load.grass` and "
            "`fuel_load.litter`."
        ),
    )
    calibration: DuetCalibration | None = Field(
        default=None,
        description=(
            "Optional calibration targets. DUET supplies the spatial pattern of "
            "surface fuels; its raw magnitudes are not physical. Without "
            "calibration the raw values are stored as-is."
        ),
    )

    @field_validator("bands")
    @classmethod
    def no_duplicate_bands(cls, v: list[DuetBand]) -> list[DuetBand]:
        return validate_no_duplicates(v)

    @model_validator(mode="after")
    def validate_calibration_covers_requested_bands(self) -> Self:
        """Reject calibration of a parameter no requested band reads.

        Calibrating `fuel_depth` while requesting only `fuel_load.*` bands runs
        the calibration and then discards it, which reads as a silent no-op.
        """
        if self.calibration is None:
            return self
        requested = {band.value.split(".", 1)[0] for band in self.bands}
        unused = [
            parameter
            for parameter in ("fuel_load", "fuel_depth", "fuel_moisture")
            if getattr(self.calibration, parameter) is not None
            and parameter not in requested
        ]
        if unused:
            raise ValueError(
                f"calibration targets {unused} have no matching requested band. "
                f"Request a {unused[0]}.* band or drop the target."
            )
        return self
