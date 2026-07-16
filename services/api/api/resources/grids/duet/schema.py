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


class DuetFuelType(StrEnum):
    """Fuel types DUET resolves separately.

    `litter` is coniferous plus deciduous; `total` additionally includes grass.
    """

    grass = "grass"
    litter = "litter"
    coniferous = "coniferous"
    deciduous = "deciduous"
    total = "total"


# The duet-tools fuel type and parameter each band is read from.
_FUEL_TYPE_SUFFIX: dict[str, DuetFuelType] = {
    "grass": DuetFuelType.grass,
    "litter": DuetFuelType.litter,
    "litter.coniferous": DuetFuelType.coniferous,
    "litter.deciduous": DuetFuelType.deciduous,
    "total": DuetFuelType.total,
}

_PARAMETER_UNITS: dict[str, str] = {
    "fuel_load": "kg/m**2",
    "fuel_depth": "m",
    "fuel_moisture": "%",
}

_PARAMETER_NOUNS: dict[str, str] = {
    "fuel_load": "Fuel Load",
    "fuel_depth": "Fuel Depth",
    "fuel_moisture": "Fuel Moisture",
}

_FUEL_TYPE_NOUNS: dict[str, str] = {
    "grass": "Grass",
    "litter": "Litter",
    "litter.coniferous": "Coniferous Litter",
    "litter.deciduous": "Deciduous Litter",
    "total": "Total Surface Fuel",
}

_PARAMETER_DESCRIPTIONS: dict[str, str] = {
    "fuel_load": "Oven-dry mass per unit ground area of {noun}.",
    "fuel_depth": "Vertical depth of the {noun} fuel bed.",
    "fuel_moisture": "Moisture content of {noun} (% of oven-dry weight).",
}


def _band_parts(band: DuetBand) -> tuple[str, str]:
    """Split a band key into its (parameter, fuel-type suffix) parts."""
    key = band.value
    for parameter in _PARAMETER_UNITS:
        if key.startswith(f"{parameter}."):
            return parameter, key[len(parameter) + 1 :]
    raise ValueError(f"Unrecognized DUET band key: {key!r}")


def duet_fuel_type(band: DuetBand) -> DuetFuelType:
    """Return the duet-tools fuel type a band is read from."""
    return _FUEL_TYPE_SUFFIX[_band_parts(band)[1]]


def duet_parameter(band: DuetBand) -> str:
    """Return the fuel parameter (`fuel_load` / `fuel_depth` / `fuel_moisture`)."""
    return _band_parts(band)[0]


def _band_definition(band: DuetBand) -> dict:
    parameter, suffix = _band_parts(band)
    noun = _FUEL_TYPE_NOUNS[suffix]
    return {
        "key": band.value,
        "name": f"{noun} {_PARAMETER_NOUNS[parameter]}",
        "description": _PARAMETER_DESCRIPTIONS[parameter].format(noun=noun.lower()),
        "type": BandType.continuous,
        "unit": _PARAMETER_UNITS[parameter],
    }


DUET_BAND_DEFS: dict[DuetBand, dict] = {b: _band_definition(b) for b in DuetBand}


def build_duet_bands(requested: list[DuetBand]) -> list[Band]:
    """Build Band objects for requested DUET bands with indices in request order."""
    return [Band(index=i, **DUET_BAND_DEFS[band]) for i, band in enumerate(requested)]


class CalibrationMethod(StrEnum):
    """How a target's values are imposed on DUET's spatial pattern.

    Both `maxmin` and `meansd` rescale only cells that already carry fuel —
    cells DUET left empty stay empty.
    """

    maxmin = "maxmin"
    meansd = "meansd"
    constant = "constant"


# Which fields each method consumes. Enforced by ValuesTarget's validator.
_METHOD_FIELDS: dict[CalibrationMethod, tuple[str, ...]] = {
    CalibrationMethod.maxmin: ("max", "min"),
    CalibrationMethod.meansd: ("mean", "sd"),
    CalibrationMethod.constant: ("value",),
}


class ValuesTarget(BaseModel):
    """Calibrate against explicit numbers.

    `method` selects which fields apply: `maxmin` reads `max`/`min`, `meansd`
    reads `mean`/`sd`, `constant` reads `value`. Supplying a field belonging to
    another method is rejected rather than ignored.

    Method-specific fields are optional here and checked by a validator rather
    than being split into one model per method. That split is the usual pattern
    (see `voxelize`'s `BiomassSource`), but it only works at the top level of a
    union: nesting a `method`-discriminated union inside this `source`-
    discriminated one makes Pydantic emit an inline schema where OpenAPI
    requires a `$ref` string, producing a spec that fails validation and breaks
    client generation.
    """

    model_config = ConfigDict(extra="forbid")

    source: Literal["values"] = "values"
    method: CalibrationMethod = Field(
        description="How the target values are imposed on DUET's pattern."
    )
    max: float | None = Field(
        default=None, ge=0, description="Target maximum (maxmin)."
    )
    min: float | None = Field(
        default=None, ge=0, description="Target minimum (maxmin)."
    )
    mean: float | None = Field(default=None, ge=0, description="Target mean (meansd).")
    sd: float | None = Field(
        default=None, ge=0, description="Target standard deviation (meansd)."
    )
    value: float | None = Field(
        default=None, ge=0, description="Target value (constant)."
    )

    @model_validator(mode="after")
    def validate_method_fields(self) -> Self:
        expected = _METHOD_FIELDS[self.method]
        # `min` defaults to 0 rather than being required, matching duet-tools.
        required = [f for f in expected if f != "min"]
        missing = [f for f in required if getattr(self, f) is None]
        if missing:
            raise ValueError(f"method '{self.method.value}' requires {missing}.")
        extra = [
            field
            for method, fields in _METHOD_FIELDS.items()
            if method != self.method
            for field in fields
            if field not in expected and getattr(self, field) is not None
        ]
        if extra:
            raise ValueError(
                f"method '{self.method.value}' does not use {sorted(set(extra))}; "
                f"it reads {list(expected)}."
            )
        if self.method is CalibrationMethod.maxmin:
            if self.min is None:
                self.min = 0.0
            if self.max < self.min:
                raise ValueError("max must be greater than or equal to min.")
        return self


# `source` is a discriminated union with one member today. Deriving targets from
# an FBFM40 grid is the intended second member (issue #449): it needs the SB40
# loading table, which currently lives in griddle and is not reachable from
# treevox, and it turns on whether SB40's timber-understory models should count
# toward the litter target — an open question for the DUET authors. Keeping the
# discriminator means adding it later is additive rather than breaking.
CalibrationTarget = Annotated[
    ValuesTarget,
    Field(discriminator="source"),
]


class ParameterCalibration(BaseModel):
    """Per-fuel-type calibration targets for one fuel parameter.

    `all` is exclusive: it calibrates every fuel type together and cannot be
    combined with a per-type target.
    """

    model_config = ConfigDict(extra="forbid")

    grass: CalibrationTarget | None = None
    coniferous: CalibrationTarget | None = None
    deciduous: CalibrationTarget | None = None
    litter: CalibrationTarget | None = None
    all: CalibrationTarget | None = None

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

    fuel_load: ParameterCalibration | None = None
    fuel_depth: ParameterCalibration | None = None
    fuel_moisture: ParameterCalibration | None = None

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
        ge=0,
        lt=360,
        description="Prevailing wind direction in whole degrees clockwise from north.",
    )
    wind_variability: int = Field(
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
        requested = {duet_parameter(band) for band in self.bands}
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
