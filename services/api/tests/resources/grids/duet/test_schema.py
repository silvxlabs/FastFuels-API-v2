"""
Unit tests for api/resources/grids/duet/schema.py.

Pure schema tests with no external dependencies.
"""

import pytest
from api.resources.grids.duet.examples import CREATE_DUET_OPENAPI_EXAMPLES
from api.resources.grids.duet.schema import (
    DUET_BAND_DEFS,
    CreateDuetRequest,
    DuetBand,
    DuetCalibration,
    DuetConstantCalibrationTarget,
    DuetMaxMinCalibrationTarget,
    DuetMeanSdCalibrationTarget,
    DuetParameterCalibration,
    DuetSource,
    build_duet_bands,
)
from api.resources.grids.schema import BandType
from pydantic import ValidationError


class TestDuetBandDefs:
    def test_every_band_has_a_definition(self):
        assert set(DUET_BAND_DEFS) == set(DuetBand)

    def test_definitions_are_complete(self):
        for band, definition in DUET_BAND_DEFS.items():
            assert definition["key"] == band.value
            assert definition["name"]
            assert definition["description"]
            assert definition["type"] == BandType.continuous

    @pytest.mark.parametrize(
        "band,unit",
        [
            (DuetBand.fuel_load_grass, "kg/m**2"),
            (DuetBand.fuel_load_total, "kg/m**2"),
            (DuetBand.fuel_depth_litter, "m"),
            (DuetBand.fuel_moisture_litter_coniferous, "%"),
        ],
    )
    def test_units_are_canonical(self, band, unit):
        # UDUNITS-2 ASCII with ** exponents — never kg/m2 or kg/m^2.
        assert DUET_BAND_DEFS[band]["unit"] == unit

    def test_build_bands_indexes_in_request_order(self):
        requested = [DuetBand.fuel_depth_litter, DuetBand.fuel_load_grass]
        bands = build_duet_bands(requested)
        assert [b.key for b in bands] == ["fuel_depth.litter", "fuel_load.grass"]
        assert [b.index for b in bands] == [0, 1]


class TestCalibrationTargets:
    def test_maxmin_defaults_min_to_zero(self):
        assert DuetMaxMinCalibrationTarget(max=5.0).min == 0.0

    def test_meansd(self):
        target = DuetMeanSdCalibrationTarget(mean=0.5, sd=0.25)
        assert (target.mean, target.sd) == (0.5, 0.25)

    def test_constant(self):
        assert DuetConstantCalibrationTarget(value=0.03).value == 0.03

    @pytest.mark.parametrize(
        "model,kwargs,missing",
        [
            (DuetMaxMinCalibrationTarget, {}, "max"),
            (DuetMeanSdCalibrationTarget, {"mean": 1.0}, "sd"),
            (DuetMeanSdCalibrationTarget, {"sd": 1.0}, "mean"),
            (DuetConstantCalibrationTarget, {}, "value"),
        ],
    )
    def test_missing_field_for_method_is_rejected(self, model, kwargs, missing):
        with pytest.raises(ValidationError, match=missing):
            model(**kwargs)

    @pytest.mark.parametrize(
        "model,kwargs",
        [
            (DuetMaxMinCalibrationTarget, {"max": 5.0, "mean": 1.0}),
            (DuetMeanSdCalibrationTarget, {"mean": 1.0, "sd": 1.0, "value": 2.0}),
            (DuetConstantCalibrationTarget, {"value": 1.0, "max": 5.0}),
        ],
    )
    def test_field_from_another_method_is_rejected_not_ignored(self, model, kwargs):
        # extra="forbid" refuses fields belonging to a different method.
        with pytest.raises(ValidationError, match="not permitted"):
            model(**kwargs)

    def test_max_below_min_is_rejected(self):
        with pytest.raises(ValidationError, match="greater than or equal to min"):
            DuetMaxMinCalibrationTarget(max=1.0, min=5.0)

    def test_negative_target_is_rejected(self):
        with pytest.raises(ValidationError):
            DuetConstantCalibrationTarget(value=-1.0)

    def test_method_selects_the_right_model(self):
        # The discriminator routes a plain dict to the matching target model.
        calibration = DuetParameterCalibration(
            grass={"method": "meansd", "mean": 0.5, "sd": 0.25},
            litter={"method": "maxmin", "max": 5.0},
        )
        assert isinstance(calibration.grass, DuetMeanSdCalibrationTarget)
        assert isinstance(calibration.litter, DuetMaxMinCalibrationTarget)


class TestDuetParameterCalibration:
    def test_per_fuel_type_targets(self):
        calibration = DuetParameterCalibration(
            grass={"method": "constant", "value": 1.0},
            litter={"method": "maxmin", "max": 5.0},
        )
        assert calibration.grass.value == 1.0
        assert calibration.litter.max == 5.0

    def test_requires_at_least_one_target(self):
        with pytest.raises(ValidationError, match="At least one fuel type"):
            DuetParameterCalibration()

    def test_all_is_exclusive(self):
        with pytest.raises(ValidationError, match="cannot be combined"):
            DuetParameterCalibration(
                all={"method": "constant", "value": 1.0},
                grass={"method": "constant", "value": 1.0},
            )

    @pytest.mark.parametrize("fuel_type", ["coniferous", "deciduous"])
    def test_litter_cannot_be_combined_with_its_parts(self, fuel_type):
        with pytest.raises(ValidationError, match="already covers"):
            DuetParameterCalibration(
                litter={"method": "constant", "value": 1.0},
                **{fuel_type: {"method": "constant", "value": 1.0}},
            )

    def test_coniferous_and_deciduous_may_be_set_together(self):
        calibration = DuetParameterCalibration(
            coniferous={"method": "constant", "value": 1.0},
            deciduous={"method": "constant", "value": 2.0},
        )
        assert calibration.coniferous.value == 1.0
        assert calibration.deciduous.value == 2.0


class TestDuetCalibration:
    def test_requires_at_least_one_parameter(self):
        with pytest.raises(ValidationError, match="at least one of"):
            DuetCalibration()

    def test_parameters_are_independent(self):
        calibration = DuetCalibration(
            fuel_load={"grass": {"method": "constant", "value": 1.0}}
        )
        assert calibration.fuel_depth is None
        assert calibration.fuel_moisture is None


class TestCreateDuetRequest:
    def test_minimal_request(self):
        body = CreateDuetRequest(source_grid_id="g1", years_since_burn=25)
        assert body.bands == [DuetBand.fuel_load_grass, DuetBand.fuel_load_litter]
        assert body.wind_direction == 270.0
        assert body.wind_variability == 30.0
        assert body.calibration is None

    def test_years_since_burn_is_required(self):
        # No default: DUET starts from the last fire, and there is no
        # defensible stand age to assume on the user's behalf.
        with pytest.raises(ValidationError, match="years_since_burn"):
            CreateDuetRequest(source_grid_id="g1")

    @pytest.mark.parametrize("years", [0, -1, 101])
    def test_years_since_burn_is_bounded(self, years):
        with pytest.raises(ValidationError):
            CreateDuetRequest(source_grid_id="g1", years_since_burn=years)

    @pytest.mark.parametrize("direction", [-1, 360, 400])
    def test_wind_direction_is_a_bearing(self, direction):
        with pytest.raises(ValidationError):
            CreateDuetRequest(
                source_grid_id="g1", years_since_burn=5, wind_direction=direction
            )

    @pytest.mark.parametrize("variability", [-1, 181])
    def test_wind_variability_is_bounded(self, variability):
        with pytest.raises(ValidationError):
            CreateDuetRequest(
                source_grid_id="g1",
                years_since_burn=5,
                wind_variability=variability,
            )

    def test_duplicate_bands_are_rejected(self):
        with pytest.raises(ValidationError, match="Duplicate"):
            CreateDuetRequest(
                source_grid_id="g1",
                years_since_burn=5,
                bands=[DuetBand.fuel_load_grass, DuetBand.fuel_load_grass],
            )

    def test_empty_bands_are_rejected(self):
        with pytest.raises(ValidationError):
            CreateDuetRequest(source_grid_id="g1", years_since_burn=5, bands=[])

    def test_unknown_field_is_rejected(self):
        # random_seed was a v1 knob and is provably inert — reject it rather
        # than accept and ignore.
        with pytest.raises(ValidationError):
            CreateDuetRequest(source_grid_id="g1", years_since_burn=5, random_seed=42)

    def test_calibrating_a_parameter_with_no_requested_band_is_rejected(self):
        with pytest.raises(ValidationError, match="no matching requested band"):
            CreateDuetRequest(
                source_grid_id="g1",
                years_since_burn=5,
                bands=[DuetBand.fuel_load_grass],
                calibration={
                    "fuel_depth": {"grass": {"method": "constant", "value": 0.1}}
                },
            )

    def test_calibration_matching_a_requested_band_is_accepted(self):
        body = CreateDuetRequest(
            source_grid_id="g1",
            years_since_burn=5,
            bands=[DuetBand.fuel_load_grass, DuetBand.fuel_depth_grass],
            calibration={
                "fuel_load": {"grass": {"method": "meansd", "mean": 0.5, "sd": 0.25}},
                "fuel_depth": {"grass": {"method": "constant", "value": 0.3}},
            },
        )
        assert body.calibration.fuel_load.grass.mean == 0.5
        assert body.calibration.fuel_depth.grass.value == 0.3

    def test_calibrating_a_fuel_type_with_no_requested_band_is_rejected(self):
        # Right parameter (fuel_load is requested), wrong fuel type: only grass
        # is requested, so the litter calibration would be computed and dropped.
        with pytest.raises(ValidationError, match="computed and then discarded"):
            CreateDuetRequest(
                source_grid_id="g1",
                years_since_burn=5,
                bands=[DuetBand.fuel_load_grass],
                calibration={"fuel_load": {"litter": {"method": "maxmin", "max": 5.0}}},
            )

    def test_litter_calibration_is_read_by_a_total_band(self):
        # `fuel_load.total` sums grass + litter, so a litter calibration is
        # reflected in it — this must be accepted, not flagged as unused.
        body = CreateDuetRequest(
            source_grid_id="g1",
            years_since_burn=5,
            bands=[DuetBand.fuel_load_total],
            calibration={"fuel_load": {"litter": {"method": "maxmin", "max": 5.0}}},
        )
        assert body.calibration.fuel_load.litter.max == 5.0

    def test_coniferous_calibration_is_read_by_the_integrated_litter_band(self):
        # `fuel_load.litter` integrates coniferous + deciduous, so calibrating a
        # single litter layer is reflected in it.
        body = CreateDuetRequest(
            source_grid_id="g1",
            years_since_burn=5,
            bands=[DuetBand.fuel_load_litter],
            calibration={
                "fuel_load": {"coniferous": {"method": "constant", "value": 1.0}}
            },
        )
        assert body.calibration.fuel_load.coniferous.value == 1.0


class TestDuetSource:
    def test_carries_the_dispatch_triple(self):
        source = DuetSource(
            source_grid_id="g1",
            years_since_burn=25,
            wind_direction=270,
            wind_variability=30,
            bands=[DuetBand.fuel_load_grass],
        )
        # treevox routes on this triple; `input` is "grid" because DUET
        # consumes a grid rather than an inventory.
        assert (source.operation, source.input, source.entity) == (
            "duet",
            "grid",
            "tree",
        )

    def test_round_trips_through_firestore_shape(self):
        source = DuetSource(
            source_grid_id="g1",
            source_grid_checksum="abc123",
            years_since_burn=25,
            wind_direction=270,
            wind_variability=30,
            bands=[DuetBand.fuel_load_grass],
            calibration={"fuel_load": {"grass": {"method": "maxmin", "max": 5.0}}},
        )
        dumped = source.model_dump(mode="json", exclude_none=True)
        assert dumped["bands"] == ["fuel_load.grass"]
        target = dumped["calibration"]["fuel_load"]["grass"]
        # The discriminated target model carries only its own method's fields, so
        # the handler reads exactly what applies (min defaults to 0.0).
        assert target == {"method": "maxmin", "max": 5.0, "min": 0.0}
        assert DuetSource(**dumped) == source


class TestExamples:
    def test_every_openapi_example_validates(self):
        for name, example in CREATE_DUET_OPENAPI_EXAMPLES.items():
            CreateDuetRequest(**example["value"]), name
