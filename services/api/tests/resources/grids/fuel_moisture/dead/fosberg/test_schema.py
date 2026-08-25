"""
Unit tests for the Fosberg 1-hr dead fuel moisture schema.

Pure schema tests with no external dependencies.
"""

import pytest
from api.resources.grids.fuel_moisture.dead.fosberg.examples import (
    ALL_FOSBERG_EXAMPLE_VALUES,
)
from api.resources.grids.fuel_moisture.dead.fosberg.schema import (
    DEAD_1HR_BAND,
    CreateFosbergFuelMoistureRequest,
    FosbergFuelMoistureSource,
    FuelMoistureMonth,
    RelativeElevation,
)
from api.resources.grids.schema import BandType
from pydantic import ValidationError

from lib.units import validate_unit


def _body(**overrides) -> dict:
    body = {
        "source_topography_grid_id": "topo-1",
        "source_irradiance_grid_id": "irr-1",
        "dry_bulb_temp": 75,
        "relative_humidity": 30,
        "time": 1200,
        "month": "June",
    }
    body.update(overrides)
    return body


class TestDead1hrBand:
    def test_band_key_and_metadata(self):
        assert DEAD_1HR_BAND.key == "fuel_moisture.dead.1hr"
        assert DEAD_1HR_BAND.type == BandType.continuous
        assert DEAD_1HR_BAND.unit == "%"
        assert DEAD_1HR_BAND.index == 0
        assert DEAD_1HR_BAND.name
        assert DEAD_1HR_BAND.description

    def test_unit_is_canonical(self):
        validate_unit(DEAD_1HR_BAND.unit)


class TestFuelMoistureMonth:
    def test_values_are_full_month_names(self):
        assert FuelMoistureMonth.june == "June"
        assert {m.value for m in FuelMoistureMonth} == {
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        }


class TestCreateFosbergFuelMoistureRequest:
    def test_minimal_valid_request(self):
        req = CreateFosbergFuelMoistureRequest(**_body())
        assert req.source_topography_grid_id == "topo-1"
        assert req.source_irradiance_grid_id == "irr-1"
        assert req.dry_bulb_temp == 75
        assert req.relative_humidity == 30
        assert req.time == 1200
        assert req.month == FuelMoistureMonth.june
        # elevation defaults to `near` (no station correction)
        assert req.elevation == RelativeElevation.near
        assert req.name == ""
        assert req.tags == []

    @pytest.mark.parametrize(
        "field",
        [
            "source_topography_grid_id",
            "source_irradiance_grid_id",
            "dry_bulb_temp",
            "relative_humidity",
            "time",
            "month",
        ],
    )
    def test_required_fields(self, field):
        body = _body()
        del body[field]
        with pytest.raises(ValidationError):
            CreateFosbergFuelMoistureRequest(**body)

    def test_temp_below_min_rejected(self):
        with pytest.raises(ValidationError):
            CreateFosbergFuelMoistureRequest(**_body(dry_bulb_temp=9))

    def test_temp_at_min_accepted(self):
        req = CreateFosbergFuelMoistureRequest(**_body(dry_bulb_temp=10))
        assert req.dry_bulb_temp == 10

    @pytest.mark.parametrize("rh", [-1, 101])
    def test_relative_humidity_out_of_range_rejected(self, rh):
        with pytest.raises(ValidationError):
            CreateFosbergFuelMoistureRequest(**_body(relative_humidity=rh))

    @pytest.mark.parametrize("rh", [0, 100])
    def test_relative_humidity_bounds_accepted(self, rh):
        req = CreateFosbergFuelMoistureRequest(**_body(relative_humidity=rh))
        assert req.relative_humidity == rh

    @pytest.mark.parametrize("time", [759, 799, 1960, 2000, 0])
    def test_time_out_of_range_rejected(self, time):
        with pytest.raises(ValidationError):
            CreateFosbergFuelMoistureRequest(**_body(time=time))

    @pytest.mark.parametrize("time", [800, 1200, 1959])
    def test_time_bounds_accepted(self, time):
        req = CreateFosbergFuelMoistureRequest(**_body(time=time))
        assert req.time == time

    def test_invalid_month_rejected(self):
        with pytest.raises(ValidationError):
            CreateFosbergFuelMoistureRequest(**_body(month="Junuary"))

    @pytest.mark.parametrize("elevation", ["below", "near", "above"])
    def test_elevation_values_accepted(self, elevation):
        req = CreateFosbergFuelMoistureRequest(**_body(elevation=elevation))
        assert req.elevation == RelativeElevation(elevation)

    def test_invalid_elevation_rejected(self):
        with pytest.raises(ValidationError):
            CreateFosbergFuelMoistureRequest(**_body(elevation="sea-level"))

    def test_unknown_field_rejected(self):
        with pytest.raises(ValidationError):
            CreateFosbergFuelMoistureRequest(**_body(resolution=10))

    def test_name_too_long_rejected(self):
        with pytest.raises(ValidationError):
            CreateFosbergFuelMoistureRequest(**_body(name="x" * 256))

    @pytest.mark.parametrize("example_name,example_value", ALL_FOSBERG_EXAMPLE_VALUES)
    def test_documented_examples_are_schema_valid(self, example_name, example_value):
        req = CreateFosbergFuelMoistureRequest(**example_value)
        assert req.source_topography_grid_id, example_name


class TestFosbergFuelMoistureSource:
    def _source(self, **overrides) -> dict:
        body = {
            "source_topography_grid_id": "topo-1",
            "source_irradiance_grid_id": "irr-1",
            "dry_bulb_temp": 75,
            "relative_humidity": 30,
            "time": 1200,
            "month": "June",
            "elevation": "near",
        }
        body.update(overrides)
        return body

    def test_name_discriminator_fixed(self):
        source = FosbergFuelMoistureSource(**self._source())
        assert source.name == "fosberg"

    def test_checksums_default_to_none(self):
        source = FosbergFuelMoistureSource(**self._source())
        assert source.source_topography_grid_checksum is None
        assert source.source_irradiance_grid_checksum is None

    def test_checksums_round_trip(self):
        source = FosbergFuelMoistureSource(
            **self._source(
                source_topography_grid_checksum="t1",
                source_irradiance_grid_checksum="i1",
            )
        )
        dumped = source.model_dump(mode="json")
        assert dumped["source_topography_grid_checksum"] == "t1"
        assert dumped["source_irradiance_grid_checksum"] == "i1"
        assert dumped["name"] == "fosberg"
        assert dumped["month"] == "June"
        assert dumped["elevation"] == "near"

    def test_extra_fields_forbidden(self):
        with pytest.raises(ValidationError):
            FosbergFuelMoistureSource(**self._source(resolution=10))
