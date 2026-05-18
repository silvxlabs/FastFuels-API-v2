"""
Unit tests for api/v2/resources/grids/uniform/schema.py

Tests the uniform grid schema models and constants.
These are pure unit tests with no external dependencies.
"""

import pytest
from api.resources.grids.schema import BandType
from api.resources.grids.uniform.examples import ALL_UNIFORM_EXAMPLE_VALUES
from api.resources.grids.uniform.schema import (
    UNIFORM_BAND_DEFS,
    CreateUniformRequest,
    UniformBand,
    UniformBandInput,
    UniformSource,
    build_uniform_bands,
)
from pydantic import ValidationError

from lib.units import validate_unit


class TestUniformBand:
    """Tests for UniformBand enum."""

    def test_has_12_members(self):
        """There are exactly 12 predefined bands."""
        assert len(UniformBand) == 12

    def test_fuel_moisture_1hr(self):
        assert UniformBand.fuel_moisture_1hr == "fuel_moisture.1hr"

    def test_fuel_moisture_10hr(self):
        assert UniformBand.fuel_moisture_10hr == "fuel_moisture.10hr"

    def test_fuel_moisture_100hr(self):
        assert UniformBand.fuel_moisture_100hr == "fuel_moisture.100hr"

    def test_fuel_moisture_live_herb(self):
        assert UniformBand.fuel_moisture_live_herb == "fuel_moisture.live_herb"

    def test_fuel_moisture_live_woody(self):
        assert UniformBand.fuel_moisture_live_woody == "fuel_moisture.live_woody"

    def test_curing(self):
        assert UniformBand.curing == "curing"

    def test_fuel_load_1hr(self):
        assert UniformBand.fuel_load_1hr == "fuel_load.1hr"

    def test_fuel_load_10hr(self):
        assert UniformBand.fuel_load_10hr == "fuel_load.10hr"

    def test_fuel_load_100hr(self):
        assert UniformBand.fuel_load_100hr == "fuel_load.100hr"

    def test_fuel_load_live_herb(self):
        assert UniformBand.fuel_load_live_herb == "fuel_load.live_herb"

    def test_fuel_load_live_woody(self):
        assert UniformBand.fuel_load_live_woody == "fuel_load.live_woody"

    def test_fuel_depth(self):
        assert UniformBand.fuel_depth == "fuel_depth"

    def test_created_from_string(self):
        """Bands can be created from their string value."""
        b = UniformBand("fuel_moisture.1hr")
        assert b == UniformBand.fuel_moisture_1hr


class TestUniformBandDefs:
    """Tests for UNIFORM_BAND_DEFS constant."""

    def test_all_bands_have_defs(self):
        """Every enum member has a definition."""
        for b in UniformBand:
            assert b in UNIFORM_BAND_DEFS

    def test_all_defs_have_key(self):
        """Every def has a key field."""
        for b, d in UNIFORM_BAND_DEFS.items():
            assert "key" in d

    def test_all_defs_have_type(self):
        """Every def has a type field."""
        for b, d in UNIFORM_BAND_DEFS.items():
            assert "type" in d

    def test_all_defs_have_unit(self):
        """Every def has a unit field."""
        for b, d in UNIFORM_BAND_DEFS.items():
            assert "unit" in d

    def test_all_defs_are_continuous(self):
        """All uniform bands are continuous."""
        for d in UNIFORM_BAND_DEFS.values():
            assert d["type"] == BandType.continuous

    def test_fuel_moisture_unit_is_percent(self):
        """Fuel moisture bands use % unit."""
        for b in [
            UniformBand.fuel_moisture_1hr,
            UniformBand.fuel_moisture_10hr,
            UniformBand.fuel_moisture_100hr,
            UniformBand.fuel_moisture_live_herb,
            UniformBand.fuel_moisture_live_woody,
        ]:
            assert UNIFORM_BAND_DEFS[b]["unit"] == "%"

    def test_curing_unit_is_percent(self):
        assert UNIFORM_BAND_DEFS[UniformBand.curing]["unit"] == "%"

    def test_fuel_load_unit_is_kg_per_m2(self):
        """Fuel load bands use kg/m**2 unit."""
        for b in [
            UniformBand.fuel_load_1hr,
            UniformBand.fuel_load_10hr,
            UniformBand.fuel_load_100hr,
            UniformBand.fuel_load_live_herb,
            UniformBand.fuel_load_live_woody,
        ]:
            assert UNIFORM_BAND_DEFS[b]["unit"] == "kg/m**2"

    def test_fuel_depth_unit_is_meters(self):
        assert UNIFORM_BAND_DEFS[UniformBand.fuel_depth]["unit"] == "m"

    def test_all_units_are_canonical(self):
        for d in UNIFORM_BAND_DEFS.values():
            validate_unit(d.get("unit"))

    def test_key_matches_enum_value(self):
        """Each def's key matches the enum member's string value."""
        for b, d in UNIFORM_BAND_DEFS.items():
            assert d["key"] == b.value


class TestUniformBandInput:
    """Tests for UniformBandInput model."""

    def test_key_required(self):
        """key field is required."""
        with pytest.raises(ValidationError):
            UniformBandInput(value=6.0)

    def test_value_required(self):
        """value field is required."""
        with pytest.raises(ValidationError):
            UniformBandInput(key="fuel_moisture.1hr")

    def test_float_value(self):
        """Float values are accepted."""
        band = UniformBandInput(key="fuel_moisture.1hr", value=6.5)
        assert band.value == 6.5

    def test_int_value(self):
        """Integer values are accepted."""
        band = UniformBandInput(key="fuel_moisture.1hr", value=6)
        assert band.value == 6

    def test_invalid_key_rejected(self):
        """Invalid key string is rejected."""
        with pytest.raises(ValidationError):
            UniformBandInput(key="not_a_band", value=6.0)

    def test_key_stored_as_enum(self):
        """Key is stored as the enum member."""
        band = UniformBandInput(key="fuel_moisture.1hr", value=6.0)
        assert band.key == UniformBand.fuel_moisture_1hr


class TestUniformSource:
    """Tests for UniformSource model."""

    def test_name_is_always_uniform(self):
        """The name field is always 'uniform'."""
        source = UniformSource(
            bands=[UniformBandInput(key="fuel_moisture.1hr", value=6.0)],
            resolution=2.0,
        )
        assert source.name == "uniform"

    def test_name_cannot_be_overridden(self):
        """The name field cannot be set to anything other than 'uniform'."""
        with pytest.raises(ValidationError):
            UniformSource(
                name="other",
                bands=[UniformBandInput(key="fuel_moisture.1hr", value=6.0)],
                resolution=2.0,
            )

    def test_bands_required(self):
        """bands field is required."""
        with pytest.raises(ValidationError):
            UniformSource(resolution=2.0)

    def test_resolution_required(self):
        """resolution field is required."""
        with pytest.raises(ValidationError):
            UniformSource(bands=[UniformBandInput(key="fuel_moisture.1hr", value=6.0)])

    def test_model_dump(self):
        """Model serializes correctly."""
        source = UniformSource(
            bands=[UniformBandInput(key="fuel_moisture.1hr", value=6.0)],
            resolution=2.0,
        )
        data = source.model_dump()
        assert data["name"] == "uniform"
        assert data["resolution"] == 2.0
        assert len(data["bands"]) == 1
        assert data["bands"][0]["key"] == "fuel_moisture.1hr"
        assert data["bands"][0]["value"] == 6.0


class TestCreateUniformRequest:
    """Tests for CreateUniformRequest model.

    domain_id now comes from the URL path parameter, not the request body.
    """

    def test_minimal_valid_request(self):
        """Minimal request with required fields."""
        request = CreateUniformRequest(
            resolution=2.0,
            bands=[{"key": "fuel_moisture.1hr", "value": 6.0}],
        )
        assert request.resolution == 2.0
        assert len(request.bands) == 1
        assert request.name == ""
        assert request.description == ""
        assert request.tags == []
        assert request.modifications == []

    def test_resolution_is_required(self):
        """resolution field is required."""
        with pytest.raises(ValidationError):
            CreateUniformRequest(
                bands=[{"key": "fuel_moisture.1hr", "value": 6.0}],
            )

    def test_resolution_must_be_ge_1(self):
        """resolution must be >= 1."""
        with pytest.raises(ValidationError):
            CreateUniformRequest(
                resolution=0.5,
                bands=[{"key": "fuel_moisture.1hr", "value": 6.0}],
            )

    def test_resolution_exactly_1_is_valid(self):
        """resolution of exactly 1 is accepted."""
        request = CreateUniformRequest(
            resolution=1.0,
            bands=[{"key": "fuel_moisture.1hr", "value": 6.0}],
        )
        assert request.resolution == 1.0

    def test_bands_is_required(self):
        """bands field is required."""
        with pytest.raises(ValidationError):
            CreateUniformRequest(resolution=2.0)

    def test_empty_bands_rejected(self):
        """Empty bands list is rejected."""
        with pytest.raises(ValidationError):
            CreateUniformRequest(
                resolution=2.0,
                bands=[],
            )

    def test_duplicate_bands_rejected(self):
        """Duplicate band keys are rejected."""
        with pytest.raises(ValidationError):
            CreateUniformRequest(
                resolution=2.0,
                bands=[
                    {"key": "fuel_moisture.1hr", "value": 6.0},
                    {"key": "fuel_moisture.1hr", "value": 8.0},
                ],
            )

    def test_multiple_unique_bands_accepted(self):
        """Multiple unique band keys are accepted."""
        request = CreateUniformRequest(
            resolution=2.0,
            bands=[
                {"key": "fuel_moisture.1hr", "value": 6.0},
                {"key": "fuel_moisture.10hr", "value": 8.0},
            ],
        )
        assert len(request.bands) == 2

    def test_full_request_with_all_fields(self):
        """Full request with all optional fields."""
        request = CreateUniformRequest(
            resolution=5.0,
            bands=[
                {"key": "fuel_load.1hr", "value": 0.15},
                {"key": "fuel_depth", "value": 0.3},
            ],
            name="Custom fuel",
            description="For sensitivity analysis",
            tags=["fuel-load", "test"],
        )
        assert request.name == "Custom fuel"
        assert request.description == "For sensitivity analysis"
        assert request.tags == ["fuel-load", "test"]


class TestBuildUniformBands:
    """Tests for build_uniform_bands function."""

    def test_single_band(self):
        """Single band input returns one Band with correct metadata."""
        inputs = [UniformBandInput(key="fuel_moisture.1hr", value=6.0)]
        bands = build_uniform_bands(inputs)

        assert len(bands) == 1
        assert bands[0].key == "fuel_moisture.1hr"
        assert bands[0].type == BandType.continuous
        assert bands[0].unit == "%"
        assert bands[0].index == 0

    def test_multiple_bands(self):
        """Multiple band inputs get sequential indices."""
        inputs = [
            UniformBandInput(key="fuel_moisture.1hr", value=6.0),
            UniformBandInput(key="fuel_load.1hr", value=0.15),
            UniformBandInput(key="fuel_depth", value=0.3),
        ]
        bands = build_uniform_bands(inputs)

        assert len(bands) == 3
        assert bands[0].key == "fuel_moisture.1hr"
        assert bands[0].unit == "%"
        assert bands[0].index == 0
        assert bands[1].key == "fuel_load.1hr"
        assert bands[1].unit == "kg/m**2"
        assert bands[1].index == 1
        assert bands[2].key == "fuel_depth"
        assert bands[2].unit == "m"
        assert bands[2].index == 2

    def test_correct_type_for_all_bands(self):
        """All bands produce continuous band type."""
        for b in UniformBand:
            inputs = [UniformBandInput(key=b, value=1.0)]
            bands = build_uniform_bands(inputs)
            assert bands[0].type == BandType.continuous


class TestExampleValidation:
    """Validate that all documented examples pass schema validation."""

    @pytest.mark.parametrize("example_name,example_value", ALL_UNIFORM_EXAMPLE_VALUES)
    def test_example_is_valid(self, example_name, example_value):
        """Each documented example should pass schema validation."""
        request = CreateUniformRequest(**example_value)
        assert request.resolution == example_value["resolution"]
        assert len(request.bands) == len(example_value["bands"])
