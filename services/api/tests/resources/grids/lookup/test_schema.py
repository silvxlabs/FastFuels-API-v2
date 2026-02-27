"""
Unit tests for api/v2/resources/grids/lookup/schema.py

Tests the FBFM40 lookup schema models and constants.
These are pure unit tests with no external dependencies.
"""

import pytest
from api.resources.grids.lookup.schema import (
    FBFM40_LOOKUP_BAND_METADATA,
    CreateFbfm40LookupRequest,
    Fbfm40LookupQuantity,
    Fbfm40LookupSource,
    LookupSource,
    get_fbfm40_lookup_band,
)
from api.resources.grids.schema import BandType
from pydantic import ValidationError


class TestLookupSource:
    """Tests for LookupSource base model."""

    def test_name_is_always_lookup(self):
        """The name field is always 'lookup'."""
        source = LookupSource(
            table="fbfm40", source_grid_id="grid-123", source_band="fbfm"
        )
        assert source.name == "lookup"

    def test_name_cannot_be_overridden(self):
        """The name field cannot be set to anything other than 'lookup'."""
        with pytest.raises(ValidationError):
            LookupSource(
                name="other",
                table="fbfm40",
                source_grid_id="grid-123",
                source_band="fbfm",
            )

    def test_table_is_required(self):
        """The table field is required."""
        with pytest.raises(ValidationError):
            LookupSource(source_grid_id="grid-123", source_band="fbfm")

    def test_source_grid_id_is_required(self):
        """The source_grid_id field is required."""
        with pytest.raises(ValidationError):
            LookupSource(table="fbfm40", source_band="fbfm")

    def test_source_band_is_required(self):
        """The source_band field is required."""
        with pytest.raises(ValidationError):
            LookupSource(table="fbfm40", source_grid_id="grid-123")

    def test_model_dump(self):
        """Model serializes correctly."""
        source = LookupSource(
            table="fbfm40", source_grid_id="grid-123", source_band="fbfm"
        )
        data = source.model_dump()
        assert data["name"] == "lookup"
        assert data["table"] == "fbfm40"
        assert data["source_grid_id"] == "grid-123"
        assert data["source_band"] == "fbfm"


class TestFbfm40LookupSource:
    """Tests for Fbfm40LookupSource model."""

    def test_table_is_always_fbfm40(self):
        """The table field is always 'fbfm40'."""
        source = Fbfm40LookupSource(source_grid_id="grid-123")
        assert source.table == "fbfm40"

    def test_table_cannot_be_overridden(self):
        """The table field cannot be set to anything other than 'fbfm40'."""
        with pytest.raises(ValidationError):
            Fbfm40LookupSource(table="other", source_grid_id="grid-123")

    def test_name_is_always_lookup(self):
        """The name field is always 'lookup'."""
        source = Fbfm40LookupSource(source_grid_id="grid-123")
        assert source.name == "lookup"

    def test_source_band_defaults_to_fbfm(self):
        """The source_band field defaults to 'fbfm'."""
        source = Fbfm40LookupSource(source_grid_id="grid-123")
        assert source.source_band == "fbfm"

    def test_source_band_can_be_overridden(self):
        """The source_band field can be set to a different value."""
        source = Fbfm40LookupSource(
            source_grid_id="grid-123", source_band="custom_band"
        )
        assert source.source_band == "custom_band"

    def test_source_grid_id_is_required(self):
        """The source_grid_id field is required."""
        with pytest.raises(ValidationError):
            Fbfm40LookupSource()

    def test_model_dump(self):
        """Model serializes correctly."""
        source = Fbfm40LookupSource(source_grid_id="grid-123")
        data = source.model_dump()
        assert data["name"] == "lookup"
        assert data["table"] == "fbfm40"
        assert data["source_grid_id"] == "grid-123"
        assert data["source_band"] == "fbfm"


class TestFbfm40LookupQuantity:
    """Tests for Fbfm40LookupQuantity enum."""

    def test_has_14_members(self):
        """There are exactly 14 predefined quantities."""
        assert len(Fbfm40LookupQuantity) == 14

    def test_fuel_load_1hr(self):
        assert Fbfm40LookupQuantity.fuel_load_1hr == "fuel_load.1hr"

    def test_fuel_load_10hr(self):
        assert Fbfm40LookupQuantity.fuel_load_10hr == "fuel_load.10hr"

    def test_fuel_load_100hr(self):
        assert Fbfm40LookupQuantity.fuel_load_100hr == "fuel_load.100hr"

    def test_fuel_load_live_herb(self):
        assert Fbfm40LookupQuantity.fuel_load_live_herb == "fuel_load.live_herb"

    def test_fuel_load_live_woody(self):
        assert Fbfm40LookupQuantity.fuel_load_live_woody == "fuel_load.live_woody"

    def test_savr_1hr(self):
        assert Fbfm40LookupQuantity.savr_1hr == "savr.1hr"

    def test_savr_10hr(self):
        assert Fbfm40LookupQuantity.savr_10hr == "savr.10hr"

    def test_savr_100hr(self):
        assert Fbfm40LookupQuantity.savr_100hr == "savr.100hr"

    def test_savr_live_herb(self):
        assert Fbfm40LookupQuantity.savr_live_herb == "savr.live_herb"

    def test_savr_live_woody(self):
        assert Fbfm40LookupQuantity.savr_live_woody == "savr.live_woody"

    def test_fuel_depth(self):
        assert Fbfm40LookupQuantity.fuel_depth == "fuel_depth"

    def test_moisture_of_extinction(self):
        assert Fbfm40LookupQuantity.moisture_of_extinction == "moisture_of_extinction"

    def test_heat_content(self):
        assert Fbfm40LookupQuantity.heat_content == "heat_content"

    def test_is_dynamic(self):
        assert Fbfm40LookupQuantity.is_dynamic == "is_dynamic"

    def test_created_from_string(self):
        """Quantities can be created from their string value."""
        q = Fbfm40LookupQuantity("fuel_load.1hr")
        assert q == Fbfm40LookupQuantity.fuel_load_1hr


class TestCreateFbfm40LookupRequest:
    """Tests for CreateFbfm40LookupRequest model."""

    def test_minimal_valid_request(self):
        """Minimal request with required fields only."""
        request = CreateFbfm40LookupRequest(
            source_grid_id="grid-123",
            quantities=["fuel_load.1hr"],
        )
        assert request.source_grid_id == "grid-123"
        assert len(request.quantities) == 1
        assert request.source_band == "fbfm"
        assert request.name == ""
        assert request.description == ""
        assert request.tags == []
        assert request.modifications == []

    def test_source_grid_id_is_required(self):
        """source_grid_id field is required."""
        with pytest.raises(ValidationError):
            CreateFbfm40LookupRequest(quantities=["fuel_load.1hr"])

    def test_quantities_is_required(self):
        """quantities field is required."""
        with pytest.raises(ValidationError):
            CreateFbfm40LookupRequest(source_grid_id="grid-123")

    def test_source_band_can_be_overridden(self):
        """source_band can be set to a different value."""
        request = CreateFbfm40LookupRequest(
            source_grid_id="grid-123",
            source_band="custom_band",
            quantities=["fuel_load.1hr"],
        )
        assert request.source_band == "custom_band"

    def test_duplicate_quantities_rejected(self):
        """Duplicate quantities are rejected with a validation error."""
        with pytest.raises(ValidationError):
            CreateFbfm40LookupRequest(
                source_grid_id="grid-123",
                quantities=["fuel_load.1hr", "fuel_load.1hr"],
            )

    def test_unique_quantities_accepted(self):
        """Multiple unique quantities are accepted."""
        request = CreateFbfm40LookupRequest(
            source_grid_id="grid-123",
            quantities=["fuel_load.1hr", "fuel_load.10hr"],
        )
        assert len(request.quantities) == 2
        assert request.quantities[0] == Fbfm40LookupQuantity.fuel_load_1hr
        assert request.quantities[1] == Fbfm40LookupQuantity.fuel_load_10hr

    def test_invalid_quantity_rejected(self):
        """Invalid quantity string is rejected."""
        with pytest.raises(ValidationError):
            CreateFbfm40LookupRequest(
                source_grid_id="grid-123",
                quantities=["not_a_quantity"],
            )

    def test_full_request_with_all_fields(self):
        """Full request with all optional fields."""
        request = CreateFbfm40LookupRequest(
            source_grid_id="grid-123",
            source_band="custom",
            quantities=["fuel_load.1hr", "fuel_depth"],
            name="Test Lookup",
            description="A test lookup grid",
            tags=["test", "lookup"],
        )
        assert request.name == "Test Lookup"
        assert request.description == "A test lookup grid"
        assert request.tags == ["test", "lookup"]
        assert len(request.quantities) == 2


class TestFbfm40LookupBandMetadata:
    """Tests for FBFM40_LOOKUP_BAND_METADATA constant."""

    def test_all_quantities_have_entries(self):
        """Every enum member has a metadata entry."""
        for q in Fbfm40LookupQuantity:
            assert q in FBFM40_LOOKUP_BAND_METADATA

    def test_fuel_load_units_are_kg_per_m2(self):
        """Fuel load quantities use kg/m² unit."""
        for q in [
            Fbfm40LookupQuantity.fuel_load_1hr,
            Fbfm40LookupQuantity.fuel_load_10hr,
            Fbfm40LookupQuantity.fuel_load_100hr,
            Fbfm40LookupQuantity.fuel_load_live_herb,
            Fbfm40LookupQuantity.fuel_load_live_woody,
        ]:
            band_type, unit = FBFM40_LOOKUP_BAND_METADATA[q]
            assert band_type == BandType.continuous
            assert unit == "kg/m²"

    def test_savr_units_are_inverse_meters(self):
        """SAVR quantities use m⁻¹ unit."""
        for q in [
            Fbfm40LookupQuantity.savr_1hr,
            Fbfm40LookupQuantity.savr_10hr,
            Fbfm40LookupQuantity.savr_100hr,
            Fbfm40LookupQuantity.savr_live_herb,
            Fbfm40LookupQuantity.savr_live_woody,
        ]:
            band_type, unit = FBFM40_LOOKUP_BAND_METADATA[q]
            assert band_type == BandType.continuous
            assert unit == "m⁻¹"

    def test_fuel_depth_unit_is_meters(self):
        """Fuel depth uses m unit."""
        band_type, unit = FBFM40_LOOKUP_BAND_METADATA[Fbfm40LookupQuantity.fuel_depth]
        assert band_type == BandType.continuous
        assert unit == "m"

    def test_moisture_of_extinction_unit_is_percent(self):
        """Moisture of extinction uses % unit."""
        band_type, unit = FBFM40_LOOKUP_BAND_METADATA[
            Fbfm40LookupQuantity.moisture_of_extinction
        ]
        assert band_type == BandType.continuous
        assert unit == "%"

    def test_heat_content_unit_is_kj_per_kg(self):
        """Heat content uses kJ/kg unit."""
        band_type, unit = FBFM40_LOOKUP_BAND_METADATA[Fbfm40LookupQuantity.heat_content]
        assert band_type == BandType.continuous
        assert unit == "kJ/kg"

    def test_is_dynamic_is_categorical(self):
        """is_dynamic is categorical with no unit."""
        band_type, unit = FBFM40_LOOKUP_BAND_METADATA[Fbfm40LookupQuantity.is_dynamic]
        assert band_type == BandType.categorical
        assert unit is None


class TestGetFbfm40LookupBand:
    """Tests for get_fbfm40_lookup_band function."""

    def test_index_matches_provided_value(self):
        """Band index matches the provided index, not a default."""
        band = get_fbfm40_lookup_band(Fbfm40LookupQuantity.fuel_load_1hr, 5)
        assert band.index == 5

    def test_all_quantities_produce_valid_bands(self):
        """Every quantity produces a valid Band object."""
        for i, q in enumerate(Fbfm40LookupQuantity):
            band = get_fbfm40_lookup_band(q, i)
            assert band.key == q.value
            assert band.index == i
            expected_type, expected_unit = FBFM40_LOOKUP_BAND_METADATA[q]
            assert band.type == expected_type
            assert band.unit == expected_unit
