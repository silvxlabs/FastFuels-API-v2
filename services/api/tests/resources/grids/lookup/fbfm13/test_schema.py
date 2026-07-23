"""
Unit tests for api/v2/resources/grids/lookup/fbfm13/schema.py

Tests the FBFM13 lookup schema models and constants. These are pure unit
tests with no external dependencies.
"""

import pytest
from api.resources.grids.lookup.fbfm13.schema import (
    FBFM13_LOOKUP_BAND_METADATA,
    CreateFbfm13LookupRequest,
    Fbfm13LookupBand,
    Fbfm13LookupSource,
    get_fbfm13_lookup_band,
)
from api.resources.grids.schema import BandType
from pydantic import ValidationError

from lib.units import validate_unit


class TestFbfm13LookupSource:
    """Tests for Fbfm13LookupSource model."""

    def test_table_is_always_fbfm13(self):
        """The table field is always 'fbfm13'."""
        source = Fbfm13LookupSource(source_grid_id="grid-123")
        assert source.table == "fbfm13"

    def test_table_cannot_be_overridden(self):
        """The table field cannot be set to anything other than 'fbfm13'."""
        with pytest.raises(ValidationError):
            Fbfm13LookupSource(table="other", source_grid_id="grid-123")

    def test_name_is_always_lookup(self):
        """The name field is always 'lookup'."""
        source = Fbfm13LookupSource(source_grid_id="grid-123")
        assert source.name == "lookup"

    def test_source_band_defaults_to_fbfm13(self):
        """The source_band field defaults to 'fbfm13'."""
        source = Fbfm13LookupSource(source_grid_id="grid-123")
        assert source.source_band == "fbfm13"

    def test_source_band_can_be_overridden(self):
        """The source_band field can be set to a different value."""
        source = Fbfm13LookupSource(
            source_grid_id="grid-123", source_band="custom_band"
        )
        assert source.source_band == "custom_band"

    def test_source_grid_id_is_required(self):
        """The source_grid_id field is required."""
        with pytest.raises(ValidationError):
            Fbfm13LookupSource()

    def test_model_dump(self):
        """Model serializes correctly."""
        source = Fbfm13LookupSource(source_grid_id="grid-123")
        data = source.model_dump()
        assert data["name"] == "lookup"
        assert data["table"] == "fbfm13"
        assert data["source_grid_id"] == "grid-123"
        assert data["source_band"] == "fbfm13"

    def test_source_grid_checksum_defaults_to_none(self):
        """source_grid_checksum defaults to None when not captured."""
        source = Fbfm13LookupSource(source_grid_id="grid-123")
        assert source.source_grid_checksum is None

    def test_source_grid_checksum_round_trips(self):
        """source_grid_checksum is carried through serialization."""
        source = Fbfm13LookupSource(
            source_grid_id="grid-123", source_grid_checksum="sum123"
        )
        assert source.source_grid_checksum == "sum123"
        assert source.model_dump()["source_grid_checksum"] == "sum123"


class TestFbfm13LookupBand:
    """Tests for Fbfm13LookupBand enum."""

    def test_has_9_members(self):
        """There are exactly 9 predefined bands."""
        assert len(Fbfm13LookupBand) == 9

    def test_fuel_load_1hr(self):
        assert Fbfm13LookupBand.fuel_load_1hr == "fuel_load.1hr"

    def test_fuel_load_10hr(self):
        assert Fbfm13LookupBand.fuel_load_10hr == "fuel_load.10hr"

    def test_fuel_load_100hr(self):
        assert Fbfm13LookupBand.fuel_load_100hr == "fuel_load.100hr"

    def test_fuel_load_live_foliage(self):
        assert Fbfm13LookupBand.fuel_load_live_foliage == "fuel_load.live_foliage"

    def test_savr_1hr(self):
        assert Fbfm13LookupBand.savr_1hr == "savr.1hr"

    def test_savr_10hr(self):
        assert Fbfm13LookupBand.savr_10hr == "savr.10hr"

    def test_savr_100hr(self):
        assert Fbfm13LookupBand.savr_100hr == "savr.100hr"

    def test_savr_live_foliage(self):
        assert Fbfm13LookupBand.savr_live_foliage == "savr.live_foliage"

    def test_fuel_depth(self):
        assert Fbfm13LookupBand.fuel_depth == "fuel_depth"

    def test_created_from_string(self):
        """Bands can be created from their string value."""
        b = Fbfm13LookupBand("fuel_load.1hr")
        assert b == Fbfm13LookupBand.fuel_load_1hr


class TestCreateFbfm13LookupRequest:
    """Tests for CreateFbfm13LookupRequest model."""

    def test_minimal_valid_request(self):
        """Minimal request with required fields only."""
        request = CreateFbfm13LookupRequest(
            source_grid_id="grid-123",
            bands=["fuel_load.1hr"],
        )
        assert request.source_grid_id == "grid-123"
        assert len(request.bands) == 1
        assert request.source_band == "fbfm13"
        assert request.name == ""
        assert request.description == ""
        assert request.tags == []
        assert request.modifications == []

    def test_source_grid_id_is_required(self):
        """source_grid_id field is required."""
        with pytest.raises(ValidationError):
            CreateFbfm13LookupRequest(bands=["fuel_load.1hr"])

    def test_bands_is_required(self):
        """bands field is required."""
        with pytest.raises(ValidationError):
            CreateFbfm13LookupRequest(source_grid_id="grid-123")

    def test_empty_bands_rejected(self):
        """An empty bands list is rejected (min_length=1)."""
        with pytest.raises(ValidationError):
            CreateFbfm13LookupRequest(source_grid_id="grid-123", bands=[])

    def test_source_band_can_be_overridden(self):
        """source_band can be set to a different value."""
        request = CreateFbfm13LookupRequest(
            source_grid_id="grid-123",
            source_band="custom_band",
            bands=["fuel_load.1hr"],
        )
        assert request.source_band == "custom_band"

    def test_duplicate_bands_rejected(self):
        """Duplicate bands are rejected with a validation error."""
        with pytest.raises(ValidationError):
            CreateFbfm13LookupRequest(
                source_grid_id="grid-123",
                bands=["fuel_load.1hr", "fuel_load.1hr"],
            )

    def test_unique_bands_accepted(self):
        """Multiple unique bands are accepted."""
        request = CreateFbfm13LookupRequest(
            source_grid_id="grid-123",
            bands=["fuel_load.1hr", "fuel_load.10hr"],
        )
        assert len(request.bands) == 2
        assert request.bands[0] == Fbfm13LookupBand.fuel_load_1hr
        assert request.bands[1] == Fbfm13LookupBand.fuel_load_10hr

    def test_invalid_band_rejected(self):
        """Invalid band string is rejected."""
        with pytest.raises(ValidationError):
            CreateFbfm13LookupRequest(
                source_grid_id="grid-123",
                bands=["not_a_band"],
            )

    def test_full_request_with_all_fields(self):
        """Full request with all optional fields."""
        request = CreateFbfm13LookupRequest(
            source_grid_id="grid-123",
            source_band="custom",
            bands=["fuel_load.1hr", "fuel_depth"],
            name="Test Lookup",
            description="A test lookup grid",
            tags=["test", "lookup"],
        )
        assert request.name == "Test Lookup"
        assert request.description == "A test lookup grid"
        assert request.tags == ["test", "lookup"]
        assert len(request.bands) == 2


class TestFbfm13LookupBandMetadata:
    """Tests for FBFM13_LOOKUP_BAND_METADATA constant."""

    def test_all_bands_have_entries(self):
        """Every enum member has a metadata entry."""
        for b in Fbfm13LookupBand:
            assert b in FBFM13_LOOKUP_BAND_METADATA

    def test_fuel_load_units_are_kg_per_m2(self):
        """Fuel load bands use kg/m**2 unit."""
        for b in [
            Fbfm13LookupBand.fuel_load_1hr,
            Fbfm13LookupBand.fuel_load_10hr,
            Fbfm13LookupBand.fuel_load_100hr,
            Fbfm13LookupBand.fuel_load_live_foliage,
        ]:
            meta = FBFM13_LOOKUP_BAND_METADATA[b]
            assert meta["type"] == BandType.continuous
            assert meta["unit"] == "kg/m**2"

    def test_savr_units_are_inverse_meters(self):
        """SAVR bands use 1/m unit."""
        for b in [
            Fbfm13LookupBand.savr_1hr,
            Fbfm13LookupBand.savr_10hr,
            Fbfm13LookupBand.savr_100hr,
            Fbfm13LookupBand.savr_live_foliage,
        ]:
            meta = FBFM13_LOOKUP_BAND_METADATA[b]
            assert meta["type"] == BandType.continuous
            assert meta["unit"] == "1/m"

    def test_fuel_depth_unit_is_meters(self):
        """Fuel depth uses m unit."""
        meta = FBFM13_LOOKUP_BAND_METADATA[Fbfm13LookupBand.fuel_depth]
        assert meta["type"] == BandType.continuous
        assert meta["unit"] == "m"

    def test_all_units_are_canonical(self):
        for meta in FBFM13_LOOKUP_BAND_METADATA.values():
            validate_unit(meta["unit"])

    def test_all_bands_have_name_and_description(self):
        """Every lookup band carries a human-readable name and description."""
        for b in Fbfm13LookupBand:
            meta = FBFM13_LOOKUP_BAND_METADATA[b]
            assert meta["name"]
            assert meta["description"]


class TestGetFbfm13LookupBand:
    """Tests for get_fbfm13_lookup_band function."""

    def test_index_matches_provided_value(self):
        """Band index matches the provided index, not a default."""
        band = get_fbfm13_lookup_band(Fbfm13LookupBand.fuel_load_1hr, 5)
        assert band.index == 5

    def test_all_bands_produce_valid_bands(self):
        """Every enum member produces a valid Band object."""
        for i, b in enumerate(Fbfm13LookupBand):
            result = get_fbfm13_lookup_band(b, i)
            assert result.key == b.value
            assert result.index == i
            meta = FBFM13_LOOKUP_BAND_METADATA[b]
            assert result.type == meta["type"]
            assert result.unit == meta["unit"]
            assert result.name == meta["name"]
            assert result.description == meta["description"]
