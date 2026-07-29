"""
Unit tests for api/v2/resources/grids/lookup/fccs/schema.py

Tests the FCCS lookup schema models and constants. These are pure unit
tests with no external dependencies.
"""

import pytest
from api.resources.grids.lookup.fccs.schema import (
    FCCS_LOOKUP_BAND_METADATA,
    CreateFccsLookupRequest,
    FccsLookupBand,
    FccsLookupSource,
    get_fccs_lookup_band,
)
from api.resources.grids.schema import BandType
from pydantic import ValidationError

from lib.units import validate_unit


class TestFccsLookupSource:
    """Tests for FccsLookupSource model."""

    def test_table_is_always_fccs(self):
        """The table field is always 'fccs'."""
        source = FccsLookupSource(source_grid_id="grid-123")
        assert source.table == "fccs"

    def test_table_cannot_be_overridden(self):
        """The table field cannot be set to anything other than 'fccs'."""
        with pytest.raises(ValidationError):
            FccsLookupSource(table="other", source_grid_id="grid-123")

    def test_name_is_always_lookup(self):
        """The name field is always 'lookup'."""
        source = FccsLookupSource(source_grid_id="grid-123")
        assert source.name == "lookup"

    def test_source_band_defaults_to_fccs(self):
        """The source_band field defaults to 'fccs'."""
        source = FccsLookupSource(source_grid_id="grid-123")
        assert source.source_band == "fccs"

    def test_source_band_can_be_overridden(self):
        """The source_band field can be set to a different value."""
        source = FccsLookupSource(source_grid_id="grid-123", source_band="custom_band")
        assert source.source_band == "custom_band"

    def test_source_grid_id_is_required(self):
        """The source_grid_id field is required."""
        with pytest.raises(ValidationError):
            FccsLookupSource()

    def test_model_dump(self):
        """Model serializes correctly."""
        source = FccsLookupSource(source_grid_id="grid-123")
        data = source.model_dump()
        assert data["name"] == "lookup"
        assert data["table"] == "fccs"
        assert data["source_grid_id"] == "grid-123"
        assert data["source_band"] == "fccs"

    def test_source_grid_checksum_defaults_to_none(self):
        """source_grid_checksum defaults to None when not captured."""
        source = FccsLookupSource(source_grid_id="grid-123")
        assert source.source_grid_checksum is None

    def test_source_grid_checksum_round_trips(self):
        """source_grid_checksum is carried through serialization."""
        source = FccsLookupSource(
            source_grid_id="grid-123", source_grid_checksum="sum123"
        )
        assert source.source_grid_checksum == "sum123"
        assert source.model_dump()["source_grid_checksum"] == "sum123"


class TestFccsLookupBand:
    """Tests for FccsLookupBand enum."""

    def test_has_12_members(self):
        """There are exactly 12 predefined bands."""
        assert len(FccsLookupBand) == 12

    def test_fuel_load_litter(self):
        assert FccsLookupBand.fuel_load_litter == "fuel_load.litter"

    def test_fuel_load_duff(self):
        assert FccsLookupBand.fuel_load_duff == "fuel_load.duff"

    def test_duff_depth(self):
        assert FccsLookupBand.duff_depth == "duff_depth"

    def test_fuel_load_live_shrub(self):
        assert FccsLookupBand.fuel_load_live_shrub == "fuel_load.live_shrub"

    def test_fuel_load_live_herb(self):
        assert FccsLookupBand.fuel_load_live_herb == "fuel_load.live_herb"

    def test_fuel_load_1hr(self):
        assert FccsLookupBand.fuel_load_1hr == "fuel_load.1hr"

    def test_fuel_load_10hr(self):
        assert FccsLookupBand.fuel_load_10hr == "fuel_load.10hr"

    def test_fuel_load_100hr(self):
        assert FccsLookupBand.fuel_load_100hr == "fuel_load.100hr"

    def test_fuel_load_1000hr_sound(self):
        assert FccsLookupBand.fuel_load_1000hr_sound == "fuel_load.1000hr_sound"

    def test_fuel_load_1000hr_rotten(self):
        assert FccsLookupBand.fuel_load_1000hr_rotten == "fuel_load.1000hr_rotten"

    def test_fuel_load_live_foliage(self):
        assert FccsLookupBand.fuel_load_live_foliage == "fuel_load.live_foliage"

    def test_fuel_load_live_branch(self):
        assert FccsLookupBand.fuel_load_live_branch == "fuel_load.live_branch"

    def test_created_from_string(self):
        """Bands can be created from their string value."""
        b = FccsLookupBand("fuel_load.litter")
        assert b == FccsLookupBand.fuel_load_litter


class TestCreateFccsLookupRequest:
    """Tests for CreateFccsLookupRequest model."""

    def test_minimal_valid_request(self):
        """Minimal request with required fields only."""
        request = CreateFccsLookupRequest(
            source_grid_id="grid-123",
            bands=["fuel_load.litter"],
        )
        assert request.source_grid_id == "grid-123"
        assert len(request.bands) == 1
        assert request.source_band == "fccs"
        assert request.name == ""
        assert request.description == ""
        assert request.tags == []
        assert request.modifications == []

    def test_source_grid_id_is_required(self):
        """source_grid_id field is required."""
        with pytest.raises(ValidationError):
            CreateFccsLookupRequest(bands=["fuel_load.litter"])

    def test_bands_is_required(self):
        """bands field is required."""
        with pytest.raises(ValidationError):
            CreateFccsLookupRequest(source_grid_id="grid-123")

    def test_empty_bands_rejected(self):
        """An empty bands list is rejected (min_length=1)."""
        with pytest.raises(ValidationError):
            CreateFccsLookupRequest(source_grid_id="grid-123", bands=[])

    def test_source_band_can_be_overridden(self):
        """source_band can be set to a different value."""
        request = CreateFccsLookupRequest(
            source_grid_id="grid-123",
            source_band="custom_band",
            bands=["fuel_load.litter"],
        )
        assert request.source_band == "custom_band"

    def test_duplicate_bands_rejected(self):
        """Duplicate bands are rejected with a validation error."""
        with pytest.raises(ValidationError):
            CreateFccsLookupRequest(
                source_grid_id="grid-123",
                bands=["fuel_load.litter", "fuel_load.litter"],
            )

    def test_unique_bands_accepted(self):
        """Multiple unique bands are accepted."""
        request = CreateFccsLookupRequest(
            source_grid_id="grid-123",
            bands=["fuel_load.litter", "fuel_load.duff"],
        )
        assert len(request.bands) == 2
        assert request.bands[0] == FccsLookupBand.fuel_load_litter
        assert request.bands[1] == FccsLookupBand.fuel_load_duff

    def test_invalid_band_rejected(self):
        """Invalid band string is rejected."""
        with pytest.raises(ValidationError):
            CreateFccsLookupRequest(
                source_grid_id="grid-123",
                bands=["not_a_band"],
            )

    def test_full_request_with_all_fields(self):
        """Full request with all optional fields."""
        request = CreateFccsLookupRequest(
            source_grid_id="grid-123",
            source_band="custom",
            bands=["fuel_load.litter", "duff_depth"],
            name="Test Lookup",
            description="A test lookup grid",
            tags=["test", "lookup"],
        )
        assert request.name == "Test Lookup"
        assert request.description == "A test lookup grid"
        assert request.tags == ["test", "lookup"]
        assert len(request.bands) == 2


class TestFccsLookupBandMetadata:
    """Tests for FCCS_LOOKUP_BAND_METADATA constant."""

    def test_all_bands_have_entries(self):
        """Every enum member has a metadata entry."""
        for b in FccsLookupBand:
            assert b in FCCS_LOOKUP_BAND_METADATA

    def test_fuel_load_units_are_kg_per_m2(self):
        """Fuel load bands use kg/m**2 unit."""
        for b in [
            FccsLookupBand.fuel_load_litter,
            FccsLookupBand.fuel_load_duff,
            FccsLookupBand.fuel_load_live_shrub,
            FccsLookupBand.fuel_load_live_herb,
            FccsLookupBand.fuel_load_1hr,
            FccsLookupBand.fuel_load_10hr,
            FccsLookupBand.fuel_load_100hr,
            FccsLookupBand.fuel_load_1000hr_sound,
            FccsLookupBand.fuel_load_1000hr_rotten,
            FccsLookupBand.fuel_load_live_foliage,
            FccsLookupBand.fuel_load_live_branch,
        ]:
            meta = FCCS_LOOKUP_BAND_METADATA[b]
            assert meta["type"] == BandType.continuous
            assert meta["unit"] == "kg/m**2"

    def test_duff_depth_unit_is_meters(self):
        """Duff depth uses m unit."""
        meta = FCCS_LOOKUP_BAND_METADATA[FccsLookupBand.duff_depth]
        assert meta["type"] == BandType.continuous
        assert meta["unit"] == "m"

    def test_all_units_are_canonical(self):
        for meta in FCCS_LOOKUP_BAND_METADATA.values():
            validate_unit(meta["unit"])

    def test_all_bands_have_name_and_description(self):
        """Every lookup band carries a human-readable name and description."""
        for b in FccsLookupBand:
            meta = FCCS_LOOKUP_BAND_METADATA[b]
            assert meta["name"]
            assert meta["description"]


class TestGetFccsLookupBand:
    """Tests for get_fccs_lookup_band function."""

    def test_index_matches_provided_value(self):
        """Band index matches the provided index, not a default."""
        band = get_fccs_lookup_band(FccsLookupBand.fuel_load_litter, 5)
        assert band.index == 5

    def test_all_bands_produce_valid_bands(self):
        """Every enum member produces a valid Band object."""
        for i, b in enumerate(FccsLookupBand):
            result = get_fccs_lookup_band(b, i)
            assert result.key == b.value
            assert result.index == i
            meta = FCCS_LOOKUP_BAND_METADATA[b]
            assert result.type == meta["type"]
            assert result.unit == meta["unit"]
            assert result.name == meta["name"]
            assert result.description == meta["description"]
