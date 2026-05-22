"""
Unit tests for api/v2/resources/grids/fbfm40/schema.py
and api/v2/resources/grids/providers/landfire.py

Tests the FBFM40 schema models, LandfireSource base, and constants.
These are pure unit tests with no external dependencies.
"""

import pytest
from api.resources.grids.fbfm40.schema import (
    FBFM40_BAND,
    NB_CODE_MAP,
    CreateLandfireFbfm40Request,
    LandfireFbfm40Source,
    NonBurnableFuelModel,
)
from api.resources.grids.providers.landfire import LandfireSource
from api.resources.grids.schema import BandType
from pydantic import ValidationError


class TestLandfireSource:
    """Tests for LandfireSource base model."""

    def test_name_is_always_landfire(self):
        """The name field is always 'landfire'."""
        source = LandfireSource(product="fbfm40", version="2022")
        assert source.name == "landfire"

    def test_name_cannot_be_overridden(self):
        """The name field cannot be set to anything other than 'landfire'."""
        with pytest.raises(ValidationError):
            LandfireSource(name="other", product="fbfm40", version="2022")

    def test_product_is_required(self):
        """The product field is required."""
        with pytest.raises(ValidationError):
            LandfireSource(version="2022")

    def test_version_is_required(self):
        """The version field is required."""
        with pytest.raises(ValidationError):
            LandfireSource(product="fbfm40")

    def test_description_defaults_to_empty_string(self):
        """The description field defaults to empty string."""
        source = LandfireSource(product="fbfm40", version="2022")
        assert source.description == ""

    def test_description_can_be_set(self):
        """The description field can be set."""
        source = LandfireSource(
            product="fbfm40",
            version="2022",
            description="Test description",
        )
        assert source.description == "Test description"

    def test_extent_buffer_cells_defaults_to_zero(self):
        """extent_buffer_cells defaults to 0 (no buffer)."""
        source = LandfireSource(product="fbfm40", version="2022")
        assert source.extent_buffer_cells == 0

    def test_extent_buffer_cells_can_be_set(self):
        """extent_buffer_cells can be set to any non-negative integer."""
        source = LandfireSource(
            product="fbfm40", version="2022", extent_buffer_cells=10
        )
        assert source.extent_buffer_cells == 10

    def test_extent_buffer_cells_rejects_negative(self):
        """extent_buffer_cells rejects negative values."""
        with pytest.raises(ValidationError):
            LandfireSource(product="fbfm40", version="2022", extent_buffer_cells=-1)


class TestLandfireFbfm40Source:
    """Tests for LandfireFbfm40Source model."""

    def test_product_is_always_fbfm40(self):
        """The product field is always 'fbfm40'."""
        source = LandfireFbfm40Source(version="2022")
        assert source.product == "fbfm40"

    def test_product_cannot_be_overridden(self):
        """The product field cannot be set to anything other than 'fbfm40'."""
        with pytest.raises(ValidationError):
            LandfireFbfm40Source(product="other", version="2022")

    def test_name_is_always_landfire(self):
        """The name field is always 'landfire'."""
        source = LandfireFbfm40Source(version="2022")
        assert source.name == "landfire"

    def test_description_is_fixed(self):
        """The description has a fixed value."""
        source = LandfireFbfm40Source(version="2022")
        assert "FBFM40" in source.description
        assert "Scott-Burgan" in source.description

    def test_version_is_required(self):
        """The version field is required."""
        with pytest.raises(ValidationError):
            LandfireFbfm40Source()

    def test_model_dump(self):
        """Model serializes correctly."""
        source = LandfireFbfm40Source(version="2022")
        data = source.model_dump()
        assert data["name"] == "landfire"
        assert data["product"] == "fbfm40"
        assert data["version"] == "2022"
        assert "description" in data


class TestCreateLandfireFbfm40Request:
    """Tests for CreateLandfireFbfm40Request model.

    domain_id now comes from the URL path parameter, not the request body.
    """

    def test_minimal_valid_request(self):
        """Minimal request with no required body fields."""
        request = CreateLandfireFbfm40Request()
        assert request.version == "2024"
        assert request.name == ""
        assert request.description == ""
        assert request.tags == []
        assert request.modifications == []

    def test_version_defaults_to_2022(self):
        """version defaults to '2024'."""
        request = CreateLandfireFbfm40Request()
        assert request.version == "2024"

    def test_version_can_be_overridden(self):
        """version can be set to a different valid value."""
        request = CreateLandfireFbfm40Request(version="2020")
        assert request.version == "2020"

    def test_invalid_version_rejected(self):
        """version must be a valid LANDFIRE FBFM40 version."""
        with pytest.raises(ValidationError):
            CreateLandfireFbfm40Request(version="2021")

    def test_full_request_with_all_fields(self):
        """Full request with all optional fields."""
        request = CreateLandfireFbfm40Request(
            version="2022",
            name="Test Grid",
            description="A test grid",
            tags=["test", "fuel"],
        )
        assert request.name == "Test Grid"
        assert request.description == "A test grid"
        assert request.tags == ["test", "fuel"]

    def test_extent_buffer_cells_defaults_to_zero(self):
        """extent_buffer_cells defaults to 0 (no buffer)."""
        request = CreateLandfireFbfm40Request()
        assert request.extent_buffer_cells == 0

    def test_extent_buffer_cells_accepts_positive(self):
        """extent_buffer_cells accepts positive integers."""
        request = CreateLandfireFbfm40Request(extent_buffer_cells=10)
        assert request.extent_buffer_cells == 10

    def test_extent_buffer_cells_accepts_zero(self):
        """extent_buffer_cells accepts 0 (explicit no buffer)."""
        request = CreateLandfireFbfm40Request(extent_buffer_cells=0)
        assert request.extent_buffer_cells == 0

    def test_extent_buffer_cells_rejects_negative(self):
        """extent_buffer_cells rejects negative values."""
        with pytest.raises(ValidationError):
            CreateLandfireFbfm40Request(extent_buffer_cells=-1)

    def test_extent_buffer_cells_rejects_above_maximum(self):
        """FBFM40 is a 30m raster, so at most 10 buffer cells are allowed."""
        with pytest.raises(ValidationError):
            CreateLandfireFbfm40Request(extent_buffer_cells=11)


class TestFbfm40Band:
    """Tests for FBFM40_BAND constant."""

    def test_key_is_fbfm(self):
        """Band key is 'fbfm'."""
        assert FBFM40_BAND.key == "fbfm"

    def test_type_is_categorical(self):
        """Band type is categorical."""
        assert FBFM40_BAND.type == BandType.categorical

    def test_unit_is_none(self):
        """Band unit is None (categorical has no unit)."""
        assert FBFM40_BAND.unit is None

    def test_index_is_zero(self):
        """Band index is 0."""
        assert FBFM40_BAND.index == 0


class TestNonBurnableFuelModel:
    """Tests for NonBurnableFuelModel enum."""

    def test_all_codes_present(self):
        """All five non-burnable fuel model codes are defined."""
        assert set(NonBurnableFuelModel) == {"NB1", "NB2", "NB3", "NB8", "NB9"}

    def test_nb_code_map_matches_enum(self):
        """NB_CODE_MAP has an entry for every enum member."""
        assert set(NB_CODE_MAP.keys()) == {m.value for m in NonBurnableFuelModel}

    def test_nb_code_map_values(self):
        """NB_CODE_MAP maps to the correct FBFM40 numeric codes."""
        assert NB_CODE_MAP == {
            "NB1": 91,
            "NB2": 92,
            "NB3": 93,
            "NB8": 98,
            "NB9": 99,
        }


class TestRemoveNonBurnable:
    """Tests for remove_non_burnable on request and source models."""

    def test_request_defaults_to_none(self):
        """remove_non_burnable defaults to None."""
        request = CreateLandfireFbfm40Request()
        assert request.remove_non_burnable is None

    def test_request_accepts_valid_codes(self):
        """A list of valid non-burnable codes is accepted."""
        request = CreateLandfireFbfm40Request(remove_non_burnable=["NB1", "NB3", "NB9"])
        assert request.remove_non_burnable == ["NB1", "NB3", "NB9"]

    def test_request_accepts_all_codes(self):
        """All five codes can be specified together."""
        request = CreateLandfireFbfm40Request(
            remove_non_burnable=["NB1", "NB2", "NB3", "NB8", "NB9"]
        )
        assert len(request.remove_non_burnable) == 5

    def test_request_accepts_empty_list(self):
        """An empty list is accepted (no codes to remove)."""
        request = CreateLandfireFbfm40Request(remove_non_burnable=[])
        assert request.remove_non_burnable == []

    def test_request_rejects_invalid_code(self):
        """An invalid non-burnable code is rejected."""
        with pytest.raises(ValidationError):
            CreateLandfireFbfm40Request(remove_non_burnable=["NB4"])

    def test_request_rejects_duplicates(self):
        """Duplicate codes are rejected."""
        with pytest.raises(ValidationError):
            CreateLandfireFbfm40Request(remove_non_burnable=["NB1", "NB1"])

    def test_source_round_trip(self):
        """remove_non_burnable survives source model_dump round-trip."""
        source = LandfireFbfm40Source(
            version="2022",
            remove_non_burnable=["NB1", "NB9"],
        )
        data = source.model_dump()
        assert data["remove_non_burnable"] == ["NB1", "NB9"]

    def test_source_none_round_trip(self):
        """None remove_non_burnable survives source model_dump round-trip."""
        source = LandfireFbfm40Source(version="2022")
        data = source.model_dump()
        assert data["remove_non_burnable"] is None
