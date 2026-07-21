"""
Unit tests for api/v2/resources/grids/fbfm13/schema.py

Tests the FBFM13 schema models and constants. LandfireSource base and
NonBurnableFuelModel/NB_CODE_MAP are shared with FBFM40 and covered there
(and in providers/landfire tests) -- not duplicated here.
"""

import pytest
from api.resources.grids.fbfm13.schema import (
    FBFM13_BAND,
    CreateLandfireFbfm13Request,
    LandfireFbfm13Source,
)
from api.resources.grids.schema import BandType
from pydantic import ValidationError


class TestLandfireFbfm13Source:
    """Tests for LandfireFbfm13Source model."""

    def test_product_is_always_fbfm13(self):
        """The product field is always 'fbfm13'."""
        source = LandfireFbfm13Source(version="2023")
        assert source.product == "fbfm13"

    def test_product_cannot_be_overridden(self):
        """The product field cannot be set to anything other than 'fbfm13'."""
        with pytest.raises(ValidationError):
            LandfireFbfm13Source(product="other", version="2023")

    def test_name_is_always_landfire(self):
        """The name field is always 'landfire'."""
        source = LandfireFbfm13Source(version="2023")
        assert source.name == "landfire"

    def test_description_is_fixed(self):
        """The description has a fixed value."""
        source = LandfireFbfm13Source(version="2023")
        assert "FBFM13" in source.description
        assert "Anderson" in source.description

    def test_version_is_required(self):
        """The version field is required."""
        with pytest.raises(ValidationError):
            LandfireFbfm13Source()

    def test_model_dump(self):
        """Model serializes correctly."""
        source = LandfireFbfm13Source(version="2023")
        data = source.model_dump()
        assert data["name"] == "landfire"
        assert data["product"] == "fbfm13"
        assert data["version"] == "2023"
        assert "description" in data


class TestCreateLandfireFbfm13Request:
    """Tests for CreateLandfireFbfm13Request model.

    domain_id comes from the URL path parameter, not the request body.
    """

    def test_minimal_valid_request(self):
        """Minimal request with no required body fields."""
        request = CreateLandfireFbfm13Request()
        assert request.version == "2024"
        assert request.name == ""
        assert request.description == ""
        assert request.tags == []
        assert request.modifications == []

    def test_version_defaults_to_2024(self):
        """version defaults to '2024'."""
        request = CreateLandfireFbfm13Request()
        assert request.version == "2024"

    def test_version_can_be_overridden(self):
        """version can be set to a different valid value."""
        request = CreateLandfireFbfm13Request(version="2023")
        assert request.version == "2023"

    def test_invalid_version_rejected(self):
        """version must be a valid LANDFIRE FBFM13 version."""
        with pytest.raises(ValidationError):
            CreateLandfireFbfm13Request(version="2021")

    def test_full_request_with_all_fields(self):
        """Full request with all optional fields."""
        request = CreateLandfireFbfm13Request(
            version="2024",
            name="Test Grid",
            description="A test grid",
            tags=["test", "fuel"],
        )
        assert request.name == "Test Grid"
        assert request.description == "A test grid"
        assert request.tags == ["test", "fuel"]

    def test_extent_buffer_cells_defaults_to_zero(self):
        """extent_buffer_cells defaults to 0 (no buffer)."""
        request = CreateLandfireFbfm13Request()
        assert request.extent_buffer_cells == 0

    def test_extent_buffer_cells_accepts_positive(self):
        """extent_buffer_cells accepts positive integers."""
        request = CreateLandfireFbfm13Request(extent_buffer_cells=10)
        assert request.extent_buffer_cells == 10

    def test_extent_buffer_cells_accepts_zero(self):
        """extent_buffer_cells accepts 0 (explicit no buffer)."""
        request = CreateLandfireFbfm13Request(extent_buffer_cells=0)
        assert request.extent_buffer_cells == 0

    def test_extent_buffer_cells_rejects_negative(self):
        """extent_buffer_cells rejects negative values."""
        with pytest.raises(ValidationError):
            CreateLandfireFbfm13Request(extent_buffer_cells=-1)

    def test_extent_buffer_cells_rejects_above_maximum(self):
        """FBFM13 is a 30m raster, so at most 10 buffer cells are allowed."""
        with pytest.raises(ValidationError):
            CreateLandfireFbfm13Request(extent_buffer_cells=11)


class TestFbfm13Band:
    """Tests for FBFM13_BAND constant."""

    def test_key_is_fbfm13(self):
        """Band key is 'fbfm13'."""
        assert FBFM13_BAND.key == "fbfm13"

    def test_type_is_categorical(self):
        """Band type is categorical."""
        assert FBFM13_BAND.type == BandType.categorical

    def test_unit_is_none(self):
        """Band unit is None (categorical has no unit)."""
        assert FBFM13_BAND.unit is None

    def test_index_is_zero(self):
        """Band index is 0."""
        assert FBFM13_BAND.index == 0

    def test_has_name_and_description(self):
        """Band carries a human-readable name and description."""
        assert FBFM13_BAND.name == "Anderson 13 Fuel Model"
        assert FBFM13_BAND.description


class TestRemoveNonBurnable:
    """Tests for remove_non_burnable on the FBFM13 request/source models."""

    def test_request_defaults_to_none(self):
        request = CreateLandfireFbfm13Request()
        assert request.remove_non_burnable is None

    def test_request_accepts_valid_codes(self):
        request = CreateLandfireFbfm13Request(remove_non_burnable=["NB1", "NB3", "NB9"])
        assert request.remove_non_burnable == ["NB1", "NB3", "NB9"]

    def test_request_accepts_all_codes(self):
        request = CreateLandfireFbfm13Request(
            remove_non_burnable=["NB1", "NB2", "NB3", "NB8", "NB9"]
        )
        assert len(request.remove_non_burnable) == 5

    def test_request_accepts_empty_list(self):
        request = CreateLandfireFbfm13Request(remove_non_burnable=[])
        assert request.remove_non_burnable == []

    def test_request_rejects_invalid_code(self):
        with pytest.raises(ValidationError):
            CreateLandfireFbfm13Request(remove_non_burnable=["NB4"])

    def test_request_rejects_duplicates(self):
        with pytest.raises(ValidationError):
            CreateLandfireFbfm13Request(remove_non_burnable=["NB1", "NB1"])

    def test_source_round_trip(self):
        source = LandfireFbfm13Source(
            version="2023",
            remove_non_burnable=["NB1", "NB9"],
        )
        data = source.model_dump()
        assert data["remove_non_burnable"] == ["NB1", "NB9"]

    def test_source_none_round_trip(self):
        source = LandfireFbfm13Source(version="2023")
        data = source.model_dump()
        assert data["remove_non_burnable"] is None
