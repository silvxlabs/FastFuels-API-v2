"""
Unit tests for api/v2/resources/grids/fccs/schema.py
and api/v2/resources/grids/providers/landfire.py

Tests the FCCS schema models, LandfireSource base, and constants.
These are pure unit tests with no external dependencies.
"""

import pytest
from api.resources.grids.fccs.schema import (
    FCCS_BAND,
    CreateLandfireFccsRequest,
    LandfireFccsSource,
)
from api.resources.grids.schema import BandType
from pydantic import ValidationError


class TestLandfireFccsSource:
    """Tests for LandfireFccsSource model."""

    def test_product_is_always_fccs(self):
        """The product field is always 'fccs'."""
        source = LandfireFccsSource(version="2023")
        assert source.product == "fccs"

    def test_product_cannot_be_overridden(self):
        """The product field cannot be set to anything other than 'fccs'."""
        with pytest.raises(ValidationError):
            LandfireFccsSource(product="other", version="2023")

    def test_name_is_always_landfire(self):
        """The name field is always 'landfire'."""
        source = LandfireFccsSource(version="2023")
        assert source.name == "landfire"

    def test_description_is_fixed(self):
        """The description has a fixed value."""
        source = LandfireFccsSource(version="2023")
        assert "FCCS" in source.description
        assert "fuelbed" in source.description

    def test_version_is_required(self):
        """The version field is required."""
        with pytest.raises(ValidationError):
            LandfireFccsSource()

    def test_model_dump(self):
        """Model serializes correctly."""
        source = LandfireFccsSource(version="2023")
        data = source.model_dump()
        assert data["name"] == "landfire"
        assert data["product"] == "fccs"
        assert data["version"] == "2023"
        assert "description" in data


class TestCreateLandfireFccsRequest:
    """Tests for CreateLandfireFccsRequest model.

    domain_id now comes from the URL path parameter, not the request body.
    """

    def test_minimal_valid_request(self):
        """Minimal request with no required body fields."""
        request = CreateLandfireFccsRequest()
        assert request.version == "2023"
        assert request.name == ""
        assert request.description == ""
        assert request.tags == []
        assert request.modifications == []

    def test_version_defaults_to_2023(self):
        """version defaults to '2023'."""
        request = CreateLandfireFccsRequest()
        assert request.version == "2023"

    def test_version_can_be_overridden(self):
        """version can be set to a different valid value."""
        request = CreateLandfireFccsRequest(version="2023")
        assert request.version == "2023"

    def test_invalid_version_rejected(self):
        """version must be a valid LANDFIRE FCCS version."""
        with pytest.raises(ValidationError):
            CreateLandfireFccsRequest(version="2021")

    def test_full_request_with_all_fields(self):
        """Full request with all optional fields."""
        request = CreateLandfireFccsRequest(
            version="2023",
            name="Test Grid",
            description="A test grid",
            tags=["test", "fuel"],
        )
        assert request.name == "Test Grid"
        assert request.description == "A test grid"
        assert request.tags == ["test", "fuel"]

    def test_alignment_defaults_to_domain_target(self):
        """alignment defaults to the domain target."""
        request = CreateLandfireFccsRequest()
        assert request.alignment.target == "domain"

    def test_extent_buffer_cells_defaults_to_zero(self):
        """extent_buffer_cells defaults to 0 (no buffer)."""
        request = CreateLandfireFccsRequest()
        assert request.extent_buffer_cells == 0

    def test_extent_buffer_cells_accepts_positive(self):
        """extent_buffer_cells accepts positive integers."""
        request = CreateLandfireFccsRequest(extent_buffer_cells=10)
        assert request.extent_buffer_cells == 10

    def test_extent_buffer_cells_accepts_zero(self):
        """extent_buffer_cells accepts 0 (explicit no buffer)."""
        request = CreateLandfireFccsRequest(extent_buffer_cells=0)
        assert request.extent_buffer_cells == 0

    def test_extent_buffer_cells_rejects_negative(self):
        """extent_buffer_cells rejects negative values."""
        with pytest.raises(ValidationError):
            CreateLandfireFccsRequest(extent_buffer_cells=-1)

    def test_extent_buffer_cells_rejects_above_maximum(self):
        """extent_buffer_cells rejects values above the maximum of 10."""
        with pytest.raises(ValidationError):
            CreateLandfireFccsRequest(extent_buffer_cells=11)


class TestFccsBand:
    """Tests for FCCS_BAND constant."""

    def test_key_is_fccs(self):
        """Band key is 'fccs'."""
        assert FCCS_BAND.key == "fccs"

    def test_type_is_categorical(self):
        """Band type is categorical."""
        assert FCCS_BAND.type == BandType.categorical

    def test_unit_is_none(self):
        """Band unit is None (categorical has no unit)."""
        assert FCCS_BAND.unit is None

    def test_index_is_zero(self):
        """Band index is 0."""
        assert FCCS_BAND.index == 0

    def test_has_name_and_description(self):
        """Band carries a human-readable name and description."""
        assert FCCS_BAND.name == "FCCS Fuelbed ID"
        assert FCCS_BAND.description


class TestRemoveBareGround:
    """Tests for remove_bare_ground on request and source models."""

    def test_request_defaults_to_false(self):
        """remove_bare_ground defaults to False."""
        request = CreateLandfireFccsRequest()
        assert request.remove_bare_ground is False

    def test_request_accepts_true(self):
        """remove_bare_ground can be set to True."""
        request = CreateLandfireFccsRequest(remove_bare_ground=True)
        assert request.remove_bare_ground is True
