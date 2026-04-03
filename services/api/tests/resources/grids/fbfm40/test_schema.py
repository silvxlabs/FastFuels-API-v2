"""
Unit tests for api/v2/resources/grids/fbfm40/schema.py
and api/v2/resources/grids/providers/landfire.py

Tests the FBFM40 schema models, LandfireSource base, and constants.
These are pure unit tests with no external dependencies.
"""

import pytest
from api.resources.grids.fbfm40.schema import (
    FBFM40_BAND,
    CreateLandfireFbfm40Request,
    LandfireFbfm40Source,
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
        assert request.version == "2022"
        assert request.name == ""
        assert request.description == ""
        assert request.tags == []
        assert request.modifications == []

    def test_version_defaults_to_2022(self):
        """version defaults to '2022'."""
        request = CreateLandfireFbfm40Request()
        assert request.version == "2022"

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
