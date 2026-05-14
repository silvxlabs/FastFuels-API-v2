"""
Unit tests for api/v2/resources/features/water/schema.py

Tests the Water schema models and source metadata.
These are pure unit tests with no external dependencies.
"""

import pytest
from api.resources.features.schema import FeatureType
from api.resources.features.water.schema import (
    CreateOsmWaterFeatureRequest,
    OsmWaterSource,
)
from pydantic import ValidationError


class TestOsmWaterSource:
    """Tests for OsmWaterSource model."""

    def test_product_is_always_osm(self):
        """The product field is always 'osm'."""
        source = OsmWaterSource()
        assert source.product == "osm"

    def test_product_cannot_be_overridden(self):
        """The product field cannot be set to anything other than 'osm'."""
        with pytest.raises(ValidationError):
            OsmWaterSource(product="custom")

    def test_description_is_fixed(self):
        """The description has a fixed value."""
        source = OsmWaterSource()
        assert source.description == "OpenStreetMap water features"

    def test_description_cannot_be_overridden(self):
        """The description field cannot be overridden."""
        with pytest.raises(ValidationError):
            OsmWaterSource(description="My custom description")

    def test_model_dump(self):
        """Model serializes correctly."""
        source = OsmWaterSource()
        data = source.model_dump()
        assert data["product"] == "osm"
        assert data["description"] == "OpenStreetMap water features"


class TestCreateOsmWaterFeatureRequest:
    """Tests for CreateOsmWaterFeatureRequest model."""

    def test_minimal_valid_request(self):
        """Minimal request only requires the correct type."""
        request = CreateOsmWaterFeatureRequest(type="water")
        assert request.type == FeatureType.water
        assert request.name == ""
        assert request.description == ""
        assert request.tags == []

    def test_type_must_be_water(self):
        """The type field cannot be set to any other feature type."""
        with pytest.raises(ValidationError):
            CreateOsmWaterFeatureRequest(type="road")

    def test_full_request_with_all_fields(self):
        """Full request with all optional metadata fields."""
        request = CreateOsmWaterFeatureRequest(
            type="water",
            name="Test Water Features",
            description="A test feature representing water bodies.",
            tags=["hydrology", "water"],
        )
        assert request.type == FeatureType.water
        assert request.name == "Test Water Features"
        assert request.description == "A test feature representing water bodies."
        assert request.tags == ["hydrology", "water"]
