"""
Unit tests for api/v2/resources/features/road/schema.py

Tests the Road schema models and source metadata base.
These are pure unit tests with no external dependencies.
"""

import pytest
from api.resources.features.road.schema import (
    CreateOsmRoadFeatureRequest,
    OsmRoadSource,
)
from api.resources.features.schema import FeatureType
from pydantic import ValidationError


class TestOsmRoadSource:
    """Tests for OsmRoadSource model."""

    def test_product_is_always_osm(self):
        """The product field is always 'osm'."""
        source = OsmRoadSource()
        assert source.product == "osm"

    def test_product_cannot_be_overridden(self):
        """The product field cannot be set to anything other than 'osm'."""
        with pytest.raises(ValidationError):
            OsmRoadSource(product="custom")

    def test_description_is_fixed(self):
        """The description has a fixed value."""
        source = OsmRoadSource()
        assert source.description == "OpenStreetMap road network"

    def test_description_cannot_be_overridden(self):
        """The description field cannot be overridden."""
        with pytest.raises(ValidationError):
            OsmRoadSource(description="My custom description")

    def test_model_dump(self):
        """Model serializes correctly."""
        source = OsmRoadSource()
        data = source.model_dump()
        assert data["product"] == "osm"
        assert data["description"] == "OpenStreetMap road network"


class TestCreateOsmRoadFeatureRequest:
    """Tests for CreateOsmRoadFeatureRequest model."""

    def test_minimal_valid_request(self):
        """Minimal request only requires the correct type."""
        request = CreateOsmRoadFeatureRequest(type="road")
        assert request.type == FeatureType.road
        assert request.name == ""
        assert request.description == ""
        assert request.tags == []

    def test_type_must_be_road(self):
        """The type field cannot be set to any other feature type."""
        with pytest.raises(ValidationError):
            CreateOsmRoadFeatureRequest(type="water")

    def test_full_request_with_all_fields(self):
        """Full request with all optional metadata fields."""
        request = CreateOsmRoadFeatureRequest(
            type="road",
            name="Test Road Network",
            description="A test feature representing roads.",
            tags=["infrastructure", "roads"],
        )
        assert request.type == FeatureType.road
        assert request.name == "Test Road Network"
        assert request.description == "A test feature representing roads."
        assert request.tags == ["infrastructure", "roads"]
