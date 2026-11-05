"""
Unit tests for api/v2/resources/features/building/schema.py

Tests the Building schema models and source metadata.
These are pure unit tests with no external dependencies.
"""

import pytest
from api.resources.features.building.schema import (
    CreateOvertureBuildingFeatureRequest,
    OvertureBuildingSource,
)
from api.resources.features.schema import FeatureType
from pydantic import ValidationError


class TestOvertureBuildingSource:
    """Tests for OvertureBuildingSource model."""

    def test_product_is_always_overture(self):
        """The product field is always 'overture'."""
        assert OvertureBuildingSource().product == "overture"

    def test_product_cannot_be_overridden(self):
        """The product field cannot be set to anything other than 'overture'."""
        with pytest.raises(ValidationError):
            OvertureBuildingSource(product="osm")

    def test_description_is_fixed(self):
        """The description has a fixed value."""
        source = OvertureBuildingSource()
        assert source.description == "Overture Maps building footprints"

    def test_description_cannot_be_overridden(self):
        """The description field cannot be overridden."""
        with pytest.raises(ValidationError):
            OvertureBuildingSource(description="My custom description")

    def test_release_defaults_to_none(self):
        """Release is unset at creation; the worker records it."""
        assert OvertureBuildingSource().release is None

    def test_model_dump(self):
        """Model serializes correctly."""
        data = OvertureBuildingSource().model_dump()
        assert data == {
            "product": "overture",
            "description": "Overture Maps building footprints",
            "extent_buffer_m": 0,
            "release": None,
        }

    def test_extent_buffer_m_default(self):
        """Buffer defaults to 0 (clip exactly to domain)."""
        assert OvertureBuildingSource().extent_buffer_m == 0

    def test_extent_buffer_m_accepts_valid_range(self):
        """Buffer accepts 0, fractional values, and the upper bound 100."""
        assert OvertureBuildingSource(extent_buffer_m=0).extent_buffer_m == 0
        assert OvertureBuildingSource(extent_buffer_m=12.5).extent_buffer_m == 12.5
        assert OvertureBuildingSource(extent_buffer_m=100).extent_buffer_m == 100

    def test_extent_buffer_m_rejects_out_of_range(self):
        """Buffer is bounded to [0, 100]."""
        with pytest.raises(ValidationError):
            OvertureBuildingSource(extent_buffer_m=-1)
        with pytest.raises(ValidationError):
            OvertureBuildingSource(extent_buffer_m=101)


class TestCreateOvertureBuildingFeatureRequest:
    """Tests for CreateOvertureBuildingFeatureRequest model."""

    def test_minimal_valid_request(self):
        """Minimal request only requires the correct type."""
        request = CreateOvertureBuildingFeatureRequest(type="building")
        assert request.type == FeatureType.building
        assert request.name == ""
        assert request.description == ""
        assert request.tags == []
        assert request.extent_buffer_m == 0

    def test_type_defaults_to_building(self):
        """The type field defaults to building."""
        assert CreateOvertureBuildingFeatureRequest().type == FeatureType.building

    def test_extent_buffer_m_accepts_valid_range(self):
        """Buffer accepts 0, fractional values, and the upper bound 100."""
        for value in (0, 12.5, 100):
            request = CreateOvertureBuildingFeatureRequest(
                type="building", extent_buffer_m=value
            )
            assert request.extent_buffer_m == value

    def test_extent_buffer_m_rejects_out_of_range(self):
        """Buffer is bounded to [0, 100]."""
        with pytest.raises(ValidationError):
            CreateOvertureBuildingFeatureRequest(type="building", extent_buffer_m=-1)
        with pytest.raises(ValidationError):
            CreateOvertureBuildingFeatureRequest(type="building", extent_buffer_m=101)

    def test_type_must_be_building(self):
        """The type field cannot be set to any other feature type."""
        with pytest.raises(ValidationError):
            CreateOvertureBuildingFeatureRequest(type="road")

    def test_release_is_not_a_request_field(self):
        """Release is chosen by the worker, never by the caller."""
        assert "release" not in CreateOvertureBuildingFeatureRequest.model_fields

    def test_full_request_with_all_fields(self):
        """Full request with all optional metadata fields."""
        request = CreateOvertureBuildingFeatureRequest(
            type="building",
            name="Test Buildings",
            description="A test feature representing buildings.",
            tags=["infrastructure", "buildings"],
            extent_buffer_m=25,
        )
        assert request.type == FeatureType.building
        assert request.name == "Test Buildings"
        assert request.description == "A test feature representing buildings."
        assert request.tags == ["infrastructure", "buildings"]
        assert request.extent_buffer_m == 25
