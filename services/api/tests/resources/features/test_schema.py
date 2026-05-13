"""
Unit tests for api/v2/resources/features/schema.py

Tests the core Feature schema models, enums, and base classes.
These are pure unit tests with no external dependencies.
"""

from datetime import datetime

import pytest
from api.resources.features.schema import (
    CreateFeatureRequestBase,
    Feature,
    FeatureGeoreference,
    FeatureSortField,
    FeatureType,
    ListFeaturesResponse,
    UpdateFeatureRequestBody,
)
from api.schema import JobStatus
from pydantic import ValidationError


class TestFeatureType:
    """Tests for FeatureType enum."""

    def test_road_value(self):
        assert FeatureType.road.value == "road"

    def test_water_value(self):
        assert FeatureType.water.value == "water"

    def test_layerset_value(self):
        assert FeatureType.layerset.value == "layerset"

    def test_enum_count(self):
        assert len(FeatureType) == 3

    def test_can_create_from_string(self):
        assert FeatureType("road") == FeatureType.road
        assert FeatureType("layerset") == FeatureType.layerset

    def test_invalid_string_raises_valueerror(self):
        with pytest.raises(ValueError):
            FeatureType("invalid")


class TestFeatureSortField:
    """Tests for FeatureSortField enum."""

    def test_values(self):
        assert FeatureSortField.created_on.value == "created_on"
        assert FeatureSortField.modified_on.value == "modified_on"
        assert FeatureSortField.name.value == "name"

    def test_enum_count(self):
        assert len(FeatureSortField) == 3


class TestFeatureGeoreference:
    """Tests for FeatureGeoreference model."""

    def test_minimal_valid_georeference(self):
        georef = FeatureGeoreference(
            crs="EPSG:4326",
            bounds=(-120.0, 40.0, -119.0, 41.0),
        )
        assert georef.crs == "EPSG:4326"
        assert georef.bounds == (-120.0, 40.0, -119.0, 41.0)

    def test_crs_is_required(self):
        with pytest.raises(ValidationError):
            FeatureGeoreference(bounds=(-120.0, 40.0, -119.0, 41.0))

    def test_bounds_is_required(self):
        with pytest.raises(ValidationError):
            FeatureGeoreference(crs="EPSG:4326")

    def test_bounds_must_have_4_elements(self):
        with pytest.raises(ValidationError):
            FeatureGeoreference(
                crs="EPSG:4326",
                bounds=(-120.0, 40.0, -119.0),  # Only 3 elements
            )

    def test_model_dump(self):
        georef = FeatureGeoreference(
            crs="EPSG:4326",
            bounds=(-120.0, 40.0, -119.0, 41.0),
        )
        data = georef.model_dump()
        assert data == {
            "crs": "EPSG:4326",
            "bounds": (-120.0, 40.0, -119.0, 41.0),
        }


class TestCreateFeatureRequestBase:
    """Tests for CreateFeatureRequestBase model."""

    def test_minimal_valid_request(self):
        request = CreateFeatureRequestBase(type="road")
        assert request.type == FeatureType.road
        assert request.name == ""
        assert request.description == ""
        assert request.tags == []

    def test_type_is_required(self):
        with pytest.raises(ValidationError):
            CreateFeatureRequestBase(name="Test")

    def test_full_request_with_all_fields(self):
        request = CreateFeatureRequestBase(
            type="water",
            name="Test Feature",
            description="A test feature",
            tags=["test", "hydro"],
        )
        assert request.type == FeatureType.water
        assert request.name == "Test Feature"
        assert request.description == "A test feature"
        assert request.tags == ["test", "hydro"]


class TestUpdateFeatureRequestBody:
    """Tests for UpdateFeatureRequestBody model."""

    def test_empty_update_is_valid(self):
        body = UpdateFeatureRequestBody()
        assert body.name is None
        assert body.description is None
        assert body.tags is None

    def test_update_all_fields(self):
        body = UpdateFeatureRequestBody(
            name="New Name",
            description="New Description",
            tags=["new", "tags"],
        )
        assert body.name == "New Name"
        assert body.description == "New Description"
        assert body.tags == ["new", "tags"]

    def test_model_dump_exclude_none(self):
        body = UpdateFeatureRequestBody(name="New Name")
        data = body.model_dump(exclude_none=True)
        assert data == {"name": "New Name"}
        assert "description" not in data
        assert "tags" not in data


class TestFeature:
    """Tests for Feature model."""

    @pytest.fixture
    def minimal_feature_data(self):
        return {
            "id": "feat123",
            "domain_id": "domain_xyz",
            "type": "road",
            "status": JobStatus.pending,
            "created_on": datetime.now(),
            "modified_on": datetime.now(),
            "source": {
                "product": "osm",
                "description": "OpenStreetMap roads",
            },
        }

    def test_minimal_valid_feature(self, minimal_feature_data):
        feature = Feature(**minimal_feature_data)
        assert feature.id == "feat123"
        assert feature.domain_id == "domain_xyz"
        assert feature.type == FeatureType.road
        assert feature.status == JobStatus.pending
        assert feature.name == ""
        assert feature.description == ""
        assert feature.tags == []
        assert feature.georeference is None

    def test_id_is_required(self, minimal_feature_data):
        del minimal_feature_data["id"]
        with pytest.raises(ValidationError):
            Feature(**minimal_feature_data)

    def test_domain_id_is_required(self, minimal_feature_data):
        del minimal_feature_data["domain_id"]
        with pytest.raises(ValidationError):
            Feature(**minimal_feature_data)

    def test_georeference_can_be_set(self, minimal_feature_data):
        minimal_feature_data["georeference"] = {
            "crs": "EPSG:32610",
            "bounds": (0.0, 0.0, 100.0, 100.0),
        }
        feature = Feature(**minimal_feature_data)
        assert isinstance(feature.georeference, FeatureGeoreference)
        assert feature.georeference.crs == "EPSG:32610"


class TestListFeaturesResponse:
    """Tests for ListFeaturesResponse model."""

    @pytest.fixture
    def sample_feature_data(self):
        return {
            "id": "feat123",
            "domain_id": "domain_xyz",
            "type": "road",
            "name": "Test Feature",
            "status": "pending",
            "source": {"product": "osm"},
        }

    def test_valid_list_response(self, sample_feature_data):
        response = ListFeaturesResponse(
            features=[sample_feature_data],
            current_page=0,
            page_size=100,
            total_items=1,
        )
        assert len(response.features) == 1
        assert isinstance(response.features[0], Feature)
        assert response.current_page == 0
        assert response.page_size == 100
        assert response.total_items == 1

    def test_features_is_required(self):
        with pytest.raises(ValidationError):
            ListFeaturesResponse(current_page=0, page_size=100, total_items=0)
