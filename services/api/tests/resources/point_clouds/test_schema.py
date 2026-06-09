"""
Unit tests for api/v2/resources/point_clouds/schema.py

Tests the PointCloud schema models and enums. These are pure unit tests with no
external dependencies.
"""

import pytest
from api.resources.point_clouds.schema import (
    DuplicatePointCloudRequest,
    ListPointCloudsResponse,
    PointCloud,
    PointCloudGeoreference,
    PointCloudSortField,
    PointCloudType,
    UpdatePointCloudRequestBody,
)
from pydantic import ValidationError


def _make_point_cloud(**overrides) -> PointCloud:
    """Construct a minimal valid PointCloud, with optional field overrides."""
    data = {
        "id": "9f1c2a7b4e0d4c8a9b2e1f3a5c6d7e8f",
        "domain_id": "1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d",
        "type": "als",
        "status": "pending",
        "source": {"name": "3dep"},
    }
    data.update(overrides)
    return PointCloud(**data)


class TestPointCloudType:
    """Tests for PointCloudType enum."""

    def test_values(self):
        assert PointCloudType.als == "als"
        assert PointCloudType.tls == "tls"

    def test_membership(self):
        assert list(PointCloudType) == [PointCloudType.als, PointCloudType.tls]


class TestPointCloudSortField:
    """Tests for PointCloudSortField enum."""

    def test_all_values(self):
        assert PointCloudSortField.created_on == "created_on"
        assert PointCloudSortField.modified_on == "modified_on"
        assert PointCloudSortField.name == "name"


class TestPointCloudGeoreference:
    """Tests for the PointCloudGeoreference model."""

    def test_construction(self):
        geo = PointCloudGeoreference(
            crs="EPSG:32612",
            bounds=[500000.0, 5060000.0, 1800.0, 501000.0, 5061000.0, 1980.0],
        )
        assert geo.crs == "EPSG:32612"
        assert geo.bounds == (500000.0, 5060000.0, 1800.0, 501000.0, 5061000.0, 1980.0)

    def test_bounds_is_3d_six_tuple(self):
        """A 3D point cloud bbox is min/max over x, y, and z — six values."""
        with pytest.raises(ValidationError):
            PointCloudGeoreference(crs="EPSG:32612", bounds=[0.0, 0.0, 1.0, 1.0])

    def test_crs_required(self):
        with pytest.raises(ValidationError):
            PointCloudGeoreference(bounds=[0.0, 0.0, 0.0, 1.0, 1.0, 1.0])


class TestPointCloud:
    """Tests for the PointCloud resource model."""

    def test_minimal_construction(self):
        pc = _make_point_cloud()
        assert pc.id == "9f1c2a7b4e0d4c8a9b2e1f3a5c6d7e8f"
        assert pc.type == PointCloudType.als
        assert pc.status == "pending"
        assert pc.source == {"name": "3dep"}

    def test_defaults(self):
        pc = _make_point_cloud()
        assert pc.name == ""
        assert pc.description == ""
        assert pc.tags == []
        assert pc.progress is None
        assert pc.georeference is None
        assert pc.checksum is None
        assert pc.error is None

    def test_georeference_round_trips(self):
        pc = _make_point_cloud(
            status="completed",
            georeference={
                "crs": "EPSG:5070",
                "bounds": [0.0, 0.0, 1.0, 10.0, 10.0, 50.0],
            },
        )
        assert pc.georeference.crs == "EPSG:5070"
        dumped = pc.model_dump()
        assert dumped["georeference"]["bounds"] == (0.0, 0.0, 1.0, 10.0, 10.0, 50.0)

    def test_checksum_round_trips(self):
        pc = _make_point_cloud(checksum="cafe" * 8)
        assert pc.checksum == "cafe" * 8
        assert pc.model_dump()["checksum"] == "cafe" * 8

    def test_type_validation(self):
        with pytest.raises(ValidationError):
            _make_point_cloud(type="mls")

    def test_json_schema_examples_are_valid(self):
        """Every documented OpenAPI example must validate as a PointCloud."""
        examples = PointCloud.model_config["json_schema_extra"]["examples"]
        assert len(examples) >= 1
        for example in examples:
            PointCloud(**example)


class TestUpdatePointCloudRequestBody:
    """Tests for UpdatePointCloudRequestBody."""

    def test_defaults_are_none(self):
        body = UpdatePointCloudRequestBody()
        assert body.name is None
        assert body.description is None
        assert body.tags is None

    def test_name_max_length(self):
        with pytest.raises(ValidationError):
            UpdatePointCloudRequestBody(name="x" * 256)

    def test_description_max_length(self):
        with pytest.raises(ValidationError):
            UpdatePointCloudRequestBody(description="x" * 2001)


class TestDuplicatePointCloudRequest:
    """Tests for DuplicatePointCloudRequest."""

    def test_defaults_are_none(self):
        body = DuplicatePointCloudRequest()
        assert body.name is None
        assert body.description is None
        assert body.tags is None

    def test_name_max_length(self):
        with pytest.raises(ValidationError):
            DuplicatePointCloudRequest(name="x" * 256)


class TestListPointCloudsResponse:
    """Tests for the paginated list response."""

    def test_construction(self):
        response = ListPointCloudsResponse(
            point_clouds=[_make_point_cloud()],
            current_page=0,
            page_size=100,
            total_items=1,
        )
        assert len(response.point_clouds) == 1
        assert response.total_items == 1
