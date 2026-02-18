"""
Unit tests for api/v2/resources/exports/schema.py

Tests the export schema models.
These are pure unit tests with no external dependencies.
"""

from datetime import datetime

import pytest
from api.resources.exports.schema import (
    Export,
    ExportGeoTiffRequest,
    ExportSortField,
    GeoTiffExportSource,
    ListExportsResponse,
    UpdateExportRequestBody,
)
from api.schema import JobStatus
from pydantic import ValidationError


class TestExport:
    """Tests for Export model."""

    def _make_export(self, **overrides):
        """Helper to build a valid Export dict."""
        data = {
            "id": "abc123",
            "domain_id": "domain_xyz",
            "name": "Test export",
            "description": "A test export",
            "status": "pending",
            "created_on": datetime(2026, 1, 1),
            "modified_on": datetime(2026, 1, 1),
            "source": {"name": "geotiff", "grid_id": "grid_123"},
            "tags": ["test"],
        }
        data.update(overrides)
        return data

    def test_minimal_valid_export(self):
        """Export with only required fields."""
        export = Export(**self._make_export())
        assert export.id == "abc123"
        assert export.domain_id == "domain_xyz"
        assert export.status == JobStatus.pending
        assert export.signed_url is None
        assert export.curl is None
        assert export.expires_on is None
        assert export.progress is None
        assert export.error is None

    def test_completed_export_with_signed_url(self):
        """Completed export has signed_url, curl, and expires_on."""
        export = Export(
            **self._make_export(
                status="completed",
                signed_url="https://storage.googleapis.com/bucket/file.tif?X-Goog-Signature=abc",
                curl="curl -o export.tif 'https://storage.googleapis.com/bucket/file.tif?X-Goog-Signature=abc'",
                expires_on=datetime(2026, 1, 8),
            )
        )
        assert export.status == JobStatus.completed
        assert (
            export.signed_url
            == "https://storage.googleapis.com/bucket/file.tif?X-Goog-Signature=abc"
        )
        assert export.curl is not None
        assert export.expires_on == datetime(2026, 1, 8)

    def test_failed_export_with_error(self):
        """Failed export has error details."""
        export = Export(
            **self._make_export(
                status="failed",
                error={
                    "code": "GRID_NOT_FOUND",
                    "message": "Source grid not found",
                },
            )
        )
        assert export.status == JobStatus.failed
        assert export.error.code == "GRID_NOT_FOUND"

    def test_id_required(self):
        data = self._make_export()
        del data["id"]
        with pytest.raises(ValidationError):
            Export(**data)

    def test_domain_id_required(self):
        data = self._make_export()
        del data["domain_id"]
        with pytest.raises(ValidationError):
            Export(**data)

    def test_status_required(self):
        data = self._make_export()
        del data["status"]
        with pytest.raises(ValidationError):
            Export(**data)

    def test_source_required(self):
        data = self._make_export()
        del data["source"]
        with pytest.raises(ValidationError):
            Export(**data)

    def test_created_on_required(self):
        data = self._make_export()
        del data["created_on"]
        with pytest.raises(ValidationError):
            Export(**data)

    def test_modified_on_required(self):
        data = self._make_export()
        del data["modified_on"]
        with pytest.raises(ValidationError):
            Export(**data)

    def test_name_defaults_to_empty(self):
        data = self._make_export()
        del data["name"]
        export = Export(**data)
        assert export.name == ""

    def test_description_defaults_to_empty(self):
        data = self._make_export()
        del data["description"]
        export = Export(**data)
        assert export.description == ""

    def test_tags_defaults_to_empty(self):
        data = self._make_export()
        del data["tags"]
        export = Export(**data)
        assert export.tags == []

    def test_source_is_dict(self):
        """Source field accepts any dict (validated by creation endpoints)."""
        export = Export(
            **self._make_export(
                source={"name": "geotiff", "grid_id": "g1", "bands": ["fbfm"]}
            )
        )
        assert export.source["name"] == "geotiff"
        assert export.source["bands"] == ["fbfm"]

    def test_all_status_values(self):
        """All JobStatus values are accepted."""
        for s in JobStatus:
            export = Export(**self._make_export(status=s.value))
            assert export.status == s

    def test_invalid_status_rejected(self):
        with pytest.raises(ValidationError):
            Export(**self._make_export(status="invalid"))


class TestExportSortField:
    """Tests for ExportSortField enum."""

    def test_created_on(self):
        assert ExportSortField.created_on == "created_on"

    def test_modified_on(self):
        assert ExportSortField.modified_on == "modified_on"

    def test_name(self):
        assert ExportSortField.name == "name"

    def test_has_3_members(self):
        assert len(ExportSortField) == 3


class TestUpdateExportRequestBody:
    """Tests for UpdateExportRequestBody model."""

    def test_all_fields_optional(self):
        body = UpdateExportRequestBody()
        assert body.name is None
        assert body.description is None
        assert body.tags is None

    def test_name_only(self):
        body = UpdateExportRequestBody(name="New name")
        assert body.name == "New name"
        assert body.description is None

    def test_tags_replaces(self):
        body = UpdateExportRequestBody(tags=["new-tag"])
        assert body.tags == ["new-tag"]


class TestListExportsResponse:
    """Tests for ListExportsResponse model."""

    def test_empty_list(self):
        response = ListExportsResponse(
            exports=[], current_page=0, page_size=100, total_items=0
        )
        assert response.exports == []
        assert response.total_items == 0

    def test_with_exports(self):
        export_data = {
            "id": "abc",
            "domain_id": "d1",
            "status": "completed",
            "created_on": datetime(2026, 1, 1),
            "modified_on": datetime(2026, 1, 1),
            "source": {"name": "geotiff", "grid_id": "g1"},
        }
        response = ListExportsResponse(
            exports=[export_data],
            current_page=0,
            page_size=100,
            total_items=1,
        )
        assert len(response.exports) == 1
        assert response.exports[0].id == "abc"


class TestExportGeoTiffRequest:
    """Tests for ExportGeoTiffRequest model."""

    def test_minimal_valid_request(self):
        request = ExportGeoTiffRequest(grid_ids=["grid_abc"])
        assert request.grid_ids == ["grid_abc"]
        assert request.bands is None
        assert request.expiration_days == 7
        assert request.name == ""
        assert request.description == ""
        assert request.tags == []

    def test_grid_ids_required(self):
        with pytest.raises(ValidationError):
            ExportGeoTiffRequest()

    def test_grid_ids_cannot_be_empty(self):
        with pytest.raises(ValidationError):
            ExportGeoTiffRequest(grid_ids=[])

    def test_with_bands(self):
        request = ExportGeoTiffRequest(
            grid_ids=["grid_abc"],
            bands=["fuel_load.1hr", "fuel_load.10hr"],
        )
        assert request.bands == ["fuel_load.1hr", "fuel_load.10hr"]

    def test_with_all_fields(self):
        request = ExportGeoTiffRequest(
            grid_ids=["grid_abc"],
            bands=["fbfm"],
            expiration_days=3,
            name="FBFM export",
            description="For QGIS",
            tags=["surface-fuel"],
        )
        assert request.name == "FBFM export"
        assert request.description == "For QGIS"
        assert request.tags == ["surface-fuel"]
        assert request.expiration_days == 3

    def test_none_bands_exports_all(self):
        """bands=None means export all bands."""
        request = ExportGeoTiffRequest(grid_ids=["g1"])
        assert request.bands is None

    def test_empty_bands_list_accepted(self):
        """Empty bands list is technically valid (handler may reject)."""
        request = ExportGeoTiffRequest(grid_ids=["g1"], bands=[])
        assert request.bands == []

    def test_expiration_days_default(self):
        request = ExportGeoTiffRequest(grid_ids=["g1"])
        assert request.expiration_days == 7

    def test_expiration_days_custom(self):
        request = ExportGeoTiffRequest(grid_ids=["g1"], expiration_days=3)
        assert request.expiration_days == 3

    def test_expiration_days_max_7(self):
        with pytest.raises(ValidationError):
            ExportGeoTiffRequest(grid_ids=["g1"], expiration_days=8)

    def test_expiration_days_min_1(self):
        with pytest.raises(ValidationError):
            ExportGeoTiffRequest(grid_ids=["g1"], expiration_days=0)


class TestGeoTiffExportSource:
    """Tests for GeoTiffExportSource model."""

    def test_name_is_always_geotiff(self):
        source = GeoTiffExportSource(grid_ids=["g1"])
        assert source.name == "geotiff"

    def test_name_cannot_be_overridden(self):
        with pytest.raises(ValidationError):
            GeoTiffExportSource(name="other", grid_ids=["g1"])

    def test_grid_ids_required(self):
        with pytest.raises(ValidationError):
            GeoTiffExportSource()

    def test_bands_optional(self):
        source = GeoTiffExportSource(grid_ids=["g1"])
        assert source.bands is None

    def test_with_bands(self):
        source = GeoTiffExportSource(grid_ids=["g1"], bands=["fuel_load.1hr"])
        assert source.bands == ["fuel_load.1hr"]

    def test_model_dump(self):
        source = GeoTiffExportSource(grid_ids=["g1"], bands=["fbfm"])
        data = source.model_dump()
        assert data == {
            "name": "geotiff",
            "grid_ids": ["g1"],
            "bands": ["fbfm"],
        }
