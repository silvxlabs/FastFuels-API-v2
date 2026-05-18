"""
Unit tests for api/v2/resources/grids/upload/netcdf/schema.py

Tests the netCDF upload grid schema models and validation.
Pure unit tests with no external dependencies.
"""

import pytest
from api.resources.grids.upload.netcdf.examples import (
    ALL_NETCDF_UPLOAD_EXAMPLE_VALUES,
)
from api.resources.grids.upload.netcdf.schema import CreateNetcdfUploadRequest
from pydantic import ValidationError


class TestCreateNetcdfUploadRequest:
    def test_minimal_request(self):
        req = CreateNetcdfUploadRequest()
        assert req.name == ""
        assert req.description == ""
        assert req.tags == []
        assert req.num_buffer_cells == 0

    def test_with_metadata(self):
        req = CreateNetcdfUploadRequest(
            name="Custom 3D grid",
            description="Voxelized bulk density",
            tags=["lidar", "external"],
        )
        assert req.name == "Custom 3D grid"
        assert req.tags == ["lidar", "external"]

    def test_with_buffer(self):
        req = CreateNetcdfUploadRequest(num_buffer_cells=3)
        assert req.num_buffer_cells == 3

    def test_num_buffer_cells_negative_raises(self):
        with pytest.raises(ValidationError):
            CreateNetcdfUploadRequest(num_buffer_cells=-1)

    def test_no_bands_field(self):
        """The netCDF route deliberately has no `bands` field."""
        assert "bands" not in CreateNetcdfUploadRequest.model_fields

    def test_no_format_field(self):
        """Format is implicit from the route, not in the body."""
        assert "format" not in CreateNetcdfUploadRequest.model_fields


class TestOpenApiExamples:
    @pytest.mark.parametrize(
        "example_name,example_value", ALL_NETCDF_UPLOAD_EXAMPLE_VALUES
    )
    def test_example_validates_against_schema(self, example_name, example_value):
        """Every OpenAPI example must parse without error."""
        CreateNetcdfUploadRequest(**example_value)
