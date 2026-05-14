"""
Unit tests for api/v2/resources/grids/upload/schema.py

Tests the upload grid schema models and validation.
Pure unit tests with no external dependencies.
"""

import pytest
from api.resources.grids.upload.examples import ALL_GRID_UPLOAD_EXAMPLE_VALUES
from api.resources.grids.upload.schema import (
    CreateGridUploadRequest,
    GridUploadFormat,
    UploadBandDefinition,
)
from pydantic import ValidationError


class TestGridUploadFormat:
    def test_geotiff_valid(self):
        assert GridUploadFormat("geotiff") == GridUploadFormat.geotiff

    def test_invalid_format_raises(self):
        with pytest.raises(ValueError):
            GridUploadFormat("shapefile")


class TestUploadBandDefinition:
    def test_continuous_band_valid(self):
        band = UploadBandDefinition(
            key="bulk_density.foliage", type="continuous", unit="kg/m3"
        )
        assert band.key == "bulk_density.foliage"
        assert band.unit == "kg/m3"

    def test_categorical_band_valid(self):
        band = UploadBandDefinition(key="fbfm", type="categorical")
        assert band.unit is None
        assert band.description is None

    def test_invalid_type_raises(self):
        with pytest.raises(ValidationError):
            UploadBandDefinition(key="x", type="nominal")


class TestCreateGridUploadRequest:
    def test_minimal_request(self):
        req = CreateGridUploadRequest(bands=[{"key": "fbfm", "type": "categorical"}])
        assert req.format == GridUploadFormat.geotiff
        assert req.name == ""
        assert req.tags == []
        assert len(req.bands) == 1

    def test_multi_band_request(self):
        req = CreateGridUploadRequest(
            bands=[
                {"key": "bulk_density.foliage", "type": "continuous", "unit": "kg/m3"},
                {
                    "key": "bulk_density.branchwood",
                    "type": "continuous",
                    "unit": "kg/m3",
                },
            ],
        )
        assert len(req.bands) == 2
        assert req.bands[0].key == "bulk_density.foliage"

    def test_with_metadata(self):
        req = CreateGridUploadRequest(
            bands=[{"key": "fbfm", "type": "categorical"}],
            name="Custom FBFM40",
            description="Derived from LiDAR",
            tags=["lidar", "2024"],
        )
        assert req.name == "Custom FBFM40"
        assert req.tags == ["lidar", "2024"]

    def test_empty_bands_raises(self):
        with pytest.raises(ValidationError):
            CreateGridUploadRequest(bands=[])

    def test_missing_bands_raises(self):
        with pytest.raises(ValidationError):
            CreateGridUploadRequest()

    def test_invalid_band_type_raises(self):
        with pytest.raises(ValidationError):
            CreateGridUploadRequest(bands=[{"key": "fbfm", "type": "nominal"}])

    def test_format_defaults_to_geotiff(self):
        req = CreateGridUploadRequest(bands=[{"key": "fbfm", "type": "categorical"}])
        assert req.format == GridUploadFormat.geotiff

    def test_duplicate_band_keys_raises(self):
        """Two bands with the same key must be rejected at validation time."""
        with pytest.raises(ValidationError):
            CreateGridUploadRequest(
                bands=[
                    {"key": "fbfm", "type": "categorical"},
                    {"key": "fbfm", "type": "categorical"},
                ]
            )

    def test_num_buffer_cells_defaults_to_zero(self):
        req = CreateGridUploadRequest(bands=[{"key": "fbfm", "type": "categorical"}])
        assert req.num_buffer_cells == 0

    def test_num_buffer_cells_accepts_positive_int(self):
        req = CreateGridUploadRequest(
            bands=[{"key": "fbfm", "type": "categorical"}],
            num_buffer_cells=5,
        )
        assert req.num_buffer_cells == 5

    def test_num_buffer_cells_negative_raises(self):
        with pytest.raises(ValidationError):
            CreateGridUploadRequest(
                bands=[{"key": "fbfm", "type": "categorical"}],
                num_buffer_cells=-1,
            )


class TestOpenApiExamples:
    @pytest.mark.parametrize(
        "example_name,example_value", ALL_GRID_UPLOAD_EXAMPLE_VALUES
    )
    def test_example_validates_against_schema(self, example_name, example_value):
        """Every OpenAPI example must parse without error."""
        req = CreateGridUploadRequest(**example_value)
        assert len(req.bands) >= 1
