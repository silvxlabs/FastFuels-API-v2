"""
Unit tests for the GeoTIFF export request schema and examples.

These are pure unit tests with no external dependencies.
"""

import pytest
from api.resources.exports.schema import (
    ExportGeoTiffRequest,
    ExportSingleGridGeoTiffRequest,
    GeoTiffExportSource,
)
from api.resources.grids.exports.examples import (
    ALL_GEOTIFF_EXAMPLE_VALUES,
    ALL_SINGLE_GRID_GEOTIFF_EXAMPLE_VALUES,
)
from pydantic import ValidationError


class TestExportGeoTiffRequestDetailed:
    """Validation tests for ExportGeoTiffRequest (domain-level, multi-grid)."""

    def test_grid_ids_is_list(self):
        request = ExportGeoTiffRequest(grid_ids=["abc123"])
        assert isinstance(request.grid_ids, list)
        assert request.grid_ids == ["abc123"]

    def test_multiple_grid_ids(self):
        request = ExportGeoTiffRequest(grid_ids=["abc123", "def456"])
        assert request.grid_ids == ["abc123", "def456"]

    def test_empty_grid_ids_rejected(self):
        with pytest.raises(ValidationError):
            ExportGeoTiffRequest(grid_ids=[])

    def test_bands_with_dot_notation(self):
        """Dot-notation band keys are accepted."""
        request = ExportGeoTiffRequest(
            grid_ids=["g1"],
            bands=["fuel_load.1hr", "savr.live_herb"],
        )
        assert request.bands == ["fuel_load.1hr", "savr.live_herb"]

    def test_single_band(self):
        request = ExportGeoTiffRequest(grid_ids=["g1"], bands=["fbfm"])
        assert request.bands == ["fbfm"]


class TestExportSingleGridGeoTiffRequestDetailed:
    """Validation tests for ExportSingleGridGeoTiffRequest (per-grid)."""

    def test_no_grid_id_field(self):
        """Request model has no grid_id — it comes from the URL path."""
        request = ExportSingleGridGeoTiffRequest()
        assert not hasattr(request, "grid_id")

    def test_bands_with_dot_notation(self):
        request = ExportSingleGridGeoTiffRequest(
            bands=["fuel_load.1hr", "savr.live_herb"],
        )
        assert request.bands == ["fuel_load.1hr", "savr.live_herb"]

    def test_defaults(self):
        request = ExportSingleGridGeoTiffRequest()
        assert request.bands is None
        assert request.name == ""
        assert request.description == ""
        assert request.tags == []


class TestGeoTiffExportSourceDetailed:
    """Validation tests for GeoTiffExportSource."""

    def test_grid_ids_list(self):
        source = GeoTiffExportSource(grid_ids=["g1", "g2"])
        assert source.grid_ids == ["g1", "g2"]

    def test_bands_none_vs_empty(self):
        """None means all bands, empty list means no bands."""
        source_none = GeoTiffExportSource(grid_ids=["g1"], bands=None)
        source_empty = GeoTiffExportSource(grid_ids=["g1"], bands=[])
        assert source_none.bands is None
        assert source_empty.bands == []


class TestExampleValidation:
    """Validate that all documented examples pass schema validation."""

    @pytest.mark.parametrize("example_name,example_value", ALL_GEOTIFF_EXAMPLE_VALUES)
    def test_domain_level_example_is_valid(self, example_name, example_value):
        """Each domain-level example should pass schema validation."""
        request = ExportGeoTiffRequest(**example_value)
        assert request.grid_ids == example_value["grid_ids"]

    @pytest.mark.parametrize(
        "example_name,example_value", ALL_SINGLE_GRID_GEOTIFF_EXAMPLE_VALUES
    )
    def test_single_grid_example_is_valid(self, example_name, example_value):
        """Each per-grid example should pass schema validation."""
        ExportSingleGridGeoTiffRequest(**example_value)
