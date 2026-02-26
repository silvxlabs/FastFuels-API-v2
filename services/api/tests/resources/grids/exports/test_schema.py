"""
Unit tests for the grid export request schema and examples.

These are pure unit tests with no external dependencies.
"""

import pytest
from api.resources.exports.schema import (
    ExportGridRequest,
    GridExportSource,
)
from api.resources.grids.exports.examples import ALL_GRID_EXPORT_EXAMPLE_VALUES


class TestExportGridRequestDetailed:
    """Validation tests for ExportGridRequest."""

    def test_no_grid_id_field(self):
        """Request model has no grid_id — it comes from the URL path."""
        request = ExportGridRequest()
        assert not hasattr(request, "grid_id")

    def test_bands_with_dot_notation(self):
        """Dot-notation band keys are accepted."""
        request = ExportGridRequest(
            bands=["fuel_load.1hr", "savr.live_herb"],
        )
        assert request.bands == ["fuel_load.1hr", "savr.live_herb"]

    def test_single_band(self):
        request = ExportGridRequest(bands=["fbfm"])
        assert request.bands == ["fbfm"]

    def test_defaults(self):
        request = ExportGridRequest()
        assert request.bands is None
        assert request.name == ""
        assert request.description == ""
        assert request.tags == []


class TestGridExportSourceDetailed:
    """Validation tests for GridExportSource."""

    def test_grid_id_string(self):
        source = GridExportSource(name="geotiff", grid_id="g1")
        assert source.grid_id == "g1"

    def test_zarr_source(self):
        source = GridExportSource(name="zarr", grid_id="g1", bands=["fbfm"])
        assert source.name == "zarr"
        assert source.grid_id == "g1"

    def test_bands_none_vs_empty(self):
        """None means all bands, empty list means no bands."""
        source_none = GridExportSource(name="geotiff", grid_id="g1", bands=None)
        source_empty = GridExportSource(name="geotiff", grid_id="g1", bands=[])
        assert source_none.bands is None
        assert source_empty.bands == []


class TestExampleValidation:
    """Validate that all documented examples pass schema validation."""

    @pytest.mark.parametrize(
        "example_name,example_value", ALL_GRID_EXPORT_EXAMPLE_VALUES
    )
    def test_grid_export_example_is_valid(self, example_name, example_value):
        """Each per-grid example should pass schema validation."""
        ExportGridRequest(**example_value)
