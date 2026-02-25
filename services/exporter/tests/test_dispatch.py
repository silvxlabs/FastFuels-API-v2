"""
Unit tests for exporter dispatch module.

Tests routing logic without external dependencies.
"""

import pytest
from exporter.dispatch import dispatch_handler
from exporter.errors import ProcessingError


class TestDispatchHandler:
    """Tests for dispatch_handler routing."""

    def _noop_progress(self, message: str, percent: int | None = None):
        pass

    def test_unknown_format_raises_processing_error(self):
        """Unknown source name raises ProcessingError with UNKNOWN_FORMAT code."""
        export = {
            "id": "test-export",
            "source": {"name": "pdf"},
        }

        with pytest.raises(ProcessingError) as exc_info:
            dispatch_handler(export, self._noop_progress)

        assert exc_info.value.code == "UNKNOWN_FORMAT"
        assert "pdf" in exc_info.value.message

    def test_geotiff_source_name_recognized(self):
        """geotiff source name is dispatched (will fail on missing grid, but not UNKNOWN_FORMAT)."""
        export = {
            "id": "test-export",
            "source": {
                "name": "geotiff",
                "grid_ids": ["nonexistent"],
            },
        }

        # This will fail because the grid doesn't exist in GCS,
        # but it should NOT raise UNKNOWN_FORMAT
        with pytest.raises(Exception) as exc_info:
            dispatch_handler(export, self._noop_progress)

        # It should be a ProcessingError about loading the grid, not about format
        if isinstance(exc_info.value, ProcessingError):
            assert exc_info.value.code != "UNKNOWN_FORMAT"

    @pytest.mark.parametrize("fmt", ["parquet", "csv", "geojson", "geopackage"])
    def test_inventory_format_recognized(self, fmt):
        """Inventory format names are dispatched (will fail on missing inventory, but not UNKNOWN_FORMAT)."""
        export = {
            "id": "test-export",
            "source": {
                "name": fmt,
                "inventory_id": "nonexistent",
            },
        }

        with pytest.raises(Exception) as exc_info:
            dispatch_handler(export, self._noop_progress)

        if isinstance(exc_info.value, ProcessingError):
            assert exc_info.value.code != "UNKNOWN_FORMAT"

    def test_missing_source_raises(self):
        """Export without source field raises KeyError."""
        export = {"id": "test-export"}

        with pytest.raises(KeyError):
            dispatch_handler(export, self._noop_progress)
