"""
Unit tests for api/v2/resources/grids/utils.py validators.

These are pure-logic tests: they call the validators directly with fabricated
grid_data dicts and assert on the raised HTTPException. No server or Firestore
is required.
"""

import pytest
from api.resources.exports.schema import GridExportFormat
from api.resources.grids.utils import validate_format_supports_grid
from fastapi import HTTPException


def _grid(shape):
    """Minimal grid_data dict carrying a georeference of the given shape."""
    return {"georeference": {"shape": shape}}


class TestValidateFormatSupportsGrid:
    """validate_format_supports_grid rejects 3D grids for 2D-only formats."""

    def test_geotiff_3d_raises_422(self):
        with pytest.raises(HTTPException) as exc:
            validate_format_supports_grid(
                _grid((5, 34, 34)), "grid-1", GridExportFormat.geotiff
            )
        assert exc.value.status_code == 422
        detail = exc.value.detail
        assert "geotiff" in detail
        assert "netcdf" in detail and "zarr" in detail

    def test_geotiff_2d_passes(self):
        # A 2D grid is fine for geotiff — no exception.
        validate_format_supports_grid(
            _grid((34, 34)), "grid-1", GridExportFormat.geotiff
        )

    def test_zarr_3d_passes(self):
        # zarr supports 3D natively.
        validate_format_supports_grid(
            _grid((5, 34, 34)), "grid-1", GridExportFormat.zarr
        )

    def test_netcdf_3d_passes(self):
        # netcdf supports 3D natively.
        validate_format_supports_grid(
            _grid((5, 34, 34)), "grid-1", GridExportFormat.netcdf
        )

    def test_missing_georeference_is_noop(self):
        # Without a georeference the dimensionality is unknown; the check is a
        # no-op (the router's completed-status fetch guarantees one in practice).
        validate_format_supports_grid({}, "grid-1", GridExportFormat.geotiff)
