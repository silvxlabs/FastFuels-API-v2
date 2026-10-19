"""Tests for griddle.storage."""

from unittest.mock import patch

import numpy as np
import pytest
import rioxarray  # noqa: F401
import xarray as xr
from griddle.storage import load_band

from lib.errors import ProcessingError


def _grid_band(values, nodata=-9999):
    """A small north-up DataArray in EPSG:5070."""
    values = np.asarray(values)
    ny, nx = values.shape
    da = xr.DataArray(
        values,
        dims=("y", "x"),
        coords={"y": -30.0 * (np.arange(ny) + 0.5), "x": 30.0 * (np.arange(nx) + 0.5)},
    )
    return da.rio.write_crs("EPSG:5070").rio.write_nodata(nodata)


class TestLoadBand:
    """Unit tests for loading one band of a stored grid."""

    @patch("griddle.storage.load_zarr")
    def test_returns_the_band_as_stored(self, mock_load):
        stored = _grid_band(np.array([[121, 131], [0, 111]], dtype=np.int16))
        mock_load.return_value = xr.Dataset({"fdist": stored})

        result = load_band("g1", "fdist")

        np.testing.assert_array_equal(result.values, stored.values)
        assert result.dtype == np.int16
        assert result.rio.nodata == -9999
        mock_load.assert_called_once_with("g1")

    @patch("griddle.storage.load_zarr")
    def test_missing_grid_raises_a_clear_error(self, mock_load):
        mock_load.side_effect = FileNotFoundError("no such store")

        with pytest.raises(ProcessingError) as exc_info:
            load_band("g1", "fdist")

        assert exc_info.value.code == "SOURCE_GRID_UNAVAILABLE"

    @patch("griddle.storage.load_zarr")
    def test_missing_band_raises_and_lists_the_available_bands(self, mock_load):
        stored = _grid_band(np.zeros((2, 2), dtype=np.int16))
        mock_load.return_value = xr.Dataset({"fbfm": stored})

        with pytest.raises(ProcessingError) as exc_info:
            load_band("g1", "fdist")

        assert exc_info.value.code == "SOURCE_BAND_NOT_FOUND"
        assert "fbfm" in exc_info.value.message
