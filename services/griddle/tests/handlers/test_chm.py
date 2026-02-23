"""
Tests for CHM handler.
"""

import os
from unittest.mock import MagicMock, patch

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rioxarray  # noqa: F401
import xarray as xr
from griddle.handlers.chm import fetch_meta_chm
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from shapely.geometry import box


def _make_mock_raster(chm_values, crs="EPSG:32611"):
    """Create a mock RasterConnection that returns a DataArray of CHM values.

    Args:
        chm_values: 2D numpy array of CHM pixel values
        crs: CRS string

    Returns:
        Mock RasterConnection instance
    """
    height, width = chm_values.shape
    transform = from_bounds(
        300000, 4100000, 300000 + width * 1, 4100000 + height * 1, width, height
    )

    da = xr.DataArray(
        chm_values[np.newaxis, :, :],  # Add band dim for squeeze
        dims=["band", "y", "x"],
        coords={
            "band": [1],
            "y": np.arange(height),
            "x": np.arange(width),
        },
    )
    da = da.rio.write_crs(crs)
    da = da.rio.write_transform(transform)

    mock_raster = MagicMock()
    mock_raster.raster_resolution = 1
    mock_raster.extract_window.return_value = da
    return mock_raster


def _make_tile_mapping():
    """Create a tile mapping GeoDataFrame covering the globe."""
    return gpd.GeoDataFrame(
        {"tile": ["test_tile_001"]},
        geometry=[box(-180, -90, 180, 90)],
        crs="EPSG:4326",
    )


def _make_roi():
    """Create a real GeoDataFrame ROI in a projected CRS."""
    return gpd.GeoDataFrame(
        geometry=[box(300000, 4100000, 300100, 4100100)],
        crs="EPSG:32611",
    )


class TestFetchMetaChm:
    """Tests for fetch_meta_chm with chm band."""

    @patch("griddle.handlers.chm.RasterConnection")
    @patch("griddle.handlers.chm.gpd.read_file")
    def test_returns_dataset(self, mock_read_file, mock_raster_cls):
        """fetch_meta_chm returns an xr.Dataset."""
        chm_values = np.array([[10.5, 20.3], [15.2, 18.7]], dtype=np.float32)
        mock_raster_cls.return_value = _make_mock_raster(chm_values)
        mock_read_file.return_value = _make_tile_mapping()
        progress = MagicMock()

        result = fetch_meta_chm(_make_roi(), "2024", progress)

        assert isinstance(result, xr.Dataset)

    @patch("griddle.handlers.chm.RasterConnection")
    @patch("griddle.handlers.chm.gpd.read_file")
    def test_has_chm_variable(self, mock_read_file, mock_raster_cls):
        """Dataset contains a 'chm' variable."""
        chm_values = np.array([[10.5, 20.3]], dtype=np.float32)
        mock_raster_cls.return_value = _make_mock_raster(chm_values)
        mock_read_file.return_value = _make_tile_mapping()
        progress = MagicMock()

        result = fetch_meta_chm(_make_roi(), "2024", progress)

        assert "chm" in result.data_vars

    @patch("griddle.handlers.chm.RasterConnection")
    @patch("griddle.handlers.chm.gpd.read_file")
    def test_chm_values_preserved(self, mock_read_file, mock_raster_cls):
        """CHM pixel values are preserved in the output."""
        chm_values = np.array([[10.5, 20.3], [15.2, 18.7]], dtype=np.float32)
        mock_raster_cls.return_value = _make_mock_raster(chm_values)
        mock_read_file.return_value = _make_tile_mapping()
        progress = MagicMock()

        result = fetch_meta_chm(_make_roi(), "2024", progress)

        np.testing.assert_array_almost_equal(result["chm"].values, chm_values)

    @patch("griddle.handlers.chm.RasterConnection")
    @patch("griddle.handlers.chm.gpd.read_file")
    def test_crs_preserved(self, mock_read_file, mock_raster_cls):
        """CRS is preserved in the output dataset."""
        chm_values = np.array([[10.5, 20.3], [15.2, 18.7]], dtype=np.float32)
        mock_raster_cls.return_value = _make_mock_raster(chm_values, crs="EPSG:32611")
        mock_read_file.return_value = _make_tile_mapping()
        progress = MagicMock()

        result = fetch_meta_chm(_make_roi(), "2024", progress)

        assert result.rio.crs == CRS.from_epsg(32611)

    @patch("griddle.handlers.chm.RasterConnection")
    @patch("griddle.handlers.chm.gpd.read_file")
    def test_dims_are_y_x(self, mock_read_file, mock_raster_cls):
        """CHM variable has (y, x) dims."""
        chm_values = np.array([[10.5, 20.3], [15.2, 18.7]], dtype=np.float32)
        mock_raster_cls.return_value = _make_mock_raster(chm_values)
        mock_read_file.return_value = _make_tile_mapping()
        progress = MagicMock()

        result = fetch_meta_chm(_make_roi(), "2024", progress)

        assert result["chm"].dims == ("y", "x")

    @patch("griddle.handlers.chm.RasterConnection")
    @patch("griddle.handlers.chm.gpd.read_file")
    def test_s3_url_constructed_from_tile(self, mock_read_file, mock_raster_cls):
        """Correct S3 URL is constructed from the tile name."""
        chm_values = np.array([[10.5]], dtype=np.float32)
        mock_raster_cls.return_value = _make_mock_raster(chm_values)
        mock_read_file.return_value = _make_tile_mapping()
        progress = MagicMock()

        fetch_meta_chm(_make_roi(), "2024", progress)

        url = mock_raster_cls.call_args[0][0]
        assert "s3://dataforgood-fb-data/forests/v1/alsgedi_global_v6_float/chm/" in url
        assert "test_tile_001.tif" in url

    @patch("griddle.handlers.chm.RasterConnection")
    @patch("griddle.handlers.chm.gpd.read_file")
    def test_aws_no_sign_request_scoped(self, mock_read_file, mock_raster_cls):
        """AWS_NO_SIGN_REQUEST is set during S3 access and restored after."""
        chm_values = np.array([[10.5]], dtype=np.float32)
        mock_raster_cls.return_value = _make_mock_raster(chm_values)
        mock_read_file.return_value = _make_tile_mapping()
        progress = MagicMock()

        os.environ.pop("AWS_NO_SIGN_REQUEST", None)

        fetch_meta_chm(_make_roi(), "2024", progress)

        assert "AWS_NO_SIGN_REQUEST" not in os.environ

    @patch("griddle.handlers.chm.RasterConnection")
    @patch("griddle.handlers.chm.gpd.read_file")
    def test_progress_called(self, mock_read_file, mock_raster_cls):
        """Progress callback is invoked during processing."""
        chm_values = np.array([[10.5]], dtype=np.float32)
        mock_raster_cls.return_value = _make_mock_raster(chm_values)
        mock_read_file.return_value = _make_tile_mapping()
        progress = MagicMock()

        fetch_meta_chm(_make_roi(), "2024", progress)

        assert progress.call_count >= 2

    @patch("griddle.handlers.chm.RasterConnection")
    @patch("griddle.handlers.chm.gpd.read_file")
    def test_no_intersecting_tiles_raises(self, mock_read_file, mock_raster_cls):
        """Raises ValueError when no tiles intersect the ROI."""
        mock_read_file.return_value = gpd.GeoDataFrame(
            {"tile": pd.Series([], dtype=str)},
            geometry=[],
            crs="EPSG:4326",
        )
        progress = MagicMock()

        with pytest.raises(ValueError, match="No Meta CHM tiles found"):
            fetch_meta_chm(_make_roi(), "2024", progress)

    @patch("griddle.handlers.chm.merge_arrays")
    @patch("griddle.handlers.chm.RasterConnection")
    @patch("griddle.handlers.chm.gpd.read_file")
    def test_multiple_tiles_merged(self, mock_read_file, mock_raster_cls, mock_merge):
        """Multiple intersecting tiles are fetched and merged."""
        chm_values = np.array([[10.5]], dtype=np.float32)
        mock_raster_cls.return_value = _make_mock_raster(chm_values)
        mock_read_file.return_value = gpd.GeoDataFrame(
            {"tile": ["tile_a", "tile_b"]},
            geometry=[box(-180, -90, 180, 90), box(-180, -90, 180, 90)],
            crs="EPSG:4326",
        )
        # merge_arrays returns a DataArray with CRS
        merged_da = _make_mock_raster(chm_values).extract_window.return_value
        merged_da = merged_da.squeeze("band", drop=True)
        mock_merge.return_value = merged_da
        progress = MagicMock()

        result = fetch_meta_chm(_make_roi(), "2024", progress)

        assert mock_raster_cls.call_count == 2
        mock_merge.assert_called_once()
        assert isinstance(result, xr.Dataset)

    @patch("griddle.handlers.chm.RasterConnection")
    @patch("griddle.handlers.chm.gpd.read_file")
    def test_tile_map_url_uses_version(self, mock_read_file, mock_raster_cls):
        """Tile mapping URL includes the version string."""
        chm_values = np.array([[10.5]], dtype=np.float32)
        mock_raster_cls.return_value = _make_mock_raster(chm_values)
        mock_read_file.return_value = _make_tile_mapping()
        progress = MagicMock()

        fetch_meta_chm(_make_roi(), "2024", progress)

        url = mock_read_file.call_args[0][0]
        assert "Meta2024" in url
