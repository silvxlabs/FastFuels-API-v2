"""
Tests for CHM handler.
"""

import io
import os
from unittest.mock import MagicMock, patch

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rioxarray  # noqa: F401
import xarray as xr
from griddle.errors import ProcessingError
from griddle.handlers.chm import fetch_meta_chm, fetch_naip_chm
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


def _make_tile_index_bytes():
    """Create a parquet index DataFrame covering the globe, serialized to bytes."""
    df = pd.DataFrame(
        {
            "tile": ["test_tile_001"],
            "bbox_xmin": [-180.0],
            "bbox_ymin": [-90.0],
            "bbox_xmax": [180.0],
            "bbox_ymax": [90.0],
        }
    )
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    return buf.getvalue()


def _make_roi():
    """Create a real GeoDataFrame ROI in a projected CRS."""
    return gpd.GeoDataFrame(
        geometry=[box(300000, 4100000, 300100, 4100100)],
        crs="EPSG:32611",
    )


def _make_naip_tile_index_bytes():
    """Create a NAIP parquet index DataFrame covering the globe, serialized to bytes."""
    df = pd.DataFrame(
        {
            "chm_url": ["http://fake-ntsg-server.com/tile_001.tif"],
            "scale_factor": [100.0],
            "bbox_xmin": [-180.0],
            "bbox_ymin": [-90.0],
            "bbox_xmax": [180.0],
            "bbox_ymax": [90.0],
        }
    )
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    return buf.getvalue()


class TestFetchMetaChm:
    """Tests for fetch_meta_chm with chm band."""

    @patch("griddle.handlers.chm.RasterConnection")
    @patch("griddle.handlers.chm._fs")
    def test_returns_dataset_and_tile_metadata(self, mock_fs, mock_raster_cls):
        """fetch_meta_chm returns a (Dataset, TileMetadata) tuple."""
        chm_values = np.array([[10.5, 20.3], [15.2, 18.7]], dtype=np.float32)
        mock_raster_cls.return_value = _make_mock_raster(chm_values)
        mock_fs.cat.return_value = _make_tile_index_bytes()
        progress = MagicMock()

        ds, tile_metadata = fetch_meta_chm(_make_roi(), "2", progress)

        assert isinstance(ds, xr.Dataset)
        assert tile_metadata["tile_count"] == 1
        assert len(tile_metadata["tiles"]) == 1

    @patch("griddle.handlers.chm.RasterConnection")
    @patch("griddle.handlers.chm._fs")
    def test_has_chm_variable(self, mock_fs, mock_raster_cls):
        """Dataset contains a 'chm' variable."""
        chm_values = np.array([[10.5, 20.3]], dtype=np.float32)
        mock_raster_cls.return_value = _make_mock_raster(chm_values)
        mock_fs.cat.return_value = _make_tile_index_bytes()
        progress = MagicMock()

        ds, _ = fetch_meta_chm(_make_roi(), "2", progress)

        assert "chm" in ds.data_vars

    @patch("griddle.handlers.chm.RasterConnection")
    @patch("griddle.handlers.chm._fs")
    def test_chm_values_preserved(self, mock_fs, mock_raster_cls):
        """CHM pixel values are preserved in the output."""
        chm_values = np.array([[10.5, 20.3], [15.2, 18.7]], dtype=np.float32)
        mock_raster_cls.return_value = _make_mock_raster(chm_values)
        mock_fs.cat.return_value = _make_tile_index_bytes()
        progress = MagicMock()

        ds, _ = fetch_meta_chm(_make_roi(), "2", progress)

        np.testing.assert_array_almost_equal(ds["chm"].values, chm_values)

    @patch("griddle.handlers.chm.RasterConnection")
    @patch("griddle.handlers.chm._fs")
    def test_crs_preserved(self, mock_fs, mock_raster_cls):
        """CRS is preserved in the output dataset."""
        chm_values = np.array([[10.5, 20.3], [15.2, 18.7]], dtype=np.float32)
        mock_raster_cls.return_value = _make_mock_raster(chm_values, crs="EPSG:32611")
        mock_fs.cat.return_value = _make_tile_index_bytes()
        progress = MagicMock()

        ds, _ = fetch_meta_chm(_make_roi(), "2", progress)

        assert ds.rio.crs == CRS.from_epsg(32611)

    @patch("griddle.handlers.chm.RasterConnection")
    @patch("griddle.handlers.chm._fs")
    def test_dims_are_y_x(self, mock_fs, mock_raster_cls):
        """CHM variable has (y, x) dims."""
        chm_values = np.array([[10.5, 20.3], [15.2, 18.7]], dtype=np.float32)
        mock_raster_cls.return_value = _make_mock_raster(chm_values)
        mock_fs.cat.return_value = _make_tile_index_bytes()
        progress = MagicMock()

        ds, _ = fetch_meta_chm(_make_roi(), "2", progress)

        assert ds["chm"].dims == ("y", "x")

    @patch("griddle.handlers.chm.RasterConnection")
    @patch("griddle.handlers.chm._fs")
    @pytest.mark.parametrize(
        "version,expected_s3_base",
        [
            ("1", "s3://dataforgood-fb-data/forests/v1/alsgedi_global_v6_float/chm/"),
            (
                "2",
                "s3://dataforgood-fb-data/forests/v2/global/dinov3_global_chm_v2_ml3/chm/",
            ),
        ],
    )
    def test_s3_url_constructed_from_tile(
        self, mock_fs, mock_raster_cls, version, expected_s3_base
    ):
        """Correct S3 URL is constructed from the tile name for each version."""
        chm_values = np.array([[10.5]], dtype=np.float32)
        mock_raster_cls.return_value = _make_mock_raster(chm_values)
        mock_fs.cat.return_value = _make_tile_index_bytes()
        progress = MagicMock()

        fetch_meta_chm(_make_roi(), version, progress)

        url = mock_raster_cls.call_args[0][0]
        assert expected_s3_base in url
        assert "test_tile_001.tif" in url

    @patch("griddle.handlers.chm.RasterConnection")
    @patch("griddle.handlers.chm._fs")
    def test_aws_no_sign_request_scoped(self, mock_fs, mock_raster_cls):
        """AWS_NO_SIGN_REQUEST is set during S3 access and restored after."""
        chm_values = np.array([[10.5]], dtype=np.float32)
        mock_raster_cls.return_value = _make_mock_raster(chm_values)
        mock_fs.cat.return_value = _make_tile_index_bytes()
        progress = MagicMock()

        os.environ.pop("AWS_NO_SIGN_REQUEST", None)

        fetch_meta_chm(_make_roi(), "2", progress)

        assert "AWS_NO_SIGN_REQUEST" not in os.environ

    @patch("griddle.handlers.chm.RasterConnection")
    @patch("griddle.handlers.chm._fs")
    def test_progress_called(self, mock_fs, mock_raster_cls):
        """Progress callback is invoked during processing."""
        chm_values = np.array([[10.5]], dtype=np.float32)
        mock_raster_cls.return_value = _make_mock_raster(chm_values)
        mock_fs.cat.return_value = _make_tile_index_bytes()
        progress = MagicMock()

        fetch_meta_chm(_make_roi(), "2", progress)

        assert progress.call_count >= 2

    @patch("griddle.handlers.chm._fs")
    def test_no_intersecting_tiles_raises(self, mock_fs):
        """Raises ProcessingError(COVERAGE_ERROR) when no tiles intersect the ROI."""
        df = pd.DataFrame(
            {
                "tile": pd.Series([], dtype=str),
                "bbox_xmin": pd.Series([], dtype=float),
                "bbox_ymin": pd.Series([], dtype=float),
                "bbox_xmax": pd.Series([], dtype=float),
                "bbox_ymax": pd.Series([], dtype=float),
            }
        )
        buf = io.BytesIO()
        df.to_parquet(buf, index=False)
        mock_fs.cat.return_value = buf.getvalue()
        progress = MagicMock()

        with pytest.raises(ProcessingError) as exc_info:
            fetch_meta_chm(_make_roi(), "2", progress)

        assert exc_info.value.code == "COVERAGE_ERROR"

    @patch("griddle.handlers.chm._fs")
    def test_index_fetch_failure_raises(self, mock_fs):
        """Raises ProcessingError(INDEX_FETCH_FAILED) when the parquet index cannot be loaded."""
        mock_fs.cat.side_effect = Exception("Network error")
        progress = MagicMock()

        with pytest.raises(ProcessingError) as exc_info:
            fetch_meta_chm(_make_roi(), "2", progress)

        assert exc_info.value.code == "INDEX_FETCH_FAILED"

    @patch("griddle.handlers.chm.merge_arrays")
    @patch("griddle.handlers.chm.RasterConnection")
    @patch("griddle.handlers.chm._fs")
    def test_multiple_tiles_merged(self, mock_fs, mock_raster_cls, mock_merge):
        """Multiple intersecting tiles are fetched and merged."""
        chm_values = np.array([[10.5]], dtype=np.float32)
        mock_raster_cls.return_value = _make_mock_raster(chm_values)

        df = pd.DataFrame(
            {
                "tile": ["tile_a", "tile_b"],
                "bbox_xmin": [-180.0, -180.0],
                "bbox_ymin": [-90.0, -90.0],
                "bbox_xmax": [180.0, 180.0],
                "bbox_ymax": [90.0, 90.0],
            }
        )
        buf = io.BytesIO()
        df.to_parquet(buf, index=False)
        mock_fs.cat.return_value = buf.getvalue()

        merged_da = _make_mock_raster(chm_values).extract_window.return_value
        merged_da = merged_da.squeeze("band", drop=True)
        mock_merge.return_value = merged_da
        progress = MagicMock()

        ds, tile_metadata = fetch_meta_chm(_make_roi(), "2", progress)

        assert mock_raster_cls.call_count == 2
        mock_merge.assert_called_once()
        assert isinstance(ds, xr.Dataset)
        assert tile_metadata["tile_count"] == 2

    @patch("griddle.handlers.chm.RasterConnection")
    @patch("griddle.handlers.chm._fs")
    @pytest.mark.parametrize(
        "version,expected_index",
        [
            ("1", "Meta2024_chm"),
            ("2", "Meta_chmv2"),
        ],
    )
    def test_tile_index_path_uses_version(
        self, mock_fs, mock_raster_cls, version, expected_index
    ):
        """Tile index path includes the correct version-specific name."""
        chm_values = np.array([[10.5]], dtype=np.float32)
        mock_raster_cls.return_value = _make_mock_raster(chm_values)
        mock_fs.cat.return_value = _make_tile_index_bytes()
        progress = MagicMock()

        fetch_meta_chm(_make_roi(), version, progress)

        path = mock_fs.cat.call_args[0][0]
        assert expected_index in path
        assert path.endswith("_optimized.parquet")

    @patch("griddle.handlers.chm.RasterConnection")
    @patch("griddle.handlers.chm._fs")
    def test_tile_metadata_native_crs(self, mock_fs, mock_raster_cls):
        """Tile metadata includes the native CRS from the mosaic."""
        chm_values = np.array([[10.5]], dtype=np.float32)
        mock_raster_cls.return_value = _make_mock_raster(chm_values, crs="EPSG:32611")
        mock_fs.cat.return_value = _make_tile_index_bytes()
        progress = MagicMock()

        _, tile_metadata = fetch_meta_chm(_make_roi(), "2", progress)

        assert tile_metadata["native_crs"] is not None
        assert "32611" in tile_metadata["native_crs"]


class TestFetchNaipChm:
    """Tests for fetch_naip_chm with chm band."""

    @patch("griddle.handlers.chm.RasterConnection")
    @patch("griddle.handlers.chm._fs")
    def test_returns_dataset_and_tile_metadata(self, mock_fs, mock_raster_cls):
        """fetch_naip_chm returns a (Dataset, TileMetadata) tuple."""
        raw_values = np.array([[1050, 2030], [1520, 1870]], dtype=np.uint16)
        mock_raster_cls.return_value = _make_mock_raster(raw_values)
        mock_fs.cat.return_value = _make_naip_tile_index_bytes()
        progress = MagicMock()

        ds, tile_metadata = fetch_naip_chm(_make_roi(), "2020", progress)

        assert isinstance(ds, xr.Dataset)
        assert tile_metadata["tile_count"] == 1
        assert len(tile_metadata["tiles"]) == 1

    @patch("griddle.handlers.chm.RasterConnection")
    @patch("griddle.handlers.chm._fs")
    def test_has_chm_variable(self, mock_fs, mock_raster_cls):
        """Dataset contains a 'chm' variable."""
        raw_values = np.array([[1050, 2030]], dtype=np.uint16)
        mock_raster_cls.return_value = _make_mock_raster(raw_values)
        mock_fs.cat.return_value = _make_naip_tile_index_bytes()
        progress = MagicMock()

        ds, _ = fetch_naip_chm(_make_roi(), "2020", progress)

        assert "chm" in ds.data_vars

    @patch("griddle.handlers.chm.RasterConnection")
    @patch("griddle.handlers.chm._fs")
    def test_chm_values_scaled(self, mock_fs, mock_raster_cls):
        """CHM pixel values are correctly divided by the scale factor (100)."""
        raw_values = np.array([[1050, 2030], [1520, 1870]], dtype=np.uint16)
        expected_values = np.array([[10.5, 20.3], [15.2, 18.7]], dtype=np.float32)

        mock_raster_cls.return_value = _make_mock_raster(raw_values)
        mock_fs.cat.return_value = _make_naip_tile_index_bytes()
        progress = MagicMock()

        ds, _ = fetch_naip_chm(_make_roi(), "2020", progress)

        np.testing.assert_array_almost_equal(ds["chm"].values, expected_values)

    @patch("griddle.handlers.chm.RasterConnection")
    @patch("griddle.handlers.chm._fs")
    def test_crs_preserved(self, mock_fs, mock_raster_cls):
        """CRS is preserved in the output dataset."""
        raw_values = np.array([[1050, 2030]], dtype=np.uint16)
        mock_raster_cls.return_value = _make_mock_raster(raw_values, crs="EPSG:32611")
        mock_fs.cat.return_value = _make_naip_tile_index_bytes()
        progress = MagicMock()

        ds, _ = fetch_naip_chm(_make_roi(), "2020", progress)

        assert ds.rio.crs == CRS.from_epsg(32611)

    @patch("griddle.handlers.chm.RasterConnection")
    @patch("griddle.handlers.chm._fs")
    def test_dims_are_y_x(self, mock_fs, mock_raster_cls):
        """CHM variable has (y, x) dims."""
        raw_values = np.array([[1050, 2030]], dtype=np.uint16)
        mock_raster_cls.return_value = _make_mock_raster(raw_values)
        mock_fs.cat.return_value = _make_naip_tile_index_bytes()
        progress = MagicMock()

        ds, _ = fetch_naip_chm(_make_roi(), "2020", progress)

        assert ds["chm"].dims == ("y", "x")

    @patch("griddle.handlers.chm.RasterConnection")
    @patch("griddle.handlers.chm._fs")
    def test_http_url_passed_to_raster_connection(self, mock_fs, mock_raster_cls):
        """Correct HTTP URL is passed directly to RasterConnection."""
        raw_values = np.array([[1050]], dtype=np.uint16)
        mock_raster_cls.return_value = _make_mock_raster(raw_values)
        mock_fs.cat.return_value = _make_naip_tile_index_bytes()
        progress = MagicMock()

        fetch_naip_chm(_make_roi(), "2020", progress)

        url = mock_raster_cls.call_args[0][0]
        assert url == "http://fake-ntsg-server.com/tile_001.tif"

    @patch("griddle.handlers.chm.RasterConnection")
    @patch("griddle.handlers.chm._fs")
    def test_progress_called(self, mock_fs, mock_raster_cls):
        """Progress callback is invoked during processing."""
        raw_values = np.array([[1050]], dtype=np.uint16)
        mock_raster_cls.return_value = _make_mock_raster(raw_values)
        mock_fs.cat.return_value = _make_naip_tile_index_bytes()
        progress = MagicMock()

        fetch_naip_chm(_make_roi(), "2020", progress)

        assert progress.call_count >= 2

    @patch("griddle.handlers.chm._fs")
    def test_no_intersecting_tiles_raises(self, mock_fs):
        """Raises ProcessingError(COVERAGE_ERROR) when no tiles intersect the ROI."""
        df = pd.DataFrame(
            {
                "chm_url": pd.Series([], dtype=str),
                "scale_factor": pd.Series([], dtype=float),
                "bbox_xmin": pd.Series([], dtype=float),
                "bbox_ymin": pd.Series([], dtype=float),
                "bbox_xmax": pd.Series([], dtype=float),
                "bbox_ymax": pd.Series([], dtype=float),
            }
        )
        buf = io.BytesIO()
        df.to_parquet(buf, index=False)
        mock_fs.cat.return_value = buf.getvalue()
        progress = MagicMock()

        with pytest.raises(ProcessingError) as exc_info:
            fetch_naip_chm(_make_roi(), "2020", progress)

        assert exc_info.value.code == "COVERAGE_ERROR"

    @patch("griddle.handlers.chm._fs")
    def test_index_fetch_failure_raises(self, mock_fs):
        """Raises ProcessingError(INDEX_FETCH_FAILED) when the parquet index cannot be loaded."""
        mock_fs.cat.side_effect = Exception("Network error")
        progress = MagicMock()

        with pytest.raises(ProcessingError) as exc_info:
            fetch_naip_chm(_make_roi(), "2020", progress)

        assert exc_info.value.code == "INDEX_FETCH_FAILED"

    @patch("griddle.handlers.chm.merge_arrays")
    @patch("griddle.handlers.chm.RasterConnection")
    @patch("griddle.handlers.chm._fs")
    def test_multiple_tiles_merged(self, mock_fs, mock_raster_cls, mock_merge):
        """Multiple intersecting tiles are fetched and merged."""
        raw_values = np.array([[1050]], dtype=np.uint16)
        mock_raster_cls.return_value = _make_mock_raster(raw_values)

        df = pd.DataFrame(
            {
                "chm_url": ["http://fake.com/a.tif", "http://fake.com/b.tif"],
                "scale_factor": [100.0, 100.0],
                "bbox_xmin": [-180.0, -180.0],
                "bbox_ymin": [-90.0, -90.0],
                "bbox_xmax": [180.0, 180.0],
                "bbox_ymax": [90.0, 90.0],
            }
        )
        buf = io.BytesIO()
        df.to_parquet(buf, index=False)
        mock_fs.cat.return_value = buf.getvalue()

        merged_da = _make_mock_raster(raw_values).extract_window.return_value
        merged_da = merged_da.squeeze("band", drop=True)
        merged_da = merged_da / 100.0
        mock_merge.return_value = merged_da

        progress = MagicMock()

        ds, tile_metadata = fetch_naip_chm(_make_roi(), "2020", progress)

        assert mock_raster_cls.call_count == 2
        mock_merge.assert_called_once()
        assert isinstance(ds, xr.Dataset)
        assert tile_metadata["tile_count"] == 2


class TestOptimizedIndexCorrectness:
    """Verify that bbox queries on optimized indexes return a superset of
    the source GeoParquet intersects() results.

    These tests use small synthetic indexes to validate the filtering logic
    without hitting GCS.
    """

    @staticmethod
    def _source_query(gdf: gpd.GeoDataFrame, roi_4326: gpd.GeoDataFrame) -> set[str]:
        """Query using the original GeoParquet approach: bbox pushdown + intersects."""
        bounds = tuple(roi_4326.total_bounds)
        # Simulate gpd.read_parquet bbox pre-filter
        xmin, ymin, xmax, ymax = bounds
        pre = gdf.cx[xmin:xmax, ymin:ymax]
        # Then exact intersects
        return set(pre[pre.intersects(roi_4326.union_all())].index)

    @staticmethod
    def _optimized_query(df: pd.DataFrame, roi_4326: gpd.GeoDataFrame) -> set:
        """Query using the optimized flat-bbox approach."""
        xmin_q, ymin_q, xmax_q, ymax_q = roi_4326.total_bounds
        mask = (
            (df["bbox_xmax"] >= xmin_q)
            & (df["bbox_xmin"] <= xmax_q)
            & (df["bbox_ymax"] >= ymin_q)
            & (df["bbox_ymin"] <= ymax_q)
        )
        return set(df[mask].index)

    @staticmethod
    def _make_grid_index() -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
        """Create a 4x4 grid of tiles spanning [-2, 2] x [-2, 2]."""
        tiles = []
        for i in range(4):
            for j in range(4):
                x0, y0 = -2 + i, -2 + j
                tiles.append(
                    {
                        "tile": f"tile_{i}_{j}",
                        "geometry": box(x0, y0, x0 + 1, y0 + 1),
                    }
                )
        gdf = gpd.GeoDataFrame(tiles, crs="EPSG:4326")
        # Optimized version with flat bbox columns
        bounds = gdf.geometry.bounds
        df = pd.DataFrame(
            {
                "tile": gdf["tile"],
                "bbox_xmin": bounds["minx"].values,
                "bbox_ymin": bounds["miny"].values,
                "bbox_xmax": bounds["maxx"].values,
                "bbox_ymax": bounds["maxy"].values,
            }
        )
        return gdf, df

    def test_single_tile_hit(self):
        """ROI fully inside one tile — both methods return it."""
        gdf, df = self._make_grid_index()
        roi = gpd.GeoDataFrame(geometry=[box(0.2, 0.2, 0.8, 0.8)], crs="EPSG:4326")

        source = self._source_query(gdf, roi)
        optimized = self._optimized_query(df, roi)

        assert source
        assert source <= optimized

    def test_multi_tile_hit(self):
        """ROI spanning tile boundaries — optimized is a superset."""
        gdf, df = self._make_grid_index()
        roi = gpd.GeoDataFrame(geometry=[box(-0.5, -0.5, 0.5, 0.5)], crs="EPSG:4326")

        source = self._source_query(gdf, roi)
        optimized = self._optimized_query(df, roi)

        assert len(source) >= 4
        assert source <= optimized

    def test_no_hits(self):
        """ROI outside all tiles — both methods return empty."""
        gdf, df = self._make_grid_index()
        roi = gpd.GeoDataFrame(geometry=[box(10, 10, 11, 11)], crs="EPSG:4326")

        source = self._source_query(gdf, roi)
        optimized = self._optimized_query(df, roi)

        assert source == set()
        assert optimized == set()

    def test_edge_touch(self):
        """ROI touching tile edges — optimized is a superset."""
        gdf, df = self._make_grid_index()
        roi = gpd.GeoDataFrame(geometry=[box(0, 0, 0, 0)], crs="EPSG:4326")

        source = self._source_query(gdf, roi)
        optimized = self._optimized_query(df, roi)

        assert source <= optimized

    def test_full_coverage(self):
        """ROI covering all tiles — both return all."""
        gdf, df = self._make_grid_index()
        roi = gpd.GeoDataFrame(geometry=[box(-3, -3, 3, 3)], crs="EPSG:4326")

        source = self._source_query(gdf, roi)
        optimized = self._optimized_query(df, roi)

        assert source == optimized
        assert len(source) == 16
