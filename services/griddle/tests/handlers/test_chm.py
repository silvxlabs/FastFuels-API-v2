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
from griddle.handlers.chm import _query_tile_index, fetch_meta_chm, fetch_naip_chm
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


def _make_roi():
    """Create a real GeoDataFrame ROI in a projected CRS."""
    return gpd.GeoDataFrame(
        geometry=[box(300000, 4100000, 300100, 4100100)],
        crs="EPSG:32611",
    )


def _make_meta_query_result(tiles=None):
    """Return a DataFrame mimicking _query_tile_index output for Meta."""
    if tiles is None:
        tiles = ["test_tile_001"]
    return pd.DataFrame(
        {
            "tile": tiles,
            "bbox_xmin": [-180.0] * len(tiles),
            "bbox_ymin": [-90.0] * len(tiles),
            "bbox_xmax": [180.0] * len(tiles),
            "bbox_ymax": [90.0] * len(tiles),
        }
    )


def _make_naip_query_result(urls=None):
    """Return a DataFrame mimicking _query_tile_index output for NAIP."""
    if urls is None:
        urls = ["http://fake-ntsg-server.com/tile_001.tif"]
    return pd.DataFrame(
        {
            "chm_url": urls,
            "scale_factor": [100.0] * len(urls),
            "bbox_xmin": [-180.0] * len(urls),
            "bbox_ymin": [-90.0] * len(urls),
            "bbox_xmax": [180.0] * len(urls),
            "bbox_ymax": [90.0] * len(urls),
        }
    )


def _serialize_df(df: pd.DataFrame) -> bytes:
    """Serialize a DataFrame to parquet bytes."""
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    return buf.getvalue()


# -- _query_tile_index tests --------------------------------------------------


class TestQueryTileIndex:
    """Tests for the _query_tile_index helper."""

    @staticmethod
    def _make_grid_index_bytes() -> bytes:
        """Create a 4x4 grid of tiles spanning [-2, 2] x [-2, 2]."""
        rows = []
        for i in range(4):
            for j in range(4):
                x0, y0 = -2 + i, -2 + j
                rows.append(
                    {
                        "tile": f"tile_{i}_{j}",
                        "bbox_xmin": float(x0),
                        "bbox_ymin": float(y0),
                        "bbox_xmax": float(x0 + 1),
                        "bbox_ymax": float(y0 + 1),
                    }
                )
        return _serialize_df(pd.DataFrame(rows))

    @staticmethod
    def _make_roi_4326(xmin, ymin, xmax, ymax):
        return gpd.GeoDataFrame(geometry=[box(xmin, ymin, xmax, ymax)], crs="EPSG:4326")

    @patch("griddle.handlers.chm.gcsfs.GCSFileSystem")
    def test_single_tile_hit(self, mock_fs_cls):
        """ROI fully inside one tile returns that tile."""
        mock_fs_cls.return_value.cat.return_value = self._make_grid_index_bytes()
        roi = self._make_roi_4326(0.2, 0.2, 0.8, 0.8)

        result = _query_tile_index("some/path.parquet", roi)

        assert len(result) >= 1
        assert "tile_2_2" in result["tile"].values

    @patch("griddle.handlers.chm.gcsfs.GCSFileSystem")
    def test_multi_tile_hit(self, mock_fs_cls):
        """ROI spanning tile boundaries returns multiple tiles."""
        mock_fs_cls.return_value.cat.return_value = self._make_grid_index_bytes()
        roi = self._make_roi_4326(-0.5, -0.5, 0.5, 0.5)

        result = _query_tile_index("some/path.parquet", roi)

        assert len(result) >= 4

    @patch("griddle.handlers.chm.gcsfs.GCSFileSystem")
    def test_no_hits(self, mock_fs_cls):
        """ROI outside all tiles returns empty DataFrame."""
        mock_fs_cls.return_value.cat.return_value = self._make_grid_index_bytes()
        roi = self._make_roi_4326(10, 10, 11, 11)

        result = _query_tile_index("some/path.parquet", roi)

        assert result.empty

    @patch("griddle.handlers.chm.gcsfs.GCSFileSystem")
    def test_full_coverage(self, mock_fs_cls):
        """ROI covering all tiles returns all 16."""
        mock_fs_cls.return_value.cat.return_value = self._make_grid_index_bytes()
        roi = self._make_roi_4326(-3, -3, 3, 3)

        result = _query_tile_index("some/path.parquet", roi)

        assert len(result) == 16

    @patch("griddle.handlers.chm.gcsfs.GCSFileSystem")
    def test_reprojects_roi_to_4326(self, mock_fs_cls):
        """ROI in a projected CRS is reprojected to EPSG:4326 for the query."""
        mock_fs_cls.return_value.cat.return_value = _serialize_df(
            pd.DataFrame(
                {
                    "tile": ["global"],
                    "bbox_xmin": [-180.0],
                    "bbox_ymin": [-90.0],
                    "bbox_xmax": [180.0],
                    "bbox_ymax": [90.0],
                }
            )
        )
        roi = _make_roi()  # EPSG:32611

        result = _query_tile_index("some/path.parquet", roi)

        assert len(result) == 1

    @patch("griddle.handlers.chm.gcsfs.GCSFileSystem")
    def test_download_failure_propagates(self, mock_fs_cls):
        """Exceptions from fs.cat propagate to the caller."""
        mock_fs_cls.return_value.cat.side_effect = Exception("Network error")
        roi = self._make_roi_4326(0, 0, 1, 1)

        with pytest.raises(Exception, match="Network error"):
            _query_tile_index("some/path.parquet", roi)

    @patch("griddle.handlers.chm.gcsfs.GCSFileSystem")
    def test_bbox_superset_of_intersects(self, mock_fs_cls):
        """Bbox filtering returns a superset of exact geometry intersects.

        This verifies that using flat bbox columns never misses tiles that
        a geometry-based intersects() query would find.
        """
        mock_fs_cls.return_value.cat.return_value = self._make_grid_index_bytes()
        roi = self._make_roi_4326(-0.5, -0.5, 0.5, 0.5)

        # Optimized result
        optimized = set(_query_tile_index("some/path.parquet", roi)["tile"])

        # Source/reference result using geopandas intersects
        rows = []
        for i in range(4):
            for j in range(4):
                x0, y0 = -2 + i, -2 + j
                rows.append(
                    {
                        "tile": f"tile_{i}_{j}",
                        "geometry": box(x0, y0, x0 + 1, y0 + 1),
                    }
                )
        gdf = gpd.GeoDataFrame(rows, crs="EPSG:4326")
        source = set(gdf[gdf.intersects(roi.union_all())]["tile"])

        assert source <= optimized

    @patch("griddle.handlers.chm.gcsfs.GCSFileSystem")
    def test_preserves_data_columns(self, mock_fs_cls):
        """Non-bbox columns are preserved in the output."""
        mock_fs_cls.return_value.cat.return_value = _serialize_df(
            pd.DataFrame(
                {
                    "chm_url": ["http://example.com/tile.tif"],
                    "scale_factor": [100.0],
                    "bbox_xmin": [-180.0],
                    "bbox_ymin": [-90.0],
                    "bbox_xmax": [180.0],
                    "bbox_ymax": [90.0],
                }
            )
        )
        roi = self._make_roi_4326(0, 0, 1, 1)

        result = _query_tile_index("some/path.parquet", roi)

        assert result.iloc[0]["chm_url"] == "http://example.com/tile.tif"
        assert result.iloc[0]["scale_factor"] == 100.0


# -- fetch_meta_chm tests -----------------------------------------------------


class TestFetchMetaChm:
    """Tests for fetch_meta_chm with chm band."""

    @patch("griddle.handlers.chm.RasterConnection")
    @patch("griddle.handlers.chm._query_tile_index")
    def test_returns_dataset_and_tile_metadata(self, mock_query, mock_raster_cls):
        """fetch_meta_chm returns a (Dataset, TileMetadata) tuple."""
        chm_values = np.array([[10.5, 20.3], [15.2, 18.7]], dtype=np.float32)
        mock_raster_cls.return_value = _make_mock_raster(chm_values)
        mock_query.return_value = _make_meta_query_result()
        progress = MagicMock()

        ds, tile_metadata = fetch_meta_chm(_make_roi(), "2", progress)

        assert isinstance(ds, xr.Dataset)
        assert tile_metadata["tile_count"] == 1
        assert len(tile_metadata["tiles"]) == 1
        call_kwargs = mock_raster_cls.return_value.extract_window.call_args[1]
        assert "projection_padding_meters" not in call_kwargs
        assert call_kwargs["interpolation_padding_cells"] == 0

    @pytest.mark.parametrize("buffer", [0, 1, 12])
    @patch("griddle.handlers.chm.RasterConnection")
    @patch("griddle.handlers.chm._query_tile_index")
    def test_extent_buffer_cells_threaded_through(
        self, mock_query, mock_raster_cls, buffer
    ):
        """Caller-supplied extent_buffer_cells reaches extract_window unchanged."""
        chm_values = np.array([[10.5, 20.3]], dtype=np.float32)
        mock_raster_cls.return_value = _make_mock_raster(chm_values)
        mock_query.return_value = _make_meta_query_result()
        progress = MagicMock()

        fetch_meta_chm(_make_roi(), "2", progress, extent_buffer_cells=buffer)

        call_kwargs = mock_raster_cls.return_value.extract_window.call_args[1]
        assert call_kwargs["interpolation_padding_cells"] == buffer

    @patch("griddle.handlers.chm.RasterConnection")
    @patch("griddle.handlers.chm._query_tile_index")
    def test_has_chm_variable(self, mock_query, mock_raster_cls):
        """Dataset contains a 'chm' variable."""
        chm_values = np.array([[10.5, 20.3]], dtype=np.float32)
        mock_raster_cls.return_value = _make_mock_raster(chm_values)
        mock_query.return_value = _make_meta_query_result()
        progress = MagicMock()

        ds, _ = fetch_meta_chm(_make_roi(), "2", progress)

        assert "chm" in ds.data_vars

    @patch("griddle.handlers.chm.RasterConnection")
    @patch("griddle.handlers.chm._query_tile_index")
    def test_chm_values_preserved(self, mock_query, mock_raster_cls):
        """CHM pixel values are preserved in the output."""
        chm_values = np.array([[10.5, 20.3], [15.2, 18.7]], dtype=np.float32)
        mock_raster_cls.return_value = _make_mock_raster(chm_values)
        mock_query.return_value = _make_meta_query_result()
        progress = MagicMock()

        ds, _ = fetch_meta_chm(_make_roi(), "2", progress)

        np.testing.assert_array_almost_equal(ds["chm"].values, chm_values)

    @patch("griddle.handlers.chm.RasterConnection")
    @patch("griddle.handlers.chm._query_tile_index")
    def test_crs_preserved(self, mock_query, mock_raster_cls):
        """CRS is preserved in the output dataset."""
        chm_values = np.array([[10.5, 20.3], [15.2, 18.7]], dtype=np.float32)
        mock_raster_cls.return_value = _make_mock_raster(chm_values, crs="EPSG:32611")
        mock_query.return_value = _make_meta_query_result()
        progress = MagicMock()

        ds, _ = fetch_meta_chm(_make_roi(), "2", progress)

        assert ds.rio.crs == CRS.from_epsg(32611)

    @patch("griddle.handlers.chm.RasterConnection")
    @patch("griddle.handlers.chm._query_tile_index")
    def test_dims_are_y_x(self, mock_query, mock_raster_cls):
        """CHM variable has (y, x) dims."""
        chm_values = np.array([[10.5, 20.3], [15.2, 18.7]], dtype=np.float32)
        mock_raster_cls.return_value = _make_mock_raster(chm_values)
        mock_query.return_value = _make_meta_query_result()
        progress = MagicMock()

        ds, _ = fetch_meta_chm(_make_roi(), "2", progress)

        assert ds["chm"].dims == ("y", "x")

    @patch("griddle.handlers.chm.RasterConnection")
    @patch("griddle.handlers.chm._query_tile_index")
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
        self, mock_query, mock_raster_cls, version, expected_s3_base
    ):
        """Correct S3 URL is constructed from the tile name for each version."""
        chm_values = np.array([[10.5]], dtype=np.float32)
        mock_raster_cls.return_value = _make_mock_raster(chm_values)
        mock_query.return_value = _make_meta_query_result()
        progress = MagicMock()

        fetch_meta_chm(_make_roi(), version, progress)

        url = mock_raster_cls.call_args[0][0]
        assert expected_s3_base in url
        assert "test_tile_001.tif" in url

    @patch("griddle.handlers.chm.RasterConnection")
    @patch("griddle.handlers.chm._query_tile_index")
    def test_aws_no_sign_request_scoped(self, mock_query, mock_raster_cls):
        """AWS_NO_SIGN_REQUEST is set during S3 access and restored after."""
        chm_values = np.array([[10.5]], dtype=np.float32)
        mock_raster_cls.return_value = _make_mock_raster(chm_values)
        mock_query.return_value = _make_meta_query_result()
        progress = MagicMock()

        os.environ.pop("AWS_NO_SIGN_REQUEST", None)

        fetch_meta_chm(_make_roi(), "2", progress)

        assert "AWS_NO_SIGN_REQUEST" not in os.environ

    @patch("griddle.handlers.chm.RasterConnection")
    @patch("griddle.handlers.chm._query_tile_index")
    def test_progress_called(self, mock_query, mock_raster_cls):
        """Progress callback is invoked during processing."""
        chm_values = np.array([[10.5]], dtype=np.float32)
        mock_raster_cls.return_value = _make_mock_raster(chm_values)
        mock_query.return_value = _make_meta_query_result()
        progress = MagicMock()

        fetch_meta_chm(_make_roi(), "2", progress)

        assert progress.call_count >= 2

    @patch("griddle.handlers.chm._query_tile_index")
    def test_no_intersecting_tiles_raises(self, mock_query):
        """Raises ProcessingError(COVERAGE_ERROR) when no tiles intersect the ROI."""
        mock_query.return_value = _make_meta_query_result(tiles=[]).iloc[0:0]
        progress = MagicMock()

        with pytest.raises(ProcessingError) as exc_info:
            fetch_meta_chm(_make_roi(), "2", progress)

        assert exc_info.value.code == "COVERAGE_ERROR"

    @patch("griddle.handlers.chm._query_tile_index")
    def test_index_fetch_failure_raises(self, mock_query):
        """Raises ProcessingError(INDEX_FETCH_FAILED) when the index cannot be loaded."""
        mock_query.side_effect = Exception("Network error")
        progress = MagicMock()

        with pytest.raises(ProcessingError) as exc_info:
            fetch_meta_chm(_make_roi(), "2", progress)

        assert exc_info.value.code == "INDEX_FETCH_FAILED"

    @patch("griddle.handlers.chm.merge_arrays")
    @patch("griddle.handlers.chm.RasterConnection")
    @patch("griddle.handlers.chm._query_tile_index")
    def test_multiple_tiles_merged(self, mock_query, mock_raster_cls, mock_merge):
        """Multiple intersecting tiles are fetched and merged."""
        chm_values = np.array([[10.5]], dtype=np.float32)
        mock_raster_cls.return_value = _make_mock_raster(chm_values)
        mock_query.return_value = _make_meta_query_result(tiles=["tile_a", "tile_b"])

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
    @patch("griddle.handlers.chm._query_tile_index")
    @pytest.mark.parametrize(
        "version,expected_index",
        [
            ("1", "Meta2024_chm"),
            ("2", "Meta_chmv2"),
        ],
    )
    def test_tile_index_path_uses_version(
        self, mock_query, mock_raster_cls, version, expected_index
    ):
        """Tile index path includes the correct version-specific name."""
        chm_values = np.array([[10.5]], dtype=np.float32)
        mock_raster_cls.return_value = _make_mock_raster(chm_values)
        mock_query.return_value = _make_meta_query_result()
        progress = MagicMock()

        fetch_meta_chm(_make_roi(), version, progress)

        path = mock_query.call_args[0][0]
        assert expected_index in path
        assert path.endswith("_optimized.parquet")

    @patch("griddle.handlers.chm.RasterConnection")
    @patch("griddle.handlers.chm._query_tile_index")
    def test_tile_metadata_native_crs(self, mock_query, mock_raster_cls):
        """Tile metadata includes the native CRS from the mosaic."""
        chm_values = np.array([[10.5]], dtype=np.float32)
        mock_raster_cls.return_value = _make_mock_raster(chm_values, crs="EPSG:32611")
        mock_query.return_value = _make_meta_query_result()
        progress = MagicMock()

        _, tile_metadata = fetch_meta_chm(_make_roi(), "2", progress)

        assert tile_metadata["native_crs"] is not None
        assert "32611" in tile_metadata["native_crs"]


# -- fetch_naip_chm tests -----------------------------------------------------


class TestFetchNaipChm:
    """Tests for fetch_naip_chm with chm band."""

    @patch("griddle.handlers.chm.RasterConnection")
    @patch("griddle.handlers.chm._query_tile_index")
    def test_returns_dataset_and_tile_metadata(self, mock_query, mock_raster_cls):
        """fetch_naip_chm returns a (Dataset, TileMetadata) tuple."""
        raw_values = np.array([[1050, 2030], [1520, 1870]], dtype=np.uint16)
        mock_raster_cls.return_value = _make_mock_raster(raw_values)
        mock_query.return_value = _make_naip_query_result()
        progress = MagicMock()

        ds, tile_metadata = fetch_naip_chm(_make_roi(), progress)

        assert isinstance(ds, xr.Dataset)
        assert tile_metadata["tile_count"] == 1
        assert len(tile_metadata["tiles"]) == 1

    @patch("griddle.handlers.chm.RasterConnection")
    @patch("griddle.handlers.chm._query_tile_index")
    def test_has_chm_variable(self, mock_query, mock_raster_cls):
        """Dataset contains a 'chm' variable."""
        raw_values = np.array([[1050, 2030]], dtype=np.uint16)
        mock_raster_cls.return_value = _make_mock_raster(raw_values)
        mock_query.return_value = _make_naip_query_result()
        progress = MagicMock()

        ds, _ = fetch_naip_chm(_make_roi(), progress)

        assert "chm" in ds.data_vars

    @patch("griddle.handlers.chm.RasterConnection")
    @patch("griddle.handlers.chm._query_tile_index")
    def test_chm_values_scaled(self, mock_query, mock_raster_cls):
        """CHM pixel values are correctly divided by the scale factor (100)."""
        raw_values = np.array([[1050, 2030], [1520, 1870]], dtype=np.uint16)
        expected_values = np.array([[10.5, 20.3], [15.2, 18.7]], dtype=np.float32)

        mock_raster_cls.return_value = _make_mock_raster(raw_values)
        mock_query.return_value = _make_naip_query_result()
        progress = MagicMock()

        ds, _ = fetch_naip_chm(_make_roi(), progress)

        np.testing.assert_array_almost_equal(ds["chm"].values, expected_values)

    @patch("griddle.handlers.chm.RasterConnection")
    @patch("griddle.handlers.chm._query_tile_index")
    def test_crs_preserved(self, mock_query, mock_raster_cls):
        """CRS is preserved in the output dataset."""
        raw_values = np.array([[1050, 2030]], dtype=np.uint16)
        mock_raster_cls.return_value = _make_mock_raster(raw_values, crs="EPSG:32611")
        mock_query.return_value = _make_naip_query_result()
        progress = MagicMock()

        ds, _ = fetch_naip_chm(_make_roi(), progress)

        assert ds.rio.crs == CRS.from_epsg(32611)

    @patch("griddle.handlers.chm.RasterConnection")
    @patch("griddle.handlers.chm._query_tile_index")
    def test_dims_are_y_x(self, mock_query, mock_raster_cls):
        """CHM variable has (y, x) dims."""
        raw_values = np.array([[1050, 2030]], dtype=np.uint16)
        mock_raster_cls.return_value = _make_mock_raster(raw_values)
        mock_query.return_value = _make_naip_query_result()
        progress = MagicMock()

        ds, _ = fetch_naip_chm(_make_roi(), progress)

        assert ds["chm"].dims == ("y", "x")

    @patch("griddle.handlers.chm.RasterConnection")
    @patch("griddle.handlers.chm._query_tile_index")
    def test_http_url_passed_to_raster_connection(self, mock_query, mock_raster_cls):
        """Correct HTTP URL is passed directly to RasterConnection."""
        raw_values = np.array([[1050]], dtype=np.uint16)
        mock_raster_cls.return_value = _make_mock_raster(raw_values)
        mock_query.return_value = _make_naip_query_result()
        progress = MagicMock()

        fetch_naip_chm(_make_roi(), progress)

        url = mock_raster_cls.call_args[0][0]
        assert url == "http://fake-ntsg-server.com/tile_001.tif"

    @patch("griddle.handlers.chm.RasterConnection")
    @patch("griddle.handlers.chm._query_tile_index")
    def test_progress_called(self, mock_query, mock_raster_cls):
        """Progress callback is invoked during processing."""
        raw_values = np.array([[1050]], dtype=np.uint16)
        mock_raster_cls.return_value = _make_mock_raster(raw_values)
        mock_query.return_value = _make_naip_query_result()
        progress = MagicMock()

        fetch_naip_chm(_make_roi(), progress)

        assert progress.call_count >= 2

    @patch("griddle.handlers.chm._query_tile_index")
    def test_no_intersecting_tiles_raises(self, mock_query):
        """Raises ProcessingError(COVERAGE_ERROR) when no tiles intersect the ROI."""
        mock_query.return_value = _make_naip_query_result(urls=[]).iloc[0:0]
        progress = MagicMock()

        with pytest.raises(ProcessingError) as exc_info:
            fetch_naip_chm(_make_roi(), progress)

        assert exc_info.value.code == "COVERAGE_ERROR"

    @patch("griddle.handlers.chm._query_tile_index")
    def test_index_fetch_failure_raises(self, mock_query):
        """Raises ProcessingError(INDEX_FETCH_FAILED) when the index cannot be loaded."""
        mock_query.side_effect = Exception("Network error")
        progress = MagicMock()

        with pytest.raises(ProcessingError) as exc_info:
            fetch_naip_chm(_make_roi(), progress)

        assert exc_info.value.code == "INDEX_FETCH_FAILED"

    @patch("griddle.handlers.chm.merge_arrays")
    @patch("griddle.handlers.chm.RasterConnection")
    @patch("griddle.handlers.chm._query_tile_index")
    def test_multiple_tiles_merged(self, mock_query, mock_raster_cls, mock_merge):
        """Multiple intersecting tiles are fetched and merged."""
        raw_values = np.array([[1050]], dtype=np.uint16)
        mock_raster_cls.return_value = _make_mock_raster(raw_values)
        mock_query.return_value = _make_naip_query_result(
            urls=["http://fake.com/a.tif", "http://fake.com/b.tif"]
        )

        merged_da = _make_mock_raster(raw_values).extract_window.return_value
        merged_da = merged_da.squeeze("band", drop=True)
        merged_da = merged_da / 100.0
        mock_merge.return_value = merged_da

        progress = MagicMock()

        ds, tile_metadata = fetch_naip_chm(_make_roi(), progress)

        assert mock_raster_cls.call_count == 2
        mock_merge.assert_called_once()
        assert isinstance(ds, xr.Dataset)
        assert tile_metadata["tile_count"] == 2
