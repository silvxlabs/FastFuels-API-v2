"""
Tests for PIM handler.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import rioxarray  # noqa: F401
import xarray as xr
from griddle.handlers.pim import TREEMAP_COLUMNS, fetch_treemap
from rasterio.crs import CRS
from rasterio.transform import from_bounds


def _make_mock_raster(tm_id_values, crs="EPSG:32611"):
    """Create a mock RasterConnection that returns a DataArray of TM_ID values.

    Args:
        tm_id_values: 2D numpy array of TM_ID pixel values
        crs: CRS string

    Returns:
        Mock RasterConnection instance
    """
    height, width = tm_id_values.shape
    transform = from_bounds(
        300000, 4100000, 300000 + width * 30, 4100000 + height * 30, width, height
    )

    da = xr.DataArray(
        tm_id_values[np.newaxis, :, :],  # Add band dim for squeeze
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
    mock_raster.raster_resolution = 30
    mock_raster.extract_window.return_value = da
    return mock_raster


def _make_tree_table(tm_ids, plt_cns, tm_col="TM_ID", cn_col="PLT_CN"):
    """Create a tree table DataFrame matching parquet schema.

    Multiple rows per TM_ID (simulating multiple trees per plot),
    but unique TM_ID -> PLT_CN mapping.
    """
    rows = []
    for tm_id, plt_cn in zip(tm_ids, plt_cns):
        # Add 3 rows per plot (simulating multiple trees)
        for _ in range(3):
            rows.append({tm_col: tm_id, cn_col: plt_cn})
    return pd.DataFrame(rows)


class TestFetchTreemapTmIdOnly:
    """Tests for fetch_treemap with tm_id band only."""

    @patch("griddle.handlers.pim.RasterConnection")
    def test_returns_dataset(self, mock_raster_cls):
        """fetch_treemap returns an xr.Dataset."""
        tm_ids = np.array([[1, 2], [3, 4]], dtype=np.int16)
        mock_raster_cls.return_value = _make_mock_raster(tm_ids)
        roi = MagicMock()
        progress = MagicMock()

        result = fetch_treemap(roi, "2022", ["tm_id"], progress)

        assert isinstance(result, xr.Dataset)

    @patch("griddle.handlers.pim.RasterConnection")
    def test_has_tm_id_variable(self, mock_raster_cls):
        """Dataset contains a 'tm_id' variable."""
        tm_ids = np.array([[1, 2], [3, 4]], dtype=np.int16)
        mock_raster_cls.return_value = _make_mock_raster(tm_ids)
        roi = MagicMock()
        progress = MagicMock()

        result = fetch_treemap(roi, "2022", ["tm_id"], progress)

        assert "tm_id" in result.data_vars

    @patch("griddle.handlers.pim.RasterConnection")
    def test_no_plt_cn_when_not_requested(self, mock_raster_cls):
        """Dataset does not contain 'plt_cn' when not requested."""
        tm_ids = np.array([[1, 2], [3, 4]], dtype=np.int16)
        mock_raster_cls.return_value = _make_mock_raster(tm_ids)
        roi = MagicMock()
        progress = MagicMock()

        result = fetch_treemap(roi, "2022", ["tm_id"], progress)

        assert "plt_cn" not in result.data_vars

    @patch("griddle.handlers.pim.RasterConnection")
    def test_tm_id_values_preserved(self, mock_raster_cls):
        """TM_ID pixel values are preserved in the output."""
        tm_ids = np.array([[100, 200], [300, 400]], dtype=np.int16)
        mock_raster_cls.return_value = _make_mock_raster(tm_ids)
        roi = MagicMock()
        progress = MagicMock()

        result = fetch_treemap(roi, "2022", ["tm_id"], progress)

        np.testing.assert_array_equal(result["tm_id"].values, tm_ids)

    @patch("griddle.handlers.pim.RasterConnection")
    def test_crs_preserved(self, mock_raster_cls):
        """CRS is preserved in the output dataset."""
        tm_ids = np.array([[1, 2], [3, 4]], dtype=np.int16)
        mock_raster_cls.return_value = _make_mock_raster(tm_ids, crs="EPSG:32611")
        roi = MagicMock()
        progress = MagicMock()

        result = fetch_treemap(roi, "2022", ["tm_id"], progress)

        assert result.rio.crs == CRS.from_epsg(32611)

    @patch("griddle.handlers.pim.RasterConnection")
    def test_raster_url_uses_version(self, mock_raster_cls):
        """RasterConnection is called with version-specific URL."""
        tm_ids = np.array([[1, 2]], dtype=np.int16)
        mock_raster_cls.return_value = _make_mock_raster(tm_ids)
        roi = MagicMock()
        progress = MagicMock()

        fetch_treemap(roi, "2016", ["tm_id"], progress)

        url = mock_raster_cls.call_args[0][0]
        assert "TreeMap2016.tif" in url

    @patch("griddle.handlers.pim.RasterConnection")
    def test_extract_window_padding(self, mock_raster_cls):
        """extract_window is called with correct padding parameters."""
        tm_ids = np.array([[1, 2]], dtype=np.int16)
        mock_raster = _make_mock_raster(tm_ids)
        mock_raster_cls.return_value = mock_raster
        roi = MagicMock()
        progress = MagicMock()

        fetch_treemap(roi, "2022", ["tm_id"], progress)

        call_kwargs = mock_raster.extract_window.call_args[1]
        assert call_kwargs["interpolation_padding_cells"] == 8


class TestFetchTreemapPltCn:
    """Tests for fetch_treemap with plt_cn band."""

    @patch("griddle.handlers.pim.pd.read_parquet")
    @patch("griddle.handlers.pim.RasterConnection")
    def test_has_plt_cn_variable(self, mock_raster_cls, mock_read_parquet):
        """Dataset contains 'plt_cn' when requested."""
        tm_ids = np.array([[1, 2], [3, 4]], dtype=np.int16)
        mock_raster_cls.return_value = _make_mock_raster(tm_ids)
        mock_read_parquet.return_value = _make_tree_table(
            [1, 2, 3, 4], [1001, 1002, 1003, 1004]
        )
        roi = MagicMock()
        progress = MagicMock()

        result = fetch_treemap(roi, "2022", ["tm_id", "plt_cn"], progress)

        assert "plt_cn" in result.data_vars

    @patch("griddle.handlers.pim.pd.read_parquet")
    @patch("griddle.handlers.pim.RasterConnection")
    def test_plt_cn_values_correct(self, mock_raster_cls, mock_read_parquet):
        """PLT_CN values are correctly mapped from TM_ID."""
        tm_ids = np.array([[1, 2], [3, 4]], dtype=np.int16)
        mock_raster_cls.return_value = _make_mock_raster(tm_ids)
        mock_read_parquet.return_value = _make_tree_table(
            [1, 2, 3, 4],
            [2232677010690, 2232677010691, 2232677010692, 2232677010693],
        )
        roi = MagicMock()
        progress = MagicMock()

        result = fetch_treemap(roi, "2022", ["tm_id", "plt_cn"], progress)

        expected = np.array(
            [[2232677010690, 2232677010691], [2232677010692, 2232677010693]],
            dtype=np.int64,
        )
        np.testing.assert_array_equal(result["plt_cn"].values, expected)

    @patch("griddle.handlers.pim.pd.read_parquet")
    @patch("griddle.handlers.pim.RasterConnection")
    def test_plt_cn_shape_matches_tm_id(self, mock_raster_cls, mock_read_parquet):
        """PLT_CN has same shape as TM_ID."""
        tm_ids = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int16)
        mock_raster_cls.return_value = _make_mock_raster(tm_ids)
        mock_read_parquet.return_value = _make_tree_table(
            [1, 2, 3, 4, 5, 6], [101, 102, 103, 104, 105, 106]
        )
        roi = MagicMock()
        progress = MagicMock()

        result = fetch_treemap(roi, "2022", ["tm_id", "plt_cn"], progress)

        assert result["plt_cn"].shape == result["tm_id"].shape

    @patch("griddle.handlers.pim.pd.read_parquet")
    @patch("griddle.handlers.pim.RasterConnection")
    def test_unmapped_tm_id_gets_zero(self, mock_raster_cls, mock_read_parquet):
        """TM_ID values not in the tree table map to 0."""
        # Raster has TM_ID=99 but tree table only has 1,2
        tm_ids = np.array([[1, 99], [2, 99]], dtype=np.int16)
        mock_raster_cls.return_value = _make_mock_raster(tm_ids)
        mock_read_parquet.return_value = _make_tree_table([1, 2], [1001, 1002])
        roi = MagicMock()
        progress = MagicMock()

        result = fetch_treemap(roi, "2022", ["tm_id", "plt_cn"], progress)

        # TM_ID=99 is within lookup range (max TM_ID=2) -> out of range -> 0
        # Actually 99 > max(1,2)=2, so it gets clipped and then zeroed
        assert result["plt_cn"].values[0, 0] == 1001  # TM_ID=1
        assert result["plt_cn"].values[1, 0] == 1002  # TM_ID=2
        assert result["plt_cn"].values[0, 1] == 0  # TM_ID=99 -> 0
        assert result["plt_cn"].values[1, 1] == 0  # TM_ID=99 -> 0

    @patch("griddle.handlers.pim.pd.read_parquet")
    @patch("griddle.handlers.pim.RasterConnection")
    def test_negative_tm_id_gets_zero(self, mock_raster_cls, mock_read_parquet):
        """Negative TM_ID values (nodata) map to 0."""
        tm_ids = np.array([[1, -1], [2, 0]], dtype=np.int16)
        mock_raster_cls.return_value = _make_mock_raster(tm_ids)
        mock_read_parquet.return_value = _make_tree_table([1, 2], [1001, 1002])
        roi = MagicMock()
        progress = MagicMock()

        result = fetch_treemap(roi, "2022", ["tm_id", "plt_cn"], progress)

        assert result["plt_cn"].values[0, 1] == 0  # TM_ID=-1 -> 0
        # TM_ID=0 -> lookup[0] which is 0 (initialized to zero)
        assert result["plt_cn"].values[1, 1] == 0

    @patch("griddle.handlers.pim.pd.read_parquet")
    @patch("griddle.handlers.pim.RasterConnection")
    def test_plt_cn_only_without_tm_id(self, mock_raster_cls, mock_read_parquet):
        """Requesting only plt_cn (without tm_id) works."""
        tm_ids = np.array([[1, 2]], dtype=np.int16)
        mock_raster_cls.return_value = _make_mock_raster(tm_ids)
        mock_read_parquet.return_value = _make_tree_table([1, 2], [1001, 1002])
        roi = MagicMock()
        progress = MagicMock()

        result = fetch_treemap(roi, "2022", ["plt_cn"], progress)

        assert "plt_cn" in result.data_vars
        assert "tm_id" not in result.data_vars

    @patch("griddle.handlers.pim.pd.read_parquet")
    @patch("griddle.handlers.pim.RasterConnection")
    def test_plt_cn_crs_preserved(self, mock_raster_cls, mock_read_parquet):
        """PLT_CN DataArray preserves CRS from source raster."""
        tm_ids = np.array([[1, 2]], dtype=np.int16)
        mock_raster_cls.return_value = _make_mock_raster(tm_ids, crs="EPSG:32610")
        mock_read_parquet.return_value = _make_tree_table([1, 2], [1001, 1002])
        roi = MagicMock()
        progress = MagicMock()

        result = fetch_treemap(roi, "2022", ["plt_cn"], progress)

        assert result.rio.crs == CRS.from_epsg(32610)

    @patch("griddle.handlers.pim.pd.read_parquet")
    @patch("griddle.handlers.pim.RasterConnection")
    def test_progress_called_for_plt_cn(self, mock_raster_cls, mock_read_parquet):
        """Progress is reported when loading tree table."""
        tm_ids = np.array([[1]], dtype=np.int16)
        mock_raster_cls.return_value = _make_mock_raster(tm_ids)
        mock_read_parquet.return_value = _make_tree_table([1], [1001])
        roi = MagicMock()
        progress = MagicMock()

        fetch_treemap(roi, "2022", ["tm_id", "plt_cn"], progress)

        progress.assert_called()
        messages = [call[0][0] for call in progress.call_args_list]
        assert any("PLT_CN" in m for m in messages)


class TestTreemapVersionColumns:
    """Tests for version-specific column name mapping."""

    def test_2022_columns(self):
        """2022 version uses TM_ID and PLT_CN."""
        assert TREEMAP_COLUMNS["2022"] == ("TM_ID", "PLT_CN")

    def test_2020_columns(self):
        """2020 version uses TM_ID and PLT_CN."""
        assert TREEMAP_COLUMNS["2020"] == ("TM_ID", "PLT_CN")

    def test_2016_columns(self):
        """2016 version uses tm_id and CN."""
        assert TREEMAP_COLUMNS["2016"] == ("tm_id", "CN")

    def test_2014_columns(self):
        """2014 version uses tl_id and CN."""
        assert TREEMAP_COLUMNS["2014"] == ("tl_id", "CN")

    @patch("griddle.handlers.pim.pd.read_parquet")
    @patch("griddle.handlers.pim.RasterConnection")
    def test_2016_uses_correct_columns(self, mock_raster_cls, mock_read_parquet):
        """2016 version reads tm_id and CN columns from parquet."""
        tm_ids = np.array([[1, 2]], dtype=np.int16)
        mock_raster_cls.return_value = _make_mock_raster(tm_ids)
        mock_read_parquet.return_value = _make_tree_table(
            [1, 2], [1001, 1002], tm_col="tm_id", cn_col="CN"
        )
        roi = MagicMock()
        progress = MagicMock()

        fetch_treemap(roi, "2016", ["plt_cn"], progress)

        # Check parquet was read with correct column names
        call_kwargs = mock_read_parquet.call_args[1]
        assert call_kwargs["columns"] == ["tm_id", "CN"]

    @patch("griddle.handlers.pim.pd.read_parquet")
    @patch("griddle.handlers.pim.RasterConnection")
    def test_table_url_uses_version(self, mock_raster_cls, mock_read_parquet):
        """Tree table parquet URL uses version string."""
        tm_ids = np.array([[1]], dtype=np.int16)
        mock_raster_cls.return_value = _make_mock_raster(tm_ids)
        mock_read_parquet.return_value = _make_tree_table(
            [1], [1001], tm_col="tm_id", cn_col="CN"
        )
        roi = MagicMock()
        progress = MagicMock()

        fetch_treemap(roi, "2016", ["plt_cn"], progress)

        url = mock_read_parquet.call_args[0][0]
        assert "TreeMap2016_tree_table.parquet" in url
