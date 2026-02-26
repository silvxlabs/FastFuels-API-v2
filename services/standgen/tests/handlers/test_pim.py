"""Tests for standgen PIM handler."""

import numpy as np
import pandas as pd
import pytest
import rioxarray  # noqa: F401 — registers .rio accessor
import xarray as xr
from standgen.errors import ProcessingError
from standgen.handlers.pim import (
    filter_and_convert_tree_table,
    raster_to_plots_gdf,
)


@pytest.fixture
def sample_dataset():
    """Create a small xarray Dataset mimicking a PIM grid."""
    x = np.array([500000.0, 500030.0, 500060.0])
    y = np.array([4500000.0, 4500030.0])
    tm_id = np.array(
        [
            [101, 102, 0],
            [103, 0, 104],
        ]
    )
    da = xr.DataArray(tm_id, dims=("y", "x"), coords={"y": y, "x": x})
    da = da.rio.write_crs("EPSG:32610")
    ds = xr.Dataset({"tm_id": da})
    ds = ds.rio.write_crs("EPSG:32610")
    return ds


@pytest.fixture
def sample_tree_table():
    """Create a sample TreeMap tree table in imperial units."""
    return pd.DataFrame(
        {
            "TM_ID": [101, 101, 102, 102, 103, 103, 104, 104],
            "SPCD": [202, 122, 202, 93, 15, 202, 122, 15],
            "STATUSCD": [1, 1, 1, 2, 1, 1, 1, 1],
            "DIA": [12.0, 6.0, 18.0, 4.0, 24.0, 10.0, 8.0, 30.0],  # inches
            "HT": [60.0, 30.0, 80.0, 20.0, 100.0, 50.0, 40.0, 110.0],  # feet
            "CR": [40.0, 60.0, 35.0, 50.0, 30.0, 45.0, 55.0, 25.0],  # percent
            "TPA_UNADJ": [10.0, 50.0, 8.0, 30.0, 5.0, 15.0, 20.0, 3.0],  # per acre
        }
    )


class TestRasterToPlots:
    def test_basic_conversion(self, sample_dataset):
        gdf = raster_to_plots_gdf(sample_dataset, "tm_id")
        # All 6 pixels (3x2 grid) including zeros
        assert len(gdf) == 6
        assert "PLOT_ID" in gdf.columns
        assert gdf.crs is not None

    def test_includes_zeros_as_anchors(self, sample_dataset):
        """Zero-value cells must be kept as zero-density anchors for interpolation."""
        gdf = raster_to_plots_gdf(sample_dataset, "tm_id")
        assert 0 in gdf["PLOT_ID"].values
        assert (gdf["PLOT_ID"] == 0).sum() == 2  # 2 zero cells in fixture

    def test_correct_plot_ids(self, sample_dataset):
        gdf = raster_to_plots_gdf(sample_dataset, "tm_id")
        expected_ids = {0, 101, 102, 103, 104}
        assert set(gdf["PLOT_ID"].values) == expected_ids

    def test_geometry_is_points(self, sample_dataset):
        gdf = raster_to_plots_gdf(sample_dataset, "tm_id")
        assert all(g.geom_type == "Point" for g in gdf.geometry)

    def test_all_zeros_returns_all_cells(self):
        """All-zero raster still returns cells (as zero-density anchors)."""
        x = np.array([500000.0, 500030.0])
        y = np.array([4500000.0, 4500030.0])
        da = xr.DataArray(
            np.zeros((2, 2), dtype=int), dims=("y", "x"), coords={"y": y, "x": x}
        )
        da = da.rio.write_crs("EPSG:32610")
        ds = xr.Dataset({"tm_id": da})
        ds = ds.rio.write_crs("EPSG:32610")

        gdf = raster_to_plots_gdf(ds, "tm_id")
        assert len(gdf) == 4
        assert all(gdf["PLOT_ID"] == 0)


class TestFilterAndConvert:
    def test_filters_to_plot_ids(self, sample_tree_table):
        result = filter_and_convert_tree_table(
            sample_tree_table, np.array([101, 102]), "2022"
        )
        assert set(result["PLOT_ID"].unique()) == {101, 102}
        assert 103 not in result["PLOT_ID"].values

    def test_unit_conversion_dia(self, sample_tree_table):
        result = filter_and_convert_tree_table(
            sample_tree_table, np.array([101]), "2022"
        )
        # 12 inches = 30.48 cm
        assert pytest.approx(result["DIA"].iloc[0], rel=0.01) == 30.48

    def test_unit_conversion_ht(self, sample_tree_table):
        result = filter_and_convert_tree_table(
            sample_tree_table, np.array([101]), "2022"
        )
        # 60 feet = 18.288 m
        assert pytest.approx(result["HT"].iloc[0], rel=0.01) == 18.288

    def test_unit_conversion_cr(self, sample_tree_table):
        result = filter_and_convert_tree_table(
            sample_tree_table, np.array([101]), "2022"
        )
        # 40% → 0.4
        assert pytest.approx(result["CR"].iloc[0]) == 0.4

    def test_unit_conversion_tpa(self, sample_tree_table):
        result = filter_and_convert_tree_table(
            sample_tree_table, np.array([101]), "2022"
        )
        # TPA_UNADJ should be renamed to TPA and converted from /acre to /m²
        assert "TPA" in result.columns
        assert "TPA_UNADJ" not in result.columns
        # 10 trees/acre ≈ 0.00247 trees/m²
        assert result["TPA"].iloc[0] < 1.0  # sanity check: much less than 1

    def test_has_standard_columns(self, sample_tree_table):
        result = filter_and_convert_tree_table(
            sample_tree_table, np.array([101]), "2022"
        )
        for col in ["PLOT_ID", "SPCD", "STATUSCD", "DIA", "HT", "CR", "TPA"]:
            assert col in result.columns

    def test_unsupported_version_raises(self, sample_tree_table):
        with pytest.raises(ProcessingError) as exc_info:
            filter_and_convert_tree_table(sample_tree_table, np.array([101]), "1999")
        assert exc_info.value.code == "UNSUPPORTED_VERSION"

    def test_version_2016_column_names(self):
        df = pd.DataFrame(
            {
                "tm_id": [101, 101],
                "SPCD": [202, 122],
                "STATUSCD": [1, 1],
                "DIA": [12.0, 6.0],
                "HT": [60.0, 30.0],
                "CR": [40.0, 60.0],
                "TPA_UNADJ": [10.0, 50.0],
            }
        )
        result = filter_and_convert_tree_table(df, np.array([101]), "2016")
        assert "PLOT_ID" in result.columns
        assert len(result) == 2

    def test_version_2014_column_names(self):
        """Version 2014 uses tl_id as both tree and plot ID."""
        df = pd.DataFrame(
            {
                "tl_id": [101, 101],
                "SPCD": [202, 122],
                "STATUSCD": [1, 1],
                "DIA": [12.0, 6.0],
                "HT": [60.0, 30.0],
                "CR": [40.0, 60.0],
                "TPA_UNADJ": [10.0, 50.0],
            }
        )
        result = filter_and_convert_tree_table(df, np.array([101]), "2014")
        assert "PLOT_ID" in result.columns
        assert "TREE_ID" in result.columns
        assert len(result) == 2
        assert set(result["PLOT_ID"].unique()) == {101}

    def test_version_2020_column_names(self):
        """Version 2020 uses TM_ID (same as 2022) but confirms mapping exists."""
        df = pd.DataFrame(
            {
                "TM_ID": [101, 101],
                "SPCD": [202, 122],
                "STATUSCD": [1, 1],
                "DIA": [12.0, 6.0],
                "HT": [60.0, 30.0],
                "CR": [40.0, 60.0],
                "TPA_UNADJ": [10.0, 50.0],
            }
        )
        result = filter_and_convert_tree_table(df, np.array([101]), "2020")
        assert "PLOT_ID" in result.columns
        assert len(result) == 2
        assert set(result["PLOT_ID"].unique()) == {101}
