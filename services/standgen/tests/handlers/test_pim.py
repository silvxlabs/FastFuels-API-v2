"""Tests for standgen PIM handler."""

import numpy as np
import pandas as pd
import pytest
import rioxarray  # noqa: F401 — registers .rio accessor
import xarray as xr
from fastfuels_core.allometry.crown_ratio import estimate_crown_base_height
from standgen.handlers import pim
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
            sample_tree_table, np.array([101, 102]), "2022", pacific_northwest=False
        )
        assert set(result["PLOT_ID"].unique()) == {101, 102}
        assert 103 not in result["PLOT_ID"].values

    def test_unit_conversion_dia(self, sample_tree_table):
        result = filter_and_convert_tree_table(
            sample_tree_table, np.array([101]), "2022", pacific_northwest=False
        )
        # 12 inches = 30.48 cm
        assert pytest.approx(result["DIA"].iloc[0], rel=0.01) == 30.48

    def test_unit_conversion_ht(self, sample_tree_table):
        result = filter_and_convert_tree_table(
            sample_tree_table, np.array([101]), "2022", pacific_northwest=False
        )
        # 60 feet = 18.288 m
        assert pytest.approx(result["HT"].iloc[0], rel=0.01) == 18.288

    def test_unit_conversion_cr(self, sample_tree_table):
        result = filter_and_convert_tree_table(
            sample_tree_table, np.array([101]), "2022", pacific_northwest=False
        )
        # 40% → 0.4
        assert pytest.approx(result["CR"].iloc[0]) == 0.4

    def test_unit_conversion_tpa(self, sample_tree_table):
        result = filter_and_convert_tree_table(
            sample_tree_table, np.array([101]), "2022", pacific_northwest=False
        )
        # TPA_UNADJ should be renamed to TPA and converted from /acre to /m²
        assert "TPA" in result.columns
        assert "TPA_UNADJ" not in result.columns
        # 10 trees/acre ≈ 0.00247 trees/m²
        assert result["TPA"].iloc[0] < 1.0  # sanity check: much less than 1

    def test_has_standard_columns(self, sample_tree_table):
        result = filter_and_convert_tree_table(
            sample_tree_table, np.array([101]), "2022", pacific_northwest=False
        )
        for col in ["PLOT_ID", "SPCD", "STATUSCD", "DIA", "HT", "CR", "TPA"]:
            assert col in result.columns

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
        result = filter_and_convert_tree_table(
            df, np.array([101]), "2016", pacific_northwest=False
        )
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
        result = filter_and_convert_tree_table(
            df, np.array([101]), "2014", pacific_northwest=False
        )
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
        result = filter_and_convert_tree_table(
            df, np.array([101]), "2020", pacific_northwest=False
        )
        assert "PLOT_ID" in result.columns
        assert len(result) == 2
        assert set(result["PLOT_ID"].unique()) == {101}

    def test_crown_base_height_column(self, sample_tree_table):
        result = filter_and_convert_tree_table(
            sample_tree_table, np.array([101, 102]), "2022", pacific_northwest=False
        )
        assert "CBH" in result.columns
        assert result["CBH"].notna().all()
        assert (result["CBH"] >= 0).all()
        assert (result["CBH"] <= result["HT"]).all()

    def test_dead_tree_crown_base_is_height(self):
        """A dead tree's missing CR becomes 0, so its crown base is its height."""
        df = pd.DataFrame(
            {
                "TM_ID": [101],
                "SPCD": [202],
                "STATUSCD": [2],
                "DIA": [12.0],
                "HT": [60.0],
                "CR": [np.nan],
                "TPA_UNADJ": [10.0],
            }
        )
        result = filter_and_convert_tree_table(
            df, np.array([101]), "2022", pacific_northwest=False
        )
        assert result["CR"].iloc[0] == 0
        assert result["CBH"].iloc[0] == pytest.approx(result["HT"].iloc[0])

    def test_crown_ratio_clipped(self):
        df = pd.DataFrame(
            {
                "TM_ID": [101, 101],
                "SPCD": [202, 202],
                "STATUSCD": [1, 1],
                "DIA": [12.0, 12.0],
                "HT": [60.0, 60.0],
                "CR": [120.0, 100.0],
                "TPA_UNADJ": [10.0, 10.0],
            }
        )
        result = filter_and_convert_tree_table(
            df, np.array([101]), "2022", pacific_northwest=False
        )
        assert result["CR"].tolist() == [1.0, 1.0]
        assert result["CBH"].iloc[0] == pytest.approx(result["CBH"].iloc[1])

    def test_published_model_matches_core(self, sample_tree_table):
        result = filter_and_convert_tree_table(
            sample_tree_table,
            np.array([101, 102]),
            "2022",
            pacific_northwest=False,
            model="monleon_2004_eq1",
        )
        expected = estimate_crown_base_height(
            result["SPCD"].to_numpy(),
            result["CR"].to_numpy(),
            result["DIA"].to_numpy(),
            result["HT"].to_numpy(),
            equation="monleon_2004_eq1",
        )
        np.testing.assert_allclose(result["CBH"].to_numpy(), expected, equal_nan=True)


class _Stop(Exception):
    pass


class TestCrownBaseHeightModel:
    def test_handle_pim_passes_source_model(self, monkeypatch, sample_dataset):
        class Snapshot:
            def to_dict(self):
                return {"source": {"product": "treemap", "version": "2022"}}

        monkeypatch.setattr(pim, "get_document", lambda *a: (None, Snapshot()))
        monkeypatch.setattr(pim, "load_grid", lambda *a: sample_dataset)
        captured = {}

        def fake_expand(*a, **kw):
            captured.update(kw)
            return {}

        monkeypatch.setattr(pim, "expand_plots", fake_expand)
        source = {
            "source_pim_grid_id": "pim-1",
            "seed": 1,
            "point_process": "inhomogeneous_poisson",
            "crown_base_height_model": {"name": "westfall_2019"},
        }
        pim.handle_pim({"id": "inv-1"}, source, None, lambda *a, **k: None)
        assert captured["crown_base_model"] == "westfall_2019"

    def test_handle_pim_defaults_to_refit(self, monkeypatch, sample_dataset):
        class Snapshot:
            def to_dict(self):
                return {"source": {}}

        monkeypatch.setattr(pim, "get_document", lambda *a: (None, Snapshot()))
        monkeypatch.setattr(pim, "load_grid", lambda *a: sample_dataset)
        captured = {}
        monkeypatch.setattr(pim, "expand_plots", lambda *a, **kw: captured.update(kw))
        source = {
            "source_pim_grid_id": "pim-1",
            "seed": 1,
            "point_process": "inhomogeneous_poisson",
        }
        pim.handle_pim({"id": "inv-1"}, source, None, lambda *a, **k: None)
        assert captured["crown_base_model"] == "fia_national_logit_2026"

    def test_expand_plots_passes_model_to_tree_table(self, monkeypatch, sample_dataset):
        captured = {}

        def fake_filter(*a, **kw):
            captured.update(kw)
            raise _Stop

        monkeypatch.setattr(pim, "load_tree_table", lambda *a: pd.DataFrame())
        monkeypatch.setattr(pim, "in_pacific_northwest", lambda gdf: False)
        monkeypatch.setattr(pim, "filter_and_convert_tree_table", fake_filter)
        plots = raster_to_plots_gdf(sample_dataset, "tm_id")
        with pytest.raises(_Stop):
            pim.expand_plots(
                {"id": "inv-1"},
                plots,
                "2022",
                None,
                lambda *a, **k: None,
                seed=1,
                point_process="inhomogeneous_poisson",
                crown_base_model="toney_reeves_2009",
            )
        assert captured["model"] == "toney_reeves_2009"
