"""Tests for the standgen PIM-CHM fusion handler."""

import geopandas as gpd
import numpy as np
import pytest
import rioxarray  # noqa: F401 — registers .rio accessor
import xarray as xr
from shapely.geometry import Point
from standgen.handlers import pim_chm_fusion
from standgen.handlers.pim_chm_fusion import _mask_fused_plots, handle_pim_chm_fusion

from lib.errors import ProcessingError

PIM_NODATA = 4294967295  # uint32 max — the TreeMap tm_id nodata sentinel


def _fused_gdf(plot_ids):
    """Build a fused-plots GeoDataFrame like sample_plots_from_hag returns."""
    n = len(plot_ids)
    xs = np.arange(n, dtype=float)
    return gpd.GeoDataFrame(
        {"PLOT_ID": np.array(plot_ids, dtype=float)},
        geometry=[Point(x, 0.0) for x in xs],
        crs="EPSG:32611",
    )


class TestMaskFusedPlots:
    """The nodata -> 0 (zero-density anchor) mapping."""

    def test_nodata_sentinel_maps_to_zero(self):
        fused = _fused_gdf([101, PIM_NODATA, 102, PIM_NODATA])
        plots = _mask_fused_plots(fused, PIM_NODATA)
        assert plots["PLOT_ID"].tolist() == [101, 0, 102, 0]

    def test_nan_cells_map_to_zero(self):
        # Cells resampling could not fill arrive as NaN; they must also read 0.
        fused = _fused_gdf([101, np.nan, 103])
        plots = _mask_fused_plots(fused, PIM_NODATA)
        assert plots["PLOT_ID"].tolist() == [101, 0, 103]

    def test_real_plot_ids_preserved(self):
        fused = _fused_gdf([101, 102, 103])
        plots = _mask_fused_plots(fused, PIM_NODATA)
        assert plots["PLOT_ID"].tolist() == [101, 102, 103]

    def test_output_shape_matches_raster_to_plots(self):
        """Only PLOT_ID (int) + Point geometry, CRS preserved."""
        fused = _fused_gdf([101, PIM_NODATA])
        plots = _mask_fused_plots(fused, PIM_NODATA)
        assert list(plots.columns) == ["PLOT_ID", "geometry"]
        assert plots["PLOT_ID"].dtype.kind == "i"
        assert plots.crs == fused.crs
        assert all(g.geom_type == "Point" for g in plots.geometry)

    def test_all_nodata_yields_no_survivors(self):
        fused = _fused_gdf([PIM_NODATA, PIM_NODATA, PIM_NODATA])
        plots = _mask_fused_plots(fused, PIM_NODATA)
        assert (plots["PLOT_ID"] != 0).sum() == 0


def _grid_ds(band: str, nodata):
    """Minimal single-band grid Dataset with a rio CRS and nodata."""
    y = np.array([10.0, 0.0])
    x = np.array([0.0, 10.0])
    da = xr.DataArray(
        np.array([[1.0, 2.0], [3.0, 4.0]]), dims=("y", "x"), coords={"y": y, "x": x}
    )
    da = da.rio.write_crs("EPSG:32611")
    da = da.rio.write_nodata(nodata)
    return xr.Dataset({band: da})


class _Snap:
    def __init__(self, data):
        self._data = data

    def to_dict(self):
        return self._data


def _install_common_stubs(monkeypatch, fused_ids):
    """Stub Firestore + grid loads + the core fusion call for handler tests."""
    pim_doc = {"source": {"product": "treemap", "version": "2022"}}

    def fake_get_document(collection, grid_id):
        if grid_id == "pim-1":
            return None, _Snap(pim_doc)
        return None, _Snap({"source": {}})

    monkeypatch.setattr(pim_chm_fusion, "get_document", fake_get_document)

    def fake_load_grid(grid_id):
        if grid_id == "pim-1":
            return _grid_ds("tm_id", PIM_NODATA)
        return _grid_ds("chm", np.nan)

    monkeypatch.setattr(pim_chm_fusion, "load_grid", fake_load_grid)
    monkeypatch.setattr(
        pim_chm_fusion,
        "sample_plots_from_hag",
        lambda *a, **k: _fused_gdf(fused_ids),
    )


def _source():
    return {
        "name": "pim",
        "fusion": ["chm"],
        "source_pim_grid_id": "pim-1",
        "source_chm_grid_id": "chm-1",
        "seed": 42,
        "point_process": "inhomogeneous_poisson",
        "method": {"name": "reimputation", "cover_threshold": 0.1},
    }


class TestHandleFusion:
    def test_empty_after_fusion_raises_with_max_cover(self, monkeypatch):
        """All cells masked -> terminal EMPTY_AFTER_FUSION reporting max cover."""
        _install_common_stubs(monkeypatch, [PIM_NODATA, PIM_NODATA])

        cover = xr.DataArray(np.array([[0.04, 0.02], [0.0, 0.03]]))
        monkeypatch.setattr(
            pim_chm_fusion, "compute_cover_from_hag", lambda *a, **k: cover
        )
        # expand_plots must never be reached on the empty path.
        monkeypatch.setattr(
            pim_chm_fusion,
            "expand_plots",
            lambda *a, **k: pytest.fail("expand_plots called on empty fusion"),
        )

        with pytest.raises(ProcessingError) as exc:
            handle_pim_chm_fusion(
                {"id": "inv-1"}, _source(), gpd.GeoDataFrame(), lambda *a, **k: None
            )
        assert exc.value.code == "EMPTY_AFTER_FUSION"
        assert "0.040" in exc.value.message  # max observed cover

    def test_survivors_call_expand_plots(self, monkeypatch):
        """Any surviving plot hands off to the shared expand_plots tail."""
        _install_common_stubs(monkeypatch, [101, PIM_NODATA, 102])

        captured = {}

        def fake_expand(inventory, plots, version, domain_gdf, progress, **kw):
            captured["plots"] = plots
            captured["version"] = version
            captured["kw"] = kw
            return {"georeference": {}, "columns": [], "forestry_metrics": None}

        monkeypatch.setattr(pim_chm_fusion, "expand_plots", fake_expand)

        result = handle_pim_chm_fusion(
            {"id": "inv-1"}, _source(), gpd.GeoDataFrame(), lambda *a, **k: None
        )
        assert result["forestry_metrics"] is None
        assert captured["version"] == "2022"
        assert captured["kw"] == {"seed": 42, "point_process": "inhomogeneous_poisson"}
        # Masked cell became a zero-density anchor; two real plots survived.
        assert captured["plots"]["PLOT_ID"].tolist() == [101, 0, 102]

    def test_unsupported_method_raises(self, monkeypatch):
        _install_common_stubs(monkeypatch, [101])
        source = _source()
        source["method"] = {"name": "surface_matching"}

        with pytest.raises(ProcessingError) as exc:
            handle_pim_chm_fusion(
                {"id": "inv-1"}, source, gpd.GeoDataFrame(), lambda *a, **k: None
            )
        assert exc.value.code == "UNSUPPORTED_FUSION_METHOD"

    def test_missing_chm_band_raises(self, monkeypatch):
        _install_common_stubs(monkeypatch, [101])
        # CHM grid loads without a 'chm' band.
        monkeypatch.setattr(
            pim_chm_fusion,
            "load_grid",
            lambda gid: (
                _grid_ds("tm_id", PIM_NODATA)
                if gid == "pim-1"
                else _grid_ds("elevation", np.nan)
            ),
        )
        with pytest.raises(ProcessingError) as exc:
            handle_pim_chm_fusion(
                {"id": "inv-1"}, _source(), gpd.GeoDataFrame(), lambda *a, **k: None
            )
        assert exc.value.code == "MISSING_BAND"

    def test_invalid_fusion_input_maps_valueerror(self, monkeypatch):
        """A core resolution/CRS ValueError becomes a handled terminal failure."""
        _install_common_stubs(monkeypatch, [101])

        def boom(*a, **k):
            raise ValueError("resolution is finer than the PIM")

        monkeypatch.setattr(pim_chm_fusion, "sample_plots_from_hag", boom)
        with pytest.raises(ProcessingError) as exc:
            handle_pim_chm_fusion(
                {"id": "inv-1"}, _source(), gpd.GeoDataFrame(), lambda *a, **k: None
            )
        assert exc.value.code == "INVALID_FUSION_INPUT"


class TestDispatch:
    def test_fusion_source_routes_to_fusion_handler(self, monkeypatch):
        from standgen import dispatch

        called = {}
        monkeypatch.setattr(
            dispatch.pim_chm_fusion,
            "handle_pim_chm_fusion",
            lambda *a, **k: called.setdefault("fusion", True) or {"ok": 1},
        )
        monkeypatch.setattr(
            dispatch.pim,
            "handle_pim",
            lambda *a, **k: called.setdefault("pim", True) or {"ok": 0},
        )
        inv = {"source": {"name": "pim", "fusion": ["chm"]}}
        dispatch.dispatch_handler(inv, gpd.GeoDataFrame(), lambda *a, **k: None)
        assert called == {"fusion": True}

    def test_plain_pim_source_routes_to_pim_handler(self, monkeypatch):
        from standgen import dispatch

        called = {}
        monkeypatch.setattr(
            dispatch.pim_chm_fusion,
            "handle_pim_chm_fusion",
            lambda *a, **k: called.setdefault("fusion", True) or {"ok": 1},
        )
        monkeypatch.setattr(
            dispatch.pim,
            "handle_pim",
            lambda *a, **k: called.setdefault("pim", True) or {"ok": 0},
        )
        inv = {"source": {"name": "pim"}}
        dispatch.dispatch_handler(inv, gpd.GeoDataFrame(), lambda *a, **k: None)
        assert called == {"pim": True}
