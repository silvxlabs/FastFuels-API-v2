"""Tests for the stable per-tree ``tree_id`` column (#611).

Generation (PIM / fusion via ``expand_plots``, CHM), preservation through
create-time and in-place modifications and treatments, and the single-compute
guarantee. Writes go to a local directory through the real fused
write-and-summarize graph; no GCP I/O.
"""

from unittest.mock import MagicMock

import dask.dataframe as dd
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rioxarray  # noqa: F401 - registers .rio accessor
import xarray as xr
from affine import Affine
from shapely.geometry import Point, box
from standgen import storage
from standgen.columns import generate_tree_ids
from standgen.handlers import chm, modifications, pim, treatments

from .handlers.conftest import BASE_INVENTORY_COLUMNS, CHM_INVENTORY_COLUMNS

TREE_ID = {"key": "tree_id", "type": "categorical"}
PIM_COLUMNS = [TREE_ID, *BASE_INVENTORY_COLUMNS]
CHM_COLUMNS = [TREE_ID, *CHM_INVENTORY_COLUMNS]
CRS = "EPSG:32610"


def _noop_progress(*_args, **_kwargs):
    pass


def _counted_source(parts: list[pd.DataFrame], calls: list[int]) -> dd.DataFrame:
    """A lazy ddf whose partition ``i`` records ``i`` in ``calls`` each time it
    is computed — a stand-in for CHM detection / PIM expansion."""

    def load(i):
        calls.append(i)
        return parts[i]

    return dd.from_map(load, list(range(len(parts))), meta=parts[0].iloc[:0])


def _local_saver(tmp_path, saved: dict):
    """A ``save_parquet_*_with_summary`` stand-in that runs the real fused
    write-and-summarize graph against a local directory."""

    def save(inventory_id, ddf, columns, inventory_type="tree", domain_gdf=None):
        path = str(tmp_path / f"{inventory_id}-{len(saved)}")
        stats, forestry = storage._build_delayed_graph(
            ddf, path, columns, inventory_type, domain_gdf, 5
        )
        saved["df"] = pd.read_parquet(path)
        saved["stats"] = stats
        saved["path"] = path
        return path, stats, forestry

    return save


def _assert_survivors_unchanged(out: pd.DataFrame, source: pd.DataFrame) -> None:
    """Every row of ``out`` is the ``source`` tree with the same ``tree_id``."""
    assert out["tree_id"].is_unique
    expected = source[source["tree_id"].isin(out["tree_id"])]
    pd.testing.assert_frame_equal(
        out.sort_values("tree_id").reset_index(drop=True),
        expected.sort_values("tree_id").reset_index(drop=True),
    )


def _trees(n: int, start: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(start)
    return pd.DataFrame(
        {
            "x": rng.uniform(0, 100, n),
            "y": rng.uniform(0, 100, n),
            "height": np.arange(start, start + n, dtype=float) + 2.0,
        }
    )


class TestGenerateTreeIds:
    def test_contiguous_across_partitions_including_empty(self):
        parts = [_trees(3), _trees(0), _trees(4, 3), _trees(2, 7)]
        ddf = dd.from_map(lambda i: parts[i], range(4), meta=parts[0].iloc[:0])

        out = generate_tree_ids(ddf).compute()

        assert out.columns[0] == "tree_id"
        assert out["tree_id"].dtype == np.int32
        assert list(out["tree_id"]) == list(range(1, 10))
        # Row order is unchanged: IDs follow the source rows.
        assert list(out["height"]) == [float(i) + 2.0 for i in range(9)]

    def test_single_compute_of_upstream_graph(self, tmp_path):
        """IDs, the Parquet write and the summaries run in one pass: each
        upstream partition is computed exactly once."""
        calls: list[int] = []
        parts = [_trees(3), _trees(0), _trees(4, 3)]
        ddf = generate_tree_ids(_counted_source(parts, calls))

        saved = {}
        _local_saver(tmp_path, saved)("inv", ddf, CHM_COLUMNS)

        assert sorted(calls) == [0, 1, 2]
        assert list(saved["df"]["tree_id"]) == list(range(1, 8))

    def test_forestry_metrics_fused_with_tree_ids(self, tmp_path):
        """Regression: with dask 2026.1.2 a Delayed in the fused compute
        misaligned its results once the graph held tree_id's cumsum. The
        forestry reductions are now expression-only."""
        pdf = _trees(6).assign(
            dbh=[10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            fia_species_code=[202, 202, 122, 122, 93, 93],
        )
        ddf = generate_tree_ids(dd.from_pandas(pdf, npartitions=3))
        domain_gdf = gpd.GeoDataFrame(geometry=[box(0, 0, 100, 100)], crs=CRS)
        columns = [TREE_ID, {"key": "dbh", "type": "continuous", "unit": "cm"}]

        stats, forestry = storage._build_delayed_graph(
            ddf, str(tmp_path / "inv"), columns, "tree", domain_gdf, 5
        )

        assert stats["tree_id"]["count"] == 6
        assert stats["dbh"]["max"] == 60.0
        assert forestry["tree_count"] == 6
        assert list(pd.read_parquet(tmp_path / "inv")["tree_id"]) == list(range(1, 7))

    def test_summary_is_categorical_with_unique_equal_to_count(self, tmp_path):
        saved = {}
        ddf = generate_tree_ids(dd.from_pandas(_trees(5), npartitions=2))
        _local_saver(tmp_path, saved)("inv", ddf, CHM_COLUMNS)

        assert saved["stats"]["tree_id"] == {
            "type": "categorical",
            "count": 5,
            "null_count": 0,
            "unique_count": 5,
        }


def _chm_inventory(mods=None):
    return {
        "id": "chm-inv",
        "domain_id": "domain",
        "type": "tree",
        "source": {"name": "chm", "source_chm_grid_id": "grid", "algorithm": {}},
        "modifications": mods or [],
        "columns": CHM_COLUMNS,
    }


class TestChmHandler:
    @pytest.fixture
    def run(self, tmp_path, monkeypatch):
        domain_gdf = gpd.GeoDataFrame(geometry=[box(0, 0, 100, 100)], crs=CRS)
        da = xr.DataArray(np.ones((4, 4)), dims=("y", "x"))
        da = da.rio.write_crs(CRS)
        da = da.rio.write_transform(Affine(1.0, 0.0, 0.0, 0.0, -1.0, 4.0))
        monkeypatch.setattr(chm, "get_document", lambda *a: (None, MagicMock()))
        monkeypatch.setattr(chm, "load_grid", lambda _id: xr.Dataset({"chm": da}))
        monkeypatch.setattr(chm, "count_inventory_rows", lambda _id: None)

        def _run(mods=None):
            calls: list[int] = []
            parts = [_trees(3), _trees(0), _trees(5, 3)]
            monkeypatch.setattr(
                chm,
                "fixed_window_filter",
                lambda **kw: _counted_source(parts, calls),
            )
            saved = {}
            monkeypatch.setattr(
                chm, "save_parquet_with_summary", _local_saver(tmp_path, saved)
            )
            inventory = _chm_inventory(mods)
            chm.handle_chm(inventory, inventory["source"], domain_gdf, _noop_progress)
            return saved["df"], calls

        return _run

    def test_ids_are_one_to_n(self, run):
        df, _ = run()
        assert list(df["tree_id"]) == list(range(1, 9))
        assert df["tree_id"].is_unique and df["tree_id"].notna().all()

    def test_detection_graph_computed_once(self, run):
        _, calls = run()
        assert sorted(calls) == [0, 1, 2]

    def test_same_request_same_ids(self, run):
        first, _ = run()
        second, _ = run()
        pd.testing.assert_frame_equal(first, second)

    def test_create_time_removal_leaves_gaps(self, run):
        full, _ = run()
        mods = [
            {
                "conditions": [{"attribute": "height", "operator": "lt", "value": 5.0}],
                "actions": [{"modifier": "remove"}],
            }
        ]
        thinned, calls = run(mods)

        assert sorted(calls) == [0, 1, 2]
        # Heights 2, 3, 4 (IDs 1-3) are removed; survivors keep their IDs.
        assert list(thinned["tree_id"]) == [4, 5, 6, 7, 8]
        pd.testing.assert_frame_equal(
            thinned.reset_index(drop=True),
            full[full["tree_id"] >= 4].reset_index(drop=True),
        )


def _plots_and_tree_table():
    """A 4 x 4 PIM raster of 30 m cells (two plots plus empty cells) and the
    matching TreeMap tree table, in the shapes ``expand_plots`` consumes."""
    xs = np.arange(4) * 30.0 + 15.0
    ys = np.arange(4) * 30.0 + 15.0
    ids = np.array(
        [[101, 101, 102, 0], [101, 102, 102, 0], [0, 101, 102, 101], [0] * 4]
    )
    points, plot_ids = [], []
    for r, y in enumerate(ys):
        for c, x in enumerate(xs):
            points.append(Point(x, y))
            plot_ids.append(int(ids[r, c]))
    plots = gpd.GeoDataFrame({"PLOT_ID": plot_ids}, geometry=points, crs=CRS)
    tree_table = pd.DataFrame(
        {
            "TM_ID": [101, 101, 101, 102, 102],
            "SPCD": [202, 122, 202, 93, 15],
            "STATUSCD": [1, 1, 1, 1, 1],
            "DIA": [12.0, 3.0, 18.0, 2.0, 24.0],
            "HT": [60.0, 20.0, 80.0, 15.0, 100.0],
            "CR": [40.0, 60.0, 35.0, 50.0, 30.0],
            "TPA_UNADJ": [60.0, 120.0, 30.0, 150.0, 20.0],
        }
    )
    return plots, tree_table


class TestExpandPlots:
    """PIM and PIM-CHM fusion share ``expand_plots``; the point process is the
    real fastfuels-core one."""

    @pytest.fixture
    def run(self, tmp_path, monkeypatch):
        plots, tree_table = _plots_and_tree_table()
        domain_gdf = gpd.GeoDataFrame(geometry=[box(0, 0, 120, 120)], crs=CRS)
        monkeypatch.setattr(pim, "load_tree_table", lambda v, ids: tree_table)

        calls: list[int] = []

        class CountingTreeSample(pim.TreeSample):
            def expand_to_roi(self, *args, **kwargs):
                ddf = super().expand_to_roi(*args, **kwargs)

                def count(df, partition_info=None):
                    calls.append(partition_info["number"])
                    return df

                return ddf.map_partitions(count, meta=ddf._meta)

        monkeypatch.setattr(pim, "TreeSample", CountingTreeSample)

        def _run(mods=None, treats=None, seed=7):
            calls.clear()
            saved = {}
            monkeypatch.setattr(
                pim, "save_parquet_with_summary", _local_saver(tmp_path, saved)
            )
            inventory = {
                "id": "pim-inv",
                "domain_id": "domain",
                "type": "tree",
                "modifications": mods or [],
                "treatments": treats or [],
                "columns": PIM_COLUMNS,
            }
            pim.expand_plots(
                inventory,
                plots,
                "2022",
                domain_gdf,
                _noop_progress,
                seed=seed,
                point_process="inhomogeneous_poisson",
            )
            return saved, list(calls)

        return _run

    def test_ids_are_one_to_n(self, run):
        saved, _ = run()
        df = saved["df"]
        assert len(df) > 10
        assert df.columns[0] == "tree_id"
        assert df["tree_id"].dtype == np.int32
        assert list(df["tree_id"]) == list(range(1, len(df) + 1))
        assert saved["stats"]["tree_id"]["unique_count"] == len(df)

    def test_expansion_graph_computed_once(self, run):
        _, calls = run()
        assert calls
        assert len(calls) == len(set(calls))

    def test_same_request_same_ids(self, run):
        first, _ = run(seed=11)
        second, _ = run(seed=11)
        pd.testing.assert_frame_equal(first["df"], second["df"])

    def test_create_time_modification_leaves_gaps(self, run):
        full = run()[0]["df"]
        mods = [
            {
                "conditions": [{"attribute": "dbh", "operator": "lt", "value": 10.0}],
                "actions": [{"modifier": "remove"}],
            }
        ]
        thinned = run(mods)[0]["df"]

        expected = full[full["dbh"] >= 10.0].reset_index(drop=True)
        assert 0 < len(thinned) < len(full)
        pd.testing.assert_frame_equal(thinned.reset_index(drop=True), expected)
        assert list(thinned["tree_id"]) != list(range(1, len(thinned) + 1))

    def test_create_time_tree_id_condition(self, run):
        full = run()[0]["df"]
        listed = [1, 3, 5]
        mods = [
            {
                "conditions": [
                    {"attribute": "tree_id", "operator": "eq", "value": listed}
                ],
                "actions": [{"modifier": "remove"}],
            }
        ]
        thinned = run(mods)[0]["df"]
        assert set(full["tree_id"]) - set(thinned["tree_id"]) == set(listed)

    @pytest.mark.parametrize(
        "treatment",
        [
            {"metric": "diameter", "method": "from_below", "value": 10.0},
            {"metric": "basal_area", "method": "from_above", "value": 1.0},
        ],
        ids=["diameter", "basal_area"],
    )
    def test_create_time_treatment_preserves_survivor_ids(self, run, treatment):
        full = run()[0]["df"]
        treated = run(treats=[treatment])[0]["df"]

        assert 0 < len(treated) < len(full)
        _assert_survivors_unchanged(treated, full)


def _stored_inventory(tmp_path) -> tuple[str, pd.DataFrame]:
    """An inventory already carrying tree IDs with gaps, as stored Parquet."""
    rng = np.random.default_rng(3)
    n = 40
    df = pd.DataFrame(
        {
            "tree_id": np.array([i * 3 for i in range(n)], dtype="int32"),
            "x": rng.uniform(500000, 501000, n),
            "y": rng.uniform(5200000, 5201000, n),
            "fia_species_code": rng.choice([93, 122, 202], n),
            "fia_status_code": [1] * n,
            "dbh": rng.uniform(1.0, 50.0, n),
            "height": rng.uniform(1.0, 30.0, n),
            "crown_ratio": rng.uniform(0.1, 0.9, n),
        }
    )
    path = str(tmp_path / "stored")
    storage._write_parquet(dd.from_pandas(df, npartitions=3), path).compute()
    return path, df


class TestInPlace:
    @pytest.fixture
    def domain_gdf(self):
        return gpd.GeoDataFrame(
            geometry=[box(500000, 5200000, 501000, 5201000)], crs="EPSG:32611"
        )

    def _inventory(self, **pending):
        return {
            "id": "inv",
            "domain_id": "domain",
            "type": "tree",
            "georeference": None,
            "columns": PIM_COLUMNS,
            **pending,
        }

    def _modify(self, tmp_path, monkeypatch, domain_gdf, mods):
        path, source = _stored_inventory(tmp_path)
        saved = {}
        monkeypatch.setattr(
            modifications, "load_inventory_parquet", lambda _id: dd.read_parquet(path)
        )
        monkeypatch.setattr(
            modifications,
            "save_parquet_replace_with_summary",
            _local_saver(tmp_path, saved),
        )
        modifications.apply_in_place_modifications(
            self._inventory(pending_modifications=mods), domain_gdf, _noop_progress
        )
        return source, saved["df"]

    def test_tree_id_eq_list_removes_exactly_listed(
        self, tmp_path, monkeypatch, domain_gdf
    ):
        listed = [0, 9, 57, 117]
        mods = [
            {
                "conditions": [
                    {"attribute": "tree_id", "operator": "eq", "value": listed}
                ],
                "actions": [{"modifier": "remove"}],
            }
        ]
        source, out = self._modify(tmp_path, monkeypatch, domain_gdf, mods)

        assert set(source["tree_id"]) - set(out["tree_id"]) == set(listed)
        expected = source[~source["tree_id"].isin(listed)].reset_index(drop=True)
        pd.testing.assert_frame_equal(out.reset_index(drop=True), expected)

    def test_tree_id_ne_list_keeps_exactly_listed(
        self, tmp_path, monkeypatch, domain_gdf
    ):
        listed = [3, 30, 60]
        mods = [
            {
                "conditions": [
                    {"attribute": "tree_id", "operator": "ne", "value": listed}
                ],
                "actions": [{"modifier": "remove"}],
            }
        ]
        _, out = self._modify(tmp_path, monkeypatch, domain_gdf, mods)
        assert sorted(out["tree_id"]) == listed

    def test_attribute_modification_preserves_ids(
        self, tmp_path, monkeypatch, domain_gdf
    ):
        mods = [
            {
                "conditions": [{"attribute": "dbh", "operator": "lt", "value": 20.0}],
                "actions": [{"modifier": "remove"}],
            },
            {
                "conditions": [],
                "actions": [
                    {"attribute": "height", "modifier": "multiply", "value": 0.5}
                ],
            },
        ]
        source, out = self._modify(tmp_path, monkeypatch, domain_gdf, mods)

        survivors = source[source["dbh"] >= 20.0].reset_index(drop=True)
        assert list(out["tree_id"]) == list(survivors["tree_id"])
        np.testing.assert_allclose(out["height"], survivors["height"] * 0.5)

    @pytest.mark.parametrize(
        "treatment",
        [
            {"metric": "diameter", "method": "from_below", "value": 20.0},
            {"metric": "basal_area", "method": "proportional", "value": 0.02},
        ],
        ids=["diameter", "basal_area"],
    )
    def test_treatment_preserves_survivor_ids(
        self, tmp_path, monkeypatch, domain_gdf, treatment
    ):
        path, source = _stored_inventory(tmp_path)
        saved = {}
        monkeypatch.setattr(
            treatments, "load_inventory_parquet", lambda _id: dd.read_parquet(path)
        )
        monkeypatch.setattr(
            treatments,
            "save_parquet_replace_with_summary",
            _local_saver(tmp_path, saved),
        )
        treatments.apply_in_place_treatments(
            self._inventory(treatments=[treatment], pending_treatments=[treatment]),
            domain_gdf,
            _noop_progress,
        )
        out = saved["df"]

        assert 0 < len(out) < len(source)
        _assert_survivors_unchanged(out, source)
