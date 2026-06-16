"""
Unit tests for standgen/summarize.py

Tests _build_column_stats_graph correctness: stat values, null handling,
and categorical unique counts.
"""

import dask
import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest
from standgen.summarize import _build_column_stats_graph


@pytest.fixture
def continuous_column():
    return {"key": "dbh", "type": "continuous", "unit": "cm"}


@pytest.fixture
def categorical_column():
    return {"key": "fia_species_code", "type": "categorical"}


def _compute(ddf, columns):
    stats_graph = _build_column_stats_graph(ddf, columns)
    flat_scalars = [
        v for k in stats_graph for s, v in stats_graph[k].items() if s != "type"
    ]
    flat_keys = [(k, s) for k in stats_graph for s in stats_graph[k] if s != "type"]
    results = dask.compute(*flat_scalars)
    stats = {}
    for i, (k, s) in enumerate(flat_keys):
        if k not in stats:
            stats[k] = {"type": stats_graph[k]["type"]}
        stats[k][s] = results[i]
    for k, v in stats.items():
        if v["type"] == "continuous" and v["count"] == 0:
            v["min"] = v["max"] = v["mean"] = v["std"] = None
    return stats


class TestContinuousColumn:
    def test_stats_match_numpy_ground_truth(self, continuous_column):
        rng = np.random.default_rng(0)
        values = rng.uniform(1.0, 100.0, 50)
        null_mask = np.zeros(50, dtype=bool)
        null_mask[:5] = True
        series = pd.Series(np.where(null_mask, np.nan, values))
        ddf = dd.from_pandas(pd.DataFrame({"dbh": series}), npartitions=2)

        stats = _compute(ddf, [continuous_column])["dbh"]

        valid = values[~null_mask]
        assert stats["type"] == "continuous"
        assert stats["count"] == 45
        assert stats["null_count"] == 5
        assert pytest.approx(stats["min"], rel=1e-6) == float(valid.min())
        assert pytest.approx(stats["max"], rel=1e-6) == float(valid.max())
        assert pytest.approx(stats["mean"], rel=1e-6) == float(valid.mean())
        assert pytest.approx(stats["std"], rel=1e-6) == float(valid.std(ddof=1))

    def test_all_null_column(self, continuous_column):
        ddf = dd.from_pandas(
            pd.DataFrame({"dbh": pd.Series([np.nan] * 20)}), npartitions=2
        )

        stats = _compute(ddf, [continuous_column])["dbh"]

        assert stats["count"] == 0
        assert stats["null_count"] == 20
        assert stats["min"] is None
        assert stats["max"] is None
        assert stats["mean"] is None
        assert stats["std"] is None


class TestCategoricalColumn:
    def test_null_count_and_unique_count(self, categorical_column):
        series = pd.Series([93, 122, 202, 93, None, None, 122])
        ddf = dd.from_pandas(pd.DataFrame({"fia_species_code": series}), npartitions=2)

        stats = _compute(ddf, [categorical_column])["fia_species_code"]

        assert stats["type"] == "categorical"
        assert stats["count"] == 5
        assert stats["null_count"] == 2
        assert stats["unique_count"] == 3

    def test_nulls_excluded_from_unique_count(self, categorical_column):
        series = pd.Series([1, None, None, None])
        ddf = dd.from_pandas(pd.DataFrame({"fia_species_code": series}), npartitions=2)

        stats = _compute(ddf, [categorical_column])["fia_species_code"]

        assert stats["unique_count"] == 1


class TestMissingColumn:
    def test_column_absent_from_ddf_is_skipped(self):
        ddf = dd.from_pandas(pd.DataFrame({"x": [1.0, 2.0]}), npartitions=1)
        columns = [{"key": "nonexistent", "type": "continuous"}]

        stats = _compute(ddf, columns)

        assert "nonexistent" not in stats
