"""
Unit tests for standgen/storage.py

Tests the Parquet write path in isolation against local directories.
No GCP I/O.
"""

import math

import dask.dataframe as dd
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest
from standgen.storage import _compute_write_and_stats, _write_parquet
from standgen.summarize import _build_column_stats_graph


@pytest.fixture
def inventory_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "x": [500000.0, 500100.0, 500200.0, 500300.0],
            "y": [4200000.0, 4200100.0, 4200200.0, 4200300.0],
            "height": [10.0, 15.0, 20.0, 25.0],
        }
    )


def _schema_names(path: str) -> list[str]:
    return pq.read_metadata(f"{path}/_metadata").schema.to_arrow_schema().names


class TestWriteParquet:
    def test_no_null_dask_index_in_schema(self, inventory_df, tmp_path):
        """dask's synthetic RangeIndex column must not be written (#335)."""
        path = str(tmp_path / "inv")
        _write_parquet(dd.from_pandas(inventory_df, npartitions=2), path).compute()

        assert _schema_names(path) == ["x", "y", "height"]

    def test_writes_metadata_file(self, inventory_df, tmp_path):
        path = str(tmp_path / "inv")
        _write_parquet(dd.from_pandas(inventory_df, npartitions=2), path).compute()

        metadata = pq.read_metadata(f"{path}/_metadata")
        assert metadata.num_rows == 4

    def test_round_trip_preserves_data(self, inventory_df, tmp_path):
        path = str(tmp_path / "inv")
        _write_parquet(dd.from_pandas(inventory_df, npartitions=2), path).compute()

        result = dd.read_parquet(path).compute().reset_index(drop=True)
        pd.testing.assert_frame_equal(result, inventory_df)

    def test_legacy_dataset_rewrites_clean(self, inventory_df, tmp_path):
        """An in-place rewrite of a legacy dataset (which carries the
        __null_dask_index__ column) must scrub the artifact."""
        legacy_path = str(tmp_path / "legacy")
        dd.from_pandas(inventory_df, npartitions=2).to_parquet(
            legacy_path, write_metadata_file=True
        )
        assert "__null_dask_index__" in _schema_names(legacy_path)

        rewritten_path = str(tmp_path / "rewritten")
        _write_parquet(dd.read_parquet(legacy_path), rewritten_path).compute()

        assert _schema_names(rewritten_path) == ["x", "y", "height"]
        result = dd.read_parquet(rewritten_path).compute().reset_index(drop=True)
        pd.testing.assert_frame_equal(result, inventory_df)


class TestComputeWriteAndStats:
    """Summary stats must never carry a non-finite float. The API serves
    inventories with Starlette's JSONResponse (allow_nan=False), so a NaN/inf
    stat persisted to Firestore would raise on serialization and 500 the GET."""

    def _stats(self, ddf, columns, tmp_path):
        write = _write_parquet(ddf, str(tmp_path / "inv"))
        return _compute_write_and_stats(write, _build_column_stats_graph(ddf, columns))

    def test_single_value_continuous_std_is_none(self, tmp_path):
        """Sample std (ddof=1) of a lone non-null value is NaN — a single-tree
        inventory. It must be sanitized to None while the finite stats stay."""
        ddf = dd.from_pandas(pd.DataFrame({"dbh": [5.0]}), npartitions=1)
        columns = [{"key": "dbh", "type": "continuous"}]

        stats = self._stats(ddf, columns, tmp_path)["dbh"]

        assert stats["count"] == 1
        assert stats["std"] is None
        assert stats["min"] == 5.0
        assert stats["max"] == 5.0
        assert stats["mean"] == 5.0

    def test_all_null_continuous_stats_are_none(self, tmp_path):
        """An all-null continuous column has no finite min/max/mean/std."""
        ddf = dd.from_pandas(
            pd.DataFrame({"dbh": [np.nan, np.nan, np.nan]}), npartitions=1
        )
        columns = [{"key": "dbh", "type": "continuous"}]

        stats = self._stats(ddf, columns, tmp_path)["dbh"]

        assert stats["count"] == 0
        assert stats["null_count"] == 3
        assert stats["min"] is None
        assert stats["max"] is None
        assert stats["mean"] is None
        assert stats["std"] is None

    def test_no_stat_is_non_finite(self, tmp_path):
        """No stat leaves this function as a NaN/inf float, for any column."""
        ddf = dd.from_pandas(pd.DataFrame({"dbh": [5.0]}), npartitions=1)
        columns = [{"key": "dbh", "type": "continuous"}]

        stats = self._stats(ddf, columns, tmp_path)["dbh"]

        for value in stats.values():
            assert not (isinstance(value, float) and not math.isfinite(value))

    def test_multi_value_continuous_std_preserved(self, tmp_path):
        """Sanitization must not clobber a well-defined std."""
        ddf = dd.from_pandas(pd.DataFrame({"dbh": [2.0, 4.0, 6.0]}), npartitions=1)
        columns = [{"key": "dbh", "type": "continuous"}]

        stats = self._stats(ddf, columns, tmp_path)["dbh"]

        assert stats["count"] == 3
        assert stats["std"] == pytest.approx(2.0)
