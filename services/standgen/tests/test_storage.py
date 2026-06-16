"""
Unit tests for standgen/storage.py

Tests the Parquet write path in isolation against local directories.
No GCP I/O.
"""

import os

import dask.dataframe as dd
import fsspec
import pandas as pd
import pyarrow.parquet as pq
import pytest
from standgen.storage import _apply_changed_partitions, _write_full, _write_parquet


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
        _write_parquet(dd.from_pandas(inventory_df, npartitions=2), path)

        assert _schema_names(path) == ["x", "y", "height"]

    def test_writes_metadata_file(self, inventory_df, tmp_path):
        path = str(tmp_path / "inv")
        _write_parquet(dd.from_pandas(inventory_df, npartitions=2), path)

        metadata = pq.read_metadata(f"{path}/_metadata")
        assert metadata.num_rows == 4

    def test_round_trip_preserves_data(self, inventory_df, tmp_path):
        path = str(tmp_path / "inv")
        _write_parquet(dd.from_pandas(inventory_df, npartitions=2), path)

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
        _write_parquet(dd.read_parquet(legacy_path), rewritten_path)

        assert _schema_names(rewritten_path) == ["x", "y", "height"]
        result = dd.read_parquet(rewritten_path).compute().reset_index(drop=True)
        pd.testing.assert_frame_equal(result, inventory_df)


class TestApplyChangedPartitions:
    """The partial-rewrite core: apply a per-partition transform and overwrite
    only the partitions that change. Exercised against the local filesystem."""

    @staticmethod
    def _write_parts(base: str, dfs: list[pd.DataFrame]) -> None:
        os.makedirs(base, exist_ok=True)
        for k, df in enumerate(dfs):
            df.to_parquet(f"{base}/part.{k}.parquet", index=False)

    @staticmethod
    def _raw(path: str) -> bytes:
        with open(path, "rb") as f:
            return f.read()

    def test_writes_only_the_changed_partition(self, tmp_path):
        """A transform scoped to one partition's rows rewrites only that
        partition; the others stay byte-identical (never re-uploaded)."""
        base = str(tmp_path / "inv")
        # Three spatially-separated partitions: x in [0,5), [10,15), [20,25).
        dfs = [
            pd.DataFrame({"x": [i * 10 + j for j in range(5)], "dbh": [10.0] * 5})
            for i in range(3)
        ]
        self._write_parts(base, dfs)
        p1_before = self._raw(f"{base}/part.1.parquet")
        p2_before = self._raw(f"{base}/part.2.parquet")

        def transform(df):
            df.loc[df["x"] < 5, "dbh"] *= 0.5  # only partition 0 has x < 5
            return df

        n = _apply_changed_partitions(fsspec.filesystem("local"), base, transform)

        assert n == 1
        assert pd.read_parquet(f"{base}/part.0.parquet")["dbh"].tolist() == [5.0] * 5
        # Untouched partitions are not rewritten at all — byte-for-byte identical.
        assert self._raw(f"{base}/part.1.parquet") == p1_before
        assert self._raw(f"{base}/part.2.parquet") == p2_before

    def test_no_change_writes_nothing(self, tmp_path):
        base = str(tmp_path / "inv")
        dfs = [
            pd.DataFrame({"x": [1.0, 2, 3], "dbh": [10.0, 11, 12]}) for _ in range(2)
        ]
        self._write_parts(base, dfs)
        before = {k: self._raw(f"{base}/part.{k}.parquet") for k in range(2)}

        n = _apply_changed_partitions(fsspec.filesystem("local"), base, lambda df: df)

        assert n == 0
        for k in range(2):
            assert self._raw(f"{base}/part.{k}.parquet") == before[k]

    def test_remove_shrinks_only_matching_partition(self, tmp_path):
        base = str(tmp_path / "inv")
        dfs = [
            pd.DataFrame({"x": [0.0, 1, 2], "dbh": [1.0, 2, 3]}),  # all dbh < 5
            pd.DataFrame({"x": [10.0, 11, 12], "dbh": [20.0, 21, 22]}),  # all >= 5
        ]
        self._write_parts(base, dfs)

        def transform(df):
            return df[df["dbh"] >= 5.0].reset_index(drop=True)

        n = _apply_changed_partitions(fsspec.filesystem("local"), base, transform)

        assert n == 1
        assert len(pd.read_parquet(f"{base}/part.0.parquet")) == 0
        assert len(pd.read_parquet(f"{base}/part.1.parquet")) == 3


class TestWriteFull:
    """The full-rewrite path (used for proportional basal-area). Replaces the
    dataset with a materialized DataFrame, dropping the now-wrong ``_metadata``
    and any stale part files so dask re-lists the directory."""

    def test_rewrites_and_drops_metadata_and_stale_parts(self, tmp_path):
        base = str(tmp_path / "inv")
        dd.from_pandas(
            pd.DataFrame({"x": range(300), "dbh": [10.0] * 300}), npartitions=3
        ).to_parquet(base, write_metadata_file=True, write_index=False)
        assert "_metadata" in os.listdir(base)

        new_df = pd.DataFrame({"x": range(50), "dbh": [10.0] * 50})
        n = _write_full(
            fsspec.filesystem("local"), base, new_df, rows_per_partition=100_000
        )

        assert n == 1
        # Only the new partition remains: stale parts, _metadata, _common_metadata
        # are all removed so readers list the directory fresh.
        assert set(os.listdir(base)) == {"part.0.parquet"}
        assert len(pd.read_parquet(base)) == 50
