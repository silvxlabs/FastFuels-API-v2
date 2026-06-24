"""
Unit tests for api/resources/inventories/cache.py

Inventory metadata comes from the dask DataFrame (footer-level counts), not the
aggregated ``_metadata`` file — so it stays correct after an in-place overwrite
that leaves ``_metadata`` stale. Exercised against local dask datasets.
"""

import glob
from unittest.mock import patch

import dask.dataframe as dd
import pandas as pd
import pytest
from api.resources.inventories import cache
from api.resources.inventories.cache import _read_metadata_sync, _read_partition_sync


@pytest.fixture
def inventory_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "x": [500000.0, 500100.0, 500200.0, 500300.0],
            "y": [4200000.0, 4200100.0, 4200200.0, 4200300.0],
            "height": [10.0, 15.0, 20.0, 25.0],
        }
    )


def _write(df: pd.DataFrame, path: str, npartitions: int = 2) -> None:
    dd.from_pandas(df, npartitions=npartitions).to_parquet(
        path, write_metadata_file=True, write_index=False
    )


class TestReadMetadata:
    def test_counts_columns_and_partitions(self, inventory_df, tmp_path):
        path = str(tmp_path / "inv")
        _write(inventory_df, path)

        with patch.object(cache, "_inventory_path", return_value=path):
            meta = _read_metadata_sync("inv")

        assert meta.num_partitions == 2
        assert meta.total_rows == 4
        assert meta.columns == ["x", "y", "height"]

    def test_counts_correct_after_in_place_overwrite(self, inventory_df, tmp_path):
        """An in-place overwrite of one partition with fewer rows leaves the
        aggregated ``_metadata`` stale; ``total_rows`` must reflect the files on
        disk, not the stale aggregate."""
        path = str(tmp_path / "inv")
        _write(inventory_df, path)  # 4 rows across 2 partitions of 2

        # Overwrite the first partition in place with a single row; do NOT touch
        # _metadata (which still claims this partition has 2 rows / 4 total).
        p0 = sorted(glob.glob(f"{path}/part.*.parquet"))[0]
        pd.read_parquet(p0).iloc[:1].to_parquet(p0, index=False)

        with patch.object(cache, "_inventory_path", return_value=path):
            meta = _read_metadata_sync("inv")

        assert meta.total_rows == 3  # 1 + 2, not the stale 4


class TestReadPartition:
    def test_reads_partition_by_index(self, inventory_df, tmp_path):
        path = str(tmp_path / "inv")
        _write(inventory_df, path)

        with patch.object(cache, "_inventory_path", return_value=path):
            p0 = _read_partition_sync("inv", 0, None)
            p1 = _read_partition_sync("inv", 1, None)

        assert len(p0) == 2
        assert len(p1) == 2
        assert list(p0.columns) == ["x", "y", "height"]

    def test_reads_column_subset(self, inventory_df, tmp_path):
        path = str(tmp_path / "inv")
        _write(inventory_df, path)

        with patch.object(cache, "_inventory_path", return_value=path):
            p0 = _read_partition_sync("inv", 0, ["x"])

        assert list(p0.columns) == ["x"]
