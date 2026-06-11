"""
Unit tests for api/resources/inventories/cache.py

Tests Parquet ``_metadata`` parsing in isolation using local dask-written
datasets. No GCP I/O.
"""

import dask.dataframe as dd
import pandas as pd
import pyarrow.parquet as pq
import pytest
from api.resources.inventories.cache import _parse_metadata


@pytest.fixture
def inventory_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "x": [500000.0, 500100.0, 500200.0, 500300.0],
            "y": [4200000.0, 4200100.0, 4200200.0, 4200300.0],
            "height": [10.0, 15.0, 20.0, 25.0],
        }
    )


def _write_dask_parquet(df: pd.DataFrame, path: str, **kwargs) -> pq.FileMetaData:
    dd.from_pandas(df, npartitions=2).to_parquet(
        path, write_metadata_file=True, **kwargs
    )
    return pq.read_metadata(f"{path}/_metadata")


class TestParseMetadata:
    def test_legacy_dataset_filters_null_dask_index(self, inventory_df, tmp_path):
        """Datasets written before write_index=False carry the dask index
        artifact in their schema; it must not surface in columns."""
        metadata = _write_dask_parquet(inventory_df, str(tmp_path / "inv"))
        # Sanity-check the fixture reproduces the bug: the artifact is
        # physically in the file schema.
        assert "__null_dask_index__" in metadata.schema.to_arrow_schema().names

        meta = _parse_metadata(metadata)

        assert meta.columns == ["x", "y", "height"]

    def test_clean_dataset_columns_pass_through(self, inventory_df, tmp_path):
        metadata = _write_dask_parquet(
            inventory_df, str(tmp_path / "inv"), write_index=False
        )

        meta = _parse_metadata(metadata)

        assert meta.columns == ["x", "y", "height"]

    def test_partitions_and_row_counts(self, inventory_df, tmp_path):
        metadata = _write_dask_parquet(inventory_df, str(tmp_path / "inv"))

        meta = _parse_metadata(metadata)

        assert meta.num_partitions == 2
        assert meta.total_rows == 4
        assert [p.index for p in meta.partitions] == [0, 1]
        assert sum(p.num_rows for p in meta.partitions) == 4
