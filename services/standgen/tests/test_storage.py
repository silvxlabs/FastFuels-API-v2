"""
Unit tests for standgen/storage.py

Tests the Parquet write path in isolation against local directories.
No GCP I/O.
"""

import dask.dataframe as dd
import pandas as pd
import pyarrow.parquet as pq
import pytest
from standgen.storage import _write_parquet


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
