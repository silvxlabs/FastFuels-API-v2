"""
Unit tests for api/resources/inventories/cache.py

Tests Parquet ``_metadata`` parsing and row-group reads in isolation using
local dask-written datasets. No GCP I/O.
"""

import asyncio

import dask.dataframe as dd
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest
from api.resources.inventories import cache
from api.resources.inventories.cache import (
    PartitionMeta,
    _parse_metadata,
    _read_row_group_sync,
    read_partition,
)

WRITER_ROW_GROUP_SIZE = 50_000


@pytest.fixture
def inventory_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "x": [500000.0, 500100.0, 500200.0, 500300.0],
            "y": [4200000.0, 4200100.0, 4200200.0, 4200300.0],
            "height": [10.0, 15.0, 20.0, 25.0],
        }
    )


@pytest.fixture
def mixed_df() -> pd.DataFrame:
    """Columns covering the dtypes inventories store, with nulls."""
    n = 37
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "tree_id": np.arange(1, n + 1, dtype="int32"),
            "x": rng.uniform(5e5, 5.1e5, n),
            "y": rng.uniform(4.2e6, 4.21e6, n),
            "height": rng.uniform(1, 40, n),
            "fia_species_code": pd.array(
                [122 if i % 3 else None for i in range(n)], dtype="Int64"
            ),
            "dbh": [None if i % 5 == 0 else float(i) for i in range(n)],
            "label": [f"t{i}" if i % 4 else None for i in range(n)],
        }
    )


def _write_dask_parquet(
    df: pd.DataFrame, path: str, npartitions: int = 2, **kwargs
) -> pq.FileMetaData:
    """Write a partitioned dataset the way standgen/uploader do (#335)."""
    dd.from_pandas(df, npartitions=npartitions).to_parquet(
        path, write_metadata_file=True, write_index=False, **kwargs
    )
    return pq.read_metadata(f"{path}/_metadata")


def _legacy_parse(metadata: pq.FileMetaData) -> list[tuple[int, int, str]]:
    """The pre-#763 parser (one partition per file), kept as ground truth."""
    partitions_by_path: dict[str, int] = {}
    for i in range(metadata.num_row_groups):
        rg = metadata.row_group(i)
        path = rg.column(0).file_path
        partitions_by_path[path] = partitions_by_path.get(path, 0) + rg.num_rows
    return [
        (idx, partitions_by_path[path], path)
        for idx, path in enumerate(sorted(partitions_by_path))
    ]


def _read_all(path: str, meta, columns=None) -> list[pd.DataFrame]:
    return [
        _read_row_group_sync(f"{path}/{p.path}", p.row_group, columns)
        for p in meta.partitions
    ]


class TestParseMetadata:
    def test_columns_match_file_schema(self, inventory_df, tmp_path):
        metadata = _write_dask_parquet(inventory_df, str(tmp_path / "inv"))

        meta = _parse_metadata(metadata)

        assert meta.columns == ["x", "y", "height"]

    def test_partitions_and_row_counts(self, inventory_df, tmp_path):
        metadata = _write_dask_parquet(inventory_df, str(tmp_path / "inv"))

        meta = _parse_metadata(metadata)

        assert meta.num_partitions == 2
        assert meta.total_rows == 4
        assert [p.index for p in meta.partitions] == [0, 1]
        assert sum(p.num_rows for p in meta.partitions) == 4

    def test_single_row_group_files_match_legacy_partitions(self, mixed_df, tmp_path):
        """Existing inventories (one row group per file) keep the exact same
        partitions. 12 files puts ``part.10`` before ``part.2``, as before."""
        metadata = _write_dask_parquet(mixed_df, str(tmp_path / "inv"), npartitions=12)
        assert metadata.num_row_groups == 12

        meta = _parse_metadata(metadata)

        assert [(p.index, p.num_rows, p.path) for p in meta.partitions] == (
            _legacy_parse(metadata)
        )
        assert [p.row_group for p in meta.partitions] == [0] * 12
        assert meta.total_rows == len(mixed_df)

    def test_multi_row_group_files_enumerate_row_groups(self, tmp_path):
        """A file over the writer cap yields one partition per row group,
        ordered by file path then row group."""
        n_big = 2 * WRITER_ROW_GROUP_SIZE + 1
        big = pd.DataFrame({"x": np.arange(n_big, dtype=float)})
        small = pd.DataFrame({"x": np.arange(10, dtype=float) + n_big})
        path = str(tmp_path / "inv")
        dd.concat(
            [dd.from_pandas(big, npartitions=1), dd.from_pandas(small, npartitions=1)]
        ).to_parquet(
            path,
            write_metadata_file=True,
            write_index=False,
            row_group_size=WRITER_ROW_GROUP_SIZE,
        )

        meta = _parse_metadata(pq.read_metadata(f"{path}/_metadata"))

        assert [
            (p.index, p.path, p.row_group, p.num_rows) for p in meta.partitions
        ] == [
            (0, "part.0.parquet", 0, WRITER_ROW_GROUP_SIZE),
            (1, "part.0.parquet", 1, WRITER_ROW_GROUP_SIZE),
            (2, "part.0.parquet", 2, 1),
            (3, "part.1.parquet", 0, 10),
        ]
        assert meta.num_partitions == 4
        assert meta.total_rows == n_big + 10
        assert sum(p.num_rows for p in meta.partitions) == meta.total_rows


class TestReadRowGroup:
    def test_single_row_group_files_match_legacy_read(self, mixed_df, tmp_path):
        """Partition bodies of existing inventories are unchanged: reading
        row group 0 equals the old whole-file ``pd.read_parquet``."""
        path = str(tmp_path / "inv")
        meta = _parse_metadata(_write_dask_parquet(mixed_df, path, npartitions=3))

        for columns in (None, ["height", "fia_species_code", "label"]):
            for p in meta.partitions:
                expected = pd.read_parquet(f"{path}/{p.path}", columns=columns)
                actual = _read_row_group_sync(f"{path}/{p.path}", p.row_group, columns)
                pd.testing.assert_frame_equal(actual, expected)

    def test_legacy_dask_index_dataset_matches_legacy_read(self, mixed_df, tmp_path):
        """Pre-#335 datasets carry ``__null_dask_index__``; it must still be
        restored as the index, not returned as a column."""
        path = str(tmp_path / "inv")
        dd.from_pandas(mixed_df, npartitions=2).to_parquet(
            path, write_metadata_file=True
        )
        meta = _parse_metadata(pq.read_metadata(f"{path}/_metadata"))

        for columns in (None, ["x", "y"]):
            for p in meta.partitions:
                expected = pd.read_parquet(f"{path}/{p.path}", columns=columns)
                actual = _read_row_group_sync(f"{path}/{p.path}", p.row_group, columns)
                pd.testing.assert_frame_equal(actual, expected)
                assert "__null_dask_index__" not in actual.columns

    def test_multi_row_group_partitions_page_through_all_rows(self, mixed_df, tmp_path):
        path = str(tmp_path / "inv")
        metadata = _write_dask_parquet(mixed_df, path, npartitions=3, row_group_size=5)
        meta = _parse_metadata(metadata)
        assert meta.num_partitions > 3

        parts = _read_all(path, meta)

        assert [len(df) for df in parts] == [p.num_rows for p in meta.partitions]
        assert sum(len(df) for df in parts) == meta.total_rows == len(mixed_df)
        assert all(len(df) <= 5 for df in parts)
        expected = pd.concat(
            [
                pd.read_parquet(f"{path}/{f}")
                for f in sorted({p.path for p in meta.partitions})
            ],
            ignore_index=True,
        )
        pd.testing.assert_frame_equal(pd.concat(parts, ignore_index=True), expected)

    def test_multi_row_group_column_subset(self, mixed_df, tmp_path):
        path = str(tmp_path / "inv")
        meta = _parse_metadata(
            _write_dask_parquet(mixed_df, path, npartitions=2, row_group_size=5)
        )

        parts = _read_all(path, meta, columns=["dbh", "x"])

        assert all(list(df.columns) == ["dbh", "x"] for df in parts)
        full = pd.concat(_read_all(path, meta), ignore_index=True)
        pd.testing.assert_frame_equal(
            pd.concat(parts, ignore_index=True), full[["dbh", "x"]]
        )


def test_read_partition_reads_its_row_group(monkeypatch):
    calls = []
    monkeypatch.setattr(cache, "INVENTORIES_BUCKET", "bucket")
    monkeypatch.setattr(
        cache,
        "_read_row_group_sync",
        lambda path, row_group, columns: calls.append((path, row_group, columns)),
    )
    partition = PartitionMeta(index=4, num_rows=10, path="part.1.parquet", row_group=2)

    asyncio.run(read_partition("inv-id", partition, columns=["x"]))

    assert calls == [("gs://bucket/inv-id/part.1.parquet", 2, ["x"])]
