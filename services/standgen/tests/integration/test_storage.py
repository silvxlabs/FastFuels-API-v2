"""
Integration tests for standgen/storage.py

Tests save_parquet_with_summary and save_parquet_replace_with_summary against
real GCS. Verifies parquet is written, summaries are returned correctly, and
each partition is materialized exactly once (single-pass guarantee).
"""

from uuid import uuid4

import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest
from standgen.storage import (
    save_parquet_replace_with_summary,
    save_parquet_with_summary,
)

from lib.config import INVENTORIES_BUCKET
from lib.gcs.blobs import delete_directory, exists

COLUMNS = [
    {"key": "x", "type": "continuous", "unit": "m"},
    {"key": "fia_species_code", "type": "categorical"},
    {"key": "dbh", "type": "continuous", "unit": "cm"},
]


@pytest.fixture
def sample_ddf():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "x": rng.uniform(0, 100, 50),
            "fia_species_code": rng.choice([93, 122, 202], 50),
            "dbh": rng.uniform(5.0, 50.0, 50),
        }
    )
    return dd.from_pandas(df, npartitions=3)


@pytest.fixture
def replacement_ddf():
    rng = np.random.default_rng(99)
    df = pd.DataFrame(
        {
            "x": rng.uniform(200, 300, 20),
            "fia_species_code": rng.choice([93, 122], 20),
            "dbh": rng.uniform(5.0, 50.0, 20),
        }
    )
    return dd.from_pandas(df, npartitions=2)


@pytest.fixture
def inventory_id():
    inv_id = f"test-{uuid4().hex}"
    yield inv_id
    path = f"gs://{INVENTORIES_BUCKET}/{inv_id}"
    if exists(path):
        delete_directory(path)


class TestSaveParquetWithSummary:
    def test_writes_parquet_to_gcs(self, sample_ddf, inventory_id, domain_gdf):
        save_parquet_with_summary(inventory_id, sample_ddf, COLUMNS, "tree", domain_gdf)
        assert exists(f"gs://{INVENTORIES_BUCKET}/{inventory_id}")

    def test_returns_correct_summary(self, sample_ddf, inventory_id, domain_gdf):
        _, stats, _ = save_parquet_with_summary(
            inventory_id, sample_ddf, COLUMNS, "tree", domain_gdf
        )

        assert "x" in stats
        assert "fia_species_code" in stats
        assert stats["x"]["type"] == "continuous"
        assert stats["x"]["count"] == 50
        assert stats["x"]["null_count"] == 0
        assert stats["x"]["min"] is not None
        assert stats["fia_species_code"]["type"] == "categorical"
        assert stats["fia_species_code"]["unique_count"] == 3

    def test_returns_forestry_metrics(self, sample_ddf, inventory_id, domain_gdf):
        _, _, forestry_metrics = save_parquet_with_summary(
            inventory_id, sample_ddf, COLUMNS, "tree", domain_gdf
        )

        assert forestry_metrics is not None
        assert forestry_metrics["tree_count"] == 50
        assert forestry_metrics["basal_area_per_area"] is not None
        assert forestry_metrics["dominant_species_groups"] != []

    def test_single_pass_guarantee(self, inventory_id, domain_gdf):
        """Each partition is materialized exactly once — not once for the write
        and once for the stats reductions."""
        npartitions = 3
        read_count = 0

        def counting_partition(df):
            nonlocal read_count
            read_count += 1
            return df

        rng = np.random.default_rng(1)
        df = pd.DataFrame({"x": rng.uniform(0, 100, 60)})
        ddf = dd.from_pandas(df, npartitions=npartitions).map_partitions(
            counting_partition, meta=pd.DataFrame({"x": pd.Series(dtype="float64")})
        )

        save_parquet_with_summary(
            inventory_id, ddf, [{"key": "x", "type": "continuous"}], "tree", domain_gdf
        )

        assert read_count == npartitions, (
            f"Expected {npartitions} partition reads, got {read_count} "
            f"(double-scan regression)"
        )


class TestSaveParquetReplaceWithSummary:
    def test_replaces_parquet_and_returns_summary(
        self, sample_ddf, replacement_ddf, inventory_id, domain_gdf
    ):
        # Write initial data
        save_parquet_with_summary(inventory_id, sample_ddf, COLUMNS, "tree", domain_gdf)

        # Replace with new data
        path, stats, _ = save_parquet_replace_with_summary(
            inventory_id, replacement_ddf, COLUMNS, "tree", domain_gdf
        )

        # Parquet reflects new data
        result = dd.read_parquet(f"gs://{INVENTORIES_BUCKET}/{inventory_id}").compute()
        assert len(result) == 20
        assert result["x"].min() >= 200

        # Summary reflects new data
        assert stats["x"]["count"] == 20
        assert stats["fia_species_code"]["unique_count"] == 2

    def test_returns_forestry_metrics(
        self, sample_ddf, replacement_ddf, inventory_id, domain_gdf
    ):
        # Write initial data
        save_parquet_with_summary(inventory_id, sample_ddf, COLUMNS, "tree", domain_gdf)

        # Replace and check forestry metrics
        _, _, forestry_metrics = save_parquet_replace_with_summary(
            inventory_id, replacement_ddf, COLUMNS, "tree", domain_gdf
        )

        assert forestry_metrics is not None
        assert forestry_metrics["tree_count"] == 20
        assert forestry_metrics["basal_area_per_area"] is not None
        assert forestry_metrics["dominant_species_groups"] != []

    def test_single_pass_guarantee(self, inventory_id, domain_gdf):
        """Each partition is materialized exactly once during the staging swap —
        not once for the write and once for the stats reductions."""
        npartitions = 3
        read_count = 0

        def counting_partition(df):
            nonlocal read_count
            read_count += 1
            return df

        rng = np.random.default_rng(2)
        df = pd.DataFrame({"x": rng.uniform(0, 100, 60)})
        initial_ddf = dd.from_pandas(df, npartitions=npartitions)
        save_parquet_with_summary(
            inventory_id,
            initial_ddf,
            [{"key": "x", "type": "continuous"}],
            "tree",
            domain_gdf,
        )

        new_ddf = dd.from_pandas(df, npartitions=npartitions).map_partitions(
            counting_partition, meta=pd.DataFrame({"x": pd.Series(dtype="float64")})
        )
        save_parquet_replace_with_summary(
            inventory_id,
            new_ddf,
            [{"key": "x", "type": "continuous"}],
            "tree",
            domain_gdf,
        )

        assert read_count == npartitions, (
            f"Expected {npartitions} partition reads, got {read_count} "
            f"(double-scan regression)"
        )
