"""
Unit tests for standgen/handlers/modifications.py

Tests the in-place modifications handler with mocked I/O (GCS). Verifies it
loads the inventory's own data, applies only the pending delta, replaces the
data in place, and carries over the existing georeference.
"""

from unittest.mock import MagicMock, patch

import dask.dataframe as dd
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box
from standgen.handlers.modifications import apply_in_place_modifications

from .conftest import BASE_INVENTORY_COLUMNS


@pytest.fixture
def sample_ddf():
    """Sample dask DataFrame that load_inventory_parquet would return."""
    rng = np.random.default_rng(42)
    n = 50
    df = pd.DataFrame(
        {
            "x": rng.uniform(500000, 501000, n),
            "y": rng.uniform(5200000, 5201000, n),
            "fia_species_code": rng.choice([93, 122, 202], n),
            "fia_status_code": [1] * n,
            "dbh": rng.uniform(1.0, 50.0, n),
            "height": rng.uniform(1.0, 30.0, n),
            "crown_ratio": rng.uniform(0.1, 0.9, n),
        }
    )
    return dd.from_pandas(df, npartitions=3)


@pytest.fixture
def domain_gdf():
    """Domain GeoDataFrame (only used to resolve spatial conditions)."""
    return gpd.GeoDataFrame(
        geometry=[box(500000, 5200000, 501000, 5201000)],
        crs="EPSG:32611",
    )


@pytest.fixture
def base_inventory():
    """A completed inventory with one pending modification delta to apply."""
    return {
        "id": "inventory-id",
        "domain_id": "domain-123",
        "georeference": {
            "crs": "EPSG:32611",
            "bounds": [500000.0, 5200000.0, 501000.0, 5201000.0],
        },
        "columns": BASE_INVENTORY_COLUMNS,
        "type": "tree",
        "source": {"name": "pim", "source_pim_grid_id": "grid-id", "seed": 1},
        "modifications": [
            {
                "conditions": [{"attribute": "dbh", "operator": "lt", "value": 5.0}],
                "actions": [{"modifier": "remove"}],
            }
        ],
        "pending_modifications": [
            {
                "conditions": [{"attribute": "dbh", "operator": "lt", "value": 5.0}],
                "actions": [{"modifier": "remove"}],
            }
        ],
    }


class TestApplyInPlaceModifications:
    @patch("standgen.handlers.modifications.save_parquet_replace_with_summary")
    @patch("standgen.handlers.modifications.load_inventory_parquet")
    def test_loads_own_data_replaces_in_place_and_carries_georeference(
        self, mock_load, mock_save, base_inventory, sample_ddf, domain_gdf
    ):
        """Loads the inventory's own data, replaces it in place, returns the
        existing georeference unchanged."""
        mock_load.return_value = sample_ddf
        mock_save.return_value = ("gs://test-bucket/inventory-id", {}, None)
        progress = MagicMock()

        result = apply_in_place_modifications(base_inventory, domain_gdf, progress)

        # Load and replace both key off the inventory's own ID (in place).
        mock_load.assert_called_once_with("inventory-id")
        mock_save.assert_called_once()
        assert mock_save.call_args[0][0] == "inventory-id"
        assert isinstance(mock_save.call_args[0][1], dd.DataFrame)

        # Georeference is carried over verbatim — not recomputed.
        assert result["georeference"] == base_inventory["georeference"]

        # Columns are passed as the third argument for summary computation.
        assert mock_save.call_args[0][2] == base_inventory["columns"]
        assert "columns" in result

    @patch("standgen.handlers.modifications.save_parquet_replace_with_summary")
    @patch("standgen.handlers.modifications.load_inventory_parquet")
    def test_applies_only_the_pending_delta(
        self, mock_load, mock_save, base_inventory, sample_ddf, domain_gdf
    ):
        """Only pending_modifications is applied to the current data."""
        mock_load.return_value = sample_ddf
        mock_save.return_value = ("gs://test-bucket/inventory-id", {}, None)
        progress = MagicMock()

        apply_in_place_modifications(base_inventory, domain_gdf, progress)

        result_df = mock_save.call_args[0][1].compute()
        # The pending delta removes dbh < 5.0.
        assert (result_df["dbh"] >= 5.0).all()
        assert len(result_df) < len(sample_ddf.compute())

    @patch("standgen.handlers.modifications.save_parquet_replace_with_summary")
    @patch("standgen.handlers.modifications.load_inventory_parquet")
    def test_reports_progress(
        self, mock_load, mock_save, base_inventory, sample_ddf, domain_gdf
    ):
        mock_load.return_value = sample_ddf
        mock_save.return_value = ("gs://test-bucket/inventory-id", {}, None)
        progress = MagicMock()

        apply_in_place_modifications(base_inventory, domain_gdf, progress)

        messages = [c[0][0] for c in progress.call_args_list]
        assert "Loading inventory..." in messages
        assert "Applying modifications..." in messages
        assert "Writing modified inventory..." in messages
        assert "Complete" in messages

    @patch("standgen.handlers.modifications.save_parquet_replace_with_summary")
    @patch("standgen.handlers.modifications.load_inventory_parquet")
    def test_multiply_delta_preserves_row_count(
        self, mock_load, mock_save, domain_gdf, sample_ddf
    ):
        inventory = {
            "id": "inventory-id",
            "domain_id": "domain-123",
            "georeference": {"crs": "EPSG:32611", "bounds": [0, 0, 1, 1]},
            "columns": BASE_INVENTORY_COLUMNS,
            "type": "tree",
            "source": {"name": "pim"},
            "modifications": [],
            "pending_modifications": [
                {
                    "conditions": [
                        {"attribute": "height", "operator": "gt", "value": 20.0}
                    ],
                    "actions": [
                        {"attribute": "height", "modifier": "multiply", "value": 0.9}
                    ],
                }
            ],
        }
        mock_load.return_value = sample_ddf
        mock_save.return_value = ("gs://test-bucket/inventory-id", {}, None)
        progress = MagicMock()

        apply_in_place_modifications(inventory, domain_gdf, progress)

        result_df = mock_save.call_args[0][1].compute()
        assert len(result_df) == len(sample_ddf.compute())

    @patch("standgen.handlers.modifications.save_parquet_replace_with_summary")
    @patch("standgen.handlers.modifications.load_inventory_parquet")
    def test_multiple_pending_modifications_applied_sequentially(
        self, mock_load, mock_save, domain_gdf, sample_ddf
    ):
        inventory = {
            "id": "inventory-id",
            "domain_id": "domain-123",
            "georeference": {"crs": "EPSG:32611", "bounds": [0, 0, 1, 1]},
            "columns": BASE_INVENTORY_COLUMNS,
            "type": "tree",
            "source": {"name": "pim"},
            "modifications": [],
            "pending_modifications": [
                {
                    "conditions": [
                        {"attribute": "dbh", "operator": "lt", "value": 3.0}
                    ],
                    "actions": [{"modifier": "remove"}],
                },
                {
                    "conditions": [
                        {"attribute": "height", "operator": "gt", "value": 25.0}
                    ],
                    "actions": [
                        {"attribute": "height", "modifier": "multiply", "value": 0.9}
                    ],
                },
            ],
        }
        mock_load.return_value = sample_ddf
        mock_save.return_value = ("gs://test-bucket/inventory-id", {}, None)
        progress = MagicMock()

        apply_in_place_modifications(inventory, domain_gdf, progress)

        result_df = mock_save.call_args[0][1].compute()
        assert (result_df["dbh"] >= 3.0).all()
        assert len(result_df) < len(sample_ddf.compute())
