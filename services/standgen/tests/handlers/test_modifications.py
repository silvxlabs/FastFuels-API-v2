"""
Unit tests for standgen/handlers/modifications.py

The in-place modifications handler resolves spatial conditions once, then hands a
per-partition transform to ``write_changed_partitions`` (which rewrites only the
partitions whose content changes). These tests mock the storage call and
exercise the captured transform against a sample partition.
"""

from unittest.mock import MagicMock, patch

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box
from standgen.handlers.modifications import apply_in_place_modifications


@pytest.fixture
def sample_partition() -> pd.DataFrame:
    """A single inventory partition, as read off Parquet."""
    rng = np.random.default_rng(42)
    n = 50
    return pd.DataFrame(
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
    @patch("standgen.handlers.modifications.write_changed_partitions")
    def test_keys_off_inventory_id_and_carries_georeference(
        self, mock_write, base_inventory, domain_gdf
    ):
        """Writes against the inventory's own ID; returns the existing
        georeference unchanged (modifications never move the footprint)."""
        result = apply_in_place_modifications(base_inventory, domain_gdf, MagicMock())

        mock_write.assert_called_once()
        assert mock_write.call_args[0][0] == "inventory-id"
        assert callable(mock_write.call_args[0][1])
        assert result["georeference"] == base_inventory["georeference"]

    @patch("standgen.handlers.modifications.write_changed_partitions")
    def test_transform_applies_pending_delta(
        self, mock_write, base_inventory, domain_gdf, sample_partition
    ):
        """The captured transform applies only the pending delta (remove dbh<5)."""
        apply_in_place_modifications(base_inventory, domain_gdf, MagicMock())

        transform = mock_write.call_args[0][1]
        out = transform(sample_partition.copy())

        assert (out["dbh"] >= 5.0).all()
        assert len(out) < len(sample_partition)

    @patch("standgen.handlers.modifications.write_changed_partitions")
    def test_reports_progress(self, mock_write, base_inventory, domain_gdf):
        progress = MagicMock()

        apply_in_place_modifications(base_inventory, domain_gdf, progress)

        messages = [c[0][0] for c in progress.call_args_list]
        assert "Applying modifications..." in messages
        assert "Writing modified inventory..." in messages
        assert "Complete" in messages

    @patch("standgen.handlers.modifications.write_changed_partitions")
    def test_multiply_delta_preserves_row_count(
        self, mock_write, domain_gdf, sample_partition
    ):
        inventory = {
            "id": "inventory-id",
            "domain_id": "domain-123",
            "georeference": {"crs": "EPSG:32611", "bounds": [0, 0, 1, 1]},
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

        apply_in_place_modifications(inventory, domain_gdf, MagicMock())

        transform = mock_write.call_args[0][1]
        out = transform(sample_partition.copy())
        assert len(out) == len(sample_partition)

    @patch("standgen.handlers.modifications.write_changed_partitions")
    def test_multiple_pending_modifications_applied_sequentially(
        self, mock_write, domain_gdf, sample_partition
    ):
        inventory = {
            "id": "inventory-id",
            "domain_id": "domain-123",
            "georeference": {"crs": "EPSG:32611", "bounds": [0, 0, 1, 1]},
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

        apply_in_place_modifications(inventory, domain_gdf, MagicMock())

        transform = mock_write.call_args[0][1]
        out = transform(sample_partition.copy())
        assert (out["dbh"] >= 3.0).all()
        assert len(out) < len(sample_partition)
