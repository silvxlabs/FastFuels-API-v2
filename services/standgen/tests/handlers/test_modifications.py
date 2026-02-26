"""
Unit tests for standgen/handlers/modifications.py

Tests the modifications handler with mocked I/O (GCS, Firestore).
Verifies the handler correctly loads source data, applies modifications,
saves results, and returns georeference.
"""

from unittest.mock import MagicMock, patch

import dask.dataframe as dd
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box
from standgen.handlers.modifications import handle_modifications


@pytest.fixture
def sample_source_ddf():
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
    """Domain GeoDataFrame for georeference computation."""
    return gpd.GeoDataFrame(
        geometry=[box(500000, 5200000, 501000, 5201000)],
        crs="EPSG:32611",
    )


@pytest.fixture
def base_inventory():
    """Base inventory document."""
    return {
        "id": "new-inventory-id",
        "domain_id": "domain-123",
        "source": {
            "name": "modifications",
            "source_inventory_id": "source-inventory-id",
            "modifications": [
                {
                    "conditions": [
                        {"attribute": "dbh", "operator": "lt", "value": 5.0}
                    ],
                    "actions": [{"modifier": "remove"}],
                }
            ],
        },
    }


class TestHandleModifications:
    @patch("standgen.handlers.modifications.save_parquet")
    @patch("standgen.handlers.modifications.load_inventory_parquet")
    def test_handler_loads_saves_and_returns_georeference(
        self, mock_load, mock_save, base_inventory, sample_source_ddf, domain_gdf
    ):
        """Handler loads source, applies mods, saves, and returns georeference."""
        mock_load.return_value = sample_source_ddf
        progress = MagicMock()

        result = handle_modifications(
            base_inventory,
            base_inventory["source"],
            domain_gdf,
            progress,
        )

        # Verify load was called with source inventory ID
        mock_load.assert_called_once_with("source-inventory-id")

        # Verify save was called with new inventory ID
        mock_save.assert_called_once()
        save_args = mock_save.call_args
        assert save_args[0][0] == "new-inventory-id"
        # Second arg should be a dask DataFrame
        saved_ddf = save_args[0][1]
        assert isinstance(saved_ddf, dd.DataFrame)

        # Verify georeference
        assert "georeference" in result
        geo = result["georeference"]
        assert "crs" in geo
        assert "bounds" in geo

    @patch("standgen.handlers.modifications.save_parquet")
    @patch("standgen.handlers.modifications.load_inventory_parquet")
    def test_handler_applies_remove_modification(
        self, mock_load, mock_save, base_inventory, sample_source_ddf, domain_gdf
    ):
        """Handler correctly removes trees matching the condition."""
        mock_load.return_value = sample_source_ddf
        progress = MagicMock()

        handle_modifications(
            base_inventory,
            base_inventory["source"],
            domain_gdf,
            progress,
        )

        # Get the saved dask DataFrame and compute it
        saved_ddf = mock_save.call_args[0][1]
        result_df = saved_ddf.compute()

        # All remaining trees should have dbh >= 5.0
        assert (result_df["dbh"] >= 5.0).all()

        # Should have fewer rows than source
        original_count = len(sample_source_ddf.compute())
        assert len(result_df) < original_count

    @patch("standgen.handlers.modifications.save_parquet")
    @patch("standgen.handlers.modifications.load_inventory_parquet")
    def test_handler_reports_progress(
        self, mock_load, mock_save, base_inventory, sample_source_ddf, domain_gdf
    ):
        """Handler calls progress callback at expected stages."""
        mock_load.return_value = sample_source_ddf
        progress = MagicMock()

        handle_modifications(
            base_inventory,
            base_inventory["source"],
            domain_gdf,
            progress,
        )

        # Verify progress was called with expected messages
        calls = [call[0] for call in progress.call_args_list]
        messages = [c[0] for c in calls]

        assert "Loading source inventory..." in messages
        assert "Applying modifications..." in messages
        assert "Writing modified inventory..." in messages
        assert "Complete" in messages

    @patch("standgen.handlers.modifications.save_parquet")
    @patch("standgen.handlers.modifications.load_inventory_parquet")
    def test_handler_with_multiply_modification(
        self, mock_load, mock_save, domain_gdf, sample_source_ddf
    ):
        """Handler applies non-remove modifications correctly."""
        inventory = {
            "id": "new-inv-id",
            "domain_id": "domain-123",
            "source": {
                "name": "modifications",
                "source_inventory_id": "source-inv-id",
                "modifications": [
                    {
                        "conditions": [
                            {"attribute": "height", "operator": "gt", "value": 20.0}
                        ],
                        "actions": [
                            {
                                "attribute": "height",
                                "modifier": "multiply",
                                "value": 0.9,
                            }
                        ],
                    }
                ],
            },
        }

        mock_load.return_value = sample_source_ddf
        progress = MagicMock()

        handle_modifications(inventory, inventory["source"], domain_gdf, progress)

        saved_ddf = mock_save.call_args[0][1]
        result_df = saved_ddf.compute()

        # Row count should be unchanged (multiply doesn't remove rows)
        assert len(result_df) == len(sample_source_ddf.compute())

    @patch("standgen.handlers.modifications.save_parquet")
    @patch("standgen.handlers.modifications.load_inventory_parquet")
    def test_handler_with_multiple_modifications(
        self, mock_load, mock_save, domain_gdf, sample_source_ddf
    ):
        """Handler applies multiple modifications sequentially."""
        inventory = {
            "id": "new-inv-id",
            "domain_id": "domain-123",
            "source": {
                "name": "modifications",
                "source_inventory_id": "source-inv-id",
                "modifications": [
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
                            {
                                "attribute": "height",
                                "modifier": "multiply",
                                "value": 0.9,
                            }
                        ],
                    },
                ],
            },
        }

        mock_load.return_value = sample_source_ddf
        progress = MagicMock()

        handle_modifications(inventory, inventory["source"], domain_gdf, progress)

        saved_ddf = mock_save.call_args[0][1]
        result_df = saved_ddf.compute()

        # Small trees removed
        assert (result_df["dbh"] >= 3.0).all()
        # Fewer rows than original
        assert len(result_df) < len(sample_source_ddf.compute())

    @patch("standgen.handlers.modifications.save_parquet")
    @patch("standgen.handlers.modifications.load_inventory_parquet")
    def test_handler_georeference_from_domain(
        self, mock_load, mock_save, base_inventory, sample_source_ddf, domain_gdf
    ):
        """Georeference should be computed from the domain GeoDataFrame."""
        mock_load.return_value = sample_source_ddf
        progress = MagicMock()

        result = handle_modifications(
            base_inventory,
            base_inventory["source"],
            domain_gdf,
            progress,
        )

        geo = result["georeference"]
        assert "32611" in geo["crs"]
        bounds = geo["bounds"]
        assert bounds[0] == pytest.approx(500000.0)
        assert bounds[1] == pytest.approx(5200000.0)
        assert bounds[2] == pytest.approx(501000.0)
        assert bounds[3] == pytest.approx(5201000.0)
