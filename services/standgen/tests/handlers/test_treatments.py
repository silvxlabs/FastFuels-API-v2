"""
Unit tests for standgen/handlers/treatments.py

Tests the in-place treatments handler with mocked I/O (GCS). Verifies it loads
the inventory's own data, applies only the pending treatment delta, replaces the
data in place, and carries over the existing georeference.
"""

from unittest.mock import MagicMock, patch

import dask.dataframe as dd
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box
from standgen.handlers.treatments import apply_in_place_treatments

from lib.errors import ProcessingError

from .conftest import BASE_INVENTORY_COLUMNS


@pytest.fixture
def sample_ddf():
    """Sample dask DataFrame that load_inventory_parquet would return."""
    rng = np.random.default_rng(42)
    n = 200
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
    """Domain GeoDataFrame (used for basal-area area sizing / spatial conditions)."""
    return gpd.GeoDataFrame(
        geometry=[box(500000, 5200000, 501000, 5201000)],
        crs="EPSG:32611",
    )


@pytest.fixture
def small_domain_gdf():
    """A 10 m × 10 m domain (0.01 ha). A small area makes a per-hectare basal-area
    target a small absolute m² target, so an inventory-wide thin actually reduces
    the synthetic stand (whose trees are sized for tests, not realistic density)."""
    return gpd.GeoDataFrame(geometry=[box(0, 0, 10, 10)], crs="EPSG:32611")


@pytest.fixture
def base_inventory():
    """A completed inventory with one pending diameter treatment delta to apply."""
    return {
        "id": "inventory-id",
        "domain_id": "domain-123",
        "georeference": {
            "crs": "EPSG:32611",
            "bounds": [500000.0, 5200000.0, 501000.0, 5201000.0],
        },
        "columns": BASE_INVENTORY_COLUMNS,
        "source": {"name": "pim", "source_pim_grid_id": "grid-id", "seed": 1},
        "treatments": [{"metric": "diameter", "method": "from_below", "value": 10.0}],
        "pending_treatments": [
            {"metric": "diameter", "method": "from_below", "value": 10.0}
        ],
    }


class TestApplyInPlaceTreatments:
    @patch("standgen.handlers.treatments.save_parquet_replace_with_summary")
    @patch("standgen.handlers.treatments.load_inventory_parquet")
    def test_loads_own_data_replaces_in_place_and_carries_georeference(
        self, mock_load, mock_save, base_inventory, sample_ddf, domain_gdf
    ):
        """Loads the inventory's own data, replaces it in place, returns the
        existing georeference unchanged."""
        mock_load.return_value = sample_ddf
        mock_save.return_value = ("gs://test-bucket/inventory-id", {})
        progress = MagicMock()

        result = apply_in_place_treatments(base_inventory, domain_gdf, progress)

        # Load and replace both key off the inventory's own ID (in place);
        # columns are passed as the third argument for summary computation.
        mock_load.assert_called_once_with("inventory-id")
        mock_save.assert_called_once()
        assert mock_save.call_args[0][0] == "inventory-id"
        assert isinstance(mock_save.call_args[0][1], dd.DataFrame)
        assert mock_save.call_args[0][2] == base_inventory["columns"]
        assert "columns" in result

        # Georeference is carried over verbatim — not recomputed.
        assert result["georeference"] == base_inventory["georeference"]

    @patch("standgen.handlers.treatments.save_parquet_replace_with_summary")
    @patch("standgen.handlers.treatments.load_inventory_parquet")
    def test_applies_only_the_pending_delta(
        self, mock_load, mock_save, base_inventory, sample_ddf, domain_gdf
    ):
        """Only pending_treatments is applied to the current data."""
        mock_load.return_value = sample_ddf
        mock_save.return_value = ("gs://test-bucket/inventory-id", {})
        progress = MagicMock()

        apply_in_place_treatments(base_inventory, domain_gdf, progress)

        result_df = mock_save.call_args[0][1].compute()
        # The pending diameter thin-from-below removes trees below 10 cm dbh.
        assert (result_df["dbh"] >= 10.0).all()
        assert len(result_df) < len(sample_ddf.compute())

    @patch("standgen.handlers.treatments.save_parquet_replace_with_summary")
    @patch("standgen.handlers.treatments.load_inventory_parquet")
    def test_reports_progress(
        self, mock_load, mock_save, base_inventory, sample_ddf, domain_gdf
    ):
        mock_load.return_value = sample_ddf
        mock_save.return_value = ("gs://test-bucket/inventory-id", {})
        progress = MagicMock()

        apply_in_place_treatments(base_inventory, domain_gdf, progress)

        messages = [c[0][0] for c in progress.call_args_list]
        assert "Loading inventory..." in messages
        assert "Applying treatments..." in messages
        assert "Writing treated inventory..." in messages
        assert "Complete" in messages

    @patch("standgen.handlers.treatments.save_parquet_replace_with_summary")
    @patch("standgen.handlers.treatments.load_inventory_parquet")
    def test_basal_area_proportional_reduces_stand(
        self, mock_load, mock_save, small_domain_gdf, sample_ddf
    ):
        """An inventory-wide proportional basal-area thin reduces the stand. This
        exercises the materialized basal-area path and the seed plumbing."""
        inventory = {
            "id": "inventory-id",
            "domain_id": "domain-123",
            "georeference": {"crs": "EPSG:32611", "bounds": [0, 0, 1, 1]},
            "columns": BASE_INVENTORY_COLUMNS,
            "source": {"name": "pim", "seed": 7},
            "treatments": [],
            "pending_treatments": [
                {"metric": "basal_area", "method": "proportional", "value": 5.0}
            ],
        }
        mock_load.return_value = sample_ddf
        mock_save.return_value = ("gs://test-bucket/inventory-id", {})
        progress = MagicMock()

        apply_in_place_treatments(inventory, small_domain_gdf, progress)

        result_df = mock_save.call_args[0][1].compute()
        assert len(result_df) < len(sample_ddf.compute())

    @patch("standgen.handlers.treatments.save_parquet_replace_with_summary")
    @patch("standgen.handlers.treatments.load_inventory_parquet")
    def test_multiple_pending_treatments_applied_sequentially(
        self, mock_load, mock_save, domain_gdf, sample_ddf
    ):
        inventory = {
            "id": "inventory-id",
            "domain_id": "domain-123",
            "georeference": {"crs": "EPSG:32611", "bounds": [0, 0, 1, 1]},
            "columns": BASE_INVENTORY_COLUMNS,
            "source": {"name": "pim", "seed": 3},
            "treatments": [],
            "pending_treatments": [
                {"metric": "diameter", "method": "from_below", "value": 5.0},
                {"metric": "diameter", "method": "from_above", "value": 45.0},
            ],
        }
        mock_load.return_value = sample_ddf
        mock_save.return_value = ("gs://test-bucket/inventory-id", {})
        progress = MagicMock()

        apply_in_place_treatments(inventory, domain_gdf, progress)

        result_df = mock_save.call_args[0][1].compute()
        # Both diameter cutoffs applied: 5.0 <= dbh <= 45.0.
        assert (result_df["dbh"] >= 5.0).all()
        assert (result_df["dbh"] <= 45.0).all()
        assert len(result_df) < len(sample_ddf.compute())

    @patch("standgen.handlers.treatments.save_parquet_replace_with_summary")
    @patch("standgen.handlers.treatments.load_inventory_parquet")
    def test_data_without_dbh_raises_actionable_error(
        self, mock_load, mock_save, base_inventory, domain_gdf
    ):
        """Data with no dbh column (e.g. a CHM-derived inventory whose document
        metadata predates the column fix) fails with an actionable
        ProcessingError before any compute or write — not a KeyError."""
        df = pd.DataFrame(
            {
                "x": [500000.0, 500500.0],
                "y": [5200000.0, 5200500.0],
                "height": [12.0, 18.0],
            }
        )
        mock_load.return_value = dd.from_pandas(df, npartitions=1)
        mock_save.return_value = ("gs://test-bucket/inventory-id", {})
        progress = MagicMock()

        with pytest.raises(ProcessingError) as exc_info:
            apply_in_place_treatments(base_inventory, domain_gdf, progress)

        assert exc_info.value.code == "TREATMENTS_REQUIRE_DBH"
        mock_save.assert_not_called()
