"""
Unit tests for standgen/handlers/treatments.py

Diameter treatments are row-local and go through ``write_changed_partitions``
(partial rewrite); basal-area treatments are whole-stand reductions, materialized
and written via ``write_full_partitions``. These tests mock the storage calls and
exercise either the captured per-partition transform (diameter) or the
materialized result (basal-area).
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

STORAGE = "standgen.handlers.treatments"


@pytest.fixture
def sample_partition() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 200
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
def sample_ddf(sample_partition):
    return dd.from_pandas(sample_partition, npartitions=3)


@pytest.fixture
def domain_gdf():
    return gpd.GeoDataFrame(
        geometry=[box(500000, 5200000, 501000, 5201000)], crs="EPSG:32611"
    )


@pytest.fixture
def small_domain_gdf():
    """A 10 m × 10 m domain (0.01 ha) so a per-hectare basal-area target is a
    small absolute m² target and an inventory-wide thin actually reduces the
    synthetic stand."""
    return gpd.GeoDataFrame(geometry=[box(0, 0, 10, 10)], crs="EPSG:32611")


@pytest.fixture
def base_inventory():
    return {
        "id": "inventory-id",
        "domain_id": "domain-123",
        "georeference": {
            "crs": "EPSG:32611",
            "bounds": [500000.0, 5200000.0, 501000.0, 5201000.0],
        },
        "source": {"name": "pim", "source_pim_grid_id": "grid-id", "seed": 1},
        "treatments": [{"metric": "diameter", "method": "from_below", "value": 10.0}],
        "pending_treatments": [
            {"metric": "diameter", "method": "from_below", "value": 10.0}
        ],
    }


class TestDiameterTreatments:
    @patch(f"{STORAGE}.write_full_partitions")
    @patch(f"{STORAGE}.write_changed_partitions")
    @patch(f"{STORAGE}.load_inventory_parquet")
    def test_routes_to_partial_rewrite_and_carries_georeference(
        self, mock_load, mock_changed, mock_full, base_inventory, sample_ddf, domain_gdf
    ):
        mock_load.return_value = sample_ddf

        result = apply_in_place_treatments(base_inventory, domain_gdf, MagicMock())

        mock_changed.assert_called_once()
        assert mock_changed.call_args[0][0] == "inventory-id"
        assert callable(mock_changed.call_args[0][1])
        mock_full.assert_not_called()  # diameter is per-partition, not a full rewrite
        assert result["georeference"] == base_inventory["georeference"]

    @patch(f"{STORAGE}.write_full_partitions")
    @patch(f"{STORAGE}.write_changed_partitions")
    @patch(f"{STORAGE}.load_inventory_parquet")
    def test_transform_thins_below_diameter(
        self,
        mock_load,
        mock_changed,
        mock_full,
        base_inventory,
        sample_ddf,
        sample_partition,
        domain_gdf,
    ):
        mock_load.return_value = sample_ddf
        apply_in_place_treatments(base_inventory, domain_gdf, MagicMock())

        transform = mock_changed.call_args[0][1]
        out = transform(sample_partition.copy())

        assert (out["dbh"] >= 10.0).all()
        assert len(out) < len(sample_partition)

    @patch(f"{STORAGE}.write_full_partitions")
    @patch(f"{STORAGE}.write_changed_partitions")
    @patch(f"{STORAGE}.load_inventory_parquet")
    def test_multiple_diameter_treatments_compose(
        self,
        mock_load,
        mock_changed,
        mock_full,
        sample_ddf,
        sample_partition,
        domain_gdf,
    ):
        inventory = {
            "id": "inventory-id",
            "domain_id": "domain-123",
            "georeference": {"crs": "EPSG:32611", "bounds": [0, 0, 1, 1]},
            "source": {"name": "pim", "seed": 3},
            "treatments": [],
            "pending_treatments": [
                {"metric": "diameter", "method": "from_below", "value": 5.0},
                {"metric": "diameter", "method": "from_above", "value": 45.0},
            ],
        }
        mock_load.return_value = sample_ddf
        apply_in_place_treatments(inventory, domain_gdf, MagicMock())

        transform = mock_changed.call_args[0][1]
        out = transform(sample_partition.copy())

        assert (out["dbh"] >= 5.0).all()
        assert (out["dbh"] <= 45.0).all()
        assert len(out) < len(sample_partition)

    @patch(f"{STORAGE}.write_full_partitions")
    @patch(f"{STORAGE}.write_changed_partitions")
    @patch(f"{STORAGE}.load_inventory_parquet")
    def test_reports_progress(
        self, mock_load, mock_changed, mock_full, base_inventory, sample_ddf, domain_gdf
    ):
        mock_load.return_value = sample_ddf
        progress = MagicMock()

        apply_in_place_treatments(base_inventory, domain_gdf, progress)

        messages = [c[0][0] for c in progress.call_args_list]
        assert "Applying treatments..." in messages
        assert "Writing treated inventory..." in messages
        assert "Complete" in messages


class TestDirectionalBasalArea:
    @patch(f"{STORAGE}.write_full_partitions")
    @patch(f"{STORAGE}.write_changed_partitions")
    @patch(f"{STORAGE}.load_inventory_parquet")
    def test_routes_to_partial_rewrite_via_cutoff(
        self,
        mock_load,
        mock_changed,
        mock_full,
        domain_gdf,
        sample_ddf,
        sample_partition,
    ):
        """A directional basal-area thin is reduced to a diameter cutoff and goes
        through the per-partition partial-rewrite path — not a full rewrite."""
        inventory = {
            "id": "inventory-id",
            "domain_id": "domain-123",
            "georeference": {"crs": "EPSG:32611", "bounds": [0, 0, 1, 1]},
            "source": {"name": "pim", "seed": 7},
            "treatments": [],
            "pending_treatments": [
                {"metric": "basal_area", "method": "from_below", "value": 0.05}
            ],
        }
        mock_load.return_value = sample_ddf

        apply_in_place_treatments(inventory, domain_gdf, MagicMock())

        mock_changed.assert_called_once()
        mock_full.assert_not_called()  # directional is a per-partition cutoff
        # The captured per-partition transform thins from below (removes the
        # smallest trees), so the partition shrinks and its minimum dbh rises.
        transform = mock_changed.call_args[0][1]
        out = transform(sample_partition.copy())
        assert len(out) < len(sample_partition)
        assert out["dbh"].min() > sample_partition["dbh"].min()


class TestProportionalBasalArea:
    @patch(f"{STORAGE}.write_full_partitions")
    @patch(f"{STORAGE}.write_changed_partitions")
    @patch(f"{STORAGE}.load_inventory_parquet")
    def test_routes_to_full_rewrite_and_reduces_stand(
        self, mock_load, mock_changed, mock_full, small_domain_gdf, sample_ddf
    ):
        """A proportional basal-area thin removes trees at random, so it cannot be
        a per-partition filter — it materializes the reduced stand and is written
        via the full-rewrite path. Also exercises the seed plumbing."""
        inventory = {
            "id": "inventory-id",
            "domain_id": "domain-123",
            "georeference": {"crs": "EPSG:32611", "bounds": [0, 0, 1, 1]},
            "source": {"name": "pim", "seed": 7},
            "treatments": [],
            "pending_treatments": [
                {"metric": "basal_area", "method": "proportional", "value": 5.0}
            ],
        }
        mock_load.return_value = sample_ddf

        apply_in_place_treatments(inventory, small_domain_gdf, MagicMock())

        mock_full.assert_called_once()
        assert mock_full.call_args[0][0] == "inventory-id"
        result_df = mock_full.call_args[0][1]
        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) < 200  # the stand was thinned
        mock_changed.assert_not_called()


class TestDbhValidation:
    @patch(f"{STORAGE}.write_full_partitions")
    @patch(f"{STORAGE}.write_changed_partitions")
    @patch(f"{STORAGE}.load_inventory_parquet")
    def test_data_without_dbh_raises_before_any_write(
        self, mock_load, mock_changed, mock_full, base_inventory, domain_gdf
    ):
        """Data with no dbh column fails with an actionable ProcessingError
        before any write — not a KeyError."""
        df = pd.DataFrame(
            {
                "x": [500000.0, 500500.0],
                "y": [5200000.0, 5200500.0],
                "height": [12.0, 18.0],
            }
        )
        mock_load.return_value = dd.from_pandas(df, npartitions=1)

        with pytest.raises(ProcessingError) as exc_info:
            apply_in_place_treatments(base_inventory, domain_gdf, MagicMock())

        assert exc_info.value.code == "TREATMENTS_REQUIRE_DBH"
        mock_changed.assert_not_called()
        mock_full.assert_not_called()
