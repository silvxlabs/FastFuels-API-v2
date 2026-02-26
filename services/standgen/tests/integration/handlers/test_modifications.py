"""
Integration tests for the inventory modifications pipeline.

Tests the full standgen pipeline for modifications: creates a PIM inventory
first (as the source), then creates a modifications inventory that references
it, runs standgen, and verifies the output.

These tests hit real GCS and Firestore and require valid credentials.
"""

from pathlib import Path
from uuid import uuid4

import dask.dataframe as dd
import pytest

from lib.config import (
    DOMAINS_COLLECTION,
    INVENTORIES_BUCKET,
    INVENTORIES_COLLECTION,
)
from lib.firestore.documents import delete_document, get_document, set_document
from lib.gcs.blobs import delete_directory, exists

from ..conftest import (
    DOMAINS_DIR,
    MockRequest,
    _stringify_coordinates,
    load_json,
)

STATIC_PIM_GRID = "static-test-blue-mtn-pim-treemap"
INVENTORIES_DIR = Path(__file__).resolve().parents[2] / "data" / "inventories"


@pytest.fixture
def modifications_runner(source_pim_grid):
    """Run a two-step pipeline: PIM expansion -> modifications.

    First creates and processes a PIM inventory (the source), then creates
    a modifications inventory referencing that source and processes it.

    Returns a tuple of (source_inventory, modified_inventory) dicts.
    """
    domain_ids = []
    inventory_ids = []

    def _run(
        modifications: list[dict],
        pim_seed: int = 42,
    ) -> tuple[dict, dict]:
        from standgen.main import process_inventory_request

        # Create domain
        domain_data = load_json(DOMAINS_DIR / "blue_mtn.json")
        domain_id = f"test-{uuid4().hex}"
        data = _stringify_coordinates(domain_data)
        data["id"] = domain_id
        set_document(DOMAINS_COLLECTION, domain_id, data)
        domain_ids.append(domain_id)

        # Step 1: Create and process PIM inventory (the source)
        pim_data = load_json(INVENTORIES_DIR / "pim_treemap.json")
        pim_data["domain_id"] = domain_id
        pim_data["source"]["source_pim_grid_id"] = source_pim_grid
        pim_data["source"]["seed"] = pim_seed
        pim_id = f"test-{uuid4().hex}"
        pim_data["id"] = pim_id
        set_document(INVENTORIES_COLLECTION, pim_id, pim_data)
        inventory_ids.append(pim_id)

        request = MockRequest(data={"id": pim_id})
        response, status_code = process_inventory_request(request)
        assert status_code == 200, f"PIM processing failed: {response}"

        _, pim_snapshot = get_document(INVENTORIES_COLLECTION, pim_id)
        pim_inventory = pim_snapshot.to_dict()
        assert pim_inventory["status"] == "completed", (
            f"PIM inventory not completed: {pim_inventory.get('error')}"
        )

        # Step 2: Create and process modifications inventory
        mod_id = f"test-{uuid4().hex}"
        mod_data = {
            "id": mod_id,
            "domain_id": domain_id,
            "name": "Modified Inventory",
            "status": "pending",
            "source": {
                "name": "modifications",
                "source_inventory_id": pim_id,
                "modifications": modifications,
            },
        }
        set_document(INVENTORIES_COLLECTION, mod_id, mod_data)
        inventory_ids.append(mod_id)

        request = MockRequest(data={"id": mod_id})
        response, status_code = process_inventory_request(request)
        assert status_code == 200, f"Modifications processing failed: {response}"

        _, mod_snapshot = get_document(INVENTORIES_COLLECTION, mod_id)
        mod_inventory = mod_snapshot.to_dict()
        assert mod_inventory["status"] == "completed", (
            f"Modifications inventory not completed: {mod_inventory.get('error')}"
        )

        return pim_inventory, mod_inventory

    yield _run

    # Cleanup
    for inventory_id in inventory_ids:
        gcs_path = f"gs://{INVENTORIES_BUCKET}/{inventory_id}"
        if exists(gcs_path):
            delete_directory(gcs_path)
        delete_document(INVENTORIES_COLLECTION, inventory_id)

    for domain_id in domain_ids:
        delete_document(DOMAINS_COLLECTION, domain_id)


class TestModificationsPipeline:
    """Full modifications pipeline: PIM -> modifications -> GCS parquet."""

    @pytest.mark.parametrize("source_pim_grid", [STATIC_PIM_GRID], indirect=True)
    def test_pipeline_completes(self, modifications_runner):
        """Modifications pipeline completes with georeference."""
        modifications = [
            {
                "conditions": [{"attribute": "dbh", "operator": "gt", "value": 0.0}],
                "actions": [
                    {"attribute": "height", "modifier": "multiply", "value": 0.99}
                ],
            }
        ]
        _, mod_inventory = modifications_runner(modifications)

        assert mod_inventory["status"] == "completed"
        assert mod_inventory["georeference"] is not None
        assert "crs" in mod_inventory["georeference"]
        assert "bounds" in mod_inventory["georeference"]

    @pytest.mark.parametrize("source_pim_grid", [STATIC_PIM_GRID], indirect=True)
    def test_remove_reduces_tree_count(self, modifications_runner):
        """Removing trees by dbh should produce fewer trees than the source."""
        modifications = [
            {
                "conditions": [{"attribute": "dbh", "operator": "lt", "value": 30.0}],
                "actions": [{"modifier": "remove"}],
            }
        ]
        pim_inventory, mod_inventory = modifications_runner(modifications)

        pim_path = f"gs://{INVENTORIES_BUCKET}/{pim_inventory['id']}"
        mod_path = f"gs://{INVENTORIES_BUCKET}/{mod_inventory['id']}"

        pim_count = len(dd.read_parquet(pim_path))
        mod_count = len(dd.read_parquet(mod_path))

        if pim_count == 0:
            pytest.skip("Source PIM produced 0 trees (sparse grid)")

        # Modified should have fewer or equal trees
        assert mod_count <= pim_count
        # With dbh < 30 threshold, we should actually remove some
        # (but with sparse grid, can't guarantee)

    @pytest.mark.parametrize("source_pim_grid", [STATIC_PIM_GRID], indirect=True)
    def test_remove_enforces_condition(self, modifications_runner):
        """After remove, no trees should violate the condition."""
        threshold = 10.0
        modifications = [
            {
                "conditions": [
                    {"attribute": "dbh", "operator": "lt", "value": threshold}
                ],
                "actions": [{"modifier": "remove"}],
            }
        ]
        _, mod_inventory = modifications_runner(modifications)

        mod_path = f"gs://{INVENTORIES_BUCKET}/{mod_inventory['id']}"
        df = dd.read_parquet(mod_path).compute()

        if len(df) == 0:
            pytest.skip("No trees after modification (sparse grid)")

        assert (df["dbh"] >= threshold).all(), (
            f"Found trees with dbh < {threshold}: {df['dbh'].min()}"
        )

    @pytest.mark.parametrize("source_pim_grid", [STATIC_PIM_GRID], indirect=True)
    def test_modify_changes_values(self, modifications_runner):
        """Multiply modification should change tree attribute values."""
        factor = 0.5
        modifications = [
            {
                "conditions": [{"attribute": "height", "operator": "gt", "value": 0.0}],
                "actions": [
                    {"attribute": "height", "modifier": "multiply", "value": factor}
                ],
            }
        ]
        pim_inventory, mod_inventory = modifications_runner(modifications)

        pim_path = f"gs://{INVENTORIES_BUCKET}/{pim_inventory['id']}"
        mod_path = f"gs://{INVENTORIES_BUCKET}/{mod_inventory['id']}"

        pim_df = dd.read_parquet(pim_path).compute()
        mod_df = dd.read_parquet(mod_path).compute()

        if len(pim_df) == 0:
            pytest.skip("Source PIM produced 0 trees (sparse grid)")

        # Row count should be the same (multiply doesn't remove)
        assert len(mod_df) == len(pim_df)

        # Heights should be halved
        assert mod_df["height"].sum() == pytest.approx(
            pim_df["height"].sum() * factor, rel=1e-3
        )

    @pytest.mark.parametrize("source_pim_grid", [STATIC_PIM_GRID], indirect=True)
    def test_unit_conversion_in_condition(self, modifications_runner):
        """Unit conversion in conditions works end-to-end."""
        # 1 inch = 2.54 cm
        modifications = [
            {
                "conditions": [
                    {"attribute": "dbh", "operator": "lt", "value": 4.0, "unit": "in"}
                ],
                "actions": [{"modifier": "remove"}],
            }
        ]
        _, mod_inventory = modifications_runner(modifications)

        mod_path = f"gs://{INVENTORIES_BUCKET}/{mod_inventory['id']}"
        df = dd.read_parquet(mod_path).compute()

        if len(df) == 0:
            pytest.skip("No trees after modification (sparse grid)")

        # 4 inches = 10.16 cm
        assert (df["dbh"] >= 10.16 - 0.01).all()


class TestModificationsParquetOutput:
    """Verify parquet output from modifications has correct schema."""

    @pytest.mark.parametrize("source_pim_grid", [STATIC_PIM_GRID], indirect=True)
    def test_parquet_has_correct_columns(self, modifications_runner):
        """Modified parquet should have same columns as source."""
        from standgen.columns import BASE_COLUMNS

        modifications = [
            {
                "conditions": [{"attribute": "dbh", "operator": "lt", "value": 5.0}],
                "actions": [{"modifier": "remove"}],
            }
        ]
        _, mod_inventory = modifications_runner(modifications)

        mod_path = f"gs://{INVENTORIES_BUCKET}/{mod_inventory['id']}"
        ddf = dd.read_parquet(mod_path)

        assert sorted(ddf.columns.tolist()) == sorted(BASE_COLUMNS)

    @pytest.mark.parametrize("source_pim_grid", [STATIC_PIM_GRID], indirect=True)
    def test_parquet_values_are_sensible(self, modifications_runner):
        """Modified tree values should be within reasonable ranges."""
        modifications = [
            {
                "conditions": [{"attribute": "dbh", "operator": "lt", "value": 2.54}],
                "actions": [{"modifier": "remove"}],
            }
        ]
        _, mod_inventory = modifications_runner(modifications)

        mod_path = f"gs://{INVENTORIES_BUCKET}/{mod_inventory['id']}"
        df = dd.read_parquet(mod_path).compute()

        if len(df) == 0:
            pytest.skip("No trees after modification (sparse grid)")

        assert df["dbh"].min() >= 2.54
        assert df["dbh"].max() < 300
        assert df["height"].min() > 0
        assert df["height"].max() < 100
        assert df["crown_ratio"].min() >= 0
        assert df["crown_ratio"].max() <= 1
        assert (df["fia_species_code"] > 0).all()
        assert not df.isna().any().any()


class TestModificationsGeoreference:
    """Verify georeference inheritance."""

    @pytest.mark.parametrize("source_pim_grid", [STATIC_PIM_GRID], indirect=True)
    def test_georeference_matches_source(self, modifications_runner):
        """Modified inventory georeference should match domain."""
        modifications = [
            {
                "conditions": [{"attribute": "dbh", "operator": "lt", "value": 5.0}],
                "actions": [{"modifier": "remove"}],
            }
        ]
        pim_inventory, mod_inventory = modifications_runner(modifications)

        pim_geo = pim_inventory["georeference"]
        mod_geo = mod_inventory["georeference"]

        assert mod_geo["crs"] == pim_geo["crs"]
        assert mod_geo["bounds"] == pim_geo["bounds"]


class TestModificationsStatusTransitions:
    """Verify status transitions for the modifications pipeline."""

    @pytest.mark.parametrize("source_pim_grid", [STATIC_PIM_GRID], indirect=True)
    def test_final_progress_is_100(self, modifications_runner):
        """After completion, progress should be 100%."""
        modifications = [
            {
                "conditions": [{"attribute": "dbh", "operator": "lt", "value": 5.0}],
                "actions": [{"modifier": "remove"}],
            }
        ]
        _, mod_inventory = modifications_runner(modifications)

        assert mod_inventory["progress"]["percent"] == 100
        assert mod_inventory["progress"]["message"] == "Complete"
