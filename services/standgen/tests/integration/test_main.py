"""
Integration tests for the standgen orchestrator (main.py).

Tests the full process_inventory_request path end-to-end with real
Firestore and GCS. Verifies status transitions, error handling, and
cleanup when using real infrastructure.
"""

from uuid import uuid4

import pytest

from lib.config import (
    DOMAINS_COLLECTION,
    INVENTORIES_BUCKET,
    INVENTORIES_COLLECTION,
)
from lib.firestore.documents import delete_document, get_document, set_document
from lib.gcs.blobs import exists

from .conftest import DOMAINS_DIR, MockRequest, _stringify_coordinates, load_json

STATIC_PIM_GRID = "static-test-blue-mtn-pim-treemap"


@pytest.fixture
def firestore_domain():
    """Create a domain in Firestore with cleanup."""
    created_ids = []

    def _create(domain_file: str = "blue_mtn.json") -> str:
        domain_data = load_json(DOMAINS_DIR / domain_file)
        domain_id = f"test-{uuid4().hex}"
        data = _stringify_coordinates(domain_data)
        data["id"] = domain_id
        set_document(DOMAINS_COLLECTION, domain_id, data)
        created_ids.append(domain_id)
        return domain_id

    yield _create

    for domain_id in created_ids:
        delete_document(DOMAINS_COLLECTION, domain_id)


@pytest.fixture
def firestore_inventory():
    """Create an inventory in Firestore with cleanup (GCS + Firestore)."""
    created_ids = []

    def _create(domain_id: str, source_pim_grid_id: str, seed: int = 42) -> str:
        inventory_id = f"test-{uuid4().hex}"
        data = {
            "id": inventory_id,
            "domain_id": domain_id,
            "name": "Test Inventory",
            "status": "pending",
            "source": {
                "name": "pim",
                "source_pim_grid_id": source_pim_grid_id,
                "point_process": "inhomogeneous_poisson",
                "seed": seed,
            },
            "columns": [
                {"key": "x", "type": "continuous", "unit": "m"},
                {"key": "y", "type": "continuous", "unit": "m"},
                {"key": "fia_species_code", "type": "categorical"},
                {"key": "fia_status_code", "type": "categorical"},
                {"key": "dbh", "type": "continuous", "unit": "cm"},
                {"key": "height", "type": "continuous", "unit": "m"},
                {"key": "crown_ratio", "type": "continuous"},
            ],
        }
        set_document(INVENTORIES_COLLECTION, inventory_id, data)
        created_ids.append(inventory_id)
        return inventory_id

    yield _create

    for inventory_id in created_ids:
        # Clean up GCS parquet data if it exists
        gcs_path = f"gs://{INVENTORIES_BUCKET}/{inventory_id}"
        if exists(gcs_path):
            from lib.gcs.blobs import delete_directory

            delete_directory(gcs_path)
        delete_document(INVENTORIES_COLLECTION, inventory_id)


class TestProcessInventoryRequest:
    """End-to-end tests for the full orchestrator."""

    @pytest.mark.parametrize("source_pim_grid", [STATIC_PIM_GRID], indirect=True)
    def test_status_transitions(
        self, source_pim_grid, firestore_domain, firestore_inventory
    ):
        """Inventory status transitions: pending -> running -> completed."""
        from standgen.main import process_inventory_request

        domain_id = firestore_domain()
        inventory_id = firestore_inventory(domain_id, source_pim_grid)

        # Verify initial status
        _, snapshot = get_document(INVENTORIES_COLLECTION, inventory_id)
        assert snapshot.to_dict()["status"] == "pending"

        # Run standgen
        request = MockRequest(data={"id": inventory_id})
        response, status_code = process_inventory_request(request)

        assert status_code == 200
        assert response == "OK"

        # Verify final status
        _, snapshot = get_document(INVENTORIES_COLLECTION, inventory_id)
        inventory = snapshot.to_dict()
        assert inventory["status"] == "completed"
        assert inventory["georeference"] is not None
        columns = inventory.get("columns", [])
        assert len(columns) > 0
        for col in columns:
            assert col["summary"] is not None
            assert col["summary"]["count"] >= 0
            assert col["summary"]["null_count"] >= 0
        # The inventory Parquet's GCS footprint is recorded on completion (#342).
        assert inventory["size_bytes"] > 0

    @pytest.mark.parametrize("source_pim_grid", [STATIC_PIM_GRID], indirect=True)
    def test_parquet_written_to_gcs(
        self, source_pim_grid, firestore_domain, firestore_inventory
    ):
        """Processing should write parquet files to GCS."""
        from standgen.main import process_inventory_request

        domain_id = firestore_domain()
        inventory_id = firestore_inventory(domain_id, source_pim_grid)

        request = MockRequest(data={"id": inventory_id})
        process_inventory_request(request)

        gcs_path = f"gs://{INVENTORIES_BUCKET}/{inventory_id}"
        assert exists(gcs_path), f"Expected parquet data at {gcs_path}"

    def test_missing_domain_marks_failed(self, firestore_inventory):
        """Referencing a nonexistent domain should mark inventory as failed."""
        from standgen.main import process_inventory_request

        # Create inventory pointing to a domain that doesn't exist
        fake_domain_id = f"test-nonexistent-{uuid4().hex}"
        inventory_id = firestore_inventory(fake_domain_id, "fake-grid-id")

        request = MockRequest(data={"id": inventory_id})
        response, status_code = process_inventory_request(request)

        assert status_code == 200

        _, snapshot = get_document(INVENTORIES_COLLECTION, inventory_id)
        inventory = snapshot.to_dict()
        assert inventory["status"] == "failed"
        assert inventory["error"]["code"] == "DOMAIN_NOT_FOUND"

    @pytest.mark.parametrize("source_pim_grid", [STATIC_PIM_GRID], indirect=True)
    def test_missing_grid_marks_failed(
        self, source_pim_grid, firestore_domain, firestore_inventory
    ):
        """Referencing a nonexistent PIM grid should mark inventory as failed."""
        from standgen.main import process_inventory_request

        domain_id = firestore_domain()
        # Point to a grid ID that doesn't exist in Firestore
        inventory_id = firestore_inventory(domain_id, f"test-nonexistent-{uuid4().hex}")

        request = MockRequest(data={"id": inventory_id})
        response, status_code = process_inventory_request(request)

        assert status_code == 200

        _, snapshot = get_document(INVENTORIES_COLLECTION, inventory_id)
        inventory = snapshot.to_dict()
        assert inventory["status"] == "failed"
        assert inventory["error"]["code"] == "SOURCE_GRID_NOT_FOUND"

    def test_deleted_inventory_returns_ok(self):
        """Processing a deleted inventory should return 200 gracefully."""
        from standgen.main import process_inventory_request

        request = MockRequest(data={"id": f"test-nonexistent-{uuid4().hex}"})
        response, status_code = process_inventory_request(request)

        assert status_code == 200

    @pytest.mark.parametrize("source_pim_grid", [STATIC_PIM_GRID], indirect=True)
    def test_progress_updates(
        self, source_pim_grid, firestore_domain, firestore_inventory
    ):
        """After completion, progress should show 100%."""
        from standgen.main import process_inventory_request

        domain_id = firestore_domain()
        inventory_id = firestore_inventory(domain_id, source_pim_grid)

        request = MockRequest(data={"id": inventory_id})
        process_inventory_request(request)

        _, snapshot = get_document(INVENTORIES_COLLECTION, inventory_id)
        inventory = snapshot.to_dict()
        assert inventory["progress"]["percent"] == 100
        assert inventory["progress"]["message"] == "Complete"
