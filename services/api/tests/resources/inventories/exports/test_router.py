"""
Integration tests for api/v2/resources/inventories/exports/router.py

These tests make real HTTP requests to the API and interact with Firestore.
"""

import pytest

from lib.config import EXPORTS_COLLECTION, INVENTORIES_COLLECTION
from tests.fixtures import make_inventory_data

# Fixtures


@pytest.fixture(scope="session")
def completed_inventory(firestore_client, domain_for_testing):
    """A completed inventory with georeference and columns, for export tests."""
    inventory_data = make_inventory_data(
        domain_id=domain_for_testing["id"],
        name="Completed inventory for export",
        status="completed",
        georeference={
            "crs": "EPSG:32611",
            "bounds": (500000.0, 5200000.0, 501000.0, 5201000.0),
        },
    )
    doc_ref = firestore_client.collection(INVENTORIES_COLLECTION).document(
        inventory_data["id"]
    )
    doc_ref.set(inventory_data)
    yield inventory_data
    doc_ref.delete()


@pytest.fixture(scope="session")
def pending_inventory(firestore_client, domain_for_testing):
    """A pending inventory (not yet completed), for validation tests."""
    inventory_data = make_inventory_data(
        domain_id=domain_for_testing["id"],
        name="Pending inventory",
        status="pending",
    )
    doc_ref = firestore_client.collection(INVENTORIES_COLLECTION).document(
        inventory_data["id"]
    )
    doc_ref.set(inventory_data)
    yield inventory_data
    doc_ref.delete()


def export_route(domain_id, inventory_id, fmt):
    """POST /domains/{domain_id}/inventories/{inventory_id}/exports/{format}"""
    return f"/domains/{domain_id}/inventories/{inventory_id}/exports/{fmt}"


def cleanup_export(firestore_client, export_id):
    """Delete an export document created during testing."""
    firestore_client.collection(EXPORTS_COLLECTION).document(export_id).delete()


class TestCreateInventoryExport:
    """Test POST /domains/{domain_id}/inventories/{inventory_id}/exports/{format}."""

    @pytest.mark.parametrize("fmt", ["parquet", "csv", "geojson", "geopackage"])
    def test_create_export_each_format(
        self, client, firestore_client, domain_for_testing, completed_inventory, fmt
    ):
        """Create export for each supported format — 201, correct source fields."""
        response = client.post(
            export_route(domain_for_testing["id"], completed_inventory["id"], fmt),
            json={
                "name": f"{fmt} export",
                "tags": ["test"],
            },
        )

        assert response.status_code == 201

        data = response.json()
        assert data["status"] == "pending"
        assert data["domain_id"] == domain_for_testing["id"]
        assert data["source"]["name"] == fmt
        assert data["source"]["inventory_id"] == completed_inventory["id"]
        assert data["source"]["columns"] is None
        assert data["source"]["crs"] == "EPSG:32611"
        assert data["name"] == f"{fmt} export"
        assert data["tags"] == ["test"]
        assert data["signed_url"] is None
        assert "id" in data
        assert "created_on" in data

        cleanup_export(firestore_client, data["id"])

    def test_create_with_column_subset(
        self, client, firestore_client, domain_for_testing, completed_inventory
    ):
        """Create export with column subset — 201, source.columns matches request."""
        response = client.post(
            export_route(domain_for_testing["id"], completed_inventory["id"], "csv"),
            json={
                "columns": ["x", "y", "dbh"],
            },
        )

        assert response.status_code == 201

        data = response.json()
        assert data["source"]["columns"] == ["x", "y", "dbh"]

        cleanup_export(firestore_client, data["id"])

    def test_create_minimal(
        self, client, firestore_client, domain_for_testing, completed_inventory
    ):
        """Minimal request with empty body."""
        response = client.post(
            export_route(
                domain_for_testing["id"], completed_inventory["id"], "parquet"
            ),
            json={},
        )

        assert response.status_code == 201

        data = response.json()
        assert data["name"] == ""
        assert data["description"] == ""
        assert data["tags"] == []
        assert data["source"]["columns"] is None

        cleanup_export(firestore_client, data["id"])

    def test_inventory_not_found_returns_404(self, client, domain_for_testing):
        """Export of non-existent inventory returns 404."""
        response = client.post(
            export_route(
                domain_for_testing["id"],
                "00000000000000000000000000000000",
                "csv",
            ),
            json={},
        )
        assert response.status_code == 404

    def test_inventory_not_completed_returns_422(
        self, client, domain_for_testing, pending_inventory
    ):
        """Export of a pending inventory returns 422."""
        response = client.post(
            export_route(domain_for_testing["id"], pending_inventory["id"], "csv"),
            json={},
        )
        assert response.status_code == 422

    def test_invalid_column_returns_422(
        self, client, domain_for_testing, completed_inventory
    ):
        """Export with non-existent column returns 422."""
        response = client.post(
            export_route(domain_for_testing["id"], completed_inventory["id"], "csv"),
            json={
                "columns": ["nonexistent_column"],
            },
        )
        assert response.status_code == 422

    def test_invalid_format_returns_422(
        self, client, domain_for_testing, completed_inventory
    ):
        """Export with unsupported format returns 422."""
        response = client.post(
            export_route(domain_for_testing["id"], completed_inventory["id"], "xlsx"),
            json={},
        )
        assert response.status_code == 422

    def test_wrong_domain_returns_404(
        self, client, domain_with_different_owner, completed_inventory
    ):
        """Export inventory from wrong domain returns 404."""
        response = client.post(
            export_route(
                domain_with_different_owner["id"],
                completed_inventory["id"],
                "csv",
            ),
            json={},
        )
        assert response.status_code == 404
