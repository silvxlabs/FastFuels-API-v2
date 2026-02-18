"""
Integration tests for api/v2/resources/exports/router.py

Tests the top-level lifecycle CRUD endpoints (GET, LIST, PATCH, DELETE).
GeoTIFF creation endpoint tests are in grids/exports/.

These tests make real HTTP requests to the API and interact with Firestore.
"""

import pytest

from lib.config import EXPORTS_COLLECTION
from tests.fixtures import make_export_data

# Fixtures


@pytest.fixture(scope="session")
def export_in_firestore(firestore_client, domain_for_testing):
    """Create an export document directly in Firestore, yield it, then delete."""
    export_data = make_export_data(
        domain_id=domain_for_testing["id"],
        name="Test Export for GET",
        description="Created by fixture for GET endpoint tests",
        tags=["test", "fixture"],
        status="completed",
        signed_url="https://storage.googleapis.com/bucket/file.tif?X-Goog-Signature=abc",
        curl="curl -o export.tif 'https://storage.googleapis.com/bucket/file.tif?X-Goog-Signature=abc'",
    )
    doc_ref = firestore_client.collection(EXPORTS_COLLECTION).document(
        export_data["id"]
    )
    doc_ref.set(export_data)
    yield export_data
    doc_ref.delete()


@pytest.fixture(scope="session")
def export_with_different_owner(firestore_client, domain_with_different_owner):
    """Create an export owned by a different user for ownership validation tests."""
    export_data = make_export_data(
        domain_id=domain_with_different_owner["id"],
        owner_id="different-owner",
        name="Other User's Export",
    )
    doc_ref = firestore_client.collection(EXPORTS_COLLECTION).document(
        export_data["id"]
    )
    doc_ref.set(export_data)
    yield export_data
    doc_ref.delete()


ROUTE = "/exports"


# GET /exports/{export_id} Tests


class TestGetExport:
    """Test the GET /exports/{export_id} endpoint."""

    def test_get_existing_export(self, client, export_in_firestore):
        """Successfully retrieve an export that exists."""
        export_id = export_in_firestore["id"]

        response = client.get(f"{ROUTE}/{export_id}")

        assert response.status_code == 200

        data = response.json()
        assert data["id"] == export_id
        assert data["name"] == "Test Export for GET"
        assert data["description"] == "Created by fixture for GET endpoint tests"
        assert data["tags"] == ["test", "fixture"]
        assert data["status"] == "completed"
        assert (
            data["signed_url"]
            == "https://storage.googleapis.com/bucket/file.tif?X-Goog-Signature=abc"
        )
        assert data["curl"] is not None
        assert "source" in data
        assert "created_on" in data
        assert "modified_on" in data

    def test_get_nonexistent_export_returns_404(self, client):
        """Fetching a non-existent export returns 404."""
        fake_id = "00000000000000000000000000000000"
        response = client.get(f"{ROUTE}/{fake_id}")
        assert response.status_code == 404

    def test_get_export_wrong_owner_returns_404(
        self, client, export_with_different_owner
    ):
        """Fetching an export owned by another user returns 404."""
        export_id = export_with_different_owner["id"]
        response = client.get(f"{ROUTE}/{export_id}")
        assert response.status_code == 404


# GET /exports Tests


class TestListExports:
    """Test the GET /exports endpoint."""

    def test_list_returns_200(self, client, export_in_firestore):
        """List endpoint returns 200 with paginated response."""
        response = client.get(ROUTE)
        assert response.status_code == 200

        data = response.json()
        assert "exports" in data
        assert "current_page" in data
        assert "page_size" in data
        assert "total_items" in data
        assert data["total_items"] >= 1

    def test_list_does_not_include_other_owners(
        self, client, export_with_different_owner
    ):
        """List should not include exports from other users."""
        response = client.get(ROUTE)
        data = response.json()
        export_ids = [e["id"] for e in data["exports"]]
        assert export_with_different_owner["id"] not in export_ids

    def test_list_filter_by_domain_id(self, client, export_in_firestore):
        """Filter exports by domain_id."""
        domain_id = export_in_firestore["domain_id"]
        response = client.get(f"{ROUTE}?domain_id={domain_id}")
        assert response.status_code == 200

        data = response.json()
        for export in data["exports"]:
            assert export["domain_id"] == domain_id

    def test_list_filter_by_source_name(self, client, export_in_firestore):
        """Filter exports by source format name."""
        response = client.get(f"{ROUTE}?source_name=geotiff")
        assert response.status_code == 200

        data = response.json()
        for export in data["exports"]:
            assert export["source"]["name"] == "geotiff"

    def test_list_filter_by_tag(self, client, export_in_firestore):
        """Filter exports by tag."""
        response = client.get(f"{ROUTE}?tag=fixture")
        assert response.status_code == 200

        data = response.json()
        for export in data["exports"]:
            assert "fixture" in export["tags"]

    def test_list_sort_by_created_on(self, client, export_in_firestore):
        """Sort exports by created_on."""
        response = client.get(f"{ROUTE}?sort_by=created_on&sort_order=descending")
        assert response.status_code == 200

    def test_list_pagination(self, client, export_in_firestore):
        """Pagination parameters work."""
        response = client.get(f"{ROUTE}?page=0&size=1")
        assert response.status_code == 200
        data = response.json()
        assert data["page_size"] == 1
        assert len(data["exports"]) <= 1


# PATCH /exports/{export_id} Tests


class TestUpdateExport:
    """Test the PATCH /exports/{export_id} endpoint."""

    def test_update_name(self, client, export_in_firestore):
        """Update export name."""
        export_id = export_in_firestore["id"]
        response = client.patch(
            f"{ROUTE}/{export_id}",
            json={"name": "Updated Export Name"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Updated Export Name"

    def test_update_description(self, client, export_in_firestore):
        """Update export description."""
        export_id = export_in_firestore["id"]
        response = client.patch(
            f"{ROUTE}/{export_id}",
            json={"description": "Updated description"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["description"] == "Updated description"

    def test_update_tags(self, client, export_in_firestore):
        """Update export tags."""
        export_id = export_in_firestore["id"]
        response = client.patch(
            f"{ROUTE}/{export_id}",
            json={"tags": ["new-tag"]},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["tags"] == ["new-tag"]

    def test_update_nonexistent_returns_404(self, client):
        """Updating a non-existent export returns 404."""
        fake_id = "00000000000000000000000000000000"
        response = client.patch(
            f"{ROUTE}/{fake_id}",
            json={"name": "new"},
        )
        assert response.status_code == 404

    def test_update_wrong_owner_returns_404(self, client, export_with_different_owner):
        """Updating another user's export returns 404."""
        export_id = export_with_different_owner["id"]
        response = client.patch(
            f"{ROUTE}/{export_id}",
            json={"name": "new"},
        )
        assert response.status_code == 404

    def test_update_modified_on_changes(self, client, export_in_firestore):
        """modified_on is updated on patch."""
        export_id = export_in_firestore["id"]
        response = client.patch(
            f"{ROUTE}/{export_id}",
            json={"name": "Timestamp test"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["modified_on"] != export_in_firestore["modified_on"].isoformat()


# DELETE /exports/{export_id} Tests


class TestDeleteExport:
    """Test the DELETE /exports/{export_id} endpoint."""

    def test_delete_export(self, client, firestore_client, domain_for_testing):
        """Successfully delete an export."""
        export_data = make_export_data(
            domain_id=domain_for_testing["id"],
            name="Export to delete",
        )
        doc_ref = firestore_client.collection(EXPORTS_COLLECTION).document(
            export_data["id"]
        )
        doc_ref.set(export_data)

        response = client.delete(f"{ROUTE}/{export_data['id']}")
        assert response.status_code == 204

        # Verify it's gone
        response = client.get(f"{ROUTE}/{export_data['id']}")
        assert response.status_code == 404

    def test_delete_nonexistent_returns_404(self, client):
        """Deleting a non-existent export returns 404."""
        fake_id = "00000000000000000000000000000000"
        response = client.delete(f"{ROUTE}/{fake_id}")
        assert response.status_code == 404

    def test_delete_wrong_owner_returns_404(self, client, export_with_different_owner):
        """Deleting another user's export returns 404."""
        export_id = export_with_different_owner["id"]
        response = client.delete(f"{ROUTE}/{export_id}")
        assert response.status_code == 404
