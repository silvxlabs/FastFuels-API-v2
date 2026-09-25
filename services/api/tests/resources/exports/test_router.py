"""
Integration tests for api/v2/resources/exports/router.py

Tests the top-level lifecycle CRUD endpoints (GET, LIST, PATCH, DELETE).
GeoTIFF creation endpoint tests are in grids/exports/.

These tests make real HTTP requests to the API and interact with Firestore.
"""

import pytest

from lib.config import DOMAINS_COLLECTION, EXPORTS_COLLECTION
from tests.fixtures import make_domain_data, make_export_data

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
    """Test the GET /exports endpoint.

    Every test lists a fresh isolated owner, never the shared owner, whose
    exports accumulate from every suite run, including runs from branches
    whose export formats this API doesn't know yet.
    """

    @pytest.fixture
    def owner_with_export(self, isolated_owner):
        """An isolated owner's client and one export in its own domain."""
        client, owner_id, seed = isolated_owner
        domain = seed(DOMAINS_COLLECTION, make_domain_data(owner_id=owner_id))
        export = seed(
            EXPORTS_COLLECTION,
            make_export_data(
                domain_id=domain["id"],
                owner_id=owner_id,
                tags=["fixture"],
                status="completed",
            ),
        )
        return client, export

    def test_list_returns_200(self, owner_with_export):
        """List endpoint returns 200 with paginated response."""
        client, export = owner_with_export
        response = client.get(ROUTE)
        assert response.status_code == 200

        data = response.json()
        assert "exports" in data
        assert "current_page" in data
        assert "page_size" in data
        assert "total_items" in data
        assert [e["id"] for e in data["exports"]] == [export["id"]]

    def test_list_does_not_include_other_owners(
        self, owner_with_export, export_with_different_owner
    ):
        """List should not include exports from other users."""
        client, _ = owner_with_export
        response = client.get(ROUTE)
        assert response.status_code == 200

        export_ids = [e["id"] for e in response.json()["exports"]]
        assert export_with_different_owner["id"] not in export_ids

    def test_list_filter_by_domain_id(self, owner_with_export):
        """Filter exports by domain_id."""
        client, export = owner_with_export
        response = client.get(f"{ROUTE}?domain_id={export['domain_id']}")
        assert response.status_code == 200

        exports = response.json()["exports"]
        assert [e["id"] for e in exports] == [export["id"]]

    def test_list_filter_by_source_name(self, owner_with_export):
        """Filter exports by source format name."""
        client, export = owner_with_export
        response = client.get(f"{ROUTE}?source_name=geotiff")
        assert response.status_code == 200

        exports = response.json()["exports"]
        assert [e["id"] for e in exports] == [export["id"]]

    def test_list_filter_by_tag(self, owner_with_export):
        """Filter exports by tag."""
        client, export = owner_with_export
        response = client.get(f"{ROUTE}?tag=fixture")
        assert response.status_code == 200

        exports = response.json()["exports"]
        assert [e["id"] for e in exports] == [export["id"]]

    @pytest.mark.parametrize("sort_by", ["created_on", "modified_on", "name"])
    @pytest.mark.parametrize("sort_order", [None, "ascending", "descending"])
    def test_list_sorting_matrix_returns_200(
        self, owner_with_export, sort_by, sort_order
    ):
        """Every sort field/direction combination is served (issue #321)."""
        client, _ = owner_with_export
        url = f"{ROUTE}?sort_by={sort_by}"
        if sort_order:
            url += f"&sort_order={sort_order}"
        response = client.get(url)
        assert response.status_code == 200

    @pytest.mark.parametrize("sort_by", ["created_on", "modified_on", "name"])
    @pytest.mark.parametrize("sort_order", [None, "ascending", "descending"])
    def test_list_sorting_matrix_with_domain_filter_returns_200(
        self, owner_with_export, sort_by, sort_order
    ):
        """Sorting combined with the domain_id filter is served (issue #321)."""
        client, export = owner_with_export
        url = f"{ROUTE}?domain_id={export['domain_id']}&sort_by={sort_by}"
        if sort_order:
            url += f"&sort_order={sort_order}"
        response = client.get(url)
        assert response.status_code == 200

    def test_list_pagination(self, owner_with_export):
        """Pagination parameters work."""
        client, _ = owner_with_export
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
