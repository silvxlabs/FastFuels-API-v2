"""
Integration tests for api/v2/resources/grids/router.py

Tests the standard CRUD endpoints (GET, LIST, PATCH, DELETE).
LANDFIRE-specific endpoint tests are in the landfire/ subdirectory.

These tests make real HTTP requests to the API and interact with Firestore.
Validation edge cases are tested at the db layer (test_documents_*.py). This file
focuses on happy paths, example verification, and HTTP-specific concerns.
"""

import pytest

from lib.config import GRIDS_COLLECTION
from tests.fixtures import make_grid_data

# Fixtures


@pytest.fixture(scope="session")
def grid_in_firestore(firestore_client, domain_for_testing):
    """Create a grid document directly in Firestore, yield it, then delete."""
    grid_data = make_grid_data(
        domain_id=domain_for_testing["id"],
        name="Test Grid for GET",
        description="Created by fixture for GET endpoint tests",
        tags=["test", "fixture"],
    )
    doc_ref = firestore_client.collection(GRIDS_COLLECTION).document(grid_data["id"])
    doc_ref.set(grid_data)
    yield grid_data
    doc_ref.delete()


@pytest.fixture(scope="session")
def grid_with_different_owner(firestore_client, domain_with_different_owner):
    """Create a grid owned by a different user for ownership validation tests."""
    grid_data = make_grid_data(
        domain_id=domain_with_different_owner["id"],
        owner_id="different-owner",
        name="Other User's Grid",
    )
    doc_ref = firestore_client.collection(GRIDS_COLLECTION).document(grid_data["id"])
    doc_ref.set(grid_data)
    yield grid_data
    doc_ref.delete()


# GET /domains/{domain_id}/grids/{grid_id} Tests


class TestGetGrid:
    """Test the GET /domains/{domain_id}/grids/{grid_id} endpoint."""

    def route(self, domain_id):
        return f"/domains/{domain_id}/grids"

    def test_get_existing_grid(self, client, grid_in_firestore, domain_for_testing):
        """Successfully retrieve a grid that exists."""
        grid_id = grid_in_firestore["id"]

        response = client.get(f"{self.route(domain_for_testing['id'])}/{grid_id}")

        assert response.status_code == 200

        data = response.json()
        assert data["id"] == grid_id
        assert data["name"] == "Test Grid for GET"
        assert data["description"] == "Created by fixture for GET endpoint tests"
        assert data["tags"] == ["test", "fixture"]
        assert "source" in data
        assert "bands" in data
        assert "georeference" in data
        assert "created_on" in data
        assert "modified_on" in data

    def test_get_nonexistent_grid_returns_404(self, client, domain_for_testing):
        """Fetching a non-existent grid returns 404."""
        fake_grid_id = "00000000000000000000000000000000"

        response = client.get(f"{self.route(domain_for_testing['id'])}/{fake_grid_id}")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_get_grid_wrong_owner_returns_404(
        self, client, grid_with_different_owner, domain_with_different_owner
    ):
        """Fetching a grid owned by another user returns 404."""
        grid_id = grid_with_different_owner["id"]

        response = client.get(
            f"{self.route(domain_with_different_owner['id'])}/{grid_id}"
        )

        assert response.status_code == 404

    def test_get_grid_wrong_domain_returns_404(
        self, client, grid_in_firestore, domain_for_testing, domain_with_different_owner
    ):
        """Fetching a grid under the wrong domain returns 404."""
        grid_id = grid_in_firestore["id"]

        # Grid belongs to domain_for_testing, but we request it under a different domain
        response = client.get(
            f"{self.route(domain_with_different_owner['id'])}/{grid_id}"
        )

        assert response.status_code == 404

    def test_get_grid_excludes_owner_id(
        self, client, grid_in_firestore, domain_for_testing
    ):
        """Response should not expose the owner_id field."""
        grid_id = grid_in_firestore["id"]

        response = client.get(f"{self.route(domain_for_testing['id'])}/{grid_id}")
        assert response.status_code == 200

        data = response.json()
        assert "owner_id" not in data


# GET /domains/-/grids (Wildcard List) Tests


class TestListGridsWildcard:
    """Test GET /domains/-/grids returns grids across all domains."""

    @pytest.fixture(scope="class")
    def grids_across_domains(self, firestore_client, domain_for_testing, second_domain):
        """Grids spread across two domains, both owned by test-owner."""
        grids = []
        for domain_id in [domain_for_testing["id"], second_domain["id"]]:
            grid_data = make_grid_data(
                domain_id=domain_id,
                name=f"Grid in {domain_id}",
                tags=["wildcard-list-test"],
            )
            doc_ref = firestore_client.collection(GRIDS_COLLECTION).document(
                grid_data["id"]
            )
            doc_ref.set(grid_data)
            grids.append(grid_data)
        yield grids
        for grid in grids:
            firestore_client.collection(GRIDS_COLLECTION).document(grid["id"]).delete()

    def route(self):
        return "/domains/-/grids"

    def test_wildcard_returns_200(self, client):
        response = client.get(self.route())
        assert response.status_code == 200

    def test_wildcard_returns_grids_from_all_domains(
        self, client, grids_across_domains
    ):
        """Grids from multiple domains are all returned."""
        response = client.get(f"{self.route()}?tag=wildcard-list-test&size=1000")
        assert response.status_code == 200

        grid_ids = [g["id"] for g in response.json()["grids"]]
        for grid in grids_across_domains:
            assert grid["id"] in grid_ids

    def test_wildcard_excludes_other_users_grids(
        self, client, grid_with_different_owner
    ):
        """Wildcard list does not return grids owned by other users."""
        response = client.get(self.route())
        assert response.status_code == 200

        grid_ids = [g["id"] for g in response.json()["grids"]]
        assert grid_with_different_owner["id"] not in grid_ids

    def test_wildcard_excludes_owner_id(self, client, grids_across_domains):
        """Wildcard list does not expose owner_id."""
        response = client.get(f"{self.route()}?tag=wildcard-list-test&size=1000")
        assert response.status_code == 200

        for grid in response.json()["grids"]:
            assert "owner_id" not in grid

    @pytest.mark.parametrize("sort_by", ["created_on", "modified_on", "name"])
    @pytest.mark.parametrize("sort_order", [None, "ascending", "descending"])
    def test_wildcard_sorting_matrix_returns_200(
        self, client, grids_across_domains, sort_by, sort_order
    ):
        """Every sort field/direction combination is served (issue #321)."""
        url = f"{self.route()}?sort_by={sort_by}"
        if sort_order:
            url += f"&sort_order={sort_order}"
        response = client.get(url)
        assert response.status_code == 200


# GET /domains/{domain_id}/grids (List) Tests


class TestListGrids:
    """Test the GET /domains/{domain_id}/grids endpoint (list all grids)."""

    def route(self, domain_id):
        return f"/domains/{domain_id}/grids"

    @pytest.fixture(scope="class")
    def grids_for_listing(self, firestore_client, domain_for_testing):
        """Create multiple grids for list testing."""
        grids = []
        for i, name in enumerate(["Alpha Grid", "Beta Grid", "Gamma Grid"]):
            grid_data = make_grid_data(
                domain_id=domain_for_testing["id"],
                name=name,
                tags=["list-test"],
            )
            doc_ref = firestore_client.collection(GRIDS_COLLECTION).document(
                grid_data["id"]
            )
            doc_ref.set(grid_data)
            grids.append(grid_data)

        yield grids

        # Cleanup
        for grid in grids:
            firestore_client.collection(GRIDS_COLLECTION).document(grid["id"]).delete()

    def test_list_returns_200(self, client, domain_for_testing):
        """List endpoint returns 200 OK."""
        response = client.get(self.route(domain_for_testing["id"]))

        assert response.status_code == 200

    def test_list_returns_paginated_response(self, client, domain_for_testing):
        """Response includes pagination metadata."""
        response = client.get(self.route(domain_for_testing["id"]))

        assert response.status_code == 200
        data = response.json()

        assert "grids" in data
        assert "current_page" in data
        assert "page_size" in data
        assert "total_items" in data
        assert isinstance(data["grids"], list)

    def test_list_returns_user_grids(
        self, client, grids_for_listing, domain_for_testing
    ):
        """List returns grids belonging to the authenticated user."""
        # Sort newest-first and ask for the max page so the fixture grids are
        # found regardless of the endpoint's default sort or how many other
        # grids share this test domain.
        response = client.get(
            f"{self.route(domain_for_testing['id'])}"
            "?sort_by=created_on&sort_order=descending&size=1000"
        )

        assert response.status_code == 200
        data = response.json()

        grid_ids = [g["id"] for g in data["grids"]]
        for grid in grids_for_listing:
            assert grid["id"] in grid_ids

    def test_list_excludes_other_users_grids(
        self, client, grid_with_different_owner, domain_for_testing
    ):
        """List does not return grids owned by other users."""
        response = client.get(self.route(domain_for_testing["id"]))

        assert response.status_code == 200
        data = response.json()

        grid_ids = [g["id"] for g in data["grids"]]
        assert grid_with_different_owner["id"] not in grid_ids

    def test_list_pagination_page_param(
        self, client, grids_for_listing, domain_for_testing
    ):
        """Page parameter controls which page is returned."""
        base = self.route(domain_for_testing["id"])

        response1 = client.get(f"{base}?page=0&size=1")
        assert response1.status_code == 200
        data1 = response1.json()
        assert data1["current_page"] == 0
        assert len(data1["grids"]) == 1

        response2 = client.get(f"{base}?page=1&size=1")
        assert response2.status_code == 200
        data2 = response2.json()
        assert data2["current_page"] == 1

    def test_list_pagination_size_param(
        self, client, grids_for_listing, domain_for_testing
    ):
        """Size parameter controls how many grids per page."""
        response = client.get(f"{self.route(domain_for_testing['id'])}?size=2")

        assert response.status_code == 200
        data = response.json()

        assert data["page_size"] == 2
        assert len(data["grids"]) <= 2

    def test_list_sorting_by_name_ascending(
        self, client, grids_for_listing, domain_for_testing
    ):
        """Sorting by name ascending returns alphabetical order."""
        response = client.get(
            f"{self.route(domain_for_testing['id'])}?sort_by=name&sort_order=ascending"
        )

        assert response.status_code == 200
        data = response.json()

        names = [g["name"] for g in data["grids"]]
        assert names == sorted(names)

    def test_list_sorting_by_name_descending(
        self, client, grids_for_listing, domain_for_testing
    ):
        """Sorting by name descending returns reverse alphabetical order."""
        response = client.get(
            f"{self.route(domain_for_testing['id'])}?sort_by=name&sort_order=descending"
        )

        assert response.status_code == 200
        data = response.json()

        names = [g["name"] for g in data["grids"]]
        assert names == sorted(names, reverse=True)

    @pytest.mark.parametrize("sort_by", ["created_on", "modified_on", "name"])
    @pytest.mark.parametrize("sort_order", [None, "ascending", "descending"])
    def test_list_sorting_matrix_returns_200(
        self, client, grids_for_listing, domain_for_testing, sort_by, sort_order
    ):
        """Every sort field/direction combination is served (issue #321)."""
        url = f"{self.route(domain_for_testing['id'])}?sort_by={sort_by}"
        if sort_order:
            url += f"&sort_order={sort_order}"
        response = client.get(url)
        assert response.status_code == 200

    def test_list_invalid_page_returns_422(self, client, domain_for_testing):
        """Negative page number returns 422."""
        response = client.get(f"{self.route(domain_for_testing['id'])}?page=-1")

        assert response.status_code == 422

    def test_list_invalid_size_too_small_returns_422(self, client, domain_for_testing):
        """Size less than 1 returns 422."""
        response = client.get(f"{self.route(domain_for_testing['id'])}?size=0")

        assert response.status_code == 422

    def test_list_invalid_size_too_large_returns_422(self, client, domain_for_testing):
        """Size greater than 1000 returns 422."""
        response = client.get(f"{self.route(domain_for_testing['id'])}?size=1001")

        assert response.status_code == 422

    def test_list_invalid_sort_by_returns_422(self, client, domain_for_testing):
        """Invalid sort_by field returns 422."""
        response = client.get(
            f"{self.route(domain_for_testing['id'])}?sort_by=invalid_field"
        )

        assert response.status_code == 422

    def test_list_invalid_sort_order_returns_422(self, client, domain_for_testing):
        """Invalid sort_order returns 422."""
        response = client.get(
            f"{self.route(domain_for_testing['id'])}?sort_order=invalid"
        )

        assert response.status_code == 422

    def test_list_grids_exclude_owner_id(
        self, client, grids_for_listing, domain_for_testing
    ):
        """Listed grids do not expose owner_id."""
        response = client.get(self.route(domain_for_testing["id"]))

        assert response.status_code == 200
        data = response.json()

        for grid in data["grids"]:
            assert "owner_id" not in grid

    def test_list_filter_by_source(self, client, grids_for_listing, domain_for_testing):
        """Filter by source name returns only matching grids."""
        response = client.get(f"{self.route(domain_for_testing['id'])}?source=landfire")

        assert response.status_code == 200
        data = response.json()

        # All returned grids should have source.name == "landfire"
        for grid in data["grids"]:
            assert grid["source"]["name"] == "landfire"

    def test_list_filter_by_source_no_results(
        self, client, grids_for_listing, domain_for_testing
    ):
        """Filter by source that doesn't exist returns empty list."""
        response = client.get(
            f"{self.route(domain_for_testing['id'])}?source=nonexistent_source"
        )

        assert response.status_code == 200
        data = response.json()

        assert data["grids"] == []
        assert data["total_items"] == 0

    def test_list_filter_by_tag(self, client, grids_for_listing, domain_for_testing):
        """Filter by tag returns only grids containing that tag."""
        response = client.get(f"{self.route(domain_for_testing['id'])}?tag=list-test")

        assert response.status_code == 200
        data = response.json()

        # All returned grids should contain the "list-test" tag
        for grid in data["grids"]:
            assert "list-test" in grid["tags"]

        # Should include our test grids
        grid_ids = [g["id"] for g in data["grids"]]
        for grid in grids_for_listing:
            assert grid["id"] in grid_ids

    def test_list_filter_by_tag_no_results(
        self, client, grids_for_listing, domain_for_testing
    ):
        """Filter by tag that doesn't exist returns empty list."""
        response = client.get(
            f"{self.route(domain_for_testing['id'])}?tag=nonexistent_tag"
        )

        assert response.status_code == 200
        data = response.json()

        assert data["grids"] == []
        assert data["total_items"] == 0

    def test_list_filter_combined_source_and_tag(
        self, client, grids_for_listing, domain_for_testing
    ):
        """Can combine source and tag filters."""
        response = client.get(
            f"{self.route(domain_for_testing['id'])}?source=landfire&tag=list-test"
        )

        assert response.status_code == 200
        data = response.json()

        # All returned grids should match both filters
        for grid in data["grids"]:
            assert grid["source"]["name"] == "landfire"
            assert "list-test" in grid["tags"]

    def test_list_filter_by_product(
        self, client, grids_for_listing, domain_for_testing
    ):
        """Filter by source product returns only matching grids."""
        response = client.get(f"{self.route(domain_for_testing['id'])}?product=fbfm40")

        assert response.status_code == 200
        data = response.json()

        # All returned grids should have source.product == "fbfm40"
        for grid in data["grids"]:
            assert grid["source"]["product"] == "fbfm40"

    def test_list_filter_combined_source_and_product(
        self, client, grids_for_listing, domain_for_testing
    ):
        """Can combine source and product filters."""
        response = client.get(
            f"{self.route(domain_for_testing['id'])}?source=landfire&product=fbfm40"
        )

        assert response.status_code == 200
        data = response.json()

        # All returned grids should match both source and product
        for grid in data["grids"]:
            assert grid["source"]["name"] == "landfire"
            assert grid["source"]["product"] == "fbfm40"


# PATCH /domains/{domain_id}/grids/{grid_id} Tests


class TestUpdateGrid:
    """Test the PATCH /domains/{domain_id}/grids/{grid_id} endpoint."""

    def route(self, domain_id):
        return f"/domains/{domain_id}/grids"

    @pytest.fixture(scope="class")
    def grid_for_update(self, firestore_client, domain_for_testing):
        """Create a grid for update tests."""
        grid_data = make_grid_data(
            domain_id=domain_for_testing["id"],
            name="Original Name",
            description="Original Description",
            tags=["original"],
        )
        doc_ref = firestore_client.collection(GRIDS_COLLECTION).document(
            grid_data["id"]
        )
        doc_ref.set(grid_data)
        yield grid_data
        doc_ref.delete()

    def test_update_name(self, client, grid_for_update, domain_for_testing):
        """Update only the name field."""
        grid_id = grid_for_update["id"]

        response = client.patch(
            f"{self.route(domain_for_testing['id'])}/{grid_id}",
            json={"name": "Updated Name"},
        )

        assert response.status_code == 200

        data = response.json()
        assert data["name"] == "Updated Name"
        assert data["description"] == "Original Description"
        assert data["tags"] == ["original"]

    def test_update_description(self, client, grid_for_update, domain_for_testing):
        """Update only the description field."""
        grid_id = grid_for_update["id"]

        response = client.patch(
            f"{self.route(domain_for_testing['id'])}/{grid_id}",
            json={"description": "Updated Description"},
        )

        assert response.status_code == 200

        data = response.json()
        assert data["description"] == "Updated Description"

    def test_update_tags(self, client, grid_for_update, domain_for_testing):
        """Update only the tags field."""
        grid_id = grid_for_update["id"]

        response = client.patch(
            f"{self.route(domain_for_testing['id'])}/{grid_id}",
            json={"tags": ["new", "tags"]},
        )

        assert response.status_code == 200

        data = response.json()
        assert data["tags"] == ["new", "tags"]

    def test_update_multiple_fields(self, client, grid_for_update, domain_for_testing):
        """Update multiple fields at once."""
        grid_id = grid_for_update["id"]

        response = client.patch(
            f"{self.route(domain_for_testing['id'])}/{grid_id}",
            json={
                "name": "Multi Update",
                "description": "Multi Update Description",
                "tags": ["multi", "update"],
            },
        )

        assert response.status_code == 200

        data = response.json()
        assert data["name"] == "Multi Update"
        assert data["description"] == "Multi Update Description"
        assert data["tags"] == ["multi", "update"]

    def test_update_modifies_modified_on(
        self, client, grid_for_update, domain_for_testing
    ):
        """Update should change the modified_on timestamp."""
        grid_id = grid_for_update["id"]
        original_modified_on = grid_for_update["modified_on"]

        response = client.patch(
            f"{self.route(domain_for_testing['id'])}/{grid_id}",
            json={"name": "Timestamp Test"},
        )

        assert response.status_code == 200

        data = response.json()
        assert data["modified_on"] != original_modified_on.isoformat()

    def test_update_preserves_immutable_fields(
        self, client, grid_for_update, domain_for_testing
    ):
        """Update should not change id, domain_id, source, bands, georeference."""
        grid_id = grid_for_update["id"]

        response = client.patch(
            f"{self.route(domain_for_testing['id'])}/{grid_id}",
            json={"name": "Immutable Test"},
        )

        assert response.status_code == 200

        data = response.json()
        assert data["id"] == grid_id
        assert data["domain_id"] == grid_for_update["domain_id"]
        assert "source" in data
        assert "bands" in data
        assert "georeference" in data

    def test_update_returns_full_grid(
        self, client, grid_for_update, domain_for_testing
    ):
        """Update response includes all grid fields."""
        grid_id = grid_for_update["id"]

        response = client.patch(
            f"{self.route(domain_for_testing['id'])}/{grid_id}",
            json={"name": "Full Response Test"},
        )

        assert response.status_code == 200

        data = response.json()
        assert "id" in data
        assert "domain_id" in data
        assert "name" in data
        assert "description" in data
        assert "status" in data
        assert "created_on" in data
        assert "modified_on" in data
        assert "source" in data
        assert "modifications" in data
        assert "bands" in data
        assert "georeference" in data
        assert "tags" in data

    def test_update_nonexistent_grid_returns_404(self, client, domain_for_testing):
        """Update a non-existent grid should return 404."""
        fake_grid_id = "00000000000000000000000000000000"

        response = client.patch(
            f"{self.route(domain_for_testing['id'])}/{fake_grid_id}",
            json={"name": "Should Fail"},
        )

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_update_wrong_owner_returns_404(
        self, client, grid_with_different_owner, domain_with_different_owner
    ):
        """Update a grid owned by another user returns 404."""
        grid_id = grid_with_different_owner["id"]

        response = client.patch(
            f"{self.route(domain_with_different_owner['id'])}/{grid_id}",
            json={"name": "Should Fail"},
        )

        assert response.status_code == 404

    def test_update_empty_body(self, client, grid_for_update, domain_for_testing):
        """Update with empty body should succeed (only updates modified_on)."""
        grid_id = grid_for_update["id"]

        response = client.patch(
            f"{self.route(domain_for_testing['id'])}/{grid_id}",
            json={},
        )

        assert response.status_code == 200

    def test_update_excludes_owner_id_from_response(
        self, client, grid_for_update, domain_for_testing
    ):
        """Response should not expose the owner_id field."""
        grid_id = grid_for_update["id"]

        response = client.patch(
            f"{self.route(domain_for_testing['id'])}/{grid_id}",
            json={"name": "Owner Test"},
        )

        assert response.status_code == 200

        data = response.json()
        assert "owner_id" not in data


# DELETE /domains/{domain_id}/grids/{grid_id} Tests


class TestDeleteGrid:
    """Test the DELETE /domains/{domain_id}/grids/{grid_id} endpoint."""

    def route(self, domain_id):
        return f"/domains/{domain_id}/grids"

    @pytest.fixture(scope="function")
    def grid_for_delete(self, firestore_client, domain_for_testing):
        """Create a grid for delete tests.

        Function-scoped because the grid gets deleted during the test.
        """
        grid_data = make_grid_data(
            domain_id=domain_for_testing["id"],
            name="Grid to Delete",
            tags=["delete-test"],
        )
        doc_ref = firestore_client.collection(GRIDS_COLLECTION).document(
            grid_data["id"]
        )
        doc_ref.set(grid_data)
        yield grid_data
        # Cleanup if not already deleted
        doc = doc_ref.get()
        if doc.exists:
            doc_ref.delete()

    def test_delete_existing_grid(
        self, client, grid_for_delete, firestore_client, domain_for_testing
    ):
        """Successfully delete an existing grid."""
        grid_id = grid_for_delete["id"]

        response = client.delete(f"{self.route(domain_for_testing['id'])}/{grid_id}")

        assert response.status_code == 204
        assert response.content == b""

        # Verify grid is actually deleted from Firestore
        doc = firestore_client.collection(GRIDS_COLLECTION).document(grid_id).get()
        assert not doc.exists

    def test_delete_nonexistent_grid_returns_404(self, client, domain_for_testing):
        """Delete a non-existent grid should return 404."""
        fake_grid_id = "00000000000000000000000000000000"

        response = client.delete(
            f"{self.route(domain_for_testing['id'])}/{fake_grid_id}"
        )

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_delete_wrong_owner_returns_404(
        self, client, grid_with_different_owner, domain_with_different_owner
    ):
        """Delete a grid owned by another user returns 404."""
        grid_id = grid_with_different_owner["id"]

        response = client.delete(
            f"{self.route(domain_with_different_owner['id'])}/{grid_id}"
        )

        assert response.status_code == 404

    def test_delete_is_permanent(self, client, grid_for_delete, domain_for_testing):
        """Deleted grid cannot be retrieved."""
        grid_id = grid_for_delete["id"]

        # Delete the grid
        delete_response = client.delete(
            f"{self.route(domain_for_testing['id'])}/{grid_id}"
        )
        assert delete_response.status_code == 204

        # Try to get it
        get_response = client.get(f"{self.route(domain_for_testing['id'])}/{grid_id}")
        assert get_response.status_code == 404

    def test_delete_twice_returns_404_second_time(
        self, client, grid_for_delete, domain_for_testing
    ):
        """Deleting the same grid twice returns 404 on second attempt."""
        grid_id = grid_for_delete["id"]

        # First delete succeeds
        response1 = client.delete(f"{self.route(domain_for_testing['id'])}/{grid_id}")
        assert response1.status_code == 204

        # Second delete returns 404
        response2 = client.delete(f"{self.route(domain_for_testing['id'])}/{grid_id}")
        assert response2.status_code == 404

    def test_delete_does_not_appear_in_list(
        self, client, grid_for_delete, domain_for_testing
    ):
        """Deleted grid does not appear in list endpoint."""
        grid_id = grid_for_delete["id"]
        list_route = self.route(domain_for_testing["id"])
        # Filter by the fixture's tag so the assertion isn't affected by
        # other grids accumulated on the session-scoped domain.
        list_params = {"tag": "delete-test"}

        # Verify it appears in list before delete
        list_response_before = client.get(list_route, params=list_params)
        grid_ids_before = [g["id"] for g in list_response_before.json()["grids"]]
        assert grid_id in grid_ids_before

        # Delete the grid
        client.delete(f"{list_route}/{grid_id}")

        # Verify it no longer appears in list
        list_response_after = client.get(list_route, params=list_params)
        grid_ids_after = [g["id"] for g in list_response_after.json()["grids"]]
        assert grid_id not in grid_ids_after

    def test_delete_returns_no_body(self, client, grid_for_delete, domain_for_testing):
        """Delete returns 204 with no response body."""
        grid_id = grid_for_delete["id"]

        response = client.delete(f"{self.route(domain_for_testing['id'])}/{grid_id}")

        assert response.status_code == 204
        assert response.content == b""
