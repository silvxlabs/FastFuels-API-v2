"""
Integration tests for api/v2/resources/inventories/router.py

Tests the standard CRUD endpoints (GET, LIST, PATCH, DELETE) and PIM creation.
These tests make real HTTP requests to the API and interact with Firestore.
"""

import pytest

from lib.config import (
    DOMAINS_COLLECTION,
    GRIDS_COLLECTION,
    INVENTORIES_COLLECTION,
)
from tests.fixtures import make_domain_data, make_grid_data, make_inventory_data

# Fixtures


@pytest.fixture(scope="session")
def pim_grid_for_inventory(firestore_client, domain_for_testing):
    """Create a completed PIM grid with tm_id band for inventory creation tests."""
    grid_data = make_grid_data(
        domain_id=domain_for_testing["id"],
        name="PIM Grid for Inventory Tests",
        status="completed",
        source={
            "name": "pim",
            "product": "treemap",
            "version": "2022",
            "bands": ["tm_id"],
            "description": "TreeMap plot imputation raster (FIA plot IDs at 30m)",
        },
        bands=[
            {"key": "tm_id", "type": "categorical", "unit": None, "index": 0},
        ],
    )
    doc_ref = firestore_client.collection(GRIDS_COLLECTION).document(grid_data["id"])
    doc_ref.set(grid_data)
    yield grid_data
    doc_ref.delete()


@pytest.fixture(scope="session")
def pim_grid_in_different_domain(firestore_client, second_domain):
    """A completed PIM grid in a different domain than domain_for_testing.

    Same owner (test-owner) but in second_domain_for_inventory. Used to test
    cross-domain protection: creating an inventory in domain A should reject
    a source grid that belongs to domain B.
    """
    grid_data = make_grid_data(
        domain_id=second_domain["id"],
        name="PIM grid in second domain",
        status="completed",
        source={
            "name": "pim",
            "product": "treemap",
            "version": "2022",
            "bands": ["tm_id"],
            "description": "TreeMap plot imputation raster (FIA plot IDs at 30m)",
        },
        bands=[
            {"key": "tm_id", "type": "categorical", "unit": None, "index": 0},
        ],
    )
    doc_ref = firestore_client.collection(GRIDS_COLLECTION).document(grid_data["id"])
    doc_ref.set(grid_data)
    yield grid_data
    doc_ref.delete()


@pytest.fixture(scope="session")
def inventory_in_firestore(firestore_client, domain_for_testing):
    """Create an inventory document directly in Firestore, yield it, then delete."""
    inv_data = make_inventory_data(
        domain_id=domain_for_testing["id"],
        name="Test Inventory for GET",
        description="Created by fixture for GET endpoint tests",
        tags=["test", "fixture"],
    )
    doc_ref = firestore_client.collection(INVENTORIES_COLLECTION).document(
        inv_data["id"]
    )
    doc_ref.set(inv_data)
    yield inv_data
    doc_ref.delete()


@pytest.fixture(scope="session")
def inventory_with_different_owner(firestore_client, domain_with_different_owner):
    """Create an inventory owned by a different user for ownership validation tests."""
    inv_data = make_inventory_data(
        domain_id=domain_with_different_owner["id"],
        owner_id="different-owner",
        name="Other User's Inventory",
    )
    doc_ref = firestore_client.collection(INVENTORIES_COLLECTION).document(
        inv_data["id"]
    )
    doc_ref.set(inv_data)
    yield inv_data
    doc_ref.delete()


# POST /domains/{domain_id}/inventories/pim Tests


class TestCreatePimInventory:
    """Test the POST /domains/{domain_id}/inventories/pim endpoint."""

    def route(self, domain_id):
        return f"/domains/{domain_id}/inventories/pim"

    @pytest.mark.xfail(reason="Standgen Cloud Run service not yet deployed")
    def test_minimal_request_creates_inventory(
        self, client, domain_for_testing, pim_grid_for_inventory
    ):
        """Minimal request creates an inventory with pending status."""
        response = client.post(
            self.route(domain_for_testing["id"]),
            json={
                "source_pim_grid_id": pim_grid_for_inventory["id"],
                "seed": 42,
            },
        )

        assert response.status_code == 201

        data = response.json()
        assert "id" in data
        assert len(data["id"]) == 32
        assert data["domain_id"] == domain_for_testing["id"]
        assert data["type"] == "tree"
        assert data["status"] == "pending"
        assert data["name"] == ""
        assert data["description"] == ""
        assert data["tags"] == []
        assert data["modifications"] == []
        assert data["georeference"] is None
        assert data["error"] is None

        # Check source
        assert data["source"]["name"] == "pim"
        assert data["source"]["source_pim_grid_id"] == pim_grid_for_inventory["id"]
        assert data["source"]["point_process"] == "inhomogeneous_poisson"
        assert data["source"]["seed"] == 42

    @pytest.mark.xfail(reason="Standgen Cloud Run service not yet deployed")
    def test_full_request_creates_inventory(
        self, client, domain_for_testing, pim_grid_for_inventory
    ):
        """Full request with all optional fields."""
        response = client.post(
            self.route(domain_for_testing["id"]),
            json={
                "source_pim_grid_id": pim_grid_for_inventory["id"],
                "seed": 12345,
                "point_process": "inhomogeneous_poisson",
                "type": "tree",
                "name": "Full PIM Inventory",
                "description": "A test PIM expansion inventory",
                "tags": ["test", "pim"],
            },
        )

        assert response.status_code == 201

        data = response.json()
        assert data["name"] == "Full PIM Inventory"
        assert data["description"] == "A test PIM expansion inventory"
        assert data["tags"] == ["test", "pim"]
        assert data["source"]["seed"] == 12345

    @pytest.mark.xfail(reason="Standgen Cloud Run service not yet deployed")
    def test_response_excludes_owner_id(
        self, client, domain_for_testing, pim_grid_for_inventory
    ):
        """Response should not expose the owner_id field."""
        response = client.post(
            self.route(domain_for_testing["id"]),
            json={
                "source_pim_grid_id": pim_grid_for_inventory["id"],
                "seed": 42,
            },
        )
        assert response.status_code == 201
        assert "owner_id" not in response.json()

    def test_nonexistent_source_grid_returns_404(self, client, domain_for_testing):
        """Referencing a non-existent source grid returns 404."""
        response = client.post(
            self.route(domain_for_testing["id"]),
            json={
                "source_pim_grid_id": "00000000000000000000000000000000",
                "seed": 42,
            },
        )
        assert response.status_code == 404

    def test_source_grid_in_different_domain_returns_404(
        self, client, domain_for_testing, pim_grid_in_different_domain
    ):
        """Source grid belonging to a different domain returns 404."""
        response = client.post(
            self.route(domain_for_testing["id"]),
            json={
                "source_pim_grid_id": pim_grid_in_different_domain["id"],
                "seed": 42,
            },
        )
        assert response.status_code == 404

    def test_source_grid_not_completed_returns_422(
        self, client, firestore_client, domain_for_testing
    ):
        """Source grid that is still pending returns 422."""
        grid_data = make_grid_data(
            domain_id=domain_for_testing["id"],
            name="Pending PIM Grid",
            status="pending",
            source={
                "name": "pim",
                "product": "treemap",
                "version": "2022",
                "bands": ["tm_id"],
                "description": "TreeMap",
            },
            bands=[
                {"key": "tm_id", "type": "categorical", "unit": None, "index": 0},
            ],
            georeference=None,
        )
        doc_ref = firestore_client.collection(GRIDS_COLLECTION).document(
            grid_data["id"]
        )
        doc_ref.set(grid_data)

        try:
            response = client.post(
                self.route(domain_for_testing["id"]),
                json={
                    "source_pim_grid_id": grid_data["id"],
                    "seed": 42,
                },
            )
            assert response.status_code == 422
        finally:
            doc_ref.delete()

    def test_non_pim_grid_returns_422(
        self, client, firestore_client, domain_for_testing
    ):
        """Source grid that is not a PIM grid returns 422."""
        grid_data = make_grid_data(
            domain_id=domain_for_testing["id"],
            name="FBFM40 Grid (not PIM)",
            status="completed",
            source={
                "name": "landfire",
                "product": "fbfm40",
                "version": "2022",
                "description": "FBFM40",
            },
            bands=[
                {"key": "fbfm", "type": "categorical", "unit": None, "index": 0},
            ],
        )
        doc_ref = firestore_client.collection(GRIDS_COLLECTION).document(
            grid_data["id"]
        )
        doc_ref.set(grid_data)

        try:
            response = client.post(
                self.route(domain_for_testing["id"]),
                json={
                    "source_pim_grid_id": grid_data["id"],
                    "seed": 42,
                },
            )
            assert response.status_code == 422
            assert "not a pim grid" in response.json()["detail"].lower()
        finally:
            doc_ref.delete()

    @pytest.mark.xfail(reason="Standgen Cloud Run service not yet deployed")
    def test_omitting_seed_generates_random_seed(
        self, client, domain_for_testing, pim_grid_for_inventory
    ):
        """Request without seed auto-generates one."""
        response = client.post(
            self.route(domain_for_testing["id"]),
            json={
                "source_pim_grid_id": pim_grid_for_inventory["id"],
            },
        )
        assert response.status_code == 201
        data = response.json()
        assert isinstance(data["source"]["seed"], int)

    def test_missing_source_pim_grid_id_returns_422(self, client, domain_for_testing):
        """Request without source_pim_grid_id returns 422."""
        response = client.post(
            self.route(domain_for_testing["id"]),
            json={"seed": 42},
        )
        assert response.status_code == 422

    def test_invalid_domain_returns_404(self, client):
        """Non-existent domain_id returns 404."""
        response = client.post(
            self.route("00000000000000000000000000000000"),
            json={"source_pim_grid_id": "grid123", "seed": 42},
        )
        assert response.status_code == 404

    def test_wrong_owner_domain_returns_404(self, client, domain_with_different_owner):
        """Domain owned by another user returns 404."""
        response = client.post(
            self.route(domain_with_different_owner["id"]),
            json={"source_pim_grid_id": "grid123", "seed": 42},
        )
        assert response.status_code == 404

    @pytest.mark.xfail(reason="Standgen Cloud Run service not yet deployed")
    @pytest.mark.parametrize(
        "example_name,example_value",
        [
            (
                "minimal",
                {
                    "source_pim_grid_id": "PLACEHOLDER",
                    "seed": 42,
                },
            ),
            (
                "full",
                {
                    "source_pim_grid_id": "PLACEHOLDER",
                    "seed": 12345,
                    "point_process": "inhomogeneous_poisson",
                    "type": "tree",
                    "name": "PIM expansion inventory",
                    "description": "Tree inventory from PIM grid expansion",
                    "tags": ["baseline"],
                },
            ),
        ],
    )
    def test_documented_example_creates_inventory(
        self,
        client,
        domain_for_testing,
        pim_grid_for_inventory,
        example_name,
        example_value,
    ):
        """Each documented PIM example should successfully create an inventory."""
        body = {**example_value}
        # Replace placeholder with real grid ID for explicit grid examples
        if body.get("source_pim_grid_id") == "PLACEHOLDER":
            body["source_pim_grid_id"] = pim_grid_for_inventory["id"]

        response = client.post(self.route(domain_for_testing["id"]), json=body)

        assert response.status_code == 201, (
            f"Example '{example_name}' failed with status {response.status_code}: "
            f"{response.json()}"
        )

        data = response.json()
        assert data["source"]["name"] == "pim"


# GET /domains/{domain_id}/inventories/{inventory_id} Tests


class TestGetInventory:
    """Test the GET /domains/{domain_id}/inventories/{inventory_id} endpoint."""

    def route(self, domain_id):
        return f"/domains/{domain_id}/inventories"

    def test_get_existing_inventory(
        self, client, inventory_in_firestore, domain_for_testing
    ):
        """Successfully retrieve an inventory that exists."""
        inv_id = inventory_in_firestore["id"]

        response = client.get(f"{self.route(domain_for_testing['id'])}/{inv_id}")

        assert response.status_code == 200

        data = response.json()
        assert data["id"] == inv_id
        assert data["name"] == "Test Inventory for GET"
        assert data["description"] == "Created by fixture for GET endpoint tests"
        assert data["tags"] == ["test", "fixture"]
        assert data["type"] == "tree"
        assert "source" in data
        assert "created_on" in data
        assert "modified_on" in data

    def test_get_nonexistent_inventory_returns_404(self, client, domain_for_testing):
        """Fetching a non-existent inventory returns 404."""
        response = client.get(
            f"{self.route(domain_for_testing['id'])}/00000000000000000000000000000000"
        )
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_get_inventory_wrong_owner_returns_404(
        self, client, inventory_with_different_owner, domain_with_different_owner
    ):
        """Fetching an inventory owned by another user returns 404."""
        inv_id = inventory_with_different_owner["id"]
        response = client.get(
            f"{self.route(domain_with_different_owner['id'])}/{inv_id}"
        )
        assert response.status_code == 404

    def test_get_inventory_wrong_domain_returns_404(
        self,
        client,
        inventory_in_firestore,
        domain_for_testing,
        domain_with_different_owner,
    ):
        """Fetching an inventory under the wrong domain returns 404."""
        inv_id = inventory_in_firestore["id"]
        response = client.get(
            f"{self.route(domain_with_different_owner['id'])}/{inv_id}"
        )
        assert response.status_code == 404

    def test_get_inventory_excludes_owner_id(
        self, client, inventory_in_firestore, domain_for_testing
    ):
        """Response should not expose the owner_id field."""
        inv_id = inventory_in_firestore["id"]
        response = client.get(f"{self.route(domain_for_testing['id'])}/{inv_id}")
        assert response.status_code == 200
        assert "owner_id" not in response.json()


# GET /domains/-/inventories (Wildcard List) Tests


class TestListInventoriesWildcard:
    """Test GET /domains/-/inventories returns inventories across all domains."""

    @pytest.fixture(scope="class")
    def inventories_across_domains(
        self, firestore_client, domain_for_testing, second_domain
    ):
        """Inventories spread across two domains, both owned by test-owner."""
        inventories = []
        for domain_id in [domain_for_testing["id"], second_domain["id"]]:
            inv_data = make_inventory_data(
                domain_id=domain_id, name=f"Inventory in {domain_id}"
            )
            doc_ref = firestore_client.collection(INVENTORIES_COLLECTION).document(
                inv_data["id"]
            )
            doc_ref.set(inv_data)
            inventories.append(inv_data)
        yield inventories
        for inv in inventories:
            firestore_client.collection(INVENTORIES_COLLECTION).document(
                inv["id"]
            ).delete()

    def route(self):
        return "/domains/-/inventories"

    def test_wildcard_returns_200(self, client):
        response = client.get(self.route())
        assert response.status_code == 200

    def test_wildcard_returns_inventories_from_all_domains(
        self, client, inventories_across_domains
    ):
        """Inventories from multiple domains are all returned."""
        response = client.get(self.route())
        assert response.status_code == 200

        inv_ids = [i["id"] for i in response.json()["inventories"]]
        for inv in inventories_across_domains:
            assert inv["id"] in inv_ids

    def test_wildcard_excludes_other_users_inventories(
        self, client, inventory_with_different_owner
    ):
        """Wildcard list does not return inventories owned by other users."""
        response = client.get(self.route())
        assert response.status_code == 200

        inv_ids = [i["id"] for i in response.json()["inventories"]]
        assert inventory_with_different_owner["id"] not in inv_ids

    def test_wildcard_excludes_owner_id(self, client, inventories_across_domains):
        """Wildcard list does not expose owner_id."""
        response = client.get(self.route())
        assert response.status_code == 200

        for inv in response.json()["inventories"]:
            assert "owner_id" not in inv


# GET /domains/{domain_id}/inventories (List) Tests


class TestListInventories:
    """Test the GET /domains/{domain_id}/inventories endpoint."""

    def route(self, domain_id):
        return f"/domains/{domain_id}/inventories"

    @pytest.fixture(scope="class")
    def inventories_for_listing(self, firestore_client, domain_for_testing):
        """Create multiple inventories for list testing."""
        inventories = []
        for name in ["Alpha Inventory", "Beta Inventory", "Gamma Inventory"]:
            inv_data = make_inventory_data(
                domain_id=domain_for_testing["id"],
                name=name,
                tags=["list-test"],
            )
            doc_ref = firestore_client.collection(INVENTORIES_COLLECTION).document(
                inv_data["id"]
            )
            doc_ref.set(inv_data)
            inventories.append(inv_data)

        yield inventories

        for inv in inventories:
            firestore_client.collection(INVENTORIES_COLLECTION).document(
                inv["id"]
            ).delete()

    def test_list_returns_200(self, client, domain_for_testing):
        """List endpoint returns 200 OK."""
        response = client.get(self.route(domain_for_testing["id"]))
        assert response.status_code == 200

    def test_list_returns_paginated_response(self, client, domain_for_testing):
        """Response includes pagination metadata."""
        response = client.get(self.route(domain_for_testing["id"]))

        assert response.status_code == 200
        data = response.json()

        assert "inventories" in data
        assert "current_page" in data
        assert "page_size" in data
        assert "total_items" in data
        assert isinstance(data["inventories"], list)

    def test_list_returns_user_inventories(
        self, client, inventories_for_listing, domain_for_testing
    ):
        """List returns inventories belonging to the authenticated user."""
        response = client.get(self.route(domain_for_testing["id"]))

        assert response.status_code == 200
        data = response.json()

        inv_ids = [i["id"] for i in data["inventories"]]
        for inv in inventories_for_listing:
            assert inv["id"] in inv_ids

    def test_list_excludes_other_users_inventories(
        self, client, inventory_with_different_owner, domain_for_testing
    ):
        """List does not return inventories owned by other users."""
        response = client.get(self.route(domain_for_testing["id"]))

        assert response.status_code == 200
        data = response.json()

        inv_ids = [i["id"] for i in data["inventories"]]
        assert inventory_with_different_owner["id"] not in inv_ids

    def test_list_inventories_exclude_owner_id(
        self, client, inventories_for_listing, domain_for_testing
    ):
        """Listed inventories do not expose owner_id."""
        response = client.get(self.route(domain_for_testing["id"]))

        assert response.status_code == 200
        data = response.json()

        for inv in data["inventories"]:
            assert "owner_id" not in inv

    def test_list_pagination_page_param(
        self, client, inventories_for_listing, domain_for_testing
    ):
        """Page parameter controls which page is returned."""
        base = self.route(domain_for_testing["id"])

        response1 = client.get(f"{base}?page=0&size=1")
        assert response1.status_code == 200
        data1 = response1.json()
        assert data1["current_page"] == 0
        assert len(data1["inventories"]) == 1

        response2 = client.get(f"{base}?page=1&size=1")
        assert response2.status_code == 200
        data2 = response2.json()
        assert data2["current_page"] == 1

    def test_list_pagination_size_param(
        self, client, inventories_for_listing, domain_for_testing
    ):
        """Size parameter controls how many inventories per page."""
        response = client.get(f"{self.route(domain_for_testing['id'])}?size=2")

        assert response.status_code == 200
        data = response.json()
        assert data["page_size"] == 2
        assert len(data["inventories"]) <= 2

    def test_list_sorting_by_name_ascending(
        self, client, inventories_for_listing, domain_for_testing
    ):
        """Sorting by name ascending returns alphabetical order."""
        response = client.get(
            f"{self.route(domain_for_testing['id'])}?sort_by=name&sort_order=ascending"
        )

        assert response.status_code == 200
        data = response.json()

        names = [i["name"] for i in data["inventories"]]
        assert names == sorted(names)

    def test_list_sorting_by_name_descending(
        self, client, inventories_for_listing, domain_for_testing
    ):
        """Sorting by name descending returns reverse alphabetical order."""
        response = client.get(
            f"{self.route(domain_for_testing['id'])}?sort_by=name&sort_order=descending"
        )

        assert response.status_code == 200
        data = response.json()

        names = [i["name"] for i in data["inventories"]]
        assert names == sorted(names, reverse=True)

    def test_list_sorting_by_created_on(
        self, client, inventories_for_listing, domain_for_testing
    ):
        """Sorting by created_on is accepted."""
        response = client.get(
            f"{self.route(domain_for_testing['id'])}?sort_by=created_on&sort_order=descending"
        )
        assert response.status_code == 200

    def test_list_sorting_by_modified_on(
        self, client, inventories_for_listing, domain_for_testing
    ):
        """Sorting by modified_on is accepted."""
        response = client.get(
            f"{self.route(domain_for_testing['id'])}?sort_by=modified_on&sort_order=ascending"
        )
        assert response.status_code == 200

    def test_list_filter_by_source(
        self, client, inventories_for_listing, domain_for_testing
    ):
        """Filter by source name returns only matching inventories."""
        response = client.get(f"{self.route(domain_for_testing['id'])}?source=pim")

        assert response.status_code == 200
        data = response.json()

        for inv in data["inventories"]:
            assert inv["source"]["name"] == "pim"

    def test_list_filter_by_source_no_results(
        self, client, inventories_for_listing, domain_for_testing
    ):
        """Filter by source that doesn't exist returns empty list."""
        response = client.get(
            f"{self.route(domain_for_testing['id'])}?source=nonexistent_source"
        )

        assert response.status_code == 200
        data = response.json()
        assert data["inventories"] == []
        assert data["total_items"] == 0

    def test_list_filter_by_type(
        self, client, inventories_for_listing, domain_for_testing
    ):
        """Filter by type returns only matching inventories."""
        response = client.get(f"{self.route(domain_for_testing['id'])}?type=tree")

        assert response.status_code == 200
        data = response.json()

        for inv in data["inventories"]:
            assert inv["type"] == "tree"

    def test_list_filter_by_tag(
        self, client, inventories_for_listing, domain_for_testing
    ):
        """Filter by tag returns only inventories containing that tag."""
        response = client.get(f"{self.route(domain_for_testing['id'])}?tag=list-test")

        assert response.status_code == 200
        data = response.json()

        for inv in data["inventories"]:
            assert "list-test" in inv["tags"]

        inv_ids = [i["id"] for i in data["inventories"]]
        for inv in inventories_for_listing:
            assert inv["id"] in inv_ids

    def test_list_filter_by_tag_no_results(
        self, client, inventories_for_listing, domain_for_testing
    ):
        """Filter by tag that doesn't exist returns empty list."""
        response = client.get(
            f"{self.route(domain_for_testing['id'])}?tag=nonexistent_tag"
        )

        assert response.status_code == 200
        data = response.json()
        assert data["inventories"] == []
        assert data["total_items"] == 0

    def test_list_filter_combined_source_and_tag(
        self, client, inventories_for_listing, domain_for_testing
    ):
        """Can combine source and tag filters."""
        response = client.get(
            f"{self.route(domain_for_testing['id'])}?source=pim&tag=list-test"
        )

        assert response.status_code == 200
        data = response.json()

        for inv in data["inventories"]:
            assert inv["source"]["name"] == "pim"
            assert "list-test" in inv["tags"]

    def test_list_filter_combined_type_and_tag(
        self, client, inventories_for_listing, domain_for_testing
    ):
        """Can combine type and tag filters."""
        response = client.get(
            f"{self.route(domain_for_testing['id'])}?type=tree&tag=list-test"
        )

        assert response.status_code == 200
        data = response.json()

        for inv in data["inventories"]:
            assert inv["type"] == "tree"
            assert "list-test" in inv["tags"]

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


# PATCH /domains/{domain_id}/inventories/{inventory_id} Tests


class TestUpdateInventory:
    """Test the PATCH /domains/{domain_id}/inventories/{inventory_id} endpoint."""

    def route(self, domain_id):
        return f"/domains/{domain_id}/inventories"

    @pytest.fixture(scope="class")
    def inventory_for_update(self, firestore_client, domain_for_testing):
        """Create an inventory for update tests."""
        inv_data = make_inventory_data(
            domain_id=domain_for_testing["id"],
            name="Original Name",
            description="Original Description",
            tags=["original"],
        )
        doc_ref = firestore_client.collection(INVENTORIES_COLLECTION).document(
            inv_data["id"]
        )
        doc_ref.set(inv_data)
        yield inv_data
        doc_ref.delete()

    def test_update_name(self, client, inventory_for_update, domain_for_testing):
        """Update only the name field."""
        inv_id = inventory_for_update["id"]

        response = client.patch(
            f"{self.route(domain_for_testing['id'])}/{inv_id}",
            json={"name": "Updated Name"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Updated Name"
        assert data["description"] == "Original Description"
        assert data["tags"] == ["original"]

    def test_update_description(self, client, inventory_for_update, domain_for_testing):
        """Update only the description field."""
        inv_id = inventory_for_update["id"]

        response = client.patch(
            f"{self.route(domain_for_testing['id'])}/{inv_id}",
            json={"description": "Updated Description"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["description"] == "Updated Description"

    def test_update_tags(self, client, inventory_for_update, domain_for_testing):
        """Update only the tags field."""
        inv_id = inventory_for_update["id"]

        response = client.patch(
            f"{self.route(domain_for_testing['id'])}/{inv_id}",
            json={"tags": ["new", "tags"]},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["tags"] == ["new", "tags"]

    def test_update_multiple_fields(
        self, client, inventory_for_update, domain_for_testing
    ):
        """Update multiple fields at once."""
        inv_id = inventory_for_update["id"]

        response = client.patch(
            f"{self.route(domain_for_testing['id'])}/{inv_id}",
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
        self, client, inventory_for_update, domain_for_testing
    ):
        """Update should change the modified_on timestamp."""
        inv_id = inventory_for_update["id"]
        original_modified_on = inventory_for_update["modified_on"]

        response = client.patch(
            f"{self.route(domain_for_testing['id'])}/{inv_id}",
            json={"name": "Timestamp Test"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["modified_on"] != original_modified_on.isoformat()

    def test_update_preserves_immutable_fields(
        self, client, inventory_for_update, domain_for_testing
    ):
        """Update should not change id, domain_id, type, source, modifications."""
        inv_id = inventory_for_update["id"]

        response = client.patch(
            f"{self.route(domain_for_testing['id'])}/{inv_id}",
            json={"name": "Immutable Test"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == inv_id
        assert data["domain_id"] == inventory_for_update["domain_id"]
        assert data["type"] == "tree"
        assert "source" in data
        assert data["modifications"] == []

    def test_update_returns_full_inventory(
        self, client, inventory_for_update, domain_for_testing
    ):
        """Update response includes all inventory fields."""
        inv_id = inventory_for_update["id"]

        response = client.patch(
            f"{self.route(domain_for_testing['id'])}/{inv_id}",
            json={"name": "Full Response Test"},
        )

        assert response.status_code == 200
        data = response.json()
        assert "id" in data
        assert "domain_id" in data
        assert "type" in data
        assert "name" in data
        assert "description" in data
        assert "status" in data
        assert "created_on" in data
        assert "modified_on" in data
        assert "source" in data
        assert "modifications" in data
        assert "georeference" in data
        assert "tags" in data

    def test_update_empty_body(self, client, inventory_for_update, domain_for_testing):
        """Update with empty body should succeed (only updates modified_on)."""
        inv_id = inventory_for_update["id"]

        response = client.patch(
            f"{self.route(domain_for_testing['id'])}/{inv_id}",
            json={},
        )

        assert response.status_code == 200

    def test_update_nonexistent_inventory_returns_404(self, client, domain_for_testing):
        """Update a non-existent inventory should return 404."""
        response = client.patch(
            f"{self.route(domain_for_testing['id'])}/00000000000000000000000000000000",
            json={"name": "Should Fail"},
        )
        assert response.status_code == 404

    def test_update_excludes_owner_id_from_response(
        self, client, inventory_for_update, domain_for_testing
    ):
        """Response should not expose the owner_id field."""
        inv_id = inventory_for_update["id"]

        response = client.patch(
            f"{self.route(domain_for_testing['id'])}/{inv_id}",
            json={"name": "Owner Test"},
        )

        assert response.status_code == 200
        assert "owner_id" not in response.json()


# DELETE /domains/{domain_id}/inventories/{inventory_id} Tests


class TestDeleteInventory:
    """Test the DELETE /domains/{domain_id}/inventories/{inventory_id} endpoint."""

    def route(self, domain_id):
        return f"/domains/{domain_id}/inventories"

    @pytest.fixture(scope="function")
    def inventory_for_delete(self, firestore_client, domain_for_testing):
        """Create an inventory for delete tests. Function-scoped."""
        inv_data = make_inventory_data(
            domain_id=domain_for_testing["id"],
            name="Inventory to Delete",
            tags=["delete-test"],
        )
        doc_ref = firestore_client.collection(INVENTORIES_COLLECTION).document(
            inv_data["id"]
        )
        doc_ref.set(inv_data)
        yield inv_data
        doc = doc_ref.get()
        if doc.exists:
            doc_ref.delete()

    def test_delete_existing_inventory(
        self, client, inventory_for_delete, firestore_client, domain_for_testing
    ):
        """Successfully delete an existing inventory."""
        inv_id = inventory_for_delete["id"]

        response = client.delete(f"{self.route(domain_for_testing['id'])}/{inv_id}")

        assert response.status_code == 204
        assert response.content == b""

        # Verify inventory is actually deleted from Firestore
        doc = firestore_client.collection(INVENTORIES_COLLECTION).document(inv_id).get()
        assert not doc.exists

    def test_delete_nonexistent_inventory_returns_404(self, client, domain_for_testing):
        """Delete a non-existent inventory should return 404."""
        response = client.delete(
            f"{self.route(domain_for_testing['id'])}/00000000000000000000000000000000"
        )
        assert response.status_code == 404

    def test_delete_wrong_owner_returns_404(
        self, client, inventory_with_different_owner, domain_with_different_owner
    ):
        """Delete an inventory owned by another user returns 404."""
        inv_id = inventory_with_different_owner["id"]
        response = client.delete(
            f"{self.route(domain_with_different_owner['id'])}/{inv_id}"
        )
        assert response.status_code == 404

    def test_delete_is_permanent(
        self, client, inventory_for_delete, domain_for_testing
    ):
        """Deleted inventory cannot be retrieved."""
        inv_id = inventory_for_delete["id"]

        delete_response = client.delete(
            f"{self.route(domain_for_testing['id'])}/{inv_id}"
        )
        assert delete_response.status_code == 204

        get_response = client.get(f"{self.route(domain_for_testing['id'])}/{inv_id}")
        assert get_response.status_code == 404

    def test_delete_twice_returns_404_second_time(
        self, client, inventory_for_delete, domain_for_testing
    ):
        """Deleting the same inventory twice returns 404 on second attempt."""
        inv_id = inventory_for_delete["id"]

        response1 = client.delete(f"{self.route(domain_for_testing['id'])}/{inv_id}")
        assert response1.status_code == 204

        response2 = client.delete(f"{self.route(domain_for_testing['id'])}/{inv_id}")
        assert response2.status_code == 404

    def test_delete_does_not_appear_in_list(
        self, client, inventory_for_delete, domain_for_testing
    ):
        """Deleted inventory does not appear in list endpoint."""
        inv_id = inventory_for_delete["id"]
        list_route = self.route(domain_for_testing["id"])

        # Verify it appears in list before delete
        list_response_before = client.get(list_route)
        inv_ids_before = [i["id"] for i in list_response_before.json()["inventories"]]
        assert inv_id in inv_ids_before

        # Delete the inventory
        client.delete(f"{list_route}/{inv_id}")

        # Verify it no longer appears in list
        list_response_after = client.get(list_route)
        inv_ids_after = [i["id"] for i in list_response_after.json()["inventories"]]
        assert inv_id not in inv_ids_after

    def test_delete_returns_no_body(
        self, client, inventory_for_delete, domain_for_testing
    ):
        """Delete returns 204 with no response body."""
        inv_id = inventory_for_delete["id"]

        response = client.delete(f"{self.route(domain_for_testing['id'])}/{inv_id}")

        assert response.status_code == 204
        assert response.content == b""


# Domain Cascade Delete Tests


class TestDomainCascadeDeleteInventories:
    """Test that domain force-delete cascade-deletes child inventories."""

    route = "/domains"

    def test_domain_with_inventory_children_returns_412(self, client, firestore_client):
        """Delete domain with child inventories without force returns 412."""
        domain_data = make_domain_data(name="Domain with inventory children")
        domain_ref = firestore_client.collection(DOMAINS_COLLECTION).document(
            domain_data["id"]
        )
        domain_ref.set(domain_data)

        inv_data = make_inventory_data(domain_id=domain_data["id"])
        inv_ref = firestore_client.collection(INVENTORIES_COLLECTION).document(
            inv_data["id"]
        )
        inv_ref.set(inv_data)

        try:
            response = client.delete(f"{self.route}/{domain_data['id']}")
            assert response.status_code == 412
            assert "child resources" in response.json()["detail"].lower()
        finally:
            inv_ref.delete()
            doc = domain_ref.get()
            if doc.exists:
                domain_ref.delete()

    def test_domain_force_delete_cascades_inventories(self, client, firestore_client):
        """Delete domain with force=true cascade-deletes child inventories."""
        domain_data = make_domain_data(name="Domain for inventory cascade")
        domain_ref = firestore_client.collection(DOMAINS_COLLECTION).document(
            domain_data["id"]
        )
        domain_ref.set(domain_data)

        inv_ids = []
        for i in range(3):
            inv_data = make_inventory_data(
                domain_id=domain_data["id"], name=f"Child inventory {i}"
            )
            inv_ref = firestore_client.collection(INVENTORIES_COLLECTION).document(
                inv_data["id"]
            )
            inv_ref.set(inv_data)
            inv_ids.append(inv_data["id"])

        response = client.delete(f"{self.route}/{domain_data['id']}?force=true")
        assert response.status_code == 204

        # Verify domain is deleted
        doc = domain_ref.get()
        assert not doc.exists

        # Verify all child inventories are deleted
        for inv_id in inv_ids:
            inv_doc = (
                firestore_client.collection(INVENTORIES_COLLECTION)
                .document(inv_id)
                .get()
            )
            assert not inv_doc.exists
