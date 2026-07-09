"""
Integration tests for api/v2/resources/inventories/tree/allometry/gdam/router.py

Tests the GDAM allometry imputation inventory creation endpoint. These tests make
real HTTP requests to the API and interact with Firestore and Cloud Tasks.
"""

import pytest
from api.resources.inventories.tree.allometry.gdam.examples import (
    ALL_GDAM_EXAMPLE_VALUES,
)

from lib.config import DOMAINS_COLLECTION, INVENTORIES_COLLECTION
from tests.fixtures import make_domain_data, make_inventory_data

# Position + height columns — the typical (uploaded) GDAM input.
_POSITION_HEIGHT_COLUMNS = [
    {"key": "x", "type": "continuous", "unit": "m"},
    {"key": "y", "type": "continuous", "unit": "m"},
    {"key": "height", "type": "continuous", "unit": "m"},
]
_SOURCE_CHECKSUM = "src-checksum-123"

# --- Fixtures ---


@pytest.fixture(scope="session")
def source_tree_inventory(firestore_client, domain_for_testing):
    """A completed position+height tree inventory in domain_for_testing."""
    inventory_data = make_inventory_data(
        domain_id=domain_for_testing["id"],
        name="Source inventory for GDAM tests",
        status="completed",
        inventory_type="tree",
    )
    inventory_data["columns"] = _POSITION_HEIGHT_COLUMNS
    inventory_data["checksum"] = _SOURCE_CHECKSUM
    doc_ref = firestore_client.collection(INVENTORIES_COLLECTION).document(
        inventory_data["id"]
    )
    doc_ref.set(inventory_data)
    yield inventory_data
    doc_ref.delete()


@pytest.fixture(scope="session")
def second_domain_for_gdam(firestore_client):
    """A second domain owned by test-owner, for cross-domain tests."""
    domain_data = make_domain_data(name="Second Domain for GDAM Tests")
    doc_ref = firestore_client.collection(DOMAINS_COLLECTION).document(
        domain_data["id"]
    )
    doc_ref.set(domain_data)
    yield domain_data
    doc_ref.delete()


@pytest.fixture(scope="session")
def source_inventory_in_different_domain(firestore_client, second_domain_for_gdam):
    """A completed tree inventory in a different domain than domain_for_testing."""
    inventory_data = make_inventory_data(
        domain_id=second_domain_for_gdam["id"],
        name="Source inventory in second domain",
        status="completed",
        inventory_type="tree",
    )
    inventory_data["columns"] = _POSITION_HEIGHT_COLUMNS
    doc_ref = firestore_client.collection(INVENTORIES_COLLECTION).document(
        inventory_data["id"]
    )
    doc_ref.set(inventory_data)
    yield inventory_data
    doc_ref.delete()


# --- Tests ---


class TestCreateGdamInventory:
    """Test the POST /domains/{domain_id}/inventories/tree/allometry/gdam endpoint."""

    def route(self, domain_id):
        return f"/domains/{domain_id}/inventories/tree/allometry/gdam"

    def test_minimal_request_creates_inventory(
        self, client, domain_for_testing, source_tree_inventory
    ):
        """Minimal request creates a pending inventory with a gdam source."""
        response = client.post(
            self.route(domain_for_testing["id"]),
            json={"source_tree_inventory_id": source_tree_inventory["id"]},
        )

        assert response.status_code == 201

        data = response.json()
        assert "id" in data
        assert len(data["id"]) == 32
        assert data["domain_id"] == domain_for_testing["id"]
        assert data["status"] == "pending"
        assert data["type"] == "tree"
        assert data["name"] == ""
        assert data["description"] == ""
        assert data["tags"] == []
        assert data["georeference"] is None

        # Source records the lineage, including the source checksum at create time.
        assert data["source"]["name"] == "gdam"
        assert data["source"]["source_tree_inventory_id"] == source_tree_inventory["id"]
        assert data["source"]["source_tree_inventory_checksum"] == _SOURCE_CHECKSUM

        # Columns reflect the source set plus exactly what GDAM imputes (all
        # three by default) — not the full base set. fia_status_code is never
        # imputed, so it must not appear.
        assert [c["key"] for c in data["columns"]] == [
            "x",
            "y",
            "height",
            "dbh",
            "crown_ratio",
            "fia_species_code",
        ]

    def test_request_with_metadata(
        self, client, domain_for_testing, source_tree_inventory
    ):
        """Request with name, description, and tags is stored verbatim."""
        request_body = {
            "source_tree_inventory_id": source_tree_inventory["id"],
            "name": "Custom GDAM Inventory",
            "description": "Testing GDAM imputation",
            "tags": ["test", "gdam"],
        }

        response = client.post(self.route(domain_for_testing["id"]), json=request_body)

        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "Custom GDAM Inventory"
        assert data["description"] == "Testing GDAM imputation"
        assert data["tags"] == ["test", "gdam"]

    def test_impute_columns_default_persisted(
        self, client, domain_for_testing, source_tree_inventory
    ):
        """Omitting impute_columns persists all three columns in the source."""
        response = client.post(
            self.route(domain_for_testing["id"]),
            json={"source_tree_inventory_id": source_tree_inventory["id"]},
        )
        assert response.status_code == 201
        assert response.json()["source"]["impute_columns"] == [
            "dbh",
            "crown_ratio",
            "fia_species_code",
        ]

    def test_impute_columns_subset_persisted(
        self, client, domain_for_testing, source_tree_inventory
    ):
        """A requested subset is persisted on the source."""
        response = client.post(
            self.route(domain_for_testing["id"]),
            json={
                "source_tree_inventory_id": source_tree_inventory["id"],
                "impute_columns": ["fia_species_code"],
            },
        )
        assert response.status_code == 201
        assert response.json()["source"]["impute_columns"] == ["fia_species_code"]

    def test_impute_columns_empty_returns_422(
        self, client, domain_for_testing, source_tree_inventory
    ):
        """An empty impute_columns list is rejected at the schema layer."""
        response = client.post(
            self.route(domain_for_testing["id"]),
            json={
                "source_tree_inventory_id": source_tree_inventory["id"],
                "impute_columns": [],
            },
        )
        assert response.status_code == 422

    def test_response_excludes_owner_id(
        self, client, domain_for_testing, source_tree_inventory
    ):
        """Response should not expose the owner_id field."""
        response = client.post(
            self.route(domain_for_testing["id"]),
            json={"source_tree_inventory_id": source_tree_inventory["id"]},
        )
        assert response.status_code == 201
        assert "owner_id" not in response.json()

    def test_invalid_domain_returns_404(self, client, source_tree_inventory):
        """Non-existent domain_id returns 404."""
        response = client.post(
            self.route("00000000000000000000000000000000"),
            json={"source_tree_inventory_id": source_tree_inventory["id"]},
        )
        assert response.status_code == 404

    def test_wrong_owner_domain_returns_404(
        self, client, domain_with_different_owner, source_tree_inventory
    ):
        """Domain owned by another user returns 404."""
        response = client.post(
            self.route(domain_with_different_owner["id"]),
            json={"source_tree_inventory_id": source_tree_inventory["id"]},
        )
        assert response.status_code == 404

    def test_nonexistent_source_inventory_returns_404(self, client, domain_for_testing):
        """Referencing a non-existent source inventory returns 404."""
        response = client.post(
            self.route(domain_for_testing["id"]),
            json={"source_tree_inventory_id": "00000000000000000000000000000000"},
        )
        assert response.status_code == 404

    def test_source_inventory_in_different_domain_returns_404(
        self, client, domain_for_testing, source_inventory_in_different_domain
    ):
        """Source inventory belonging to a different domain returns 404."""
        response = client.post(
            self.route(domain_for_testing["id"]),
            json={
                "source_tree_inventory_id": source_inventory_in_different_domain["id"]
            },
        )
        assert response.status_code == 404

    def test_source_inventory_not_completed_returns_422(
        self, client, firestore_client, domain_for_testing
    ):
        """Source inventory that is still pending returns 422 (status mismatch)."""
        inventory_data = make_inventory_data(
            domain_id=domain_for_testing["id"],
            name="Pending source inventory",
            status="pending",
            inventory_type="tree",
        )
        inventory_data["columns"] = _POSITION_HEIGHT_COLUMNS
        doc_ref = firestore_client.collection(INVENTORIES_COLLECTION).document(
            inventory_data["id"]
        )
        doc_ref.set(inventory_data)

        try:
            response = client.post(
                self.route(domain_for_testing["id"]),
                json={"source_tree_inventory_id": inventory_data["id"]},
            )
            assert response.status_code == 422
        finally:
            doc_ref.delete()

    def test_source_missing_required_columns_returns_422(
        self, client, firestore_client, domain_for_testing
    ):
        """Source inventory missing height returns 422."""
        inventory_data = make_inventory_data(
            domain_id=domain_for_testing["id"],
            name="Source missing height",
            status="completed",
            inventory_type="tree",
        )
        # Drop the required 'height' column.
        inventory_data["columns"] = [
            {"key": "x", "type": "continuous", "unit": "m"},
            {"key": "y", "type": "continuous", "unit": "m"},
        ]
        doc_ref = firestore_client.collection(INVENTORIES_COLLECTION).document(
            inventory_data["id"]
        )
        doc_ref.set(inventory_data)

        try:
            response = client.post(
                self.route(domain_for_testing["id"]),
                json={"source_tree_inventory_id": inventory_data["id"]},
            )
            assert response.status_code == 422
            assert "height" in str(response.json()["detail"]).lower()
        finally:
            doc_ref.delete()

    def test_non_tree_source_returns_422(
        self, client, firestore_client, domain_for_testing
    ):
        """A non-tree source inventory returns 422 (defensive guard)."""
        inventory_data = make_inventory_data(
            domain_id=domain_for_testing["id"],
            name="Non-tree source",
            status="completed",
            inventory_type="tree",
        )
        inventory_data["columns"] = _POSITION_HEIGHT_COLUMNS
        # Write a non-tree type directly (bypasses the create-time schema enum).
        inventory_data["type"] = "shrub"
        doc_ref = firestore_client.collection(INVENTORIES_COLLECTION).document(
            inventory_data["id"]
        )
        doc_ref.set(inventory_data)

        try:
            response = client.post(
                self.route(domain_for_testing["id"]),
                json={"source_tree_inventory_id": inventory_data["id"]},
            )
            assert response.status_code == 422
            assert "tree" in str(response.json()["detail"]).lower()
        finally:
            doc_ref.delete()

    @pytest.mark.parametrize("example_name,example_value", ALL_GDAM_EXAMPLE_VALUES)
    def test_documented_example_creates_inventory(
        self,
        client,
        domain_for_testing,
        source_tree_inventory,
        example_name,
        example_value,
    ):
        """Each documented GDAM example should successfully create an inventory."""
        body = {**example_value}
        if body.get("source_tree_inventory_id") == "PLACEHOLDER_INVENTORY_ID":
            body["source_tree_inventory_id"] = source_tree_inventory["id"]

        response = client.post(self.route(domain_for_testing["id"]), json=body)

        assert response.status_code == 201, (
            f"Example '{example_name}' failed with status {response.status_code}: "
            f"{response.json()}"
        )
        data = response.json()
        assert data["source"]["name"] == "gdam"
        assert data["status"] == "pending"
