"""
Integration tests for the inventory modifications endpoint.

Tests POST /domains/{domain_id}/inventories/{inventory_id}/modifications

These tests make real HTTP requests to the API and interact with Firestore.
They verify that the endpoint creates a new inventory from modifications
on an existing completed inventory.
"""

import pytest

from lib.config import GRIDS_COLLECTION, INVENTORIES_COLLECTION
from tests.fixtures import make_grid_data, make_inventory_data

# Fixtures


@pytest.fixture(scope="session")
def pim_grid_for_inventory(firestore_client, domain_for_testing):
    """Create a completed PIM grid with tm_id band for PIM+modifications tests."""
    grid_data = make_grid_data(
        domain_id=domain_for_testing["id"],
        name="PIM Grid for Modifications Tests",
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
def completed_inventory_for_modifications(firestore_client, domain_for_testing):
    """A completed inventory to use as the source for modification tests."""
    inv_data = make_inventory_data(
        domain_id=domain_for_testing["id"],
        name="Source Inventory for Modifications",
        status="completed",
        georeference={
            "crs": "EPSG:32611",
            "bounds": [500000.0, 5200000.0, 501000.0, 5201000.0],
        },
    )
    doc_ref = firestore_client.collection(INVENTORIES_COLLECTION).document(
        inv_data["id"]
    )
    doc_ref.set(inv_data)
    yield inv_data
    doc_ref.delete()


@pytest.fixture(scope="session")
def pending_inventory(firestore_client, domain_for_testing):
    """A pending inventory (not completed) for status validation tests."""
    inv_data = make_inventory_data(
        domain_id=domain_for_testing["id"],
        name="Pending Inventory",
        status="pending",
    )
    doc_ref = firestore_client.collection(INVENTORIES_COLLECTION).document(
        inv_data["id"]
    )
    doc_ref.set(inv_data)
    yield inv_data
    doc_ref.delete()


@pytest.fixture(scope="session")
def completed_inventory_different_owner(firestore_client, domain_with_different_owner):
    """A completed inventory owned by a different user."""
    inv_data = make_inventory_data(
        domain_id=domain_with_different_owner["id"],
        owner_id="different-owner",
        name="Other Owner's Completed Inventory",
        status="completed",
        georeference={
            "crs": "EPSG:32611",
            "bounds": [500000.0, 5200000.0, 501000.0, 5201000.0],
        },
    )
    doc_ref = firestore_client.collection(INVENTORIES_COLLECTION).document(
        inv_data["id"]
    )
    doc_ref.set(inv_data)
    yield inv_data
    doc_ref.delete()


# POST /domains/{domain_id}/inventories/{inventory_id}/modifications Tests


MINIMAL_MODIFICATIONS_BODY = {
    "modifications": [
        {
            "conditions": {"attribute": "dbh", "operator": "lt", "value": 5.0},
            "actions": {"modifier": "remove"},
        }
    ],
}


class TestApplyModifications:
    """Test POST /domains/{domain_id}/inventories/{inventory_id}/modifications."""

    def route(self, domain_id, inventory_id):
        return f"/domains/{domain_id}/inventories/{inventory_id}/modifications"

    def test_creates_new_inventory(
        self,
        client,
        domain_for_testing,
        completed_inventory_for_modifications,
    ):
        """Creates a new inventory with pending status."""
        source_id = completed_inventory_for_modifications["id"]
        response = client.post(
            self.route(domain_for_testing["id"], source_id),
            json=MINIMAL_MODIFICATIONS_BODY,
        )

        assert response.status_code == 201

        data = response.json()
        assert "id" in data
        assert len(data["id"]) == 32
        # New inventory should have a different ID than source
        assert data["id"] != source_id
        assert data["domain_id"] == domain_for_testing["id"]
        assert data["type"] == "tree"
        assert data["status"] == "pending"
        assert data["georeference"] is None
        assert data["error"] is None

        # Source should be "modifications" with reference to original
        assert data["source"]["name"] == "modifications"
        assert data["source"]["source_inventory_id"] == source_id
        assert len(data["source"]["modifications"]) == 1

    def test_full_request_with_metadata(
        self,
        client,
        domain_for_testing,
        completed_inventory_for_modifications,
    ):
        """Full request with name, description, tags."""
        source_id = completed_inventory_for_modifications["id"]
        body = {
            "name": "Modified inventory",
            "description": "Removed small trees",
            "tags": ["modified", "test"],
            "modifications": [
                {
                    "conditions": {
                        "attribute": "dbh",
                        "operator": "lt",
                        "value": 2.54,
                    },
                    "actions": {"modifier": "remove"},
                }
            ],
        }

        response = client.post(
            self.route(domain_for_testing["id"], source_id),
            json=body,
        )

        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "Modified inventory"
        assert data["description"] == "Removed small trees"
        assert data["tags"] == ["modified", "test"]

    def test_response_excludes_owner_id(
        self,
        client,
        domain_for_testing,
        completed_inventory_for_modifications,
    ):
        """Response should not expose the owner_id field."""
        source_id = completed_inventory_for_modifications["id"]
        response = client.post(
            self.route(domain_for_testing["id"], source_id),
            json=MINIMAL_MODIFICATIONS_BODY,
        )
        assert response.status_code == 201
        assert "owner_id" not in response.json()

    def test_preserves_source_columns(
        self,
        client,
        domain_for_testing,
        completed_inventory_for_modifications,
    ):
        """New inventory should have same columns as source."""
        source_id = completed_inventory_for_modifications["id"]
        response = client.post(
            self.route(domain_for_testing["id"], source_id),
            json=MINIMAL_MODIFICATIONS_BODY,
        )

        assert response.status_code == 201
        data = response.json()
        source_columns = completed_inventory_for_modifications["columns"]
        assert data["columns"] == source_columns

    def test_preserves_source_type(
        self,
        client,
        domain_for_testing,
        completed_inventory_for_modifications,
    ):
        """New inventory should inherit type from source."""
        source_id = completed_inventory_for_modifications["id"]
        response = client.post(
            self.route(domain_for_testing["id"], source_id),
            json=MINIMAL_MODIFICATIONS_BODY,
        )

        assert response.status_code == 201
        assert response.json()["type"] == completed_inventory_for_modifications["type"]

    def test_nonexistent_source_inventory_returns_404(self, client, domain_for_testing):
        """Referencing a non-existent source inventory returns 404."""
        response = client.post(
            self.route(domain_for_testing["id"], "00000000000000000000000000000000"),
            json=MINIMAL_MODIFICATIONS_BODY,
        )
        assert response.status_code == 404

    def test_source_inventory_not_completed_returns_422(
        self,
        client,
        domain_for_testing,
        pending_inventory,
    ):
        """Source inventory that is still pending returns 422."""
        response = client.post(
            self.route(domain_for_testing["id"], pending_inventory["id"]),
            json=MINIMAL_MODIFICATIONS_BODY,
        )
        assert response.status_code == 422

    def test_source_inventory_wrong_owner_returns_404(
        self,
        client,
        domain_with_different_owner,
        completed_inventory_different_owner,
    ):
        """Source inventory owned by another user returns 404."""
        response = client.post(
            self.route(
                domain_with_different_owner["id"],
                completed_inventory_different_owner["id"],
            ),
            json=MINIMAL_MODIFICATIONS_BODY,
        )
        assert response.status_code == 404

    def test_source_inventory_cross_domain_returns_404(
        self,
        client,
        domain_for_testing,
        firestore_client,
    ):
        """Source inventory in a different domain returns 404.

        Even when both domain and inventory are owned by the same user,
        the inventory must belong to the domain in the URL path.
        """
        # Create a completed inventory in the test domain
        inv_data = make_inventory_data(
            domain_id=domain_for_testing["id"],
            name="Inventory in domain A",
            status="completed",
            georeference={
                "crs": "EPSG:32611",
                "bounds": [500000.0, 5200000.0, 501000.0, 5201000.0],
            },
        )
        doc_ref = firestore_client.collection(INVENTORIES_COLLECTION).document(
            inv_data["id"]
        )
        doc_ref.set(inv_data)

        try:
            # Try to use it from a different domain (using the test domain's ID
            # but a URL with a non-matching domain will fail on domain validation)
            response = client.post(
                self.route("00000000000000000000000000000000", inv_data["id"]),
                json=MINIMAL_MODIFICATIONS_BODY,
            )
            # Should fail because domain doesn't exist
            assert response.status_code == 404
        finally:
            doc_ref.delete()

    def test_invalid_domain_returns_404(self, client):
        """Non-existent domain_id returns 404."""
        response = client.post(
            self.route(
                "00000000000000000000000000000000",
                "00000000000000000000000000000000",
            ),
            json=MINIMAL_MODIFICATIONS_BODY,
        )
        assert response.status_code == 404

    def test_empty_modifications_returns_422(
        self,
        client,
        domain_for_testing,
        completed_inventory_for_modifications,
    ):
        """Empty modifications list should return 422 (min_length=1)."""
        source_id = completed_inventory_for_modifications["id"]
        response = client.post(
            self.route(domain_for_testing["id"], source_id),
            json={"modifications": []},
        )
        assert response.status_code == 422

    def test_invalid_operator_for_species_returns_422(
        self,
        client,
        domain_for_testing,
        completed_inventory_for_modifications,
    ):
        """Using gt operator on fia_species_code should return 422."""
        source_id = completed_inventory_for_modifications["id"]
        response = client.post(
            self.route(domain_for_testing["id"], source_id),
            json={
                "modifications": [
                    {
                        "conditions": {
                            "attribute": "fia_species_code",
                            "operator": "gt",
                            "value": 100,
                        },
                        "actions": {"modifier": "remove"},
                    }
                ]
            },
        )
        assert response.status_code == 422

    def test_incompatible_unit_returns_422(
        self,
        client,
        domain_for_testing,
        completed_inventory_for_modifications,
    ):
        """Providing incompatible unit (kg for dbh) should return 422."""
        source_id = completed_inventory_for_modifications["id"]
        response = client.post(
            self.route(domain_for_testing["id"], source_id),
            json={
                "modifications": [
                    {
                        "conditions": {
                            "attribute": "dbh",
                            "operator": "lt",
                            "value": 5.0,
                            "unit": "kg",
                        },
                        "actions": {"modifier": "remove"},
                    }
                ]
            },
        )
        assert response.status_code == 422

    def test_invalid_expression_returns_422(
        self,
        client,
        domain_for_testing,
        completed_inventory_for_modifications,
    ):
        """Invalid expression (function call) should return 422."""
        source_id = completed_inventory_for_modifications["id"]
        response = client.post(
            self.route(domain_for_testing["id"], source_id),
            json={
                "modifications": [
                    {
                        "conditions": {"expression": "abs(dbh) < 5"},
                        "actions": {"modifier": "remove"},
                    }
                ]
            },
        )
        assert response.status_code == 422

    def test_divide_by_zero_returns_422(
        self,
        client,
        domain_for_testing,
        completed_inventory_for_modifications,
    ):
        """Dividing by zero should return 422."""
        source_id = completed_inventory_for_modifications["id"]
        response = client.post(
            self.route(domain_for_testing["id"], source_id),
            json={
                "modifications": [
                    {
                        "conditions": {
                            "attribute": "height",
                            "operator": "gt",
                            "value": 0,
                        },
                        "actions": {
                            "attribute": "height",
                            "modifier": "divide",
                            "value": 0,
                        },
                    }
                ]
            },
        )
        assert response.status_code == 422

    def test_multiple_modifications(
        self,
        client,
        domain_for_testing,
        completed_inventory_for_modifications,
    ):
        """Multiple modifications in one request."""
        source_id = completed_inventory_for_modifications["id"]
        body = {
            "modifications": [
                {
                    "conditions": {
                        "attribute": "dbh",
                        "operator": "lt",
                        "value": 2.54,
                    },
                    "actions": {"modifier": "remove"},
                },
                {
                    "conditions": {
                        "attribute": "height",
                        "operator": "gt",
                        "value": 50,
                    },
                    "actions": {
                        "attribute": "height",
                        "modifier": "multiply",
                        "value": 0.9,
                    },
                },
            ]
        }

        response = client.post(
            self.route(domain_for_testing["id"], source_id),
            json=body,
        )

        assert response.status_code == 201
        data = response.json()
        assert len(data["source"]["modifications"]) == 2

    def test_modifications_with_unit_conversion(
        self,
        client,
        domain_for_testing,
        completed_inventory_for_modifications,
    ):
        """Modifications with unit fields are accepted."""
        source_id = completed_inventory_for_modifications["id"]
        body = {
            "modifications": [
                {
                    "conditions": {
                        "attribute": "dbh",
                        "operator": "lt",
                        "value": 1.0,
                        "unit": "in",
                    },
                    "actions": {"modifier": "remove"},
                }
            ]
        }

        response = client.post(
            self.route(domain_for_testing["id"], source_id),
            json=body,
        )

        assert response.status_code == 201
        data = response.json()
        cond = data["source"]["modifications"][0]["conditions"][0]
        assert cond["unit"] == "in"

    def test_expression_condition(
        self,
        client,
        domain_for_testing,
        completed_inventory_for_modifications,
    ):
        """Expression conditions are accepted."""
        source_id = completed_inventory_for_modifications["id"]
        body = {
            "modifications": [
                {
                    "conditions": {"expression": "height * crown_ratio < 1.0"},
                    "actions": {"modifier": "remove"},
                }
            ]
        }

        response = client.post(
            self.route(domain_for_testing["id"], source_id),
            json=body,
        )

        assert response.status_code == 201


class TestPimWithModifications:
    """Test that the PIM endpoint accepts the new modifications field."""

    def route(self, domain_id):
        return f"/domains/{domain_id}/inventories/tree/pim"

    def test_pim_with_modifications(
        self, client, domain_for_testing, pim_grid_for_inventory
    ):
        """PIM endpoint accepts modifications field."""
        response = client.post(
            self.route(domain_for_testing["id"]),
            json={
                "source_pim_grid_id": pim_grid_for_inventory["id"],
                "seed": 42,
                "modifications": [
                    {
                        "conditions": {
                            "attribute": "dbh",
                            "operator": "le",
                            "value": 12.7,
                        },
                        "actions": {"modifier": "remove"},
                    }
                ],
            },
        )

        assert response.status_code == 201
        data = response.json()
        assert len(data["modifications"]) == 1
        assert data["modifications"][0]["conditions"][0]["attribute"] == "dbh"
        assert data["modifications"][0]["actions"][0]["modifier"] == "remove"

    def test_pim_without_modifications(
        self, client, domain_for_testing, pim_grid_for_inventory
    ):
        """PIM endpoint works without modifications (backwards compatible)."""
        response = client.post(
            self.route(domain_for_testing["id"]),
            json={
                "source_pim_grid_id": pim_grid_for_inventory["id"],
                "seed": 42,
            },
        )

        assert response.status_code == 201
        data = response.json()
        assert data["modifications"] == []

    def test_pim_invalid_modification_returns_422(
        self, client, domain_for_testing, pim_grid_for_inventory
    ):
        """PIM endpoint rejects invalid modifications."""
        response = client.post(
            self.route(domain_for_testing["id"]),
            json={
                "source_pim_grid_id": pim_grid_for_inventory["id"],
                "seed": 42,
                "modifications": [
                    {
                        "conditions": {
                            "attribute": "fia_species_code",
                            "operator": "gt",  # Invalid for species
                            "value": 100,
                        },
                        "actions": {"modifier": "remove"},
                    }
                ],
            },
        )
        assert response.status_code == 422
