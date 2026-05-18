"""
Integration tests for api/resources/grids/voxelize/inventory/tree/router.py.

Tests the POST /domains/{domain_id}/grids/voxelize/inventory/tree endpoint.
These tests make real HTTP requests to the API and interact with Firestore
and Cloud Tasks.
"""

import pytest
from api.resources.grids.voxelize.inventory.tree.examples import (
    ALL_TREE_INVENTORY_EXAMPLE_VALUES,
)

from lib.config import DOMAINS_COLLECTION, INVENTORIES_COLLECTION
from tests.fixtures import make_domain_data, make_inventory_data

# --- Fixtures ---


@pytest.fixture(scope="session")
def tree_inventory_for_voxelization(firestore_client, domain_for_testing):
    """A completed tree inventory in domain_for_testing."""
    inventory_data = make_inventory_data(
        domain_id=domain_for_testing["id"],
        name="Tree inventory for voxelization tests",
        status="completed",
        inventory_type="tree",
    )
    doc_ref = firestore_client.collection(INVENTORIES_COLLECTION).document(
        inventory_data["id"]
    )
    doc_ref.set(inventory_data)
    yield inventory_data
    doc_ref.delete()


@pytest.fixture(scope="session")
def second_domain_for_tree_voxelization(firestore_client):
    """A second domain owned by test-owner, for cross-domain tests."""
    domain_data = make_domain_data(name="Second Domain for Tree Voxelization")
    doc_ref = firestore_client.collection(DOMAINS_COLLECTION).document(
        domain_data["id"]
    )
    doc_ref.set(domain_data)
    yield domain_data
    doc_ref.delete()


@pytest.fixture(scope="session")
def tree_inventory_in_different_domain(
    firestore_client, second_domain_for_tree_voxelization
):
    """A completed tree inventory in a different domain than domain_for_testing."""
    inventory_data = make_inventory_data(
        domain_id=second_domain_for_tree_voxelization["id"],
        name="Tree inventory in second domain",
        status="completed",
        inventory_type="tree",
    )
    doc_ref = firestore_client.collection(INVENTORIES_COLLECTION).document(
        inventory_data["id"]
    )
    doc_ref.set(inventory_data)
    yield inventory_data
    doc_ref.delete()


# --- Tests ---


class TestCreateTreeInventoryGrid:
    """POST /domains/{domain_id}/grids/voxelize/inventory/tree."""

    def route(self, domain_id):
        return f"/domains/{domain_id}/grids/voxelize/inventory/tree"

    def test_minimal_request_creates_grid(
        self, client, domain_for_testing, tree_inventory_for_voxelization
    ):
        """Minimal request creates a pending grid with resolved defaults."""
        body = {
            "source_inventory_id": tree_inventory_for_voxelization["id"],
        }
        response = client.post(self.route(domain_for_testing["id"]), json=body)

        assert response.status_code == 201

        data = response.json()
        assert "id" in data
        assert len(data["id"]) == 32
        assert data["domain_id"] == domain_for_testing["id"]
        assert data["status"] == "pending"
        assert data["name"] == ""
        assert data["description"] == ""
        assert data["tags"] == []
        assert data["modifications"] == []

        # Source metadata captures every resolved default.
        source = data["source"]
        assert source["operation"] == "voxelize"
        assert source["input"] == "inventory"
        assert source["entity"] == "tree"
        assert source["source_inventory_id"] == tree_inventory_for_voxelization["id"]
        assert source["resolution"] == {"horizontal": 2.0, "vertical": 1.0}
        assert source["bands"] == ["bulk_density.foliage.live"]
        assert source["crown_profile_model"] == "purves"
        assert source["biomass_source"] == {
            "type": "allometry",
            "equations": "nsvb",
            "components": ["foliage"],
            "component_states": {"foliage": {"live": 1.0, "dead": 0.0}},
        }
        assert "moisture_model" not in source

        # Single continuous band with index 0.
        assert len(data["bands"]) == 1
        assert data["bands"][0]["key"] == "bulk_density.foliage.live"
        assert data["bands"][0]["type"] == "continuous"
        assert data["bands"][0]["unit"] == "kg/m**3"
        assert data["bands"][0]["index"] == 0

    def test_all_bands_request_creates_grid(
        self, client, domain_for_testing, tree_inventory_for_voxelization
    ):
        """Request with all six TreeBand values succeeds."""
        body = {
            "source_inventory_id": tree_inventory_for_voxelization["id"],
            "resolution": {"horizontal": 2.0, "vertical": 1.0},
            "bands": [
                "bulk_density.foliage.live",
                "fuel_moisture.live",
                "savr.foliage",
                "spcd",
                "tree_id",
                "volume_fraction",
            ],
        }
        response = client.post(self.route(domain_for_testing["id"]), json=body)
        assert response.status_code == 201

        data = response.json()
        assert len(data["bands"]) == 6
        assert [b["index"] for b in data["bands"]] == [0, 1, 2, 3, 4, 5]
        # fuel_moisture.live in bands triggers default uniform moisture model.
        assert data["source"]["moisture_model"] == {
            "live": {"method": "uniform", "value": 100.0},
        }

    def test_request_with_metadata(
        self, client, domain_for_testing, tree_inventory_for_voxelization
    ):
        """Name, description, tags round-trip through the endpoint."""
        body = {
            "source_inventory_id": tree_inventory_for_voxelization["id"],
            "resolution": {"horizontal": 1.5, "vertical": 0.75},
            "bands": ["bulk_density.foliage.live"],
            "name": "Named tree grid",
            "description": "A tree grid with metadata",
            "tags": ["tree", "test"],
        }
        response = client.post(self.route(domain_for_testing["id"]), json=body)
        assert response.status_code == 201

        data = response.json()
        assert data["name"] == "Named tree grid"
        assert data["description"] == "A tree grid with metadata"
        assert data["tags"] == ["tree", "test"]

    def test_georeference_is_null_on_creation(
        self, client, domain_for_testing, tree_inventory_for_voxelization
    ):
        """Georeference is null until Treevox completes voxelization."""
        body = {
            "source_inventory_id": tree_inventory_for_voxelization["id"],
            "resolution": {"horizontal": 2.0, "vertical": 1.0},
            "bands": ["bulk_density.foliage.live"],
        }
        response = client.post(self.route(domain_for_testing["id"]), json=body)
        assert response.status_code == 201
        assert response.json()["georeference"] is None

    def test_chunks_is_null_on_creation(
        self, client, domain_for_testing, tree_inventory_for_voxelization
    ):
        """chunks is null until Treevox computes the 3D chunk layout."""
        body = {
            "source_inventory_id": tree_inventory_for_voxelization["id"],
            "resolution": {"horizontal": 2.0, "vertical": 1.0},
            "bands": ["bulk_density.foliage.live"],
        }
        response = client.post(self.route(domain_for_testing["id"]), json=body)
        assert response.status_code == 201
        assert response.json()["chunks"] is None

    def test_response_excludes_owner_id(
        self, client, domain_for_testing, tree_inventory_for_voxelization
    ):
        body = {
            "source_inventory_id": tree_inventory_for_voxelization["id"],
            "resolution": {"horizontal": 2.0, "vertical": 1.0},
            "bands": ["bulk_density.foliage.live"],
        }
        response = client.post(self.route(domain_for_testing["id"]), json=body)
        assert response.status_code == 201
        assert "owner_id" not in response.json()

    # --- Domain validation ---

    def test_invalid_domain_returns_404(self, client, tree_inventory_for_voxelization):
        body = {
            "source_inventory_id": tree_inventory_for_voxelization["id"],
            "resolution": {"horizontal": 2.0, "vertical": 1.0},
            "bands": ["bulk_density.foliage.live"],
        }
        response = client.post(
            self.route("00000000000000000000000000000000"), json=body
        )
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_wrong_owner_domain_returns_404(
        self,
        client,
        domain_with_different_owner,
        tree_inventory_for_voxelization,
    ):
        body = {
            "source_inventory_id": tree_inventory_for_voxelization["id"],
            "resolution": {"horizontal": 2.0, "vertical": 1.0},
            "bands": ["bulk_density.foliage.live"],
        }
        response = client.post(self.route(domain_with_different_owner["id"]), json=body)
        assert response.status_code == 404

    # --- Source inventory validation ---

    def test_nonexistent_source_inventory_returns_404(self, client, domain_for_testing):
        body = {
            "source_inventory_id": "00000000000000000000000000000000",
            "resolution": {"horizontal": 2.0, "vertical": 1.0},
            "bands": ["bulk_density.foliage.live"],
        }
        response = client.post(self.route(domain_for_testing["id"]), json=body)
        assert response.status_code == 404

    def test_source_inventory_in_different_domain_returns_404(
        self,
        client,
        domain_for_testing,
        tree_inventory_in_different_domain,
    ):
        body = {
            "source_inventory_id": tree_inventory_in_different_domain["id"],
            "resolution": {"horizontal": 2.0, "vertical": 1.0},
            "bands": ["bulk_density.foliage.live"],
        }
        response = client.post(self.route(domain_for_testing["id"]), json=body)
        assert response.status_code == 404

    def test_source_inventory_not_completed_returns_422(
        self, client, firestore_client, domain_for_testing
    ):
        """A pending source inventory cannot be voxelized."""
        pending_inv = make_inventory_data(
            domain_id=domain_for_testing["id"],
            name="Pending inventory",
            status="pending",
            inventory_type="tree",
        )
        doc_ref = firestore_client.collection(INVENTORIES_COLLECTION).document(
            pending_inv["id"]
        )
        doc_ref.set(pending_inv)
        try:
            body = {
                "source_inventory_id": pending_inv["id"],
                "resolution": {"horizontal": 2.0, "vertical": 1.0},
                "bands": ["bulk_density.foliage.live"],
            }
            response = client.post(self.route(domain_for_testing["id"]), json=body)
            assert response.status_code == 422
        finally:
            doc_ref.delete()

    # --- Request body validation ---

    def test_missing_source_inventory_id_returns_422(self, client, domain_for_testing):
        body = {
            "resolution": {"horizontal": 2.0, "vertical": 1.0},
            "bands": ["bulk_density.foliage.live"],
        }
        response = client.post(self.route(domain_for_testing["id"]), json=body)
        assert response.status_code == 422

    def test_missing_resolution_uses_default(
        self, client, domain_for_testing, tree_inventory_for_voxelization
    ):
        body = {
            "source_inventory_id": tree_inventory_for_voxelization["id"],
            "bands": ["bulk_density.foliage.live"],
        }
        response = client.post(self.route(domain_for_testing["id"]), json=body)
        assert response.status_code == 201
        assert response.json()["source"]["resolution"] == {
            "horizontal": 2.0,
            "vertical": 1.0,
        }

    def test_missing_bands_uses_default(
        self, client, domain_for_testing, tree_inventory_for_voxelization
    ):
        body = {
            "source_inventory_id": tree_inventory_for_voxelization["id"],
            "resolution": {"horizontal": 2.0, "vertical": 1.0},
        }
        response = client.post(self.route(domain_for_testing["id"]), json=body)
        assert response.status_code == 201
        assert response.json()["source"]["bands"] == ["bulk_density.foliage.live"]

    def test_empty_bands_returns_422(
        self, client, domain_for_testing, tree_inventory_for_voxelization
    ):
        body = {
            "source_inventory_id": tree_inventory_for_voxelization["id"],
            "resolution": {"horizontal": 2.0, "vertical": 1.0},
            "bands": [],
        }
        response = client.post(self.route(domain_for_testing["id"]), json=body)
        assert response.status_code == 422

    def test_duplicate_bands_returns_422(
        self, client, domain_for_testing, tree_inventory_for_voxelization
    ):
        body = {
            "source_inventory_id": tree_inventory_for_voxelization["id"],
            "resolution": {"horizontal": 2.0, "vertical": 1.0},
            "bands": ["bulk_density.foliage.live", "bulk_density.foliage.live"],
        }
        response = client.post(self.route(domain_for_testing["id"]), json=body)
        assert response.status_code == 422

    def test_invalid_band_name_returns_422(
        self, client, domain_for_testing, tree_inventory_for_voxelization
    ):
        body = {
            "source_inventory_id": tree_inventory_for_voxelization["id"],
            "resolution": {"horizontal": 2.0, "vertical": 1.0},
            "bands": ["not_a_real_band"],
        }
        response = client.post(self.route(domain_for_testing["id"]), json=body)
        assert response.status_code == 422

    @pytest.mark.parametrize(
        "resolution",
        [
            {"horizontal": 0.0, "vertical": 1.0},
            {"horizontal": -1.0, "vertical": 1.0},
            {"horizontal": 2.0, "vertical": 0.0},
        ],
    )
    def test_non_positive_resolution_returns_422(
        self,
        client,
        domain_for_testing,
        tree_inventory_for_voxelization,
        resolution,
    ):
        body = {
            "source_inventory_id": tree_inventory_for_voxelization["id"],
            "resolution": resolution,
            "bands": ["bulk_density.foliage.live"],
        }
        response = client.post(self.route(domain_for_testing["id"]), json=body)
        assert response.status_code == 422

    def test_invalid_crown_profile_returns_422(
        self, client, domain_for_testing, tree_inventory_for_voxelization
    ):
        body = {
            "source_inventory_id": tree_inventory_for_voxelization["id"],
            "resolution": {"horizontal": 2.0, "vertical": 1.0},
            "bands": ["bulk_density.foliage.live"],
            "crown_profile_model": "watershed",
        }
        response = client.post(self.route(domain_for_testing["id"]), json=body)
        assert response.status_code == 422

    def test_invalid_biomass_source_returns_422(
        self, client, domain_for_testing, tree_inventory_for_voxelization
    ):
        body = {
            "source_inventory_id": tree_inventory_for_voxelization["id"],
            "resolution": {"horizontal": 2.0, "vertical": 1.0},
            "bands": ["bulk_density.foliage.live"],
            "biomass_source": {
                "type": "allometry",
                "equations": "allometric",
                "components": ["foliage"],
            },
        }
        response = client.post(self.route(domain_for_testing["id"]), json=body)
        assert response.status_code == 422

    def test_invalid_moisture_method_returns_422(
        self, client, domain_for_testing, tree_inventory_for_voxelization
    ):
        body = {
            "source_inventory_id": tree_inventory_for_voxelization["id"],
            "resolution": {"horizontal": 2.0, "vertical": 1.0},
            "bands": ["bulk_density.foliage.live", "fuel_moisture.live"],
            "moisture_model": {"live": {"method": "fosberg", "value": 100.0}},
        }
        response = client.post(self.route(domain_for_testing["id"]), json=body)
        assert response.status_code == 422

    def test_band_component_mismatch_returns_422(
        self, client, domain_for_testing, tree_inventory_for_voxelization
    ):
        """API rejects bulk_density bands whose component isn't configured."""
        body = {
            "source_inventory_id": tree_inventory_for_voxelization["id"],
            "resolution": {"horizontal": 2.0, "vertical": 1.0},
            "bands": ["bulk_density.foliage.live"],
            "biomass_source": {
                "type": "allometry",
                "equations": "nsvb",
                "components": ["branchwood"],
            },
        }
        response = client.post(self.route(domain_for_testing["id"]), json=body)
        assert response.status_code == 422
        assert "biomass_source.components" in str(response.json()["detail"]).lower()

    # --- Default-resolution behavior ---

    def test_fuel_moisture_live_auto_populates_default_moisture_model(
        self, client, domain_for_testing, tree_inventory_for_voxelization
    ):
        body = {
            "source_inventory_id": tree_inventory_for_voxelization["id"],
            "resolution": {"horizontal": 2.0, "vertical": 1.0},
            "bands": ["bulk_density.foliage.live", "fuel_moisture.live"],
        }
        response = client.post(self.route(domain_for_testing["id"]), json=body)
        assert response.status_code == 201
        assert response.json()["source"]["moisture_model"] == {
            "live": {"method": "uniform", "value": 100.0},
        }

    def test_fuel_moisture_dead_auto_populates_default_moisture_model(
        self, client, domain_for_testing, tree_inventory_for_voxelization
    ):
        body = {
            "source_inventory_id": tree_inventory_for_voxelization["id"],
            "resolution": {"horizontal": 2.0, "vertical": 1.0},
            "bands": ["bulk_density.foliage.dead", "fuel_moisture.dead"],
        }
        response = client.post(self.route(domain_for_testing["id"]), json=body)
        assert response.status_code == 201
        assert response.json()["source"]["moisture_model"] == {
            "dead": {"method": "uniform", "value": 10.0},
        }

    def test_moisture_model_stripped_when_fuel_moisture_band_absent(
        self, client, domain_for_testing, tree_inventory_for_voxelization
    ):
        """moisture_model should be stripped when fuel_moisture.live is not requested."""
        body = {
            "source_inventory_id": tree_inventory_for_voxelization["id"],
            "resolution": {"horizontal": 2.0, "vertical": 1.0},
            "bands": ["bulk_density.foliage.live"],
            "moisture_model": {"live": {"method": "uniform", "value": 50.0}},
        }
        response = client.post(self.route(domain_for_testing["id"]), json=body)
        assert response.status_code == 201
        assert "moisture_model" not in response.json()["source"]

    def test_nested_biomass_source_is_stored(
        self, client, domain_for_testing, tree_inventory_for_voxelization
    ):
        body = {
            "source_inventory_id": tree_inventory_for_voxelization["id"],
            "resolution": {"horizontal": 2.0, "vertical": 1.0},
            "bands": ["bulk_density.fine.live"],
            "biomass_source": {
                "type": "allometry",
                "equations": "jenkins",
                "components": ["fine"],
                "fine": {
                    "recipe": "foliage_plus_branchwood_fraction",
                    "branchwood_fraction": 0.1,
                },
            },
        }
        response = client.post(self.route(domain_for_testing["id"]), json=body)
        assert response.status_code == 201
        expected = {
            **body["biomass_source"],
            "component_states": {"fine": {"live": 1.0, "dead": 0.0}},
        }
        assert response.json()["source"]["biomass_source"] == expected

    def test_inventory_column_biomass_source_is_stored(
        self, client, domain_for_testing, tree_inventory_for_voxelization
    ):
        body = {
            "source_inventory_id": tree_inventory_for_voxelization["id"],
            "resolution": {"horizontal": 2.0, "vertical": 1.0},
            "bands": ["bulk_density.foliage.live"],
            "biomass_source": {
                "type": "inventory_columns",
                "columns": {
                    "foliage": {
                        "column": "my_fuel_load_col",
                        "unit": "kg",
                    }
                },
                "components": ["foliage"],
            },
        }
        response = client.post(self.route(domain_for_testing["id"]), json=body)
        assert response.status_code == 201
        expected = {
            **body["biomass_source"],
            "component_states": {"foliage": {"live": 1.0, "dead": 0.0}},
        }
        assert response.json()["source"]["biomass_source"] == expected

    def test_non_foliage_output_bands_are_accepted_for_async_failure(
        self, client, domain_for_testing, tree_inventory_for_voxelization
    ):
        body = {
            "source_inventory_id": tree_inventory_for_voxelization["id"],
            "resolution": {"horizontal": 2.0, "vertical": 1.0},
            "bands": ["bulk_density.fine.live"],
            "biomass_source": {
                "type": "allometry",
                "equations": "nsvb",
                "components": ["fine"],
                "fine": {
                    "recipe": "foliage_plus_branchwood_fraction",
                    "branchwood_fraction": 0.1,
                },
            },
        }
        response = client.post(self.route(domain_for_testing["id"]), json=body)
        assert response.status_code == 201
        data = response.json()
        assert data["source"]["bands"] == ["bulk_density.fine.live"]
        assert data["bands"][0]["key"] == "bulk_density.fine.live"

    @pytest.mark.parametrize(
        "example_name,example_value", ALL_TREE_INVENTORY_EXAMPLE_VALUES
    )
    def test_documented_example_creates_grid(
        self,
        client,
        domain_for_testing,
        tree_inventory_for_voxelization,
        example_name,
        example_value,
    ):
        """Every documented example must produce a successful request."""
        body = {**example_value}
        if body.get("source_inventory_id") == "PLACEHOLDER_INVENTORY_ID":
            body["source_inventory_id"] = tree_inventory_for_voxelization["id"]

        response = client.post(self.route(domain_for_testing["id"]), json=body)
        assert response.status_code == 201, (
            f"Example '{example_name}' failed with status "
            f"{response.status_code}: {response.json()}"
        )
        data = response.json()
        assert data["source"]["operation"] == "voxelize"
        assert data["source"]["input"] == "inventory"
        assert data["source"]["entity"] == "tree"
        assert data["status"] == "pending"
