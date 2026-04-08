"""
Integration tests for api/v2/resources/inventories/tree/chm/router.py

Tests the CHM extraction inventory creation endpoint.
These tests make real HTTP requests to the API and interact with Firestore and Cloud Tasks.
"""

import pytest
from api.resources.inventories.tree.chm.examples import ALL_CHM_EXAMPLE_VALUES

from lib.config import DOMAINS_COLLECTION, GRIDS_COLLECTION
from tests.fixtures import make_domain_data, make_grid_data

# --- Fixtures ---


@pytest.fixture(scope="session")
def chm_grid_for_inventory(firestore_client, domain_for_testing):
    """Create a completed CHM grid with 'chm' band for inventory creation tests."""
    grid_data = make_grid_data(
        domain_id=domain_for_testing["id"],
        name="CHM Grid for Inventory Tests",
        status="completed",
        source={
            "name": "chm",
            "product": "naip",
            "description": "NAIP high-resolution canopy height model",
        },
        bands=[
            {"key": "chm", "type": "continuous", "unit": "m", "index": 0},
        ],
    )
    doc_ref = firestore_client.collection(GRIDS_COLLECTION).document(grid_data["id"])
    doc_ref.set(grid_data)
    yield grid_data
    doc_ref.delete()


@pytest.fixture(scope="session")
def chm_grid_in_different_domain(firestore_client, second_domain_for_inventory):
    """A completed CHM grid in a different domain than domain_for_testing."""
    grid_data = make_grid_data(
        domain_id=second_domain_for_inventory["id"],
        name="CHM grid in second domain",
        status="completed",
        source={
            "name": "chm",
            "product": "meta",
            "description": "Meta CHM",
        },
        bands=[
            {"key": "chm", "type": "continuous", "unit": "m", "index": 0},
        ],
    )
    doc_ref = firestore_client.collection(GRIDS_COLLECTION).document(grid_data["id"])
    doc_ref.set(grid_data)
    yield grid_data
    doc_ref.delete()


@pytest.fixture(scope="session")
def second_domain_for_inventory(firestore_client):
    """A second domain owned by test-owner, used for cross-domain tests."""
    domain_data = make_domain_data(name="Second Domain for Inventory Tests")
    doc_ref = firestore_client.collection(DOMAINS_COLLECTION).document(
        domain_data["id"]
    )
    doc_ref.set(domain_data)
    yield domain_data
    doc_ref.delete()


# --- Tests ---


class TestCreateChmInventory:
    """Test the POST /domains/{domain_id}/inventories/tree/chm endpoint."""

    def route(self, domain_id):
        return f"/domains/{domain_id}/inventories/tree/chm"

    def test_minimal_request_creates_inventory(
        self, client, domain_for_testing, chm_grid_for_inventory
    ):
        """Minimal request creates an inventory with default LMF parameters."""
        response = client.post(
            self.route(domain_for_testing["id"]),
            json={"source_chm_grid_id": chm_grid_for_inventory["id"]},
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

        # Check source and default algorithm parameters
        assert data["source"]["name"] == "chm"
        assert data["source"]["source_chm_grid_id"] == chm_grid_for_inventory["id"]
        assert data["source"]["algorithm"]["name"] == "lmf"
        assert data["source"]["algorithm"]["min_height"] == 2.0
        assert data["source"]["algorithm"]["footprint_size"] == 3

    def test_request_with_metadata(
        self, client, domain_for_testing, chm_grid_for_inventory
    ):
        """Request with custom algorithm params, name, description, and tags."""
        request_body = {
            "source_chm_grid_id": chm_grid_for_inventory["id"],
            "algorithm": {
                "name": "lmf",
                "min_height": 5.5,
                "footprint_size": 7,
            },
            "name": "Custom CHM Inventory",
            "description": "Testing custom LMF extraction",
            "tags": ["test", "lidar"],
        }

        response = client.post(self.route(domain_for_testing["id"]), json=request_body)

        assert response.status_code == 201

        data = response.json()
        assert data["name"] == "Custom CHM Inventory"
        assert data["description"] == "Testing custom LMF extraction"
        assert data["tags"] == ["test", "lidar"]
        assert data["source"]["algorithm"]["min_height"] == 5.5
        assert data["source"]["algorithm"]["footprint_size"] == 7

    def test_georeference_is_null_on_creation(
        self, client, domain_for_testing, chm_grid_for_inventory
    ):
        """Georeference is null until backend populates it."""
        response = client.post(
            self.route(domain_for_testing["id"]),
            json={"source_chm_grid_id": chm_grid_for_inventory["id"]},
        )
        assert response.status_code == 201

        data = response.json()
        assert data["georeference"] is None

    def test_invalid_domain_returns_404(self, client, chm_grid_for_inventory):
        """Non-existent domain_id returns 404."""
        response = client.post(
            self.route("00000000000000000000000000000000"),
            json={"source_chm_grid_id": chm_grid_for_inventory["id"]},
        )

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_wrong_owner_domain_returns_404(
        self, client, domain_with_different_owner, chm_grid_for_inventory
    ):
        """Domain owned by another user returns 404."""
        response = client.post(
            self.route(domain_with_different_owner["id"]),
            json={"source_chm_grid_id": chm_grid_for_inventory["id"]},
        )

        assert response.status_code == 404

    def test_response_excludes_owner_id(
        self, client, domain_for_testing, chm_grid_for_inventory
    ):
        """Response should not expose the owner_id field."""
        response = client.post(
            self.route(domain_for_testing["id"]),
            json={"source_chm_grid_id": chm_grid_for_inventory["id"]},
        )
        assert response.status_code == 201

        data = response.json()
        assert "owner_id" not in data

    @pytest.mark.parametrize("example_name,example_value", ALL_CHM_EXAMPLE_VALUES)
    def test_documented_example_creates_inventory(
        self,
        client,
        domain_for_testing,
        chm_grid_for_inventory,
        example_name,
        example_value,
    ):
        """Each documented CHM example should successfully create an inventory."""
        body = {**example_value}
        # Inject the valid grid ID for tests
        if body.get("source_chm_grid_id") == "PLACEHOLDER_GRID_ID":
            body["source_chm_grid_id"] = chm_grid_for_inventory["id"]

        response = client.post(self.route(domain_for_testing["id"]), json=body)

        assert response.status_code == 201, (
            f"Example '{example_name}' failed with status {response.status_code}: "
            f"{response.json()}"
        )

        data = response.json()
        assert data["source"]["name"] == "chm"
        assert data["status"] == "pending"

    # --- CHM Specific Validation Tests ---

    def test_nonexistent_source_grid_returns_404(self, client, domain_for_testing):
        """Referencing a non-existent source grid returns 404."""
        response = client.post(
            self.route(domain_for_testing["id"]),
            json={"source_chm_grid_id": "00000000000000000000000000000000"},
        )
        assert response.status_code == 404

    def test_source_grid_in_different_domain_returns_404(
        self, client, domain_for_testing, chm_grid_in_different_domain
    ):
        """Source grid belonging to a different domain returns 404."""
        response = client.post(
            self.route(domain_for_testing["id"]),
            json={"source_chm_grid_id": chm_grid_in_different_domain["id"]},
        )
        assert response.status_code == 404

    def test_source_grid_not_completed_returns_422(
        self, client, firestore_client, domain_for_testing
    ):
        """Source grid that is still pending returns 422."""
        grid_data = make_grid_data(
            domain_id=domain_for_testing["id"],
            name="Pending CHM Grid",
            status="pending",
            source={"name": "chm", "product": "naip"},
            bands=[{"key": "chm", "type": "continuous", "unit": "m", "index": 0}],
            georeference=None,
        )
        doc_ref = firestore_client.collection(GRIDS_COLLECTION).document(
            grid_data["id"]
        )
        doc_ref.set(grid_data)

        try:
            response = client.post(
                self.route(domain_for_testing["id"]),
                json={"source_chm_grid_id": grid_data["id"]},
            )
            assert response.status_code == 422
        finally:
            doc_ref.delete()

    def test_non_chm_grid_returns_422(
        self, client, firestore_client, domain_for_testing
    ):
        """Source grid that is not a CHM grid (missing source name and band) returns 422."""
        grid_data = make_grid_data(
            domain_id=domain_for_testing["id"],
            name="PIM Grid (not CHM)",
            status="completed",
            source={"name": "pim", "product": "treemap"},
            bands=[{"key": "tm_id", "type": "categorical"}],
        )
        doc_ref = firestore_client.collection(GRIDS_COLLECTION).document(
            grid_data["id"]
        )
        doc_ref.set(grid_data)

        try:
            response = client.post(
                self.route(domain_for_testing["id"]),
                json={"source_chm_grid_id": grid_data["id"]},
            )
            assert response.status_code == 422
            assert "not a chm grid" in response.json()["detail"].lower()
        finally:
            doc_ref.delete()

    def test_invalid_lmf_footprint_returns_422(
        self, client, domain_for_testing, chm_grid_for_inventory
    ):
        """An even footprint_size for the LMF algorithm returns 422."""
        response = client.post(
            self.route(domain_for_testing["id"]),
            json={
                "source_chm_grid_id": chm_grid_for_inventory["id"],
                "algorithm": {
                    "name": "lmf",
                    "footprint_size": 4,  # Invalid: must be odd
                },
            },
        )
        assert response.status_code == 422
        assert "must be an odd integer" in str(response.json()["detail"]).lower()

    def test_request_creates_inventory_with_vwf_algorithm(
        self, client, domain_for_testing, chm_grid_for_inventory
    ):
        """Request with VWF algorithm payload correctly parses and stores parameters."""
        request_body = {
            "source_chm_grid_id": chm_grid_for_inventory["id"],
            "algorithm": {
                "name": "vwf",
                "min_height": 5.0,
                "crown_ratio": 0.15,
            },
        }

        response = client.post(self.route(domain_for_testing["id"]), json=request_body)
        assert response.status_code == 201

        data = response.json()
        assert data["source"]["algorithm"]["name"] == "vwf"
        assert data["source"]["algorithm"]["min_height"] == 5.0
        assert data["source"]["algorithm"]["crown_ratio"] == 0.15
        assert (
            data["source"]["algorithm"]["crown_offset"] == 1.0
        )  # Proves default was set

    def test_invalid_algorithm_name_returns_422(
        self, client, domain_for_testing, chm_grid_for_inventory
    ):
        """Using an unsupported algorithm name triggers Pydantic discriminator 422 error."""
        request_body = {
            "source_chm_grid_id": chm_grid_for_inventory["id"],
            "algorithm": {
                "name": "watershed",  # Does not exist in the Annotated Union
                "min_height": 2.0,
            },
        }

        response = client.post(self.route(domain_for_testing["id"]), json=request_body)
        assert response.status_code == 422

        # FastAPI/Pydantic returns a validation error detailing the discriminator failure
        detail = response.json()["detail"]
        assert any(
            "discriminator" in str(error).lower() or "tag" in str(error).lower()
            for error in detail
        )
