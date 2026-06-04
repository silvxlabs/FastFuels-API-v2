"""
Integration tests for api/v2/resources/grids/lookup/router.py

Tests the FBFM40 lookup endpoint (POST /domains/{domain_id}/grids/lookup/fbfm40).
These tests make real HTTP requests to the API and interact with Firestore.

The lookup endpoint requires a source grid that:
1. Exists and is owned by the requesting user
2. Has status "completed"
3. Contains an "fbfm" band
4. Belongs to the same domain as the URL path
"""

import pytest
from api.resources.grids.lookup.examples import ALL_FBFM40_LOOKUP_EXAMPLE_VALUES

from lib.config import DOMAINS_COLLECTION, GRIDS_COLLECTION
from tests.fixtures import make_domain_data, make_grid_data


@pytest.fixture(scope="session")
def complete_fbfm40_grid(firestore_client, domain_for_testing):
    """A complete LANDFIRE FBFM40 grid for use as a lookup source.

    This grid has status "completed", an "fbfm" band, and a georeference —
    the minimum requirements for a valid lookup source grid.
    """
    grid_data = make_grid_data(
        domain_id=domain_for_testing["id"],
        name="FBFM40 source for lookup tests",
        status="completed",
        source={
            "name": "landfire",
            "product": "fbfm40",
            "version": "2022",
            "description": "LANDFIRE FBFM40 fuel model codes (Scott-Burgan 40 classification)",
        },
        bands=[
            {"key": "fbfm", "type": "categorical", "unit": None, "index": 0},
        ],
    )
    doc_ref = firestore_client.collection(GRIDS_COLLECTION).document(grid_data["id"])
    doc_ref.set(grid_data)
    yield grid_data
    doc_ref.delete()


@pytest.fixture(scope="session")
def pending_fbfm40_grid(firestore_client, domain_for_testing):
    """An FBFM40 grid with status "pending" (not yet complete)."""
    grid_data = make_grid_data(
        domain_id=domain_for_testing["id"],
        name="Pending FBFM40 grid",
        status="pending",
        bands=[
            {"key": "fbfm", "type": "categorical", "unit": None, "index": 0},
        ],
        georeference=None,
    )
    doc_ref = firestore_client.collection(GRIDS_COLLECTION).document(grid_data["id"])
    doc_ref.set(grid_data)
    yield grid_data
    doc_ref.delete()


@pytest.fixture(scope="session")
def grid_without_fbfm_band(firestore_client, domain_for_testing):
    """A complete grid that does not have an "fbfm" band."""
    grid_data = make_grid_data(
        domain_id=domain_for_testing["id"],
        name="3DEP elevation grid",
        status="completed",
        source={"name": "3dep", "resolution": "10m", "version": "2023"},
        bands=[
            {"key": "elevation", "type": "continuous", "unit": "m", "index": 0},
        ],
    )
    doc_ref = firestore_client.collection(GRIDS_COLLECTION).document(grid_data["id"])
    doc_ref.set(grid_data)
    yield grid_data
    doc_ref.delete()


@pytest.fixture(scope="session")
def second_domain_for_lookup(firestore_client):
    """A second domain owned by test-owner, used for cross-domain tests."""
    domain_data = make_domain_data(name="Second Domain for Lookup Tests")
    doc_ref = firestore_client.collection(DOMAINS_COLLECTION).document(
        domain_data["id"]
    )
    doc_ref.set(domain_data)
    yield domain_data
    doc_ref.delete()


@pytest.fixture(scope="session")
def grid_in_different_domain(firestore_client, second_domain_for_lookup):
    """A complete FBFM40 grid in a different domain than domain_for_testing.

    Same owner (test-owner) but in second_domain_for_lookup. Used to test
    cross-domain protection: a lookup in domain A should reject a source
    grid that belongs to domain B.
    """
    grid_data = make_grid_data(
        domain_id=second_domain_for_lookup["id"],
        name="FBFM40 grid in second domain",
        status="completed",
        source={
            "name": "landfire",
            "product": "fbfm40",
            "version": "2022",
            "description": "LANDFIRE FBFM40 fuel model codes (Scott-Burgan 40 classification)",
        },
        bands=[
            {"key": "fbfm", "type": "categorical", "unit": None, "index": 0},
        ],
    )
    doc_ref = firestore_client.collection(GRIDS_COLLECTION).document(grid_data["id"])
    doc_ref.set(grid_data)
    yield grid_data
    doc_ref.delete()


class TestCreateFbfm40Lookup:
    """Test the POST /domains/{domain_id}/grids/lookup/fbfm40 endpoint."""

    def route(self, domain_id):
        return f"/domains/{domain_id}/grids/lookup/fbfm40"

    def test_minimal_request_creates_grid(
        self, client, domain_for_testing, complete_fbfm40_grid
    ):
        """Minimal request with required fields creates a lookup grid."""
        request_body = {
            "source_grid_id": complete_fbfm40_grid["id"],
            "bands": ["fuel_load.1hr"],
        }

        response = client.post(self.route(domain_for_testing["id"]), json=request_body)

        assert response.status_code == 201

        data = response.json()
        assert "id" in data
        assert len(data["id"]) == 32
        assert data["domain_id"] == domain_for_testing["id"]
        assert data["status"] == "pending"
        assert data["name"] == ""
        assert data["description"] == ""
        assert data["tags"] == []

        # Check source
        assert data["source"]["name"] == "lookup"
        assert data["source"]["table"] == "fbfm40"
        assert data["source"]["source_grid_id"] == complete_fbfm40_grid["id"]
        assert data["source"]["source_band"] == "fbfm"

        # Check bands
        assert len(data["bands"]) == 1
        assert data["bands"][0]["key"] == "fuel_load.1hr"
        assert data["bands"][0]["type"] == "continuous"
        assert data["bands"][0]["unit"] == "kg/m**2"
        assert data["bands"][0]["index"] == 0

    def test_request_with_metadata(
        self, client, domain_for_testing, complete_fbfm40_grid
    ):
        """Request with name, description, and tags."""
        request_body = {
            "source_grid_id": complete_fbfm40_grid["id"],
            "bands": ["fuel_load.1hr", "fuel_depth"],
            "name": "Surface fuels from FBFM40",
            "description": "Fuel parameters for baseline scenario",
            "tags": ["baseline", "surface-fuel"],
        }

        response = client.post(self.route(domain_for_testing["id"]), json=request_body)

        assert response.status_code == 201

        data = response.json()
        assert data["name"] == "Surface fuels from FBFM40"
        assert data["description"] == "Fuel parameters for baseline scenario"
        assert data["tags"] == ["baseline", "surface-fuel"]

    def test_multiple_bands_creates_correct_indices(
        self, client, domain_for_testing, complete_fbfm40_grid
    ):
        """Requesting multiple bands creates bands with correct indices."""
        bands = [
            "fuel_load.1hr",
            "savr.1hr",
            "fuel_depth",
        ]
        request_body = {
            "source_grid_id": complete_fbfm40_grid["id"],
            "bands": bands,
        }

        response = client.post(self.route(domain_for_testing["id"]), json=request_body)

        assert response.status_code == 201

        data = response.json()
        assert len(data["bands"]) == 3
        for i, band_key in enumerate(bands):
            assert data["bands"][i]["key"] == band_key
            assert data["bands"][i]["index"] == i

    def test_inherits_georeference_from_source(
        self, client, domain_for_testing, complete_fbfm40_grid
    ):
        """Output grid inherits georeference from the source grid."""
        request_body = {
            "source_grid_id": complete_fbfm40_grid["id"],
            "bands": ["fuel_load.1hr"],
        }

        response = client.post(self.route(domain_for_testing["id"]), json=request_body)

        assert response.status_code == 201

        data = response.json()
        expected = complete_fbfm40_grid["georeference"]
        assert data["georeference"]["crs"] == expected["crs"]
        assert data["georeference"]["transform"] == list(expected["transform"])
        assert data["georeference"]["shape"] == list(expected["shape"])

    def test_response_excludes_owner_id(
        self, client, domain_for_testing, complete_fbfm40_grid
    ):
        """Response should not expose the owner_id field."""
        request_body = {
            "source_grid_id": complete_fbfm40_grid["id"],
            "bands": ["fuel_load.1hr"],
        }

        response = client.post(self.route(domain_for_testing["id"]), json=request_body)
        assert response.status_code == 201

        data = response.json()
        assert "owner_id" not in data

    # Validation tests

    def test_nonexistent_source_grid_returns_404(self, client, domain_for_testing):
        """Non-existent source_grid_id returns 404."""
        request_body = {
            "source_grid_id": "00000000000000000000000000000000",
            "bands": ["fuel_load.1hr"],
        }

        response = client.post(self.route(domain_for_testing["id"]), json=request_body)

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_source_grid_not_complete_returns_422(
        self, client, domain_for_testing, pending_fbfm40_grid
    ):
        """Source grid with status != 'complete' returns 422."""
        request_body = {
            "source_grid_id": pending_fbfm40_grid["id"],
            "bands": ["fuel_load.1hr"],
        }

        response = client.post(self.route(domain_for_testing["id"]), json=request_body)

        assert response.status_code == 422
        assert "status" in response.json()["detail"].lower()

    def test_source_grid_missing_fbfm_band_returns_422(
        self, client, domain_for_testing, grid_without_fbfm_band
    ):
        """Source grid without an 'fbfm' band returns 422."""
        request_body = {
            "source_grid_id": grid_without_fbfm_band["id"],
            "bands": ["fuel_load.1hr"],
        }

        response = client.post(self.route(domain_for_testing["id"]), json=request_body)

        assert response.status_code == 422
        assert "fbfm" in response.json()["detail"].lower()

    def test_invalid_band_returns_422(
        self, client, domain_for_testing, complete_fbfm40_grid
    ):
        """Invalid band value returns 422 (Pydantic validation)."""
        request_body = {
            "source_grid_id": complete_fbfm40_grid["id"],
            "bands": ["not_a_real_band"],
        }

        response = client.post(self.route(domain_for_testing["id"]), json=request_body)

        assert response.status_code == 422

    def test_source_grid_in_different_domain_returns_404(
        self, client, domain_for_testing, grid_in_different_domain
    ):
        """Source grid belonging to a different domain returns 404."""
        request_body = {
            "source_grid_id": grid_in_different_domain["id"],
            "bands": ["fuel_load.1hr"],
        }

        response = client.post(self.route(domain_for_testing["id"]), json=request_body)

        assert response.status_code == 404

    # Example tests

    @pytest.mark.parametrize(
        "example_name,example_value", ALL_FBFM40_LOOKUP_EXAMPLE_VALUES
    )
    def test_documented_example_creates_grid(
        self,
        client,
        domain_for_testing,
        complete_fbfm40_grid,
        example_name,
        example_value,
    ):
        """Each documented lookup example should successfully create a grid."""
        request_body = {
            **example_value,
            "source_grid_id": complete_fbfm40_grid["id"],
        }

        response = client.post(self.route(domain_for_testing["id"]), json=request_body)

        assert response.status_code == 201, (
            f"Example '{example_name}' failed with status {response.status_code}: "
            f"{response.json()}"
        )

        data = response.json()
        assert data["source"]["name"] == "lookup"
        assert data["source"]["table"] == "fbfm40"
        assert data["source"]["source_grid_id"] == complete_fbfm40_grid["id"]
