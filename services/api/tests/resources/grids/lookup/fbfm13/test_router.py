"""
Integration tests for api/v2/resources/grids/lookup/fbfm13/router.py

Tests the FBFM13 lookup endpoint (POST /domains/{domain_id}/grids/lookup/fbfm13).
These tests make real HTTP requests to the API and interact with Firestore.

The lookup endpoint requires a source grid that:
1. Exists and is owned by the requesting user
2. Has status "completed"
3. Contains an "fbfm13" band
4. Belongs to the same domain as the URL path
"""

import pytest
from api.resources.grids.lookup.fbfm13.examples import (
    ALL_FBFM13_LOOKUP_EXAMPLE_VALUES,
)

from lib.config import GRIDS_COLLECTION
from tests.resources.grids.lookup.conftest import (
    _make_cross_domain_lookup_grid,
    _make_landfire_grid_data,
    _persist_and_cleanup,
)


@pytest.fixture(scope="session")
def complete_fbfm13_grid(firestore_client, domain_for_testing):
    """A complete LANDFIRE FBFM13 grid for use as a lookup source."""
    grid_data = _make_landfire_grid_data(
        domain_for_testing["id"],
        product="fbfm13",
        version="2023",
        band_key="fbfm13",
        description="LANDFIRE FBFM13 fuel model codes (Anderson 13 classification)",
        name="FBFM13 source for lookup tests",
        status="completed",
    )
    yield from _persist_and_cleanup(firestore_client, GRIDS_COLLECTION, grid_data)


@pytest.fixture(scope="session")
def pending_fbfm13_grid(firestore_client, domain_for_testing):
    """An FBFM13 grid with status "pending" (not yet complete)."""
    grid_data = _make_landfire_grid_data(
        domain_for_testing["id"],
        product="fbfm13",
        version="2023",
        band_key="fbfm13",
        description="LANDFIRE FBFM13 fuel model codes (Anderson 13 classification)",
        name="Pending FBFM13 grid",
        status="pending",
        georeference=None,
    )
    yield from _persist_and_cleanup(firestore_client, GRIDS_COLLECTION, grid_data)


@pytest.fixture(scope="session")
def fbfm13_grid_in_different_domain(firestore_client, second_domain_for_lookup):
    """A complete FBFM13 grid in a different domain than domain_for_testing."""
    yield from _make_cross_domain_lookup_grid(
        firestore_client,
        second_domain_for_lookup,
        product="fbfm13",
        version="2023",
        band_key="fbfm13",
        description="LANDFIRE FBFM13 fuel model codes (Anderson 13 classification)",
        name="FBFM13 grid in second domain",
    )


class TestCreateFbfm13Lookup:
    """Test the POST /domains/{domain_id}/grids/lookup/fbfm13 endpoint."""

    def route(self, domain_id):
        return f"/domains/{domain_id}/grids/lookup/fbfm13"

    def test_minimal_request_creates_grid(
        self, client, domain_for_testing, complete_fbfm13_grid
    ):
        """Minimal request with required fields creates a lookup grid."""
        request_body = {
            "source_grid_id": complete_fbfm13_grid["id"],
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

        assert data["source"]["name"] == "lookup"
        assert data["source"]["table"] == "fbfm13"
        assert data["source"]["source_grid_id"] == complete_fbfm13_grid["id"]
        assert data["source"]["source_band"] == "fbfm13"

        assert len(data["bands"]) == 1
        assert data["bands"][0]["key"] == "fuel_load.1hr"
        assert data["bands"][0]["type"] == "continuous"
        assert data["bands"][0]["unit"] == "kg/m**2"
        assert data["bands"][0]["index"] == 0

    def test_request_with_metadata(
        self, client, domain_for_testing, complete_fbfm13_grid
    ):
        """Request with name, description, and tags."""
        request_body = {
            "source_grid_id": complete_fbfm13_grid["id"],
            "bands": ["fuel_load.1hr", "fuel_depth"],
            "name": "Surface fuels from FBFM13",
            "description": "Fuel parameters for baseline scenario",
            "tags": ["baseline", "surface-fuel"],
        }

        response = client.post(self.route(domain_for_testing["id"]), json=request_body)

        assert response.status_code == 201

        data = response.json()
        assert data["name"] == "Surface fuels from FBFM13"
        assert data["description"] == "Fuel parameters for baseline scenario"
        assert data["tags"] == ["baseline", "surface-fuel"]

    def test_multiple_bands_creates_correct_indices(
        self, client, domain_for_testing, complete_fbfm13_grid
    ):
        """Requesting multiple bands creates bands with correct indices."""
        bands = [
            "fuel_load.1hr",
            "savr.1hr",
            "fuel_depth",
        ]
        request_body = {
            "source_grid_id": complete_fbfm13_grid["id"],
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
        self, client, domain_for_testing, complete_fbfm13_grid
    ):
        """Output grid inherits georeference from the source grid."""
        request_body = {
            "source_grid_id": complete_fbfm13_grid["id"],
            "bands": ["fuel_load.1hr"],
        }

        response = client.post(self.route(domain_for_testing["id"]), json=request_body)

        assert response.status_code == 201

        data = response.json()
        expected = complete_fbfm13_grid["georeference"]
        assert data["georeference"]["crs"] == expected["crs"]
        assert data["georeference"]["transform"] == list(expected["transform"])
        assert data["georeference"]["shape"] == list(expected["shape"])

    def test_response_excludes_owner_id(
        self, client, domain_for_testing, complete_fbfm13_grid
    ):
        """Response should not expose the owner_id field."""
        request_body = {
            "source_grid_id": complete_fbfm13_grid["id"],
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
        self, client, domain_for_testing, pending_fbfm13_grid
    ):
        """Source grid with status != 'complete' returns 422."""
        request_body = {
            "source_grid_id": pending_fbfm13_grid["id"],
            "bands": ["fuel_load.1hr"],
        }

        response = client.post(self.route(domain_for_testing["id"]), json=request_body)

        assert response.status_code == 422
        assert "status" in response.json()["detail"].lower()

    def test_source_grid_missing_fbfm13_band_returns_422(
        self, client, domain_for_testing, grid_without_fbfm_band
    ):
        """Source grid without an 'fbfm13' band returns 422."""
        request_body = {
            "source_grid_id": grid_without_fbfm_band["id"],
            "bands": ["fuel_load.1hr"],
        }

        response = client.post(self.route(domain_for_testing["id"]), json=request_body)

        assert response.status_code == 422
        assert "fbfm13" in response.json()["detail"].lower()

    def test_invalid_band_returns_422(
        self, client, domain_for_testing, complete_fbfm13_grid
    ):
        """Invalid band value returns 422 (Pydantic validation)."""
        request_body = {
            "source_grid_id": complete_fbfm13_grid["id"],
            "bands": ["not_a_real_band"],
        }

        response = client.post(self.route(domain_for_testing["id"]), json=request_body)

        assert response.status_code == 422

    def test_source_grid_in_different_domain_returns_404(
        self, client, domain_for_testing, fbfm13_grid_in_different_domain
    ):
        """Source grid belonging to a different domain returns 404."""
        request_body = {
            "source_grid_id": fbfm13_grid_in_different_domain["id"],
            "bands": ["fuel_load.1hr"],
        }

        response = client.post(self.route(domain_for_testing["id"]), json=request_body)

        assert response.status_code == 404

    # Example tests

    @pytest.mark.parametrize(
        "example_name,example_value", ALL_FBFM13_LOOKUP_EXAMPLE_VALUES
    )
    def test_documented_example_creates_grid(
        self,
        client,
        domain_for_testing,
        complete_fbfm13_grid,
        example_name,
        example_value,
    ):
        """Each documented lookup example should successfully create a grid."""
        request_body = {
            **example_value,
            "source_grid_id": complete_fbfm13_grid["id"],
        }

        response = client.post(self.route(domain_for_testing["id"]), json=request_body)

        assert response.status_code == 201, (
            f"Example '{example_name}' failed with status {response.status_code}: "
            f"{response.json()}"
        )

        data = response.json()
        assert data["source"]["name"] == "lookup"
        assert data["source"]["table"] == "fbfm13"
        assert data["source"]["source_grid_id"] == complete_fbfm13_grid["id"]
