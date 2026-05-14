"""
Integration tests for api/v2/resources/grids/fccs/router.py

Tests the FCCS LANDFIRE endpoint.
These tests make real HTTP requests to the API and interact with Firestore.
"""

import pytest
from api.resources.grids.fccs.examples import (
    ALL_FCCS_EXAMPLE_VALUES,
)


class TestCreateLandfireFccs:
    """Test the POST /domains/{domain_id}/grids/fccs/landfire endpoint."""

    def route(self, domain_id):
        return f"/domains/{domain_id}/grids/fccs/landfire"

    def test_minimal_request_creates_grid(self, client, domain_for_testing):
        """Minimal request with required fields creates a grid."""
        response = client.post(self.route(domain_for_testing["id"]), json={})

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
        assert data["source"]["name"] == "landfire"
        assert data["source"]["product"] == "fccs"
        assert data["source"]["version"] == "2023"

        # Check single fccs band
        assert len(data["bands"]) == 1
        assert data["bands"][0]["key"] == "fccs"
        assert data["bands"][0]["type"] == "categorical"
        assert data["bands"][0]["unit"] is None

    def test_request_with_metadata(self, client, domain_for_testing):
        """Request with name, description, and tags."""
        request_body = {
            "name": "FCCS fuelbed IDs",
            "description": "Fuelbed IDs for baseline",
            "tags": ["baseline", "surface-fuel"],
            "version": "2023",
        }

        response = client.post(self.route(domain_for_testing["id"]), json=request_body)

        assert response.status_code == 201

        data = response.json()
        assert data["name"] == "FCCS fuelbed IDs"
        assert data["description"] == "Fuelbed IDs for baseline"
        assert data["tags"] == ["baseline", "surface-fuel"]
        assert data["source"]["version"] == "2023"

    def test_georeference_is_null_on_creation(self, client, domain_for_testing):
        """Georeference is null until backend populates it."""
        response = client.post(self.route(domain_for_testing["id"]), json={})
        assert response.status_code == 201

        data = response.json()
        assert data["georeference"] is None

    def test_invalid_domain_returns_404(self, client):
        """Non-existent domain_id returns 404."""
        response = client.post(self.route("00000000000000000000000000000000"), json={})

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_wrong_owner_domain_returns_404(self, client, domain_with_different_owner):
        """Domain owned by another user returns 404."""
        response = client.post(self.route(domain_with_different_owner["id"]), json={})

        assert response.status_code == 404

    def test_response_excludes_owner_id(self, client, domain_for_testing):
        """Response should not expose the owner_id field."""
        response = client.post(self.route(domain_for_testing["id"]), json={})
        assert response.status_code == 201

        data = response.json()
        assert "owner_id" not in data

    @pytest.mark.parametrize("example_name,example_value", ALL_FCCS_EXAMPLE_VALUES)
    def test_documented_example_creates_grid(
        self, client, domain_for_testing, example_name, example_value
    ):
        """Each documented FCCS example should successfully create a grid."""
        response = client.post(self.route(domain_for_testing["id"]), json=example_value)

        assert response.status_code == 201, (
            f"Example '{example_name}' failed with status {response.status_code}: "
            f"{response.json()}"
        )

        data = response.json()
        assert data["source"]["name"] == "landfire"
        assert data["source"]["product"] == "fccs"
