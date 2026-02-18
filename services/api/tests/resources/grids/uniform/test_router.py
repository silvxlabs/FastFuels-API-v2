"""
Integration tests for api/v2/resources/grids/uniform/router.py

Tests the POST /domains/{domain_id}/grids/uniform endpoint.
These tests make real HTTP requests to the API and interact with Firestore.
"""

import pytest
from api.resources.grids.uniform.examples import ALL_UNIFORM_EXAMPLE_VALUES


class TestCreateUniformGrid:
    """Test the POST /domains/{domain_id}/grids/uniform endpoint."""

    def _route(self, domain_id):
        return f"/domains/{domain_id}/grids/uniform"

    def test_minimal_request_creates_grid(self, client, domain_for_testing):
        """Minimal request with required fields creates a grid."""
        route = self._route(domain_for_testing["id"])
        request_body = {
            "resolution": 2.0,
            "bands": [{"quantity": "fuel_moisture.1hr", "value": 6.0}],
        }

        response = client.post(route, json=request_body)

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
        assert data["source"]["name"] == "uniform"
        assert data["source"]["resolution"] == 2.0
        assert len(data["source"]["bands"]) == 1
        assert data["source"]["bands"][0]["quantity"] == "fuel_moisture.1hr"
        assert data["source"]["bands"][0]["value"] == 6.0

        # Check band metadata
        assert len(data["bands"]) == 1
        assert data["bands"][0]["key"] == "fuel_moisture.1hr"
        assert data["bands"][0]["type"] == "continuous"
        assert data["bands"][0]["unit"] == "%"
        assert data["bands"][0]["index"] == 0

    def test_multiple_bands(self, client, domain_for_testing):
        """Request with multiple bands creates correct band metadata."""
        route = self._route(domain_for_testing["id"])
        request_body = {
            "resolution": 2.0,
            "bands": [
                {"quantity": "fuel_moisture.1hr", "value": 6.0},
                {"quantity": "fuel_moisture.10hr", "value": 8.0},
                {"quantity": "fuel_moisture.100hr", "value": 12.0},
            ],
        }

        response = client.post(route, json=request_body)

        assert response.status_code == 201

        data = response.json()
        assert len(data["bands"]) == 3
        assert data["bands"][0]["key"] == "fuel_moisture.1hr"
        assert data["bands"][0]["index"] == 0
        assert data["bands"][1]["key"] == "fuel_moisture.10hr"
        assert data["bands"][1]["index"] == 1
        assert data["bands"][2]["key"] == "fuel_moisture.100hr"
        assert data["bands"][2]["index"] == 2

        # Source stores the band values
        assert len(data["source"]["bands"]) == 3
        assert data["source"]["bands"][0]["value"] == 6.0
        assert data["source"]["bands"][1]["value"] == 8.0
        assert data["source"]["bands"][2]["value"] == 12.0

    def test_request_with_metadata(self, client, domain_for_testing):
        """Request with name, description, and tags."""
        route = self._route(domain_for_testing["id"])
        request_body = {
            "resolution": 5.0,
            "bands": [{"quantity": "fuel_load.1hr", "value": 0.15}],
            "name": "Fuel load scenario",
            "description": "Test fuel load grid",
            "tags": ["fuel-load", "test"],
        }

        response = client.post(route, json=request_body)

        assert response.status_code == 201

        data = response.json()
        assert data["name"] == "Fuel load scenario"
        assert data["description"] == "Test fuel load grid"
        assert data["tags"] == ["fuel-load", "test"]

    def test_source_stores_resolution(self, client, domain_for_testing):
        """Source stores the requested resolution for reproducibility."""
        route = self._route(domain_for_testing["id"])
        request_body = {
            "resolution": 10.0,
            "bands": [{"quantity": "fuel_depth", "value": 0.3}],
        }

        response = client.post(route, json=request_body)

        assert response.status_code == 201
        assert response.json()["source"]["resolution"] == 10.0

    def test_georeference_is_null_on_creation(self, client, domain_for_testing):
        """Georeference is null until backend populates it."""
        route = self._route(domain_for_testing["id"])
        request_body = {
            "resolution": 2.0,
            "bands": [{"quantity": "fuel_moisture.1hr", "value": 6.0}],
        }

        response = client.post(route, json=request_body)
        assert response.status_code == 201

        data = response.json()
        assert data["georeference"] is None

    def test_invalid_domain_returns_404(self, client):
        """Non-existent domain_id in URL returns 404."""
        route = self._route("00000000000000000000000000000000")
        request_body = {
            "resolution": 2.0,
            "bands": [{"quantity": "fuel_moisture.1hr", "value": 6.0}],
        }

        response = client.post(route, json=request_body)

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_wrong_owner_domain_returns_404(self, client, domain_with_different_owner):
        """Domain owned by another user returns 404."""
        route = self._route(domain_with_different_owner["id"])
        request_body = {
            "resolution": 2.0,
            "bands": [{"quantity": "fuel_moisture.1hr", "value": 6.0}],
        }

        response = client.post(route, json=request_body)

        assert response.status_code == 404

    def test_response_excludes_owner_id(self, client, domain_for_testing):
        """Response should not expose the owner_id field."""
        route = self._route(domain_for_testing["id"])
        request_body = {
            "resolution": 2.0,
            "bands": [{"quantity": "fuel_moisture.1hr", "value": 6.0}],
        }

        response = client.post(route, json=request_body)
        assert response.status_code == 201

        data = response.json()
        assert "owner_id" not in data

    @pytest.mark.parametrize("example_name,example_value", ALL_UNIFORM_EXAMPLE_VALUES)
    def test_documented_example_creates_grid(
        self, client, domain_for_testing, example_name, example_value
    ):
        """Each documented uniform example should successfully create a grid."""
        route = self._route(domain_for_testing["id"])

        response = client.post(route, json=example_value)

        assert response.status_code == 201, (
            f"Example '{example_name}' failed with status {response.status_code}: "
            f"{response.json()}"
        )

        data = response.json()
        assert data["source"]["name"] == "uniform"
