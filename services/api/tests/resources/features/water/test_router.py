"""
Integration tests for api/v2/resources/features/water/router.py

Tests the POST /domains/{domain_id}/features/water/osm endpoint.
These tests make real HTTP requests to the API and interact with Firestore.
"""

from unittest.mock import patch

import pytest
from api.resources.features.water.examples import WATER_EXAMPLE_VALUES


# Mock out Cloud Tasks so tests don't attempt to enqueue real background jobs
@pytest.fixture(autouse=True)
def mock_create_task():
    with patch("api.resources.features.water.router.create_http_task_async") as mock:
        yield mock


class TestCreateOsmWaterFeature:
    """Test the POST /domains/{domain_id}/features/water/osm endpoint."""

    def route(self, domain_id):
        return f"/domains/{domain_id}/features/water/osm"

    def test_minimal_request_creates_feature(self, client, domain_for_testing):
        """Minimal request creates a feature document in pending state."""
        response = client.post(
            self.route(domain_for_testing["id"]), json={"type": "water"}
        )

        assert response.status_code == 201

        data = response.json()
        assert "id" in data
        assert len(data["id"]) == 32
        assert data["domain_id"] == domain_for_testing["id"]
        assert data["type"] == "water"

        # Background worker hasn't run yet
        assert data["status"] == "pending"
        assert data["georeference"] is None

        # Defaults
        assert data["name"] == ""
        assert data["description"] == ""
        assert data["tags"] == []

        # Source is locked to OSM with no buffer by default
        assert data["source"]["product"] == "osm"
        assert data["source"]["extent_buffer_m"] == 0

    def test_request_with_extent_buffer_m(self, client, domain_for_testing):
        """extent_buffer_m is persisted into the source dict."""
        response = client.post(
            self.route(domain_for_testing["id"]),
            json={"type": "water", "extent_buffer_m": 25},
        )

        assert response.status_code == 201
        assert response.json()["source"]["extent_buffer_m"] == 25

    def test_request_with_out_of_range_buffer_rejected(
        self, client, domain_for_testing
    ):
        """extent_buffer_m above 100 is rejected by request validation."""
        response = client.post(
            self.route(domain_for_testing["id"]),
            json={"type": "water", "extent_buffer_m": 101},
        )

        assert response.status_code == 422

    def test_request_with_metadata(self, client, domain_for_testing):
        """Request accepts name, description, and tags."""
        request_body = {
            "type": "water",
            "name": "Project Alpha Hydrology",
            "description": "OSM-derived hydrology layer",
            "tags": ["osm", "water", "alpha"],
        }

        response = client.post(self.route(domain_for_testing["id"]), json=request_body)

        assert response.status_code == 201

        data = response.json()
        assert data["name"] == "Project Alpha Hydrology"
        assert data["description"] == "OSM-derived hydrology layer"
        assert data["tags"] == ["osm", "water", "alpha"]

    def test_invalid_domain_returns_404(self, client):
        """Non-existent domain_id returns 404."""
        response = client.post(
            self.route("00000000000000000000000000000000"), json={"type": "water"}
        )

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_wrong_owner_domain_returns_404(self, client, domain_with_different_owner):
        """Domain owned by another user returns 404."""
        response = client.post(
            self.route(domain_with_different_owner["id"]), json={"type": "water"}
        )

        assert response.status_code == 404

    def test_response_excludes_owner_id(self, client, domain_for_testing):
        """Response should not expose the owner_id field to the client."""
        response = client.post(
            self.route(domain_for_testing["id"]), json={"type": "water"}
        )
        assert response.status_code == 201

        data = response.json()
        assert "owner_id" not in data

    @pytest.mark.parametrize("example_name,example_value", WATER_EXAMPLE_VALUES)
    def test_documented_example_creates_feature(
        self, client, domain_for_testing, example_name, example_value
    ):
        """Each documented WATER example payload should successfully create a feature."""
        response = client.post(self.route(domain_for_testing["id"]), json=example_value)

        assert response.status_code == 201, (
            f"Example '{example_name}' failed with status {response.status_code}: "
            f"{response.json()}"
        )

        data = response.json()
        assert data["type"] == "water"
        assert data["source"]["product"] == "osm"
