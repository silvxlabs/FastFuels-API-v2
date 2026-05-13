"""
Integration tests for api/v2/resources/features/road/router.py

Tests the POST /domains/{domain_id}/features/road endpoint.
These tests make real HTTP requests to the API and interact with Firestore.
"""

from unittest.mock import patch

import pytest
from api.resources.features.road.examples import ROAD_EXAMPLE_VALUES


# Mock out Cloud Tasks so tests don't attempt to enqueue real background jobs
@pytest.fixture(autouse=True)
def mock_create_task():
    with patch("api.resources.features.road.router.create_http_task_async") as mock:
        yield mock


class TestCreateOsmRoadFeature:
    """Test the POST /domains/{domain_id}/features/road endpoint."""

    def route(self, domain_id):
        return f"/domains/{domain_id}/features/road"

    def test_minimal_request_creates_feature(self, client, domain_for_testing):
        """Minimal request creates a feature document in pending state."""
        response = client.post(
            self.route(domain_for_testing["id"]), json={"type": "road"}
        )

        assert response.status_code == 201

        data = response.json()
        assert "id" in data
        assert len(data["id"]) == 32
        assert data["domain_id"] == domain_for_testing["id"]
        assert data["type"] == "road"

        # Background worker hasn't run yet
        assert data["status"] == "pending"
        assert data["georeference"] is None

        # Defaults
        assert data["name"] == ""
        assert data["description"] == ""
        assert data["tags"] == []

        # Source is locked to OSM
        assert data["source"]["product"] == "osm"

    def test_request_with_metadata(self, client, domain_for_testing):
        """Request accepts name, description, and tags."""
        request_body = {
            "type": "road",
            "name": "Project Alpha Roads",
            "description": "Custom clipped road boundaries",
            "tags": ["osm", "roads", "alpha"],
        }

        response = client.post(self.route(domain_for_testing["id"]), json=request_body)

        assert response.status_code == 201

        data = response.json()
        assert data["name"] == "Project Alpha Roads"
        assert data["description"] == "Custom clipped road boundaries"
        assert data["tags"] == ["osm", "roads", "alpha"]

    def test_invalid_domain_returns_404(self, client):
        """Non-existent domain_id returns 404."""
        response = client.post(
            self.route("00000000000000000000000000000000"), json={"type": "road"}
        )

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_wrong_owner_domain_returns_404(self, client, domain_with_different_owner):
        """Domain owned by another user returns 404."""
        response = client.post(
            self.route(domain_with_different_owner["id"]), json={"type": "road"}
        )

        assert response.status_code == 404

    def test_response_excludes_owner_id(self, client, domain_for_testing):
        """Response should not expose the owner_id field to the client."""
        response = client.post(
            self.route(domain_for_testing["id"]), json={"type": "road"}
        )
        assert response.status_code == 201

        data = response.json()
        assert "owner_id" not in data

    @pytest.mark.parametrize("example_name,example_value", ROAD_EXAMPLE_VALUES)
    def test_documented_example_creates_feature(
        self, client, domain_for_testing, example_name, example_value
    ):
        """Each documented ROAD example payload should successfully create a feature."""
        response = client.post(self.route(domain_for_testing["id"]), json=example_value)

        assert response.status_code == 201, (
            f"Example '{example_name}' failed with status {response.status_code}: "
            f"{response.json()}"
        )

        data = response.json()
        assert data["type"] == "road"
        assert data["source"]["product"] == "osm"
