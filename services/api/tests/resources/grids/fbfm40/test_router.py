"""
Integration tests for api/v2/resources/grids/fbfm40/router.py

Tests the FBFM40 LANDFIRE endpoint.
These tests make real HTTP requests to the API and interact with Firestore.
"""

import pytest
from api.resources.grids.fbfm40.examples import (
    ALL_FBFM40_EXAMPLE_VALUES,
)


class TestCreateLandfireFbfm40:
    """Test the POST /domains/{domain_id}/grids/fbfm40/landfire endpoint."""

    def route(self, domain_id):
        return f"/domains/{domain_id}/grids/fbfm40/landfire"

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
        assert data["source"]["product"] == "fbfm40"
        assert data["source"]["version"] == "2024"

        # Check single fbfm band
        assert len(data["bands"]) == 1
        assert data["bands"][0]["key"] == "fbfm"
        assert data["bands"][0]["type"] == "categorical"
        assert data["bands"][0]["unit"] is None

    def test_request_with_metadata(self, client, domain_for_testing):
        """Request with name, description, and tags."""
        request_body = {
            "name": "FBFM40 Codes",
            "description": "Fuel model codes for baseline",
            "tags": ["baseline", "surface-fuel"],
            "version": "2022",
        }

        response = client.post(self.route(domain_for_testing["id"]), json=request_body)

        assert response.status_code == 201

        data = response.json()
        assert data["name"] == "FBFM40 Codes"
        assert data["description"] == "Fuel model codes for baseline"
        assert data["tags"] == ["baseline", "surface-fuel"]
        assert data["source"]["version"] == "2022"

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

    @pytest.mark.parametrize("example_name,example_value", ALL_FBFM40_EXAMPLE_VALUES)
    def test_documented_example_creates_grid(
        self, client, domain_for_testing, example_name, example_value
    ):
        """Each documented FBFM40 example should successfully create a grid."""
        response = client.post(self.route(domain_for_testing["id"]), json=example_value)

        assert response.status_code == 201, (
            f"Example '{example_name}' failed with status {response.status_code}: "
            f"{response.json()}"
        )

        data = response.json()
        assert data["source"]["name"] == "landfire"
        assert data["source"]["product"] == "fbfm40"

    def test_extent_buffer_cells_defaults_to_zero(self, client, domain_for_testing):
        """Omitting extent_buffer_cells resolves to no buffer."""
        response = client.post(self.route(domain_for_testing["id"]), json={})

        assert response.status_code == 201
        assert response.json()["source"]["extent_buffer_cells"] == 0

    @pytest.mark.parametrize("buffer", [0, 10])
    def test_extent_buffer_cells_explicit_value_persisted(
        self, client, domain_for_testing, buffer
    ):
        """Explicit extent_buffer_cells (including 0) is persisted in source."""
        response = client.post(
            self.route(domain_for_testing["id"]),
            json={"extent_buffer_cells": buffer},
        )

        assert response.status_code == 201
        assert response.json()["source"]["extent_buffer_cells"] == buffer

    def test_extent_buffer_cells_negative_rejected(self, client, domain_for_testing):
        """Negative extent_buffer_cells is rejected with 422."""
        response = client.post(
            self.route(domain_for_testing["id"]),
            json={"extent_buffer_cells": -1},
        )

        assert response.status_code == 422

    def test_extent_buffer_cells_above_maximum_rejected(
        self, client, domain_for_testing
    ):
        response = client.post(
            self.route(domain_for_testing["id"]),
            json={"extent_buffer_cells": 11},
        )

        assert response.status_code == 422
