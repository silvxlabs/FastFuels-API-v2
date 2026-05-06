"""
Integration tests for api/v2/resources/grids/chm/router.py

Tests the Meta CHM endpoint.
These tests make real HTTP requests to the API and interact with Firestore.
"""

import pytest
from api.resources.grids.chm.examples import (
    META_CHM_EXAMPLE_VALUES,
    NAIP_CHM_EXAMPLE_VALUES,
)
from api.resources.grids.chm.schema import MetaCHMVersion


class TestCreateMetaChm:
    """Test the POST /domains/{domain_id}/grids/chm/meta endpoint."""

    def route(self, domain_id):
        return f"/domains/{domain_id}/grids/chm/meta"

    def test_minimal_request_creates_grid(self, client, domain_for_testing):
        """Minimal request creates a grid with default chm band."""
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
        assert data["source"]["version"] == "2"

        # Check source
        assert data["source"]["name"] == "chm"
        assert data["source"]["product"] == "meta"

        # Check single continuous band
        assert len(data["bands"]) == 1
        assert data["bands"][0]["key"] == "chm"
        assert data["bands"][0]["type"] == "continuous"
        assert data["bands"][0]["unit"] == "m"
        assert data["bands"][0]["index"] == 0

    def test_request_with_metadata(self, client, domain_for_testing):
        """Request with name, description, and tags."""
        request_body = {
            "name": "Meta canopy height",
            "description": "Global canopy height model for inventory",
            "tags": ["chm", "meta"],
        }

        response = client.post(self.route(domain_for_testing["id"]), json=request_body)

        assert response.status_code == 201

        data = response.json()
        assert data["name"] == "Meta canopy height"
        assert data["description"] == "Global canopy height model for inventory"
        assert data["tags"] == ["chm", "meta"]
        assert data["source"]["version"] == "2"

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

    @pytest.mark.parametrize("example_name,example_value", META_CHM_EXAMPLE_VALUES)
    def test_documented_example_creates_grid(
        self, client, domain_for_testing, example_name, example_value
    ):
        """Each documented Meta CHM example should successfully create a grid."""
        response = client.post(self.route(domain_for_testing["id"]), json=example_value)

        assert response.status_code == 201, (
            f"Example '{example_name}' failed with status {response.status_code}: "
            f"{response.json()}"
        )

        data = response.json()
        assert data["source"]["name"] == "chm"
        assert data["source"]["product"] == "meta"

    @pytest.mark.parametrize("version", [v.value for v in MetaCHMVersion])
    def test_version_can_be_set(self, client, domain_for_testing, version):
        """Each valid version creates a grid with the correct version set."""
        response = client.post(
            self.route(domain_for_testing["id"]), json={"version": version}
        )
        assert response.status_code == 201
        assert response.json()["source"]["version"] == version

    def test_extent_buffer_cells_defaults_to_zero(self, client, domain_for_testing):
        response = client.post(self.route(domain_for_testing["id"]), json={})
        assert response.status_code == 201
        assert response.json()["source"]["extent_buffer_cells"] == 0

    @pytest.mark.parametrize("buffer", [0, 10])
    def test_extent_buffer_cells_explicit_value_persisted(
        self, client, domain_for_testing, buffer
    ):
        response = client.post(
            self.route(domain_for_testing["id"]),
            json={"extent_buffer_cells": buffer},
        )
        assert response.status_code == 201
        assert response.json()["source"]["extent_buffer_cells"] == buffer

    def test_extent_buffer_cells_negative_rejected(self, client, domain_for_testing):
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


class TestCreateNaipChm:
    """Test the POST /domains/{domain_id}/grids/chm/naip endpoint."""

    def route(self, domain_id):
        return f"/domains/{domain_id}/grids/chm/naip"

    def test_minimal_request_creates_grid(self, client, domain_for_testing):
        """Minimal request creates a grid with default chm band."""
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
        assert data["source"]["name"] == "chm"
        assert data["source"]["product"] == "naip"

        # Check single continuous band
        assert len(data["bands"]) == 1
        assert data["bands"][0]["key"] == "chm"
        assert data["bands"][0]["type"] == "continuous"
        assert data["bands"][0]["unit"] == "m"
        assert data["bands"][0]["index"] == 0

    def test_request_with_metadata(self, client, domain_for_testing):
        """Request with name, description, and tags."""
        request_body = {
            "name": "NAIP canopy height",
            "description": "High-res canopy height model",
            "tags": ["chm", "naip"],
        }

        response = client.post(self.route(domain_for_testing["id"]), json=request_body)

        assert response.status_code == 201

        data = response.json()
        assert data["name"] == "NAIP canopy height"
        assert data["description"] == "High-res canopy height model"
        assert data["tags"] == ["chm", "naip"]

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

    @pytest.mark.parametrize("example_name,example_value", NAIP_CHM_EXAMPLE_VALUES)
    def test_documented_example_creates_grid(
        self, client, domain_for_testing, example_name, example_value
    ):
        """Each documented NAIP CHM example should successfully create a grid."""
        response = client.post(self.route(domain_for_testing["id"]), json=example_value)

        assert response.status_code == 201, (
            f"Example '{example_name}' failed with status {response.status_code}: "
            f"{response.json()}"
        )

        data = response.json()
        assert data["source"]["name"] == "chm"
        assert data["source"]["product"] == "naip"

    def test_extent_buffer_cells_defaults_to_zero(self, client, domain_for_testing):
        response = client.post(self.route(domain_for_testing["id"]), json={})
        assert response.status_code == 201
        assert response.json()["source"]["extent_buffer_cells"] == 0

    @pytest.mark.parametrize("buffer", [0, 10])
    def test_extent_buffer_cells_explicit_value_persisted(
        self, client, domain_for_testing, buffer
    ):
        response = client.post(
            self.route(domain_for_testing["id"]),
            json={"extent_buffer_cells": buffer},
        )
        assert response.status_code == 201
        assert response.json()["source"]["extent_buffer_cells"] == buffer

    def test_extent_buffer_cells_negative_rejected(self, client, domain_for_testing):
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
