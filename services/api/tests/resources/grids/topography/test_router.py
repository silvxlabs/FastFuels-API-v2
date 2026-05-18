"""
Integration tests for api/v2/resources/grids/topography/router.py

Tests the Topography LANDFIRE endpoint.
These tests make real HTTP requests to the API and interact with Firestore.
"""

import pytest
from api.resources.grids.topography.examples import (
    ALL_3DEP_EXAMPLE_VALUES,
    ALL_TOPOGRAPHY_EXAMPLE_VALUES,
)


class TestCreateLandfireTopography:
    """Test the POST /domains/{domain_id}/grids/topography/landfire endpoint."""

    def route(self, domain_id):
        return f"/domains/{domain_id}/grids/topography/landfire"

    def test_minimal_request_creates_grid(self, client, domain_for_testing):
        """Minimal request creates a grid with all three topography bands."""
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
        assert data["source"]["product"] == "topography"
        assert data["source"]["version"] == "2020"
        assert data["source"]["bands"] == ["elevation", "slope", "aspect"]

        # Check three continuous bands
        assert len(data["bands"]) == 3
        assert data["bands"][0]["key"] == "elevation"
        assert data["bands"][0]["type"] == "continuous"
        assert data["bands"][0]["unit"] == "m"
        assert data["bands"][0]["index"] == 0
        assert data["bands"][1]["key"] == "slope"
        assert data["bands"][1]["type"] == "continuous"
        assert data["bands"][1]["unit"] == "deg"
        assert data["bands"][1]["index"] == 1
        assert data["bands"][2]["key"] == "aspect"
        assert data["bands"][2]["type"] == "continuous"
        assert data["bands"][2]["unit"] == "deg"
        assert data["bands"][2]["index"] == 2

    def test_elevation_only(self, client, domain_for_testing):
        """Request with only elevation band."""
        request_body = {
            "bands": ["elevation"],
        }

        response = client.post(self.route(domain_for_testing["id"]), json=request_body)

        assert response.status_code == 201

        data = response.json()
        assert len(data["bands"]) == 1
        assert data["bands"][0]["key"] == "elevation"
        assert data["bands"][0]["index"] == 0
        assert data["source"]["bands"] == ["elevation"]

    def test_request_with_metadata(self, client, domain_for_testing):
        """Request with name, description, and tags."""
        request_body = {
            "name": "Terrain data",
            "description": "Test terrain data",
            "tags": ["topography"],
            "version": "2020",
            "bands": ["elevation", "slope"],
        }

        response = client.post(self.route(domain_for_testing["id"]), json=request_body)

        assert response.status_code == 201

        data = response.json()
        assert data["name"] == "Terrain data"
        assert data["description"] == "Test terrain data"
        assert data["tags"] == ["topography"]
        assert data["source"]["version"] == "2020"
        assert len(data["bands"]) == 2

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

    @pytest.mark.parametrize(
        "example_name,example_value", ALL_TOPOGRAPHY_EXAMPLE_VALUES
    )
    def test_documented_example_creates_grid(
        self, client, domain_for_testing, example_name, example_value
    ):
        """Each documented topography example should successfully create a grid."""
        response = client.post(self.route(domain_for_testing["id"]), json=example_value)

        assert response.status_code == 201, (
            f"Example '{example_name}' failed with status {response.status_code}: "
            f"{response.json()}"
        )

        data = response.json()
        assert data["source"]["name"] == "landfire"
        assert data["source"]["product"] == "topography"

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


class TestCreate3DepTopography:
    """Test the POST /domains/{domain_id}/grids/topography/3dep endpoint."""

    def route(self, domain_id):
        return f"/domains/{domain_id}/grids/topography/3dep"

    def test_minimal_request_creates_grid(self, client, domain_for_testing):
        """Minimal request creates a grid with 10m elevation (defaults)."""
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
        assert data["source"]["name"] == "3dep"
        assert data["source"]["product"] == "topography"
        assert data["source"]["source_resolution"] == 10
        assert data["source"]["bands"] == ["elevation"]

        # Check single continuous band
        assert len(data["bands"]) == 1
        assert data["bands"][0]["key"] == "elevation"
        assert data["bands"][0]["type"] == "continuous"
        assert data["bands"][0]["unit"] == "m"
        assert data["bands"][0]["index"] == 0

    def test_all_bands(self, client, domain_for_testing):
        """Request with all three bands."""
        request_body = {
            "bands": ["elevation", "slope", "aspect"],
        }

        response = client.post(self.route(domain_for_testing["id"]), json=request_body)

        assert response.status_code == 201

        data = response.json()
        assert len(data["bands"]) == 3
        assert data["bands"][0]["key"] == "elevation"
        assert data["bands"][1]["key"] == "slope"
        assert data["bands"][2]["key"] == "aspect"
        assert data["source"]["bands"] == ["elevation", "slope", "aspect"]

    def test_1m_source_resolution(self, client, domain_for_testing):
        """Request with 1m source product."""
        request_body = {"source_resolution": 1, "bands": ["elevation"]}

        response = client.post(self.route(domain_for_testing["id"]), json=request_body)

        assert response.status_code == 201
        assert response.json()["source"]["source_resolution"] == 1

    def test_30m_source_resolution(self, client, domain_for_testing):
        """Request with 30m source product."""
        request_body = {"source_resolution": 30}

        response = client.post(self.route(domain_for_testing["id"]), json=request_body)

        assert response.status_code == 201
        assert response.json()["source"]["source_resolution"] == 30

    def test_invalid_source_resolution_rejected(self, client, domain_for_testing):
        """Invalid source_resolution returns 422."""
        request_body = {"source_resolution": 5}

        response = client.post(self.route(domain_for_testing["id"]), json=request_body)

        assert response.status_code == 422

    def test_with_metadata(self, client, domain_for_testing):
        """Request with name, description, and tags."""
        request_body = {
            "name": "High-res terrain",
            "description": "10m DEM for fire modeling",
            "tags": ["3dep", "topography"],
            "source_resolution": 10,
            "bands": ["elevation", "slope"],
        }

        response = client.post(self.route(domain_for_testing["id"]), json=request_body)

        assert response.status_code == 201

        data = response.json()
        assert data["name"] == "High-res terrain"
        assert data["description"] == "10m DEM for fire modeling"
        assert data["tags"] == ["3dep", "topography"]
        assert data["source"]["source_resolution"] == 10
        assert len(data["bands"]) == 2

    def test_georeference_is_null_on_creation(self, client, domain_for_testing):
        """Georeference is null until backend populates it."""
        response = client.post(self.route(domain_for_testing["id"]), json={})
        assert response.status_code == 201
        assert response.json()["georeference"] is None

    def test_response_excludes_owner_id(self, client, domain_for_testing):
        """Response should not expose the owner_id field."""
        response = client.post(self.route(domain_for_testing["id"]), json={})
        assert response.status_code == 201
        assert "owner_id" not in response.json()

    @pytest.mark.parametrize("example_name,example_value", ALL_3DEP_EXAMPLE_VALUES)
    def test_documented_example_creates_grid(
        self, client, domain_for_testing, example_name, example_value
    ):
        """Each documented 3DEP example should successfully create a grid."""
        response = client.post(self.route(domain_for_testing["id"]), json=example_value)

        assert response.status_code == 201, (
            f"Example '{example_name}' failed with status {response.status_code}: "
            f"{response.json()}"
        )

        data = response.json()
        assert data["source"]["name"] == "3dep"
        assert data["source"]["product"] == "topography"

    def test_extent_buffer_cells_defaults_to_zero(self, client, domain_for_testing):
        """Omitted extent_buffer_cells persists 0 regardless of band selection."""
        response = client.post(self.route(domain_for_testing["id"]), json={})
        assert response.status_code == 201
        assert response.json()["source"]["extent_buffer_cells"] == 0

    def test_extent_buffer_cells_default_unchanged_by_derivatives(
        self, client, domain_for_testing
    ):
        """Slope/aspect do NOT bump the persisted buffer; gradient overhead is internal."""
        response = client.post(
            self.route(domain_for_testing["id"]),
            json={"bands": ["elevation", "slope", "aspect"]},
        )
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
