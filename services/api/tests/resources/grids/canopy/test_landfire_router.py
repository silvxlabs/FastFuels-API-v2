"""
Integration tests for the LANDFIRE canopy endpoint
(POST /domains/{domain_id}/grids/canopy/landfire).

These tests make real HTTP requests to the API and interact with Firestore.
"""

import pytest
from api.resources.grids.canopy.examples import (
    ALL_LANDFIRE_CANOPY_EXAMPLE_VALUES,
)
from api.resources.grids.canopy.schema import LandfireCanopyVersion


class TestCreateLandfireCanopy:
    """Test the POST /domains/{domain_id}/grids/canopy/landfire endpoint."""

    def route(self, domain_id):
        return f"/domains/{domain_id}/grids/canopy/landfire"

    def test_minimal_request_creates_grid(self, client, domain_for_testing):
        """Minimal request creates a grid with all four canopy bands."""
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
        assert data["source"]["name"] == "canopy"
        assert data["source"]["product"] == "landfire"
        assert data["source"]["version"] == "2024"
        assert data["source"]["bands"] == ["chm", "cbd", "cbh", "cc"]

        # Check four continuous bands at correct indices and units
        assert len(data["bands"]) == 4
        assert data["bands"][0]["key"] == "chm"
        assert data["bands"][0]["type"] == "continuous"
        assert data["bands"][0]["unit"] == "m"
        assert data["bands"][0]["index"] == 0
        assert data["bands"][1]["key"] == "cbd"
        assert data["bands"][1]["unit"] == "kg/m**3"
        assert data["bands"][1]["index"] == 1
        assert data["bands"][2]["key"] == "cbh"
        assert data["bands"][2]["unit"] == "m"
        assert data["bands"][2]["index"] == 2
        assert data["bands"][3]["key"] == "cc"
        assert data["bands"][3]["unit"] == "%"
        assert data["bands"][3]["index"] == 3

    def test_single_band_cbd(self, client, domain_for_testing):
        """Request with only the CBD band."""
        response = client.post(
            self.route(domain_for_testing["id"]), json={"bands": ["cbd"]}
        )

        assert response.status_code == 201

        data = response.json()
        assert len(data["bands"]) == 1
        assert data["bands"][0]["key"] == "cbd"
        assert data["bands"][0]["index"] == 0
        assert data["source"]["bands"] == ["cbd"]

    def test_crown_fire_inputs(self, client, domain_for_testing):
        """Two-band request returning CBD and CBH for crown fire modeling."""
        response = client.post(
            self.route(domain_for_testing["id"]),
            json={"bands": ["cbd", "cbh"]},
        )

        assert response.status_code == 201

        data = response.json()
        assert [b["key"] for b in data["bands"]] == ["cbd", "cbh"]
        assert [b["index"] for b in data["bands"]] == [0, 1]
        assert data["source"]["bands"] == ["cbd", "cbh"]

    def test_chm_only_mirrors_meta_naip_band_key(self, client, domain_for_testing):
        """A LANDFIRE-source grid with only the chm band uses the same band
        key as the Meta/NAIP CHM endpoints, so downstream consumers (e.g. the
        CHM-tree inventory) can use it interchangeably."""
        response = client.post(
            self.route(domain_for_testing["id"]), json={"bands": ["chm"]}
        )

        assert response.status_code == 201

        data = response.json()
        assert len(data["bands"]) == 1
        assert data["bands"][0]["key"] == "chm"
        assert data["bands"][0]["unit"] == "m"

    def test_request_with_metadata(self, client, domain_for_testing):
        """Request with name, description, tags, and explicit version."""
        request_body = {
            "name": "Canopy fuels",
            "description": "Test canopy data",
            "tags": ["canopy", "landfire"],
            "version": "2024",
            "bands": ["chm", "cbd"],
        }

        response = client.post(self.route(domain_for_testing["id"]), json=request_body)

        assert response.status_code == 201

        data = response.json()
        assert data["name"] == "Canopy fuels"
        assert data["description"] == "Test canopy data"
        assert data["tags"] == ["canopy", "landfire"]
        assert data["source"]["version"] == "2024"
        assert len(data["bands"]) == 2

    def test_georeference_is_null_on_creation(self, client, domain_for_testing):
        """Georeference is null until the backend populates it."""
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
        "example_name,example_value", ALL_LANDFIRE_CANOPY_EXAMPLE_VALUES
    )
    def test_documented_example_creates_grid(
        self, client, domain_for_testing, example_name, example_value
    ):
        """Each documented LANDFIRE canopy example should create a grid."""
        response = client.post(self.route(domain_for_testing["id"]), json=example_value)

        assert response.status_code == 201, (
            f"Example '{example_name}' failed with status {response.status_code}: "
            f"{response.json()}"
        )

        data = response.json()
        assert data["source"]["name"] == "canopy"
        assert data["source"]["product"] == "landfire"

    @pytest.mark.parametrize("version", [v.value for v in LandfireCanopyVersion])
    def test_version_can_be_set(self, client, domain_for_testing, version):
        """Each valid version creates a grid with the correct version set."""
        response = client.post(
            self.route(domain_for_testing["id"]),
            json={"version": version, "bands": ["chm"]},
        )
        assert response.status_code == 201
        assert response.json()["source"]["version"] == version

    def test_invalid_version_returns_422(self, client, domain_for_testing):
        response = client.post(
            self.route(domain_for_testing["id"]),
            json={"version": "2022", "bands": ["chm"]},
        )
        assert response.status_code == 422

    def test_empty_bands_returns_422(self, client, domain_for_testing):
        response = client.post(
            self.route(domain_for_testing["id"]),
            json={"bands": []},
        )
        assert response.status_code == 422

    def test_duplicate_bands_returns_422(self, client, domain_for_testing):
        response = client.post(
            self.route(domain_for_testing["id"]),
            json={"bands": ["chm", "chm"]},
        )
        assert response.status_code == 422

    def test_unknown_band_returns_422(self, client, domain_for_testing):
        response = client.post(
            self.route(domain_for_testing["id"]),
            json={"bands": ["lai"]},
        )
        assert response.status_code == 422

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
