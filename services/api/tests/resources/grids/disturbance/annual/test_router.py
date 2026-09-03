"""
Integration tests for api/v2/resources/grids/disturbance/annual/router.py

Tests the LANDFIRE Limited Annual Disturbance endpoint.
These tests make real HTTP requests to the API and interact with Firestore.
"""

import pytest
from api.resources.grids.disturbance.annual.examples import (
    LFPS_DISTURBANCE_EXAMPLE_VALUES,
)

from lib.landfire import LANDFIRE_VERSIONS


class TestCreateLandfireDisturbance:
    """Test the POST /domains/{domain_id}/grids/disturbance/annual/landfire
    endpoint.

    annual_disturbance has no staged national release -- every request goes
    through a real LFPS coverage check. Unlike fbfm40/fccs's current-year
    on-demand versions, LDist's geo_areas is always "All" (it's a single
    national dataset, not rolled out region by region), so any domain has
    real coverage -- domain_for_testing works fine, same as the staged
    products' tests.
    """

    def route(self, domain_id):
        return f"/domains/{domain_id}/grids/disturbance/annual/landfire"

    def test_minimal_request_creates_grid(self, client, domain_for_testing):
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

        assert data["source"]["name"] == "landfire"
        assert data["source"]["product"] == "annual_disturbance"
        assert (
            data["source"]["version"]
            == LANDFIRE_VERSIONS["annual_disturbance"]["default"]
        )

        assert len(data["bands"]) == 1
        assert data["bands"][0]["key"] == "annual_disturbance"
        assert data["bands"][0]["type"] == "categorical"
        assert data["bands"][0]["unit"] is None

    def test_request_with_metadata(self, client, domain_for_testing):
        request_body = {
            "name": "Disturbance codes",
            "description": "Recent disturbance codes for baseline",
            "tags": ["baseline", "disturbance"],
        }
        response = client.post(self.route(domain_for_testing["id"]), json=request_body)

        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "Disturbance codes"
        assert data["description"] == "Recent disturbance codes for baseline"
        assert data["tags"] == ["baseline", "disturbance"]

    def test_georeference_is_null_on_creation(self, client, domain_for_testing):
        response = client.post(self.route(domain_for_testing["id"]), json={})
        assert response.status_code == 201
        assert response.json()["georeference"] is None

    def test_invalid_domain_returns_404(self, client):
        response = client.post(self.route("00000000000000000000000000000000"), json={})
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_wrong_owner_domain_returns_404(self, client, domain_with_different_owner):
        response = client.post(self.route(domain_with_different_owner["id"]), json={})
        assert response.status_code == 404

    def test_response_excludes_owner_id(self, client, domain_for_testing):
        response = client.post(self.route(domain_for_testing["id"]), json={})
        assert response.status_code == 201
        assert "owner_id" not in response.json()

    @pytest.mark.parametrize(
        "example_name,example_value", LFPS_DISTURBANCE_EXAMPLE_VALUES
    )
    def test_documented_example_creates_grid(
        self, client, domain_for_testing, example_name, example_value
    ):
        """Each documented example creates a grid against real LFPS
        coverage -- proving the coverage-check path works end-to-end."""
        response = client.post(self.route(domain_for_testing["id"]), json=example_value)

        assert response.status_code == 201, (
            f"Example '{example_name}' failed with status {response.status_code}: "
            f"{response.json()}"
        )
        data = response.json()
        assert data["source"]["name"] == "landfire"
        assert data["source"]["product"] == "annual_disturbance"

    def test_extent_buffer_cells_defaults_to_zero(self, client, domain_for_testing):
        response = client.post(self.route(domain_for_testing["id"]), json={})
        assert response.status_code == 201
        assert response.json()["source"]["extent_buffer_cells"] == 0

    @pytest.mark.parametrize("buffer", [0, 10])
    def test_extent_buffer_cells_explicit_value_persisted(
        self, client, domain_for_testing, buffer
    ):
        response = client.post(
            self.route(domain_for_testing["id"]), json={"extent_buffer_cells": buffer}
        )
        assert response.status_code == 201
        assert response.json()["source"]["extent_buffer_cells"] == buffer

    def test_extent_buffer_cells_negative_rejected(self, client, domain_for_testing):
        response = client.post(
            self.route(domain_for_testing["id"]), json={"extent_buffer_cells": -1}
        )
        assert response.status_code == 422

    def test_extent_buffer_cells_above_maximum_rejected(
        self, client, domain_for_testing
    ):
        response = client.post(
            self.route(domain_for_testing["id"]), json={"extent_buffer_cells": 11}
        )
        assert response.status_code == 422

    def test_invalid_version_rejected(self, client, domain_for_testing):
        response = client.post(
            self.route(domain_for_testing["id"]), json={"version": "2019"}
        )
        assert response.status_code == 422

    def test_alignment_defaults_to_domain_target(self, client, domain_for_testing):
        response = client.post(self.route(domain_for_testing["id"]), json={})
        assert response.status_code == 201
        assert response.json()["source"]["alignment"]["target"] == "domain"
