"""
Integration tests for api/v2/resources/grids/fbfm40/router.py

Tests the FBFM40 LANDFIRE endpoint.
These tests make real HTTP requests to the API and interact with Firestore.
"""

import pytest
from api.resources.grids.fbfm40.examples import (
    LFPS_FBFM40_EXAMPLE_VALUES,
    STAGED_FBFM40_EXAMPLE_VALUES,
)

from lib.landfire import LANDFIRE_VERSIONS, SEASON_CODES


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

    @pytest.mark.parametrize("example_name,example_value", STAGED_FBFM40_EXAMPLE_VALUES)
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


class TestLfpsCoverage:
    """Real LFPS-coverage tests for both the seasonal and annual LFPS-sourced
    examples -- everything routed through validate_lfps_coverage."""

    def route(self, domain_id):
        return f"/domains/{domain_id}/grids/fbfm40/landfire"

    @pytest.mark.parametrize("example_name,example_value", LFPS_FBFM40_EXAMPLE_VALUES)
    def test_lfps_documented_example_creates_grid(
        self, client, lfps_covered_domain, example_name, example_value
    ):
        """Confirms validate_lfps_coverage succeeds against real LFPS for a
        domain with known coverage -- proving the coverage-check path works
        end-to-end, not just under mocks."""

        response = client.post(
            self.route(lfps_covered_domain["id"]), json=example_value
        )
        assert response.status_code == 201, (
            f"Example '{example_name}' failed with status {response.status_code}: "
            f"{response.json()}"
        )


class TestSeasonalCoverage:
    """Router-level tests for the `season` field on grid creation."""

    def route(self, domain_id):
        return f"/domains/{domain_id}/grids/fbfm40/landfire"

    def test_annual_version_with_season_rejected(self, client, domain_for_testing):
        """An annual-only version with season set fails schema validation
        (422) before any LFPS call is made -- deterministic, no network."""
        response = client.post(
            self.route(domain_for_testing["id"]),
            json={"version": "2024", "season": "SP"},
        )
        assert response.status_code == 422

    def test_season_without_lfps_version_rejected(self, client, domain_for_testing):
        """season set with the default annual version (2024) is rejected."""
        response = client.post(
            self.route(domain_for_testing["id"]),
            json={"season": "SP"},
        )
        assert response.status_code == 422

    def test_season_omitted_persists_none_and_annual_year(
        self, client, domain_for_testing
    ):
        """Without season, the grid is created with source.season == None and
        source.year == the annual vintage (the version year -- no LFPS call)."""
        response = client.post(self.route(domain_for_testing["id"]), json={})
        assert response.status_code == 201
        source = response.json()["source"]
        assert source["season"] is None
        assert source["year"] == 2024  # default version


class TestLandfireCoverage:
    """Live coverage pre-flight against a domain with known LFPS coverage."""

    def route(self, domain_id):
        return f"/domains/{domain_id}/grids/fbfm40/landfire/coverage"

    def test_reports_latest_release_with_create_link(self, client, lfps_covered_domain):
        response = client.get(self.route(lfps_covered_domain["id"]))
        assert response.status_code == 200, response.json()
        body = response.json()

        assert body["product"] == "fbfm40"
        latest = body["latest"]
        assert latest["coverage"] == "full"
        create = latest["links"]["create"]
        assert create["method"] == "POST"
        assert (
            create["href"]
            == f"/domains/{lfps_covered_domain['id']}/grids/fbfm40/landfire"
        )
        assert create["body"]["version"] == latest["version"]

    def test_lists_every_registry_version_with_staged_ones_full(
        self, client, lfps_covered_domain
    ):
        body = client.get(self.route(lfps_covered_domain["id"])).json()
        annual = {r["version"]: r for r in body["releases"] if r["season"] is None}

        versions = LANDFIRE_VERSIONS["fbfm40"]
        assert set(annual) == set(versions["available"]) | set(
            versions["lfps_available"]
        )
        for version in versions["available"]:
            assert annual[version]["coverage"] == "full"
            assert annual[version]["year"] == int(version)
            assert annual[version]["links"]["create"]["body"] == {"version": version}
        # The Kingman domain sits in the SW GeoArea LFPS serves the current year for.
        for version in versions["lfps_available"]:
            assert annual[version]["coverage"] == "full"

    def test_lists_every_season_of_the_on_demand_vintage(
        self, client, lfps_covered_domain
    ):
        body = client.get(self.route(lfps_covered_domain["id"])).json()
        seasonal = [r for r in body["releases"] if r["season"] is not None]

        on_demand = LANDFIRE_VERSIONS["fbfm40"]["lfps_available"]
        assert {(r["version"], r["season"]) for r in seasonal} == {
            (version, season) for version in on_demand for season in SEASON_CODES
        }
        for release in seasonal:
            if release["coverage"] == "unpublished":
                assert release["year"] is None
                assert release["links"]["create"] is None
            else:
                assert release["year"] > int(release["version"])
