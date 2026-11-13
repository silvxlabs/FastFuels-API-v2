"""
Integration tests for the 3DEP coverage pre-flight endpoint.

GET /domains/{domain_id}/grids/topography/3dep/coverage

Tests use three domain fixtures with known 3DEP coverage characteristics:
- Bondurant, WY (EPSG:32612): Has S1M 1m coverage
- Blue Mountain, MT (EPSG:32611): No S1M 1m coverage, but has 1m lidar
  project tiles (so 1m via the project fallback, and 2m), and 10m/30m
- Pacific offshore, CA (EPSG:32610): No 1m coverage of either kind

These tests make real HTTP requests to the API, real S3 calls for 1m
resolution, and real TNM Access searches for 1m and 2m. They require a
running API server and valid GCP/AWS credentials.
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest
from api.resources.grids.topography.router import check_3dep_coverage

from lib.config import DOMAINS_COLLECTION
from lib.testing import SHARED_TEST_DOMAINS_DIR
from lib.threedep import TnmSearchError


def _load_domain(path: Path, owner_id: str) -> dict:
    """Load a shared domain JSON and prepare it for Firestore."""
    with open(path) as f:
        data = json.load(f)
    data["id"] = f"test-{uuid.uuid4().hex}"
    data["owner_id"] = owner_id
    data["created_on"] = datetime.now()
    data["modified_on"] = datetime.now()
    # Stringify coordinates for Firestore
    for feature in data.get("features", []):
        coords = feature.get("geometry", {}).get("coordinates")
        if isinstance(coords, list):
            feature["geometry"]["coordinates"] = json.dumps(coords)
    return data


@pytest.fixture(scope="session")
def bondurant_domain(firestore_client, test_owner_id):
    """Bondurant, WY — known S1M 1m coverage."""
    domain_data = _load_domain(
        SHARED_TEST_DOMAINS_DIR / "bondurant.json", test_owner_id
    )
    doc_ref = firestore_client.collection(DOMAINS_COLLECTION).document(
        domain_data["id"]
    )
    doc_ref.set(domain_data)
    yield domain_data
    doc_ref.delete()


@pytest.fixture(scope="session")
def blue_mtn_domain(firestore_client, test_owner_id):
    """Blue Mountain, MT — no S1M, has 1m lidar project tiles and 10m/30m."""
    domain_data = _load_domain(SHARED_TEST_DOMAINS_DIR / "blue_mtn.json", test_owner_id)
    doc_ref = firestore_client.collection(DOMAINS_COLLECTION).document(
        domain_data["id"]
    )
    doc_ref.set(domain_data)
    yield domain_data
    doc_ref.delete()


@pytest.fixture(scope="session")
def offshore_domain(firestore_client, test_owner_id):
    """Pacific offshore, CA — no 1m coverage of either kind."""
    domain_data = _load_domain(
        SHARED_TEST_DOMAINS_DIR / "pacific_offshore.json", test_owner_id
    )
    doc_ref = firestore_client.collection(DOMAINS_COLLECTION).document(
        domain_data["id"]
    )
    doc_ref.set(domain_data)
    yield domain_data
    doc_ref.delete()


def _route(domain_id: str) -> str:
    return f"/domains/{domain_id}/grids/topography/3dep/coverage"


class TestThreeDepCoverage10m:
    """Tests for 10m resolution coverage (pure math, no S3 I/O)."""

    def test_returns_available_tiles(self, client, blue_mtn_domain):
        """10m coverage should always be available for CONUS domains."""
        response = client.get(_route(blue_mtn_domain["id"]), params={"resolution": 10})

        assert response.status_code == 200
        data = response.json()
        assert data["resolution"] == 10
        assert data["available"] is True
        assert data["tile_count"] >= 1
        assert len(data["tiles"]) == data["tile_count"]
        assert all(url.endswith(".tif") for url in data["tiles"])
        assert data["acquisition_dates"] is None

    def test_tile_urls_use_correct_product_code(self, client, blue_mtn_domain):
        """10m tiles should use product code '13'."""
        response = client.get(_route(blue_mtn_domain["id"]), params={"resolution": 10})

        assert response.status_code == 200
        for url in response.json()["tiles"]:
            assert "/13/TIFF/current/" in url
            assert "USGS_13_" in url


class TestThreeDepCoverage30m:
    """Tests for 30m resolution coverage (pure math, no S3 I/O)."""

    def test_returns_available_tiles(self, client, blue_mtn_domain):
        """30m coverage should always be available for CONUS domains."""
        response = client.get(_route(blue_mtn_domain["id"]), params={"resolution": 30})

        assert response.status_code == 200
        data = response.json()
        assert data["resolution"] == 30
        assert data["available"] is True
        assert data["tile_count"] >= 1
        assert all(url.endswith(".tif") for url in data["tiles"])
        assert data["acquisition_dates"] is None

    def test_tile_urls_use_correct_product_code(self, client, blue_mtn_domain):
        """30m tiles should use product code '1'."""
        response = client.get(_route(blue_mtn_domain["id"]), params={"resolution": 30})

        assert response.status_code == 200
        for url in response.json()["tiles"]:
            assert "/1/TIFF/current/" in url
            assert "USGS_1_" in url


class TestThreeDepCoverage1m:
    """Tests for 1m resolution coverage (real S3 I/O)."""

    def test_available_when_s1m_coverage_exists(self, client, bondurant_domain):
        """Bondurant, WY has known S1M coverage — should return tiles."""
        response = client.get(_route(bondurant_domain["id"]), params={"resolution": 1})

        assert response.status_code == 200
        data = response.json()
        assert data["resolution"] == 1
        assert data["available"] is True
        assert data["tile_count"] >= 1
        assert len(data["tiles"]) == data["tile_count"]
        assert all(url.endswith(".tif") for url in data["tiles"])
        assert data["acquisition_dates"] is not None
        assert len(data["acquisition_dates"]) >= 1

    def test_falls_back_to_lidar_projects_without_s1m(self, client, blue_mtn_domain):
        """Blue Mountain, MT has no S1M coverage, but has lidar project tiles."""
        response = client.get(_route(blue_mtn_domain["id"]), params={"resolution": 1})

        assert response.status_code == 200
        data = response.json()
        assert data["resolution"] == 1
        assert data["available"] is True
        assert data["tile_count"] == len(data["tiles"]) >= 1
        assert all("/1m/Projects/" in url for url in data["tiles"])
        assert data["acquisition_dates"]

    def test_not_available_without_any_1m_coverage(self, client, offshore_domain):
        """Offshore has neither S1M nor lidar project tiles — should return empty."""
        response = client.get(_route(offshore_domain["id"]), params={"resolution": 1})

        assert response.status_code == 200
        data = response.json()
        assert data["resolution"] == 1
        assert data["available"] is False
        assert data["tile_count"] == 0
        assert data["tiles"] == []
        assert data["acquisition_dates"] is None

    def test_defaults_to_1m_resolution(self, client, bondurant_domain):
        """Resolution should default to 1 when not specified."""
        response = client.get(_route(bondurant_domain["id"]))

        assert response.status_code == 200
        assert response.json()["resolution"] == 1


class TestThreeDepCoverage2m:
    """Tests for 2m resolution coverage (real TNM Access search)."""

    def test_available_from_lidar_projects(self, client, blue_mtn_domain):
        response = client.get(_route(blue_mtn_domain["id"]), params={"resolution": 2})

        assert response.status_code == 200
        data = response.json()
        assert data["resolution"] == 2
        assert data["available"] is True
        assert data["tile_count"] == len(data["tiles"]) >= 1
        assert all("/1m/Projects/" in url for url in data["tiles"])
        assert data["acquisition_dates"]

    def test_not_available_without_lidar_projects(self, client, offshore_domain):
        response = client.get(_route(offshore_domain["id"]), params={"resolution": 2})

        assert response.status_code == 200
        data = response.json()
        assert data["available"] is False
        assert data["tiles"] == []
        assert data["acquisition_dates"] is None


class TestThreeDepCoverageValidation:
    """Tests for input validation and auth."""

    def test_invalid_resolution_returns_422(self, client, blue_mtn_domain):
        """Resolution must be 1, 2, 10, or 30."""
        response = client.get(_route(blue_mtn_domain["id"]), params={"resolution": 5})

        assert response.status_code == 422

    def test_invalid_domain_returns_404(self, client):
        """Non-existent domain_id returns 404."""
        response = client.get(
            _route("00000000000000000000000000000000"),
            params={"resolution": 10},
        )

        assert response.status_code == 404

    def test_wrong_owner_returns_404(self, client, domain_with_different_owner):
        """Domain owned by another user returns 404 (not 403)."""
        response = client.get(
            _route(domain_with_different_owner["id"]),
            params={"resolution": 10},
        )

        assert response.status_code == 404


class TestThreeDepCoverageResponseStructure:
    """Verify the response schema is correct."""

    def test_response_has_all_fields(self, client, blue_mtn_domain):
        """Response should have all expected fields."""
        response = client.get(_route(blue_mtn_domain["id"]), params={"resolution": 10})

        assert response.status_code == 200
        data = response.json()
        assert "resolution" in data
        assert "available" in data
        assert "tile_count" in data
        assert "tiles" in data
        assert "acquisition_dates" in data

    def test_types_are_correct(self, client, blue_mtn_domain):
        """Response field types should match the schema."""
        response = client.get(_route(blue_mtn_domain["id"]), params={"resolution": 10})

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data["resolution"], int)
        assert isinstance(data["available"], bool)
        assert isinstance(data["tile_count"], int)
        assert isinstance(data["tiles"], list)


class TestThreeDepCoverageSearchUnavailable:
    """A failed TNM search reports unknown coverage, not an error.

    Router tests hit a live server that patches can't reach, so these call
    the route function directly.
    """

    ROUTER = "api.resources.grids.topography.router"

    @staticmethod
    def _domain() -> dict:
        with open(SHARED_TEST_DOMAINS_DIR / "blue_mtn.json") as f:
            return json.load(f)

    def _check(self, resolution: int):
        return asyncio.run(check_3dep_coverage(self._domain(), resolution))

    @pytest.mark.parametrize("resolution", [1, 2])
    def test_search_failure_reports_unknown_coverage(self, resolution, caplog):
        with (
            patch(f"{self.ROUTER}.discover_s1m_tiles", return_value=([], [])),
            patch(
                f"{self.ROUTER}.discover_lidar_project_tiles",
                side_effect=TnmSearchError("read timed out"),
            ),
            caplog.at_level(logging.WARNING, logger=self.ROUTER),
        ):
            response = self._check(resolution)

        data = response.model_dump()
        assert data["resolution"] == resolution
        assert data["available"] is None
        assert data["unavailable_reason"]
        assert data["tile_count"] == 0
        assert data["tiles"] == []
        assert data["acquisition_dates"] is None
        assert "read timed out" in caplog.text

    def test_s1m_hit_skips_search(self):
        s1m_urls = ["https://prd-tnm.s3.amazonaws.com/S1M/tile.tif"]
        with (
            patch(
                f"{self.ROUTER}.discover_s1m_tiles",
                return_value=(s1m_urls, ["20230515"]),
            ),
            patch(f"{self.ROUTER}.discover_lidar_project_tiles") as search,
        ):
            response = self._check(1)

        search.assert_not_called()
        assert response.available is True
        assert response.unavailable_reason is None

    @pytest.mark.parametrize("resolution", [10, 30])
    def test_arc_second_never_searches(self, resolution):
        with patch(f"{self.ROUTER}.discover_lidar_project_tiles") as search:
            response = self._check(resolution)

        search.assert_not_called()
        assert response.available is True
        assert response.unavailable_reason is None
