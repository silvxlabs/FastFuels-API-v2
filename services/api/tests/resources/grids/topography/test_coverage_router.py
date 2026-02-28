"""
Integration tests for the 3DEP coverage pre-flight endpoint.

GET /domains/{domain_id}/grids/topography/3dep/coverage

Tests use two domain fixtures with known 3DEP coverage characteristics:
- Bondurant, WY (EPSG:32612): Has S1M 1m coverage
- Blue Mountain, MT (EPSG:32611): No S1M 1m coverage, but has 10m/30m

These tests make real HTTP requests to the API and real S3 calls for 1m
resolution. They require a running API server and valid GCP/AWS credentials.
"""

import json
import uuid
from datetime import datetime
from pathlib import Path

import pytest

from lib.config import DOMAINS_COLLECTION
from lib.testing import SHARED_TEST_DOMAINS_DIR


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
    """Blue Mountain, MT — no S1M 1m coverage, has 10m/30m."""
    domain_data = _load_domain(SHARED_TEST_DOMAINS_DIR / "blue_mtn.json", test_owner_id)
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

    def test_not_available_when_no_s1m_coverage(self, client, blue_mtn_domain):
        """Blue Mountain, MT has no S1M coverage — should return empty."""
        response = client.get(_route(blue_mtn_domain["id"]), params={"resolution": 1})

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


class TestThreeDepCoverageValidation:
    """Tests for input validation and auth."""

    def test_invalid_resolution_returns_422(self, client, blue_mtn_domain):
        """Resolution must be 1, 10, or 30."""
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
