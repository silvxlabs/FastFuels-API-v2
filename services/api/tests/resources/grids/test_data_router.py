"""
Integration tests for grid data streaming endpoints.

Tests the grid data streaming endpoints:
  GET /domains/{domain_id}/grids/{grid_id}/chunks/{chunk_index}
  GET /domains/{domain_id}/grids/{grid_id}/data/{band}/{chunk_index}

These tests use the static-test-blue-mtn-landfire-fbfm40 fixture which has
real zarr data on GCS.
"""

import json

import pytest

from lib.config import GRIDS_COLLECTION
from lib.testing import SHARED_TEST_GRIDS_DIR
from tests.fixtures import make_grid_data

STATIC_NAME = "static-test-blue-mtn-landfire-fbfm40"


def _load_static_template(static_name: str) -> dict:
    path = SHARED_TEST_GRIDS_DIR / f"{static_name}.json"
    with open(path) as f:
        return json.load(f)


# Fixtures


@pytest.fixture(scope="session")
def static_grid_in_firestore(firestore_client, test_owner_id, domain_for_testing):
    """Register the static FBFM40 fixture as a Firestore grid doc."""
    template = _load_static_template(STATIC_NAME)
    template["id"] = STATIC_NAME
    template["owner_id"] = test_owner_id
    template["domain_id"] = domain_for_testing["id"]
    doc_ref = firestore_client.collection(GRIDS_COLLECTION).document(STATIC_NAME)
    doc_ref.set(template)
    yield template
    doc_ref.delete()


@pytest.fixture(scope="session")
def pending_grid_in_firestore(firestore_client, domain_for_testing):
    """A pending grid (not completed) for validation tests."""
    grid_data = make_grid_data(
        domain_id=domain_for_testing["id"],
        name="Pending grid for data tests",
        status="pending",
        georeference=None,
    )
    doc_ref = firestore_client.collection(GRIDS_COLLECTION).document(grid_data["id"])
    doc_ref.set(grid_data)
    yield grid_data
    doc_ref.delete()


@pytest.fixture(scope="session")
def grid_with_different_owner(firestore_client, domain_with_different_owner):
    """A grid owned by a different user."""
    grid_data = make_grid_data(
        domain_id=domain_with_different_owner["id"],
        owner_id="different-owner",
        name="Other User's Grid",
        status="completed",
    )
    doc_ref = firestore_client.collection(GRIDS_COLLECTION).document(grid_data["id"])
    doc_ref.set(grid_data)
    yield grid_data
    doc_ref.delete()


# Helpers


def chunk_route(domain_id, grid_id, chunk_index):
    return f"/domains/{domain_id}/grids/{grid_id}/chunks/{chunk_index}"


def data_route(domain_id, grid_id, band, chunk_index=0, **params):
    return f"/domains/{domain_id}/grids/{grid_id}/data/{band}/{chunk_index}", params


# GET /domains/{domain_id}/grids/{grid_id}/chunks/{chunk_index}


class TestGetGridDataChunkMetadata:
    def test_chunk_0_returns_200(
        self, client, domain_for_testing, static_grid_in_firestore
    ):
        """Chunk 0 of static fixture returns correct metadata."""
        response = client.get(chunk_route(domain_for_testing["id"], STATIC_NAME, 0))
        assert response.status_code == 200

        data = response.json()
        georef = static_grid_in_firestore["georeference"]
        # Static fixture fits in a single chunk (chunk_shape is 512×512), so
        # chunk 0 covers the entire grid.
        assert data["index"] == 0
        assert data["shape"] == list(georef["shape"])
        assert data["offset"] == [0, 0]
        assert len(data["transform"]) == 6
        for i in range(6):
            assert data["transform"][i] == pytest.approx(georef["transform"][i])

    def test_chunk_out_of_range_returns_422(
        self, client, domain_for_testing, static_grid_in_firestore
    ):
        """Chunk index beyond total chunks returns 422."""
        response = client.get(chunk_route(domain_for_testing["id"], STATIC_NAME, 1))
        assert response.status_code == 422

    def test_grid_not_completed_returns_422(
        self, client, domain_for_testing, pending_grid_in_firestore
    ):
        response = client.get(
            chunk_route(domain_for_testing["id"], pending_grid_in_firestore["id"], 0)
        )
        assert response.status_code == 422

    def test_grid_not_found_returns_404(self, client, domain_for_testing):
        response = client.get(
            chunk_route(
                domain_for_testing["id"],
                "00000000000000000000000000000000",
                0,
            )
        )
        assert response.status_code == 404

    def test_grid_wrong_owner_returns_404(
        self, client, domain_for_testing, grid_with_different_owner
    ):
        response = client.get(
            chunk_route(
                domain_for_testing["id"],
                grid_with_different_owner["id"],
                0,
            )
        )
        assert response.status_code == 404

    def test_grid_wrong_domain_returns_404(
        self, client, domain_with_different_owner, static_grid_in_firestore
    ):
        response = client.get(
            chunk_route(domain_with_different_owner["id"], STATIC_NAME, 0)
        )
        assert response.status_code == 404


# GET /domains/{domain_id}/grids/{grid_id}/data/{band}/{chunk_index}


class TestGetGridData:
    def test_json_format_returns_200(
        self, client, domain_for_testing, static_grid_in_firestore
    ):
        """JSON response with band data."""
        url, params = data_route(
            domain_for_testing["id"],
            STATIC_NAME,
            band="fbfm",
            format="json",
        )
        response = client.get(url, params=params)
        assert response.status_code == 200

        data = response.json()
        expected_shape = list(static_grid_in_firestore["georeference"]["shape"])
        assert data["shape"] == expected_shape
        assert data["order"] == "C"
        assert isinstance(data["data"], list)
        assert len(data["data"]) == expected_shape[0] * expected_shape[1]

    def test_binary_format_returns_200(
        self, client, domain_for_testing, static_grid_in_firestore
    ):
        """Binary response has correct headers."""
        url, params = data_route(
            domain_for_testing["id"],
            STATIC_NAME,
            band="fbfm",
            format="binary",
        )
        response = client.get(url, params=params)
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/octet-stream"
        assert "X-Data-Shape" in response.headers
        assert "X-Data-Dtype" in response.headers
        assert response.headers["X-Data-Order"] == "C"
        expected_shape = static_grid_in_firestore["georeference"]["shape"]
        assert response.headers["X-Data-Shape"] == ",".join(
            str(s) for s in expected_shape
        )

    def test_default_params(self, client, domain_for_testing, static_grid_in_firestore):
        """Defaults: format=json, order=C."""
        url, params = data_route(domain_for_testing["id"], STATIC_NAME, band="fbfm")
        response = client.get(url, params=params)
        assert response.status_code == 200

        data = response.json()
        assert data["order"] == "C"
        assert data["shape"] == list(static_grid_in_firestore["georeference"]["shape"])

    def test_missing_band_returns_404(
        self, client, domain_for_testing, static_grid_in_firestore
    ):
        """Omitting band from the path returns 404 (no matching route)."""
        response = client.get(
            f"/domains/{domain_for_testing['id']}/grids/{STATIC_NAME}/data"
        )
        assert response.status_code == 404

    def test_invalid_band_returns_422(
        self, client, domain_for_testing, static_grid_in_firestore
    ):
        """Non-existent band returns 422."""
        url, params = data_route(
            domain_for_testing["id"],
            STATIC_NAME,
            band="nonexistent_band",
        )
        response = client.get(url, params=params)
        assert response.status_code == 422

    def test_chunk_out_of_range_returns_422(
        self, client, domain_for_testing, static_grid_in_firestore
    ):
        url, params = data_route(
            domain_for_testing["id"],
            STATIC_NAME,
            chunk_index=99,
            band="fbfm",
        )
        response = client.get(url, params=params)
        assert response.status_code == 422

    def test_grid_not_completed_returns_422(
        self, client, domain_for_testing, pending_grid_in_firestore
    ):
        url, params = data_route(
            domain_for_testing["id"],
            pending_grid_in_firestore["id"],
            band="fbfm",
        )
        response = client.get(url, params=params)
        assert response.status_code == 422

    def test_grid_not_found_returns_404(self, client, domain_for_testing):
        url, params = data_route(
            domain_for_testing["id"],
            "00000000000000000000000000000000",
            band="fbfm",
        )
        response = client.get(url, params=params)
        assert response.status_code == 404

    def test_grid_wrong_owner_returns_404(
        self, client, domain_for_testing, grid_with_different_owner
    ):
        url, params = data_route(
            domain_for_testing["id"],
            grid_with_different_owner["id"],
            band="fbfm",
        )
        response = client.get(url, params=params)
        assert response.status_code == 404

    def test_grid_wrong_domain_returns_404(
        self, client, domain_with_different_owner, static_grid_in_firestore
    ):
        url, params = data_route(
            domain_with_different_owner["id"],
            STATIC_NAME,
            band="fbfm",
        )
        response = client.get(url, params=params)
        assert response.status_code == 404
