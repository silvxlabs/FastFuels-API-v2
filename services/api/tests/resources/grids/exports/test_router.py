"""
Integration tests for api/v2/resources/grids/exports/router.py

Tests the per-grid export endpoint: POST /domains/{domain_id}/grids/{grid_id}/exports/{format}

These tests make real HTTP requests to the API and interact with Firestore.
"""

import pytest

from lib.config import EXPORTS_COLLECTION, GRIDS_COLLECTION
from tests.fixtures import make_grid_data

# Fixtures


@pytest.fixture(scope="session")
def completed_grid(firestore_client, domain_for_testing):
    """A completed grid with georeference, for export tests."""
    grid_data = make_grid_data(
        domain_id=domain_for_testing["id"],
        name="Completed grid for export",
        status="completed",
        bands=[
            {"key": "fbfm", "type": "categorical", "unit": None, "index": 0},
            {"key": "fuel_load.1hr", "type": "continuous", "unit": "kg/m²", "index": 1},
            {
                "key": "fuel_load.10hr",
                "type": "continuous",
                "unit": "kg/m²",
                "index": 2,
            },
        ],
    )
    doc_ref = firestore_client.collection(GRIDS_COLLECTION).document(grid_data["id"])
    doc_ref.set(grid_data)
    yield grid_data
    doc_ref.delete()


@pytest.fixture(scope="session")
def pending_grid(firestore_client, domain_for_testing):
    """A pending grid (not yet completed), for validation tests."""
    grid_data = make_grid_data(
        domain_id=domain_for_testing["id"],
        name="Pending grid",
        status="pending",
        georeference=None,
    )
    doc_ref = firestore_client.collection(GRIDS_COLLECTION).document(grid_data["id"])
    doc_ref.set(grid_data)
    yield grid_data
    doc_ref.delete()


def grid_export_route(domain_id, grid_id, fmt="geotiff"):
    """Per-grid: POST /domains/{domain_id}/grids/{grid_id}/exports/{format}"""
    return f"/domains/{domain_id}/grids/{grid_id}/exports/{fmt}"


def cleanup_export(firestore_client, export_id):
    """Delete an export document created during testing."""
    firestore_client.collection(EXPORTS_COLLECTION).document(export_id).delete()


# POST /domains/{domain_id}/grids/{grid_id}/exports/{format}


class TestCreateGridExportGeotiff:
    """Test the POST /domains/{domain_id}/grids/{grid_id}/exports/geotiff endpoint."""

    def test_create_all_bands(
        self, client, firestore_client, domain_for_testing, completed_grid
    ):
        """Create export with all bands (no body or empty body)."""
        response = client.post(
            grid_export_route(domain_for_testing["id"], completed_grid["id"]),
            json={
                "name": "All bands export",
                "tags": ["test"],
            },
        )

        assert response.status_code == 201

        data = response.json()
        assert data["status"] == "pending"
        assert data["domain_id"] == domain_for_testing["id"]
        assert data["source"]["name"] == "geotiff"
        assert data["source"]["grid_id"] == completed_grid["id"]
        assert data["source"]["bands"] is None
        assert data["name"] == "All bands export"
        assert "id" in data

        cleanup_export(firestore_client, data["id"])

    def test_create_band_subset(
        self, client, firestore_client, domain_for_testing, completed_grid
    ):
        """Create export with a subset of bands."""
        response = client.post(
            grid_export_route(domain_for_testing["id"], completed_grid["id"]),
            json={
                "bands": ["fuel_load.1hr", "fuel_load.10hr"],
            },
        )

        assert response.status_code == 201

        data = response.json()
        assert data["source"]["bands"] == ["fuel_load.1hr", "fuel_load.10hr"]

        cleanup_export(firestore_client, data["id"])

    def test_create_minimal(
        self, client, firestore_client, domain_for_testing, completed_grid
    ):
        """Minimal request with empty body."""
        response = client.post(
            grid_export_route(domain_for_testing["id"], completed_grid["id"]),
            json={},
        )

        assert response.status_code == 201

        data = response.json()
        assert data["name"] == ""
        assert data["description"] == ""
        assert data["tags"] == []
        assert data["source"]["grid_id"] == completed_grid["id"]

        cleanup_export(firestore_client, data["id"])

    def test_grid_not_found_returns_404(self, client, domain_for_testing):
        """Export of non-existent grid returns 404."""
        response = client.post(
            grid_export_route(
                domain_for_testing["id"], "00000000000000000000000000000000"
            ),
            json={},
        )
        assert response.status_code == 404

    def test_grid_not_completed_returns_422(
        self, client, domain_for_testing, pending_grid
    ):
        """Export of a pending grid returns 422."""
        response = client.post(
            grid_export_route(domain_for_testing["id"], pending_grid["id"]),
            json={},
        )
        assert response.status_code == 422

    def test_invalid_band_returns_422(self, client, domain_for_testing, completed_grid):
        """Export with non-existent band returns 422."""
        response = client.post(
            grid_export_route(domain_for_testing["id"], completed_grid["id"]),
            json={
                "bands": ["nonexistent_band"],
            },
        )
        assert response.status_code == 422

    def test_invalid_format_returns_422(
        self, client, domain_for_testing, completed_grid
    ):
        """Export with invalid format returns 422."""
        response = client.post(
            grid_export_route(
                domain_for_testing["id"], completed_grid["id"], fmt="pdf"
            ),
            json={},
        )
        assert response.status_code == 422


class TestCreateGridExportZarr:
    """Test the POST /domains/{domain_id}/grids/{grid_id}/exports/zarr endpoint."""

    def test_create_all_bands(
        self, client, firestore_client, domain_for_testing, completed_grid
    ):
        """Create zarr export with all bands."""
        response = client.post(
            grid_export_route(
                domain_for_testing["id"], completed_grid["id"], fmt="zarr"
            ),
            json={
                "name": "All bands zarr export",
                "tags": ["test"],
            },
        )

        assert response.status_code == 201

        data = response.json()
        assert data["status"] == "pending"
        assert data["source"]["name"] == "zarr"
        assert data["source"]["grid_id"] == completed_grid["id"]
        assert data["source"]["bands"] is None

        cleanup_export(firestore_client, data["id"])

    def test_create_band_subset(
        self, client, firestore_client, domain_for_testing, completed_grid
    ):
        """Create zarr export with a subset of bands."""
        response = client.post(
            grid_export_route(
                domain_for_testing["id"], completed_grid["id"], fmt="zarr"
            ),
            json={
                "bands": ["fuel_load.1hr"],
            },
        )

        assert response.status_code == 201

        data = response.json()
        assert data["source"]["bands"] == ["fuel_load.1hr"]

        cleanup_export(firestore_client, data["id"])

    def test_create_minimal(
        self, client, firestore_client, domain_for_testing, completed_grid
    ):
        """Minimal zarr export request."""
        response = client.post(
            grid_export_route(
                domain_for_testing["id"], completed_grid["id"], fmt="zarr"
            ),
            json={},
        )

        assert response.status_code == 201

        data = response.json()
        assert data["source"]["name"] == "zarr"
        assert data["source"]["grid_id"] == completed_grid["id"]

        cleanup_export(firestore_client, data["id"])
