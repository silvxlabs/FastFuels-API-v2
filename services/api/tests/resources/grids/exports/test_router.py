"""
Integration tests for api/v2/resources/grids/exports/router.py

Tests both the domain-level multi-grid endpoint and the per-grid endpoint.

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


def domain_route(domain_id):
    """Domain-level: POST /domains/{domain_id}/grids/exports/geotiff"""
    return f"/domains/{domain_id}/grids/exports/geotiff"


def grid_route(domain_id, grid_id):
    """Per-grid: POST /domains/{domain_id}/grids/{grid_id}/exports/geotiff"""
    return f"/domains/{domain_id}/grids/{grid_id}/exports/geotiff"


def cleanup_export(firestore_client, export_id):
    """Delete an export document created during testing."""
    firestore_client.collection(EXPORTS_COLLECTION).document(export_id).delete()


# POST /domains/{domain_id}/grids/exports/geotiff (domain-level, multi-grid)


class TestCreateGeoTiffExport:
    """Test the domain-level POST /domains/{domain_id}/grids/exports/geotiff endpoint."""

    def test_create_all_bands(
        self, client, firestore_client, domain_for_testing, completed_grid
    ):
        """Create export with all bands (bands=null)."""
        response = client.post(
            domain_route(domain_for_testing["id"]),
            json={
                "grid_ids": [completed_grid["id"]],
                "name": "All bands export",
                "tags": ["test"],
            },
        )

        assert response.status_code == 201

        data = response.json()
        assert data["status"] == "pending"
        assert data["domain_id"] == domain_for_testing["id"]
        assert data["source"]["name"] == "geotiff"
        assert data["source"]["grid_ids"] == [completed_grid["id"]]
        assert data["source"]["bands"] is None
        assert data["name"] == "All bands export"
        assert data["tags"] == ["test"]
        assert data["signed_url"] is None
        assert "id" in data
        assert "created_on" in data

        cleanup_export(firestore_client, data["id"])

    def test_create_band_subset(
        self, client, firestore_client, domain_for_testing, completed_grid
    ):
        """Create export with a subset of bands."""
        response = client.post(
            domain_route(domain_for_testing["id"]),
            json={
                "grid_ids": [completed_grid["id"]],
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
        """Minimal request with only grid_ids."""
        response = client.post(
            domain_route(domain_for_testing["id"]),
            json={"grid_ids": [completed_grid["id"]]},
        )

        assert response.status_code == 201

        data = response.json()
        assert data["name"] == ""
        assert data["description"] == ""
        assert data["tags"] == []

        cleanup_export(firestore_client, data["id"])

    def test_grid_not_found_returns_404(self, client, domain_for_testing):
        """Export of non-existent grid returns 404."""
        response = client.post(
            domain_route(domain_for_testing["id"]),
            json={"grid_ids": ["00000000000000000000000000000000"]},
        )
        assert response.status_code == 404

    def test_grid_not_completed_returns_422(
        self, client, domain_for_testing, pending_grid
    ):
        """Export of a pending grid returns 422."""
        response = client.post(
            domain_route(domain_for_testing["id"]),
            json={"grid_ids": [pending_grid["id"]]},
        )
        assert response.status_code == 422

    def test_invalid_band_returns_422(self, client, domain_for_testing, completed_grid):
        """Export with non-existent band returns 422."""
        response = client.post(
            domain_route(domain_for_testing["id"]),
            json={
                "grid_ids": [completed_grid["id"]],
                "bands": ["nonexistent_band"],
            },
        )
        assert response.status_code == 422

    def test_grid_wrong_domain_returns_404(
        self, client, domain_with_different_owner, completed_grid
    ):
        """Export grid from wrong domain returns 404 (domain ownership check)."""
        response = client.post(
            domain_route(domain_with_different_owner["id"]),
            json={"grid_ids": [completed_grid["id"]]},
        )
        assert response.status_code == 404

    def test_empty_grid_ids_returns_422(self, client, domain_for_testing):
        """Request with empty grid_ids list returns 422."""
        response = client.post(
            domain_route(domain_for_testing["id"]),
            json={"grid_ids": []},
        )
        assert response.status_code == 422

    def test_missing_grid_ids_returns_422(self, client, domain_for_testing):
        """Request without grid_ids returns 422."""
        response = client.post(
            domain_route(domain_for_testing["id"]),
            json={},
        )
        assert response.status_code == 422


# POST /domains/{domain_id}/grids/{grid_id}/exports/geotiff (per-grid)


class TestCreateSingleGridGeoTiffExport:
    """Test the per-grid POST /domains/{domain_id}/grids/{grid_id}/exports/geotiff endpoint."""

    def test_create_all_bands(
        self, client, firestore_client, domain_for_testing, completed_grid
    ):
        """Create export with all bands (no body or empty body)."""
        response = client.post(
            grid_route(domain_for_testing["id"], completed_grid["id"]),
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
        assert data["source"]["grid_ids"] == [completed_grid["id"]]
        assert data["source"]["bands"] is None
        assert data["name"] == "All bands export"
        assert "id" in data

        cleanup_export(firestore_client, data["id"])

    def test_create_band_subset(
        self, client, firestore_client, domain_for_testing, completed_grid
    ):
        """Create export with a subset of bands."""
        response = client.post(
            grid_route(domain_for_testing["id"], completed_grid["id"]),
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
            grid_route(domain_for_testing["id"], completed_grid["id"]),
            json={},
        )

        assert response.status_code == 201

        data = response.json()
        assert data["name"] == ""
        assert data["description"] == ""
        assert data["tags"] == []
        assert data["source"]["grid_ids"] == [completed_grid["id"]]

        cleanup_export(firestore_client, data["id"])

    def test_grid_not_found_returns_404(self, client, domain_for_testing):
        """Export of non-existent grid returns 404."""
        response = client.post(
            grid_route(domain_for_testing["id"], "00000000000000000000000000000000"),
            json={},
        )
        assert response.status_code == 404

    def test_grid_not_completed_returns_422(
        self, client, domain_for_testing, pending_grid
    ):
        """Export of a pending grid returns 422."""
        response = client.post(
            grid_route(domain_for_testing["id"], pending_grid["id"]),
            json={},
        )
        assert response.status_code == 422

    def test_invalid_band_returns_422(self, client, domain_for_testing, completed_grid):
        """Export with non-existent band returns 422."""
        response = client.post(
            grid_route(domain_for_testing["id"], completed_grid["id"]),
            json={
                "bands": ["nonexistent_band"],
            },
        )
        assert response.status_code == 422
