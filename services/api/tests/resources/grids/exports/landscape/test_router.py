"""
Integration tests for api/v2/resources/grids/exports/landscape/router.py

Tests the landscape combined export endpoint:
    POST /domains/{domain_id}/grids/exports/landscape

These tests make real HTTP requests to the API and interact with Firestore.
"""

import pytest

from lib.config import EXPORTS_COLLECTION, GRIDS_COLLECTION
from tests.fixtures import make_grid_data

# The shared test domain bbox is (500000, 5200000, 501000, 5201000) in
# EPSG:32611 — 1000 m per side. At the default 30 m landscape resolution the
# lattice pads up to 34x34 cells anchored at the bbox's lower-left, so the
# padded north edge sits at 5200000 + 34 * 30 = 5201020. Role grids are
# seeded on exactly that lattice, matching what domain-target grid alignment
# produces for LANDFIRE-sourced grids.
RES = 30.0
N30 = 34
WEST = 500000.0
NORTH_PADDED = 5200000.0 + N30 * RES  # 5201020
TRANSFORM_30M = (RES, 0.0, WEST, 0.0, -RES, NORTH_PADDED)
GEOREF_30M = {
    "crs": "EPSG:32611",
    "transform": TRANSFORM_30M,
    "shape": (N30, N30),
}

_TOPO_BANDS = [
    {"key": "elevation", "type": "continuous", "unit": "m", "index": 0},
    {"key": "slope", "type": "continuous", "unit": "deg", "index": 1},
    {"key": "aspect", "type": "continuous", "unit": "deg", "index": 2},
]
_FBFM_BANDS = [{"key": "fbfm", "type": "categorical", "unit": None, "index": 0}]
_CANOPY_BANDS = [
    {"key": "cc", "type": "continuous", "unit": "%", "index": 0},
    {"key": "chm", "type": "continuous", "unit": "m", "index": 1},
    {"key": "cbh", "type": "continuous", "unit": "m", "index": 2},
    {"key": "cbd", "type": "continuous", "unit": "kg/m**3", "index": 3},
]


def _seed_grid(firestore_client, data: dict) -> dict:
    firestore_client.collection(GRIDS_COLLECTION).document(data["id"]).set(data)
    return data


@pytest.fixture(scope="session")
def topo_grid(firestore_client, domain_for_testing):
    """2D 30 m topography grid with elevation + slope + aspect."""
    data = make_grid_data(
        domain_id=domain_for_testing["id"],
        name="Landscape test topo grid",
        status="completed",
        source={"name": "3dep"},
        bands=_TOPO_BANDS,
        georeference=GEOREF_30M,
    )
    yield _seed_grid(firestore_client, data)
    firestore_client.collection(GRIDS_COLLECTION).document(data["id"]).delete()


@pytest.fixture(scope="session")
def fbfm_grid(firestore_client, domain_for_testing):
    """2D 30 m built-in LANDFIRE FBFM40 grid."""
    data = make_grid_data(
        domain_id=domain_for_testing["id"],
        name="Landscape test fbfm40 grid",
        status="completed",
        source={"name": "landfire", "product": "fbfm40", "version": "2024"},
        bands=_FBFM_BANDS,
        georeference=GEOREF_30M,
    )
    yield _seed_grid(firestore_client, data)
    firestore_client.collection(GRIDS_COLLECTION).document(data["id"]).delete()


@pytest.fixture(scope="session")
def canopy_grid(firestore_client, domain_for_testing):
    """2D 30 m canopy grid with cc + chm + cbh + cbd."""
    data = make_grid_data(
        domain_id=domain_for_testing["id"],
        name="Landscape test canopy grid",
        status="completed",
        source={"name": "landfire", "product": "canopy", "version": "2024"},
        bands=_CANOPY_BANDS,
        georeference=GEOREF_30M,
    )
    yield _seed_grid(firestore_client, data)
    firestore_client.collection(GRIDS_COLLECTION).document(data["id"]).delete()


@pytest.fixture(scope="session")
def uploaded_fuel_grid(firestore_client, domain_for_testing):
    """2D 30 m custom fuel model grid (uploaded provenance, no product)."""
    data = make_grid_data(
        domain_id=domain_for_testing["id"],
        name="Landscape test uploaded fuel grid",
        status="completed",
        source={"name": "upload"},
        bands=_FBFM_BANDS,
        georeference=GEOREF_30M,
    )
    yield _seed_grid(firestore_client, data)
    firestore_client.collection(GRIDS_COLLECTION).document(data["id"]).delete()


@pytest.fixture(scope="session")
def pending_topo_grid(firestore_client, domain_for_testing):
    """A topo-shaped grid still in pending status, for status-check tests."""
    data = make_grid_data(
        domain_id=domain_for_testing["id"],
        name="Landscape pending topo grid",
        status="pending",
        source={"name": "3dep"},
        bands=_TOPO_BANDS,
        georeference=None,
    )
    yield _seed_grid(firestore_client, data)
    firestore_client.collection(GRIDS_COLLECTION).document(data["id"]).delete()


@pytest.fixture(scope="session")
def three_d_topo_shaped_grid(firestore_client, domain_for_testing):
    """3D grid that *has* an elevation band but is the wrong dimensionality."""
    data = make_grid_data(
        domain_id=domain_for_testing["id"],
        name="Landscape 3D topo-shaped grid",
        status="completed",
        source={"operation": "voxelize", "input": "inventory", "entity": "tree"},
        bands=[{"key": "elevation", "type": "continuous", "unit": "m", "index": 0}],
        georeference={
            "crs": "EPSG:32611",
            "transform": TRANSFORM_30M,
            "shape": (5, N30, N30),
            "z_origin": 0.0,
            "z_resolution": 1.0,
        },
    )
    yield _seed_grid(firestore_client, data)
    firestore_client.collection(GRIDS_COLLECTION).document(data["id"]).delete()


def _route(domain_id: str) -> str:
    return f"/domains/{domain_id}/grids/exports/landscape"


def _minimal_body(topo_id: str, fbfm_id: str, canopy_id: str) -> dict:
    return {
        "fire_behavior_fuel_model": "fbfm40",
        "elevation": {"grid_id": topo_id, "band": "elevation"},
        "slope": {"grid_id": topo_id, "band": "slope"},
        "aspect": {"grid_id": topo_id, "band": "aspect"},
        "fuel_model": {"grid_id": fbfm_id, "band": "fbfm"},
        "canopy_cover": {"grid_id": canopy_id, "band": "cc"},
        "canopy_height": {"grid_id": canopy_id, "band": "chm"},
        "canopy_base_height": {"grid_id": canopy_id, "band": "cbh"},
        "canopy_bulk_density": {"grid_id": canopy_id, "band": "cbd"},
    }


def _cleanup_export(firestore_client, export_id: str) -> None:
    firestore_client.collection(EXPORTS_COLLECTION).document(export_id).delete()


# Happy path


class TestCreateLandscapeExport:
    def test_minimal(
        self,
        client,
        firestore_client,
        domain_for_testing,
        topo_grid,
        fbfm_grid,
        canopy_grid,
    ):
        body = _minimal_body(topo_grid["id"], fbfm_grid["id"], canopy_grid["id"])
        body["name"] = "Landscape minimal"

        response = client.post(_route(domain_for_testing["id"]), json=body)
        assert response.status_code == 201, response.text

        data = response.json()
        assert data["status"] == "pending"
        assert data["domain_id"] == domain_for_testing["id"]
        assert data["name"] == "Landscape minimal"
        assert data["source"]["name"] == "landscape"
        assert data["source"]["domain_id"] == domain_for_testing["id"]
        assert data["source"]["fire_behavior_fuel_model"] == "fbfm40"

        lattice = data["source"]["resolved"]["landscape_grid"]
        assert lattice["nx"] == N30
        assert lattice["ny"] == N30
        assert lattice["dx"] == RES
        assert lattice["crs"] == "EPSG:32611"
        assert lattice["transform"][2] == WEST
        assert lattice["transform"][5] == NORTH_PADDED
        # Default alignment is recorded on the persisted source.
        assert data["source"]["alignment"]["target"] == "domain"
        assert data["source"]["alignment"]["resolution"] == RES

        _cleanup_export(firestore_client, data["id"])

    def test_fbfm13_with_uploaded_grid(
        self,
        client,
        firestore_client,
        domain_for_testing,
        topo_grid,
        uploaded_fuel_grid,
        canopy_grid,
    ):
        body = _minimal_body(
            topo_grid["id"], uploaded_fuel_grid["id"], canopy_grid["id"]
        )
        body["fire_behavior_fuel_model"] = "fbfm13"

        response = client.post(_route(domain_for_testing["id"]), json=body)
        assert response.status_code == 201, response.text

        data = response.json()
        assert data["source"]["fire_behavior_fuel_model"] == "fbfm13"

        _cleanup_export(firestore_client, data["id"])

    def test_grid_target_alignment(
        self,
        client,
        firestore_client,
        domain_for_testing,
        topo_grid,
        fbfm_grid,
        canopy_grid,
    ):
        body = _minimal_body(topo_grid["id"], fbfm_grid["id"], canopy_grid["id"])
        body["alignment"] = {"target": "grid", "grid_id": topo_grid["id"]}

        response = client.post(_route(domain_for_testing["id"]), json=body)
        assert response.status_code == 201, response.text

        data = response.json()
        assert data["source"]["alignment"]["target"] == "grid"
        assert data["source"]["alignment"]["grid_id"] == topo_grid["id"]
        lattice = data["source"]["resolved"]["landscape_grid"]
        assert lattice["nx"] == N30
        assert lattice["ny"] == N30
        assert lattice["dx"] == RES

        _cleanup_export(firestore_client, data["id"])

    def test_grid_target_not_found_returns_404(
        self,
        client,
        domain_for_testing,
        topo_grid,
        fbfm_grid,
        canopy_grid,
    ):
        body = _minimal_body(topo_grid["id"], fbfm_grid["id"], canopy_grid["id"])
        body["alignment"] = {
            "target": "grid",
            "grid_id": "00000000000000000000000000000000",
        }

        response = client.post(_route(domain_for_testing["id"]), json=body)
        assert response.status_code == 404


# Negative paths


class TestLandscapeExportValidation:
    def test_missing_required_role_returns_422(
        self,
        client,
        domain_for_testing,
        topo_grid,
        fbfm_grid,
        canopy_grid,
    ):
        body = _minimal_body(topo_grid["id"], fbfm_grid["id"], canopy_grid["id"])
        body.pop("canopy_bulk_density")

        response = client.post(_route(domain_for_testing["id"]), json=body)
        assert response.status_code == 422

    def test_grid_not_found_returns_404(
        self,
        client,
        domain_for_testing,
        topo_grid,
        fbfm_grid,
        canopy_grid,
    ):
        body = _minimal_body(topo_grid["id"], fbfm_grid["id"], canopy_grid["id"])
        body["elevation"] = {
            "grid_id": "00000000000000000000000000000000",
            "band": "elevation",
        }

        response = client.post(_route(domain_for_testing["id"]), json=body)
        assert response.status_code == 404

    def test_pending_grid_returns_422(
        self,
        client,
        domain_for_testing,
        pending_topo_grid,
        fbfm_grid,
        canopy_grid,
    ):
        body = _minimal_body(
            pending_topo_grid["id"], fbfm_grid["id"], canopy_grid["id"]
        )

        response = client.post(_route(domain_for_testing["id"]), json=body)
        assert response.status_code == 422

    def test_missing_band_returns_422(
        self,
        client,
        domain_for_testing,
        topo_grid,
        fbfm_grid,
        canopy_grid,
    ):
        body = _minimal_body(topo_grid["id"], fbfm_grid["id"], canopy_grid["id"])
        body["elevation"]["band"] = "dem"  # not on the grid

        response = client.post(_route(domain_for_testing["id"]), json=body)
        assert response.status_code == 422
        assert "missing" in response.text.lower() or "band" in response.text.lower()

    def test_wrong_unit_returns_422(
        self,
        client,
        domain_for_testing,
        topo_grid,
        fbfm_grid,
        canopy_grid,
    ):
        # Use the canopy grid's chm (m) where canopy_cover expects %.
        # Both grid + band exist; only the unit is wrong.
        body = _minimal_body(topo_grid["id"], fbfm_grid["id"], canopy_grid["id"])
        body["canopy_cover"] = {"grid_id": canopy_grid["id"], "band": "chm"}

        response = client.post(_route(domain_for_testing["id"]), json=body)
        assert response.status_code == 422
        assert "unit" in response.text.lower()

    def test_wrong_dimensionality_returns_422(
        self,
        client,
        domain_for_testing,
        three_d_topo_shaped_grid,
        topo_grid,
        fbfm_grid,
        canopy_grid,
    ):
        body = _minimal_body(topo_grid["id"], fbfm_grid["id"], canopy_grid["id"])
        body["elevation"] = {
            "grid_id": three_d_topo_shaped_grid["id"],
            "band": "elevation",
        }

        response = client.post(_route(domain_for_testing["id"]), json=body)
        assert response.status_code == 422
        assert "2D" in response.text or "3D" in response.text

    def test_fbfm13_declared_for_builtin_fbfm40_returns_422(
        self,
        client,
        domain_for_testing,
        topo_grid,
        fbfm_grid,
        canopy_grid,
    ):
        body = _minimal_body(topo_grid["id"], fbfm_grid["id"], canopy_grid["id"])
        body["fire_behavior_fuel_model"] = "fbfm13"

        response = client.post(_route(domain_for_testing["id"]), json=body)
        assert response.status_code == 422
        detail = response.json()["detail"]
        assert "FBFM40" in detail
        assert "fbfm13" in detail

    def test_resolution_mismatch_returns_422(
        self,
        client,
        domain_for_testing,
        topo_grid,
        fbfm_grid,
        canopy_grid,
    ):
        # Ask for a 10 m landscape while every role grid is 30 m.
        body = _minimal_body(topo_grid["id"], fbfm_grid["id"], canopy_grid["id"])
        body["alignment"] = {"target": "domain", "resolution": 10.0}

        response = client.post(_route(domain_for_testing["id"]), json=body)
        assert response.status_code == 422
        detail = response.json()["detail"]
        assert "resolution mismatch" in detail.lower()
        assert '"alignment": {"resolution": 30.0}' in detail
