"""
Integration tests for api/v2/resources/grids/exports/quicfire/router.py

Tests the QUIC-Fire combined export endpoint:
    POST /domains/{domain_id}/grids/exports/quicfire

These tests make real HTTP requests to the API and interact with Firestore.
"""

import pytest

from lib.config import EXPORTS_COLLECTION, GRIDS_COLLECTION
from tests.fixtures import make_grid_data

# All fixture grids share the same 2D footprint (origin + dx) so the
# resolution-alignment check passes by default. The "bad_dx" grid uses a
# different dx to exercise the mismatch path.
DOMAIN_ORIGIN = (500000.0, 5201000.0)
DX = 2.0
NX = NY = 500
NZ = 10
TRANSFORM_2D = (DX, 0.0, DOMAIN_ORIGIN[0], 0.0, -DX, DOMAIN_ORIGIN[1])
TRANSFORM_3D = TRANSFORM_2D
GEOREF_2D = {"crs": "EPSG:32611", "transform": TRANSFORM_2D, "shape": (NY, NX)}
GEOREF_3D = {
    "crs": "EPSG:32611",
    "transform": TRANSFORM_3D,
    "shape": (NZ, NY, NX),
    "z_origin": 0.0,
    "z_resolution": 1.0,
}


def _seed_grid(firestore_client, data: dict) -> dict:
    firestore_client.collection(GRIDS_COLLECTION).document(data["id"]).set(data)
    return data


# Fixtures: one canopy grid, one surface grid, plus aux grids for SAVR /
# topography / negative cases. All session-scoped to keep the test suite fast.


@pytest.fixture(scope="session")
def canopy_grid(firestore_client, domain_for_testing):
    """3D tree grid with bulk_density.foliage.live + fuel_moisture.live + savr.foliage."""
    data = make_grid_data(
        domain_id=domain_for_testing["id"],
        name="QUIC-Fire test canopy grid",
        status="completed",
        source={"operation": "voxelize", "input": "inventory", "entity": "tree"},
        bands=[
            {
                "key": "bulk_density.foliage.live",
                "type": "continuous",
                "unit": "kg/m**3",
                "index": 0,
            },
            {
                "key": "fuel_moisture.live",
                "type": "continuous",
                "unit": "%",
                "index": 1,
            },
            {
                "key": "savr.foliage",
                "type": "continuous",
                "unit": "1/m",
                "index": 2,
            },
        ],
        georeference=GEOREF_3D,
    )
    yield _seed_grid(firestore_client, data)
    firestore_client.collection(GRIDS_COLLECTION).document(data["id"]).delete()


@pytest.fixture(scope="session")
def surface_grid(firestore_client, domain_for_testing):
    """2D lookup grid with fuel_load.1hr + fuel_depth + savr.1hr."""
    data = make_grid_data(
        domain_id=domain_for_testing["id"],
        name="QUIC-Fire test surface grid",
        status="completed",
        source={"name": "lookup", "table": "fbfm40"},
        bands=[
            {
                "key": "fuel_load.1hr",
                "type": "continuous",
                "unit": "kg/m**2",
                "index": 0,
            },
            {"key": "fuel_depth", "type": "continuous", "unit": "m", "index": 1},
            {"key": "savr.1hr", "type": "continuous", "unit": "1/m", "index": 2},
        ],
        georeference=GEOREF_2D,
    )
    yield _seed_grid(firestore_client, data)
    firestore_client.collection(GRIDS_COLLECTION).document(data["id"]).delete()


@pytest.fixture(scope="session")
def surface_moisture_grid(firestore_client, domain_for_testing):
    """2D uniform grid with fuel_moisture.1hr."""
    data = make_grid_data(
        domain_id=domain_for_testing["id"],
        name="QUIC-Fire test surface moisture grid",
        status="completed",
        source={"name": "uniform"},
        bands=[
            {
                "key": "fuel_moisture.1hr",
                "type": "continuous",
                "unit": "%",
                "index": 0,
            },
        ],
        georeference=GEOREF_2D,
    )
    yield _seed_grid(firestore_client, data)
    firestore_client.collection(GRIDS_COLLECTION).document(data["id"]).delete()


@pytest.fixture(scope="session")
def topography_grid(firestore_client, domain_for_testing):
    """2D topography grid with elevation."""
    data = make_grid_data(
        domain_id=domain_for_testing["id"],
        name="QUIC-Fire test topography grid",
        status="completed",
        source={"name": "3dep"},
        bands=[
            {"key": "elevation", "type": "continuous", "unit": "m", "index": 0},
        ],
        georeference=GEOREF_2D,
    )
    yield _seed_grid(firestore_client, data)
    firestore_client.collection(GRIDS_COLLECTION).document(data["id"]).delete()


@pytest.fixture(scope="session")
def pending_canopy_grid(firestore_client, domain_for_testing):
    """A canopy-shaped grid still in pending status, for status-check tests."""
    data = make_grid_data(
        domain_id=domain_for_testing["id"],
        name="QUIC-Fire pending canopy grid",
        status="pending",
        source={"operation": "voxelize", "input": "inventory", "entity": "tree"},
        bands=[
            {
                "key": "bulk_density.foliage.live",
                "type": "continuous",
                "unit": "kg/m**3",
                "index": 0,
            },
            {
                "key": "fuel_moisture.live",
                "type": "continuous",
                "unit": "%",
                "index": 1,
            },
        ],
        georeference=None,
    )
    yield _seed_grid(firestore_client, data)
    firestore_client.collection(GRIDS_COLLECTION).document(data["id"]).delete()


@pytest.fixture(scope="session")
def coarse_surface_grid(firestore_client, domain_for_testing):
    """2D lookup grid at 30 m resolution — for cell-size mismatch tests."""
    data = make_grid_data(
        domain_id=domain_for_testing["id"],
        name="QUIC-Fire 30m surface grid",
        status="completed",
        source={"name": "lookup", "table": "fbfm40"},
        bands=[
            {
                "key": "fuel_load.1hr",
                "type": "continuous",
                "unit": "kg/m**2",
                "index": 0,
            },
            {"key": "fuel_depth", "type": "continuous", "unit": "m", "index": 1},
        ],
        georeference={
            "crs": "EPSG:32611",
            "transform": (30.0, 0.0, DOMAIN_ORIGIN[0], 0.0, -30.0, DOMAIN_ORIGIN[1]),
            "shape": (34, 34),
        },
    )
    yield _seed_grid(firestore_client, data)
    firestore_client.collection(GRIDS_COLLECTION).document(data["id"]).delete()


@pytest.fixture(scope="session")
def two_d_canopy_shaped_grid(firestore_client, domain_for_testing):
    """2D grid that *has* the canopy band but is the wrong dimensionality.

    Used to test the dimensionality validator (canopy_bulk_density expects 3D).
    """
    data = make_grid_data(
        domain_id=domain_for_testing["id"],
        name="QUIC-Fire 2D canopy-shaped grid",
        status="completed",
        source={"operation": "voxelize", "input": "inventory", "entity": "tree"},
        bands=[
            {
                "key": "bulk_density.foliage.live",
                "type": "continuous",
                "unit": "kg/m**3",
                "index": 0,
            },
        ],
        georeference=GEOREF_2D,
    )
    yield _seed_grid(firestore_client, data)
    firestore_client.collection(GRIDS_COLLECTION).document(data["id"]).delete()


def _route(domain_id: str) -> str:
    return f"/domains/{domain_id}/grids/exports/quicfire"


def _minimal_body(canopy_id: str, surface_id: str, moisture_id: str) -> dict:
    return {
        "canopy_bulk_density": {
            "grid_id": canopy_id,
            "band": "bulk_density.foliage.live",
        },
        "canopy_moisture": {"grid_id": canopy_id, "band": "fuel_moisture.live"},
        "surface_fuel_load": {"grid_id": surface_id, "band": "fuel_load.1hr"},
        "surface_fuel_depth": {"grid_id": surface_id, "band": "fuel_depth"},
        "surface_moisture": {"grid_id": moisture_id, "band": "fuel_moisture.1hr"},
    }


def _cleanup_export(firestore_client, export_id: str) -> None:
    firestore_client.collection(EXPORTS_COLLECTION).document(export_id).delete()


# Happy path


class TestCreateQuicfireExport:
    def test_minimal(
        self,
        client,
        firestore_client,
        domain_for_testing,
        canopy_grid,
        surface_grid,
        surface_moisture_grid,
    ):
        body = _minimal_body(
            canopy_grid["id"], surface_grid["id"], surface_moisture_grid["id"]
        )
        body["name"] = "QUIC-Fire minimal"

        response = client.post(_route(domain_for_testing["id"]), json=body)
        assert response.status_code == 201, response.text

        data = response.json()
        assert data["status"] == "pending"
        assert data["domain_id"] == domain_for_testing["id"]
        assert data["name"] == "QUIC-Fire minimal"
        assert data["source"]["name"] == "quicfire"
        assert data["source"]["domain_id"] == domain_for_testing["id"]
        assert data["source"]["topography"] is None
        assert data["source"]["canopy_savr"] is None
        assert data["source"]["surface_savr"] is None

        fire_grid = data["source"]["resolved"]["fire_grid"]
        assert fire_grid["nx"] == NX
        assert fire_grid["ny"] == NY
        assert fire_grid["nz"] == NZ
        # Alignment was omitted → defaults to QF-recommended 2 m / 1 m.
        assert fire_grid["dx"] == DX
        assert fire_grid["dy"] == DX
        assert fire_grid["dz"] == 1.0
        # Default alignment is recorded on the persisted source.
        assert data["source"]["alignment"]["target"] == "domain"
        assert data["source"]["alignment"]["dx"] == DX

        _cleanup_export(firestore_client, data["id"])

    def test_with_topography(
        self,
        client,
        firestore_client,
        domain_for_testing,
        canopy_grid,
        surface_grid,
        surface_moisture_grid,
        topography_grid,
    ):
        body = _minimal_body(
            canopy_grid["id"], surface_grid["id"], surface_moisture_grid["id"]
        )
        body["topography"] = {"grid_id": topography_grid["id"], "band": "elevation"}

        response = client.post(_route(domain_for_testing["id"]), json=body)
        assert response.status_code == 201, response.text

        data = response.json()
        assert data["source"]["topography"]["grid_id"] == topography_grid["id"]
        assert data["source"]["topography"]["band"] == "elevation"

        _cleanup_export(firestore_client, data["id"])

    def test_with_savr(
        self,
        client,
        firestore_client,
        domain_for_testing,
        canopy_grid,
        surface_grid,
        surface_moisture_grid,
    ):
        body = _minimal_body(
            canopy_grid["id"], surface_grid["id"], surface_moisture_grid["id"]
        )
        body["canopy_savr"] = {"grid_id": canopy_grid["id"], "band": "savr.foliage"}
        body["surface_savr"] = {"grid_id": surface_grid["id"], "band": "savr.1hr"}

        response = client.post(_route(domain_for_testing["id"]), json=body)
        assert response.status_code == 201, response.text

        data = response.json()
        assert data["source"]["canopy_savr"]["band"] == "savr.foliage"
        assert data["source"]["surface_savr"]["band"] == "savr.1hr"
        assert data["source"]["canopy_savr"]["grid_id"] == canopy_grid["id"]

        _cleanup_export(firestore_client, data["id"])

    def test_explicit_domain_alignment(
        self,
        client,
        firestore_client,
        domain_for_testing,
        canopy_grid,
        surface_grid,
        surface_moisture_grid,
    ):
        """Explicit Domain alignment with default values matches omitting it."""
        body = _minimal_body(
            canopy_grid["id"], surface_grid["id"], surface_moisture_grid["id"]
        )
        body["alignment"] = {"target": "domain", "dx": DX, "dy": DX, "dz": 1.0}

        response = client.post(_route(domain_for_testing["id"]), json=body)
        assert response.status_code == 201, response.text

        fire_grid = response.json()["source"]["resolved"]["fire_grid"]
        assert fire_grid["dx"] == DX
        assert fire_grid["dy"] == DX
        assert fire_grid["dz"] == 1.0
        assert fire_grid["nx"] == NX
        assert fire_grid["ny"] == NY

        _cleanup_export(firestore_client, response.json()["id"])

    def test_grid_target_alignment(
        self,
        client,
        firestore_client,
        domain_for_testing,
        canopy_grid,
        surface_grid,
        surface_moisture_grid,
    ):
        """target='grid' anchors the fire grid to a referenced grid's lattice."""
        body = _minimal_body(
            canopy_grid["id"], surface_grid["id"], surface_moisture_grid["id"]
        )
        # Anchor the fire grid to the surface grid's lattice. Since all
        # fixtures share the same transform/shape, the resulting fire grid
        # matches the Domain-padded one cell-for-cell.
        body["alignment"] = {"target": "grid", "grid_id": surface_grid["id"]}

        response = client.post(_route(domain_for_testing["id"]), json=body)
        assert response.status_code == 201, response.text

        data = response.json()
        assert data["source"]["alignment"]["target"] == "grid"
        assert data["source"]["alignment"]["grid_id"] == surface_grid["id"]
        fire_grid = data["source"]["resolved"]["fire_grid"]
        assert fire_grid["nx"] == NX
        assert fire_grid["ny"] == NY
        assert fire_grid["dx"] == DX
        assert fire_grid["crs"] == "EPSG:32611"

        _cleanup_export(firestore_client, data["id"])

    def test_grid_target_not_found_returns_404(
        self,
        client,
        domain_for_testing,
        canopy_grid,
        surface_grid,
        surface_moisture_grid,
    ):
        body = _minimal_body(
            canopy_grid["id"], surface_grid["id"], surface_moisture_grid["id"]
        )
        body["alignment"] = {
            "target": "grid",
            "grid_id": "00000000000000000000000000000000",
        }

        response = client.post(_route(domain_for_testing["id"]), json=body)
        assert response.status_code == 404


# Negative paths


class TestQuicfireExportValidation:
    def test_missing_required_role_returns_422(
        self,
        client,
        domain_for_testing,
        canopy_grid,
        surface_grid,
        surface_moisture_grid,
    ):
        body = _minimal_body(
            canopy_grid["id"], surface_grid["id"], surface_moisture_grid["id"]
        )
        body.pop("surface_moisture")

        response = client.post(_route(domain_for_testing["id"]), json=body)
        assert response.status_code == 422

    def test_grid_not_found_returns_404(
        self,
        client,
        domain_for_testing,
        canopy_grid,
        surface_grid,
        surface_moisture_grid,
    ):
        body = _minimal_body(
            canopy_grid["id"], surface_grid["id"], surface_moisture_grid["id"]
        )
        body["topography"] = {
            "grid_id": "00000000000000000000000000000000",
            "band": "elevation",
        }

        response = client.post(_route(domain_for_testing["id"]), json=body)
        assert response.status_code == 404

    def test_pending_grid_returns_422(
        self,
        client,
        domain_for_testing,
        pending_canopy_grid,
        surface_grid,
        surface_moisture_grid,
    ):
        body = _minimal_body(
            pending_canopy_grid["id"], surface_grid["id"], surface_moisture_grid["id"]
        )

        response = client.post(_route(domain_for_testing["id"]), json=body)
        assert response.status_code == 422

    def test_missing_band_returns_422(
        self,
        client,
        domain_for_testing,
        canopy_grid,
        surface_grid,
        surface_moisture_grid,
    ):
        body = _minimal_body(
            canopy_grid["id"], surface_grid["id"], surface_moisture_grid["id"]
        )
        body["canopy_bulk_density"]["band"] = (
            "bulk_density.fine.live"  # not on the grid
        )

        response = client.post(_route(domain_for_testing["id"]), json=body)
        assert response.status_code == 422
        assert "missing" in response.text.lower() or "band" in response.text.lower()

    def test_wrong_unit_returns_422(
        self,
        client,
        domain_for_testing,
        canopy_grid,
        surface_grid,
        surface_moisture_grid,
    ):
        # Use the surface grid's fuel_load.1hr (kg/m**2) where canopy_bulk_density
        # expects kg/m**3. Both grid + band exist; only the unit is wrong.
        body = _minimal_body(
            canopy_grid["id"], surface_grid["id"], surface_moisture_grid["id"]
        )
        body["canopy_bulk_density"] = {
            "grid_id": surface_grid["id"],
            "band": "fuel_load.1hr",
        }

        response = client.post(_route(domain_for_testing["id"]), json=body)
        assert response.status_code == 422
        assert "unit" in response.text.lower()

    def test_wrong_dimensionality_returns_422(
        self,
        client,
        domain_for_testing,
        two_d_canopy_shaped_grid,
        surface_grid,
        surface_moisture_grid,
    ):
        # canopy_bulk_density expects a 3D grid; this fixture is 2D.
        body = _minimal_body(
            two_d_canopy_shaped_grid["id"],
            surface_grid["id"],
            surface_moisture_grid["id"],
        )

        response = client.post(_route(domain_for_testing["id"]), json=body)
        assert response.status_code == 422
        assert "2D" in response.text or "3D" in response.text

    def test_cell_size_mismatch_returns_422(
        self,
        client,
        domain_for_testing,
        canopy_grid,
        coarse_surface_grid,
        surface_moisture_grid,
    ):
        # Canopy is 2 m, coarse surface is 30 m.
        body = _minimal_body(
            canopy_grid["id"],
            coarse_surface_grid["id"],
            surface_moisture_grid["id"],
        )

        response = client.post(_route(domain_for_testing["id"]), json=body)
        assert response.status_code == 422
        detail = response.json()["detail"]
        assert "resolution mismatch" in detail.lower()
        assert '"alignment": {"dx": 30.0, "dy": 30.0}' in detail

    def test_dz_mismatch_returns_422(
        self,
        client,
        domain_for_testing,
        canopy_grid,
        surface_grid,
        surface_moisture_grid,
    ):
        # Tree grid is 1 m vertical; request asks for a 0.5 m fire grid.
        # alignment.dz is now honored, so the mismatch is rejected loudly
        # instead of being silently ignored (issue #377).
        body = _minimal_body(
            canopy_grid["id"], surface_grid["id"], surface_moisture_grid["id"]
        )
        body["alignment"] = {"target": "domain", "dx": DX, "dy": DX, "dz": 0.5}

        response = client.post(_route(domain_for_testing["id"]), json=body)
        assert response.status_code == 422
        detail = response.json()["detail"]
        assert "vertical resolution mismatch" in detail.lower()
        assert '"alignment": {"dz": 1.0}' in detail

    def test_savr_pair_required_canopy_only_returns_422(
        self,
        client,
        domain_for_testing,
        canopy_grid,
        surface_grid,
        surface_moisture_grid,
    ):
        body = _minimal_body(
            canopy_grid["id"], surface_grid["id"], surface_moisture_grid["id"]
        )
        body["canopy_savr"] = {"grid_id": canopy_grid["id"], "band": "savr.foliage"}

        response = client.post(_route(domain_for_testing["id"]), json=body)
        assert response.status_code == 422
        assert "canopy_savr" in response.text or "surface_savr" in response.text

    def test_savr_pair_required_surface_only_returns_422(
        self,
        client,
        domain_for_testing,
        canopy_grid,
        surface_grid,
        surface_moisture_grid,
    ):
        body = _minimal_body(
            canopy_grid["id"], surface_grid["id"], surface_moisture_grid["id"]
        )
        body["surface_savr"] = {"grid_id": surface_grid["id"], "band": "savr.1hr"}

        response = client.post(_route(domain_for_testing["id"]), json=body)
        assert response.status_code == 422
        assert "canopy_savr" in response.text or "surface_savr" in response.text
