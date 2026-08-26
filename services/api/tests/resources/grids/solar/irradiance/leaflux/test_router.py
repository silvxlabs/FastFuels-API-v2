"""
Integration tests for api/resources/grids/solar/irradiance/leaflux/router.py.

Tests the POST /domains/{domain_id}/grids/solar/irradiance/leaflux endpoint.
These tests make real HTTP requests to the API and interact with Firestore
and Cloud Tasks.
"""

import pytest
from api.resources.grids.solar.irradiance.leaflux.examples import (
    ALL_LEAFLUX_IRRADIANCE_EXAMPLE_VALUES,
)

from lib.config import GRIDS_COLLECTION
from tests.fixtures import make_grid_data

DATE_TIME = "2025-07-01T19:00:00Z"
CANOPY = "irradiance.canopy.relative"
SURFACE = "irradiance.surface.relative"
TRANSFORM = (2.0, 0.0, 500000.0, 0.0, -2.0, 5201000.0)

# --- Fixtures ---


@pytest.fixture
def grid_factory(firestore_client):
    """Create source-grid documents in Firestore and clean them up after.

    Only the fields the router/validators read are set explicitly: `bands`
    (list of {"key": ...}), `georeference.shape` (drives the dimensionality
    check), `status`, and `checksum`. Owner defaults to the test owner via
    make_grid_data so get_document_async(owner_id=...) matches the caller.
    """
    created = []

    def _make(
        domain_id,
        *,
        status="completed",
        bands=("leaf_area_density",),
        shape=(6, 40, 40),
        checksum="src-checksum",
        crs="EPSG:32611",
        transform=TRANSFORM,
    ):
        data = make_grid_data(domain_id=domain_id, name="source grid", status=status)
        data["bands"] = [{"key": key} for key in bands]
        data["georeference"] = {
            "crs": crs,
            "transform": list(transform),
            "shape": list(shape),
        }
        data["checksum"] = checksum
        firestore_client.collection(GRIDS_COLLECTION).document(data["id"]).set(data)
        created.append(data["id"])
        return data

    yield _make

    for grid_id in created:
        firestore_client.collection(GRIDS_COLLECTION).document(grid_id).delete()


@pytest.fixture
def source_lad_grid(grid_factory, domain_for_testing):
    """A completed 3D grid with a leaf_area_density band in domain_for_testing."""
    return grid_factory(
        domain_for_testing["id"], bands=("leaf_area_density",), shape=(6, 40, 40)
    )


@pytest.fixture
def terrain_grid(grid_factory, domain_for_testing):
    """A completed 2D grid with an elevation band in domain_for_testing."""
    return grid_factory(domain_for_testing["id"], bands=("elevation",), shape=(40, 40))


# --- Tests ---


class TestCreateLeafluxIrradianceGrid:
    """POST /domains/{domain_id}/grids/solar/irradiance/leaflux."""

    def route(self, domain_id):
        return f"/domains/{domain_id}/grids/solar/irradiance/leaflux"

    # --- Success paths ---

    def test_minimal_request_creates_grid(
        self, client, domain_for_testing, source_lad_grid
    ):
        """Minimal request creates a pending grid with resolved defaults."""
        body = {"source_lad_grid_id": source_lad_grid["id"], "date_time": DATE_TIME}
        response = client.post(self.route(domain_for_testing["id"]), json=body)
        assert response.status_code == 201

        data = response.json()
        assert len(data["id"]) == 32
        assert data["domain_id"] == domain_for_testing["id"]
        assert data["status"] == "pending"
        assert data["name"] == ""
        assert data["description"] == ""
        assert data["tags"] == []
        assert data["modifications"] == []
        assert data["georeference"] is None
        assert data["chunks"] is None

        source = data["source"]
        assert source["operation"] == "irradiance"
        assert source["input"] == "grid"
        assert source["entity"] == "solar"
        assert source["source_lad_grid_id"] == source_lad_grid["id"]
        assert "source_terrain_grid_id" not in source
        assert source["bands"] == [SURFACE]
        assert source["extinction_coefficient"] == 0.5
        assert source["source_lad_grid_checksum"] == source_lad_grid["checksum"]

        assert len(data["bands"]) == 1
        assert data["bands"][0]["key"] == SURFACE
        assert data["bands"][0]["type"] == "continuous"
        assert data["bands"][0]["index"] == 0

    def test_full_request_with_terrain(
        self, client, domain_for_testing, source_lad_grid, terrain_grid
    ):
        body = {
            "name": "Midday irradiance",
            "description": "Canopy and surface.",
            "tags": ["solar", "irradiance"],
            "source_lad_grid_id": source_lad_grid["id"],
            "source_terrain_grid_id": terrain_grid["id"],
            "bands": [CANOPY, SURFACE],
            "date_time": DATE_TIME,
            "extinction_coefficient": 0.4,
        }
        response = client.post(self.route(domain_for_testing["id"]), json=body)
        assert response.status_code == 201

        data = response.json()
        assert data["name"] == "Midday irradiance"
        assert data["tags"] == ["solar", "irradiance"]
        source = data["source"]
        assert source["source_terrain_grid_id"] == terrain_grid["id"]
        assert source["bands"] == [CANOPY, SURFACE]
        assert source["extinction_coefficient"] == 0.4
        assert [b["index"] for b in data["bands"]] == [0, 1]

    def test_canopy_only_request(self, client, domain_for_testing, source_lad_grid):
        body = {
            "source_lad_grid_id": source_lad_grid["id"],
            "bands": [CANOPY],
            "date_time": DATE_TIME,
        }
        response = client.post(self.route(domain_for_testing["id"]), json=body)
        assert response.status_code == 201
        assert response.json()["source"]["bands"] == [CANOPY]

    def test_surface_only_no_terrain(self, client, domain_for_testing, source_lad_grid):
        body = {
            "source_lad_grid_id": source_lad_grid["id"],
            "bands": [SURFACE],
            "date_time": DATE_TIME,
        }
        response = client.post(self.route(domain_for_testing["id"]), json=body)
        assert response.status_code == 201
        assert "source_terrain_grid_id" not in response.json()["source"]

    def test_surface_only_with_terrain(
        self, client, domain_for_testing, source_lad_grid, terrain_grid
    ):
        body = {
            "source_lad_grid_id": source_lad_grid["id"],
            "source_terrain_grid_id": terrain_grid["id"],
            "bands": [SURFACE],
            "date_time": DATE_TIME,
        }
        response = client.post(self.route(domain_for_testing["id"]), json=body)
        assert response.status_code == 201
        assert response.json()["source"]["source_terrain_grid_id"] == terrain_grid["id"]

    def test_missing_bands_uses_default(
        self, client, domain_for_testing, source_lad_grid
    ):
        body = {"source_lad_grid_id": source_lad_grid["id"], "date_time": DATE_TIME}
        response = client.post(self.route(domain_for_testing["id"]), json=body)
        assert response.status_code == 201
        assert response.json()["source"]["bands"] == [SURFACE]

    def test_response_excludes_owner_id(
        self, client, domain_for_testing, source_lad_grid
    ):
        body = {"source_lad_grid_id": source_lad_grid["id"], "date_time": DATE_TIME}
        response = client.post(self.route(domain_for_testing["id"]), json=body)
        assert response.status_code == 201
        assert "owner_id" not in response.json()

    # --- Domain validation ---

    def test_invalid_domain_returns_404(self, client, source_lad_grid):
        body = {"source_lad_grid_id": source_lad_grid["id"], "date_time": DATE_TIME}
        response = client.post(
            self.route("00000000000000000000000000000000"), json=body
        )
        assert response.status_code == 404

    def test_wrong_owner_domain_returns_404(
        self, client, domain_with_different_owner, source_lad_grid
    ):
        body = {"source_lad_grid_id": source_lad_grid["id"], "date_time": DATE_TIME}
        response = client.post(self.route(domain_with_different_owner["id"]), json=body)
        assert response.status_code == 404

    # --- Source grid validation ---

    def test_nonexistent_source_grid_returns_404(self, client, domain_for_testing):
        body = {
            "source_lad_grid_id": "00000000000000000000000000000000",
            "date_time": DATE_TIME,
        }
        response = client.post(self.route(domain_for_testing["id"]), json=body)
        assert response.status_code == 404

    def test_source_grid_not_completed_returns_422(
        self, client, domain_for_testing, grid_factory
    ):
        pending = grid_factory(domain_for_testing["id"], status="pending")
        body = {"source_lad_grid_id": pending["id"], "date_time": DATE_TIME}
        response = client.post(self.route(domain_for_testing["id"]), json=body)
        assert response.status_code == 422

    def test_source_grid_in_another_domain_returns_404(
        self, client, domain_for_testing, second_domain, grid_factory
    ):
        other_domain_lad = grid_factory(
            second_domain["id"], bands=("leaf_area_density",), shape=(6, 40, 40)
        )
        body = {
            "source_lad_grid_id": other_domain_lad["id"],
            "date_time": DATE_TIME,
        }
        response = client.post(self.route(domain_for_testing["id"]), json=body)
        assert response.status_code == 404

    def test_source_grid_missing_lad_band_returns_422(
        self, client, domain_for_testing, grid_factory
    ):
        no_lad = grid_factory(
            domain_for_testing["id"],
            bands=("bulk_density.foliage.live",),
            shape=(6, 40, 40),
        )
        body = {"source_lad_grid_id": no_lad["id"], "date_time": DATE_TIME}
        response = client.post(self.route(domain_for_testing["id"]), json=body)
        assert response.status_code == 422

    def test_source_grid_not_3d_returns_422(
        self, client, domain_for_testing, grid_factory
    ):
        two_d = grid_factory(
            domain_for_testing["id"], bands=("leaf_area_density",), shape=(40, 40)
        )
        body = {"source_lad_grid_id": two_d["id"], "date_time": DATE_TIME}
        response = client.post(self.route(domain_for_testing["id"]), json=body)
        assert response.status_code == 422

    # --- Terrain grid validation ---

    def test_nonexistent_terrain_grid_returns_404(
        self, client, domain_for_testing, source_lad_grid
    ):
        body = {
            "source_lad_grid_id": source_lad_grid["id"],
            "source_terrain_grid_id": "00000000000000000000000000000000",
            "bands": [SURFACE],
            "date_time": DATE_TIME,
        }
        response = client.post(self.route(domain_for_testing["id"]), json=body)
        assert response.status_code == 404

    def test_terrain_grid_not_completed_returns_422(
        self, client, domain_for_testing, source_lad_grid, grid_factory
    ):
        pending_terrain = grid_factory(
            domain_for_testing["id"],
            status="pending",
            bands=("elevation",),
            shape=(40, 40),
        )
        body = {
            "source_lad_grid_id": source_lad_grid["id"],
            "source_terrain_grid_id": pending_terrain["id"],
            "bands": [SURFACE],
            "date_time": DATE_TIME,
        }
        response = client.post(self.route(domain_for_testing["id"]), json=body)
        assert response.status_code == 422

    def test_terrain_grid_in_another_domain_returns_404(
        self,
        client,
        domain_for_testing,
        second_domain,
        source_lad_grid,
        grid_factory,
    ):
        other_domain_terrain = grid_factory(
            second_domain["id"], bands=("elevation",), shape=(40, 40)
        )
        body = {
            "source_lad_grid_id": source_lad_grid["id"],
            "source_terrain_grid_id": other_domain_terrain["id"],
            "bands": [SURFACE],
            "date_time": DATE_TIME,
        }
        response = client.post(self.route(domain_for_testing["id"]), json=body)
        assert response.status_code == 404

    def test_terrain_grid_missing_elevation_band_returns_422(
        self, client, domain_for_testing, source_lad_grid, grid_factory
    ):
        no_elev = grid_factory(
            domain_for_testing["id"], bands=("leaf_area_density",), shape=(40, 40)
        )
        body = {
            "source_lad_grid_id": source_lad_grid["id"],
            "source_terrain_grid_id": no_elev["id"],
            "bands": [SURFACE],
            "date_time": DATE_TIME,
        }
        response = client.post(self.route(domain_for_testing["id"]), json=body)
        assert response.status_code == 422

    def test_terrain_grid_not_2d_returns_422(
        self, client, domain_for_testing, source_lad_grid, grid_factory
    ):
        three_d_terrain = grid_factory(
            domain_for_testing["id"], bands=("elevation",), shape=(6, 40, 40)
        )
        body = {
            "source_lad_grid_id": source_lad_grid["id"],
            "source_terrain_grid_id": three_d_terrain["id"],
            "bands": [SURFACE],
            "date_time": DATE_TIME,
        }
        response = client.post(self.route(domain_for_testing["id"]), json=body)
        assert response.status_code == 422

    @pytest.mark.parametrize(
        ("terrain_overrides", "detail_fragment"),
        [
            ({"shape": (20, 20)}, "shape"),
            (
                {
                    "transform": (
                        2.0,
                        0.0,
                        500001.0,
                        0.0,
                        -2.0,
                        5201000.0,
                    )
                },
                "transform",
            ),
            ({"crs": "EPSG:4326"}, "CRS"),
        ],
    )
    def test_terrain_grid_not_aligned_with_lad_returns_422(
        self,
        client,
        domain_for_testing,
        source_lad_grid,
        grid_factory,
        terrain_overrides,
        detail_fragment,
    ):
        terrain = grid_factory(
            domain_for_testing["id"],
            bands=("elevation",),
            shape=terrain_overrides.get("shape", (40, 40)),
            crs=terrain_overrides.get("crs", "EPSG:32611"),
            transform=terrain_overrides.get("transform", TRANSFORM),
        )
        body = {
            "source_lad_grid_id": source_lad_grid["id"],
            "source_terrain_grid_id": terrain["id"],
            "bands": [SURFACE],
            "date_time": DATE_TIME,
        }
        response = client.post(self.route(domain_for_testing["id"]), json=body)
        assert response.status_code == 422
        assert detail_fragment in response.json()["detail"]

    # --- Request body validation ---

    def test_missing_source_lad_grid_id_returns_422(self, client, domain_for_testing):
        response = client.post(
            self.route(domain_for_testing["id"]), json={"date_time": DATE_TIME}
        )
        assert response.status_code == 422

    def test_missing_date_time_returns_422(
        self, client, domain_for_testing, source_lad_grid
    ):
        body = {"source_lad_grid_id": source_lad_grid["id"]}
        response = client.post(self.route(domain_for_testing["id"]), json=body)
        assert response.status_code == 422

    def test_empty_bands_returns_422(self, client, domain_for_testing, source_lad_grid):
        body = {
            "source_lad_grid_id": source_lad_grid["id"],
            "bands": [],
            "date_time": DATE_TIME,
        }
        response = client.post(self.route(domain_for_testing["id"]), json=body)
        assert response.status_code == 422

    def test_duplicate_bands_returns_422(
        self, client, domain_for_testing, source_lad_grid
    ):
        body = {
            "source_lad_grid_id": source_lad_grid["id"],
            "bands": [SURFACE, SURFACE],
            "date_time": DATE_TIME,
        }
        response = client.post(self.route(domain_for_testing["id"]), json=body)
        assert response.status_code == 422

    def test_invalid_band_returns_422(
        self, client, domain_for_testing, source_lad_grid
    ):
        body = {
            "source_lad_grid_id": source_lad_grid["id"],
            "bands": ["not_a_band"],
            "date_time": DATE_TIME,
        }
        response = client.post(self.route(domain_for_testing["id"]), json=body)
        assert response.status_code == 422

    @pytest.mark.parametrize("extinction", [0.0, -0.5])
    def test_non_positive_extinction_returns_422(
        self, client, domain_for_testing, source_lad_grid, extinction
    ):
        body = {
            "source_lad_grid_id": source_lad_grid["id"],
            "date_time": DATE_TIME,
            "extinction_coefficient": extinction,
        }
        response = client.post(self.route(domain_for_testing["id"]), json=body)
        assert response.status_code == 422

    # --- Documented examples ---

    @pytest.mark.parametrize(
        "example_name,example_value", ALL_LEAFLUX_IRRADIANCE_EXAMPLE_VALUES
    )
    def test_documented_example_creates_grid(
        self,
        client,
        domain_for_testing,
        source_lad_grid,
        terrain_grid,
        example_name,
        example_value,
    ):
        """Every documented example must produce a successful request."""
        body = {**example_value}
        if body.get("source_lad_grid_id") == "LAD_GRID_ID":
            body["source_lad_grid_id"] = source_lad_grid["id"]
        if body.get("source_terrain_grid_id") == "TERRAIN_GRID_ID":
            body["source_terrain_grid_id"] = terrain_grid["id"]

        response = client.post(self.route(domain_for_testing["id"]), json=body)
        assert response.status_code == 201, (
            f"Example '{example_name}' failed with status "
            f"{response.status_code}: {response.json()}"
        )
        source = response.json()["source"]
        assert source["operation"] == "irradiance"
        assert source["input"] == "grid"
        assert source["entity"] == "solar"
