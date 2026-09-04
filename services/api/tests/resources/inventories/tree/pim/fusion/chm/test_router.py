"""
Integration tests for api/v2/resources/inventories/tree/pim/fusion/chm/router.py

Tests the PIM-CHM fusion inventory creation endpoint. These tests make real
HTTP requests to the API and interact with Firestore and Cloud Tasks, so they
require a running API server (see the testing guide).
"""

import pytest
from api.resources.inventories.tree.pim.fusion.chm.examples import (
    ALL_PIM_CHM_FUSION_EXAMPLE_VALUES,
)

from lib.config import DOMAINS_COLLECTION, GRIDS_COLLECTION
from tests.fixtures import make_domain_data, make_grid_data

# A CHM georeference finer than the default 7.5 m reimputation resolution, so
# the cell-size ordering (chm cell <= resolution <= pim cell) holds by default.
CHM_GEOREFERENCE_1M = {
    "crs": "EPSG:32611",
    "transform": (1.0, 0.0, 500000.0, 0.0, -1.0, 5201000.0),
    "shape": (1024, 1024),
}

# --- Fixtures ---


@pytest.fixture(scope="session")
def pim_grid_for_fusion(firestore_client, domain_for_testing):
    """A completed TreeMap PIM grid (30 m) with the tm_id plot-id band."""
    grid_data = make_grid_data(
        domain_id=domain_for_testing["id"],
        name="PIM Grid for Fusion Tests",
        status="completed",
        source={"name": "pim", "product": "treemap", "description": "TreeMap PIM"},
        bands=[{"key": "tm_id", "type": "categorical", "unit": None, "index": 0}],
    )
    doc_ref = firestore_client.collection(GRIDS_COLLECTION).document(grid_data["id"])
    doc_ref.set(grid_data)
    yield grid_data
    doc_ref.delete()


@pytest.fixture(scope="session")
def chm_grid_for_fusion(firestore_client, domain_for_testing):
    """A completed CHM grid (1 m) with a 'chm' band in meters."""
    grid_data = make_grid_data(
        domain_id=domain_for_testing["id"],
        name="CHM Grid for Fusion Tests",
        status="completed",
        source={"name": "canopy", "product": "naip", "description": "NAIP CHM"},
        bands=[{"key": "chm", "type": "continuous", "unit": "m", "index": 0}],
        georeference=CHM_GEOREFERENCE_1M,
    )
    doc_ref = firestore_client.collection(GRIDS_COLLECTION).document(grid_data["id"])
    doc_ref.set(grid_data)
    yield grid_data
    doc_ref.delete()


@pytest.fixture(scope="session")
def second_domain_for_fusion(firestore_client):
    """A second domain owned by test-owner, used for cross-domain tests."""
    domain_data = make_domain_data(name="Second Domain for Fusion Tests")
    doc_ref = firestore_client.collection(DOMAINS_COLLECTION).document(
        domain_data["id"]
    )
    doc_ref.set(domain_data)
    yield domain_data
    doc_ref.delete()


@pytest.fixture(scope="session")
def pim_grid_in_different_domain(firestore_client, second_domain_for_fusion):
    """A completed PIM grid in a different domain than domain_for_testing."""
    grid_data = make_grid_data(
        domain_id=second_domain_for_fusion["id"],
        name="PIM grid in second domain",
        status="completed",
        source={"name": "pim", "product": "treemap"},
        bands=[{"key": "tm_id", "type": "categorical", "unit": None, "index": 0}],
    )
    doc_ref = firestore_client.collection(GRIDS_COLLECTION).document(grid_data["id"])
    doc_ref.set(grid_data)
    yield grid_data
    doc_ref.delete()


@pytest.fixture(scope="session")
def chm_grid_in_different_domain(firestore_client, second_domain_for_fusion):
    """A completed CHM grid in a different domain than domain_for_testing."""
    grid_data = make_grid_data(
        domain_id=second_domain_for_fusion["id"],
        name="CHM grid in second domain",
        status="completed",
        source={"name": "canopy", "product": "meta"},
        bands=[{"key": "chm", "type": "continuous", "unit": "m", "index": 0}],
        georeference=CHM_GEOREFERENCE_1M,
    )
    doc_ref = firestore_client.collection(GRIDS_COLLECTION).document(grid_data["id"])
    doc_ref.set(grid_data)
    yield grid_data
    doc_ref.delete()


# --- Tests ---


class TestCreatePimChmFusionInventory:
    """Test the POST /domains/{domain_id}/inventories/tree/pim/fusion/chm endpoint."""

    def route(self, domain_id):
        return f"/domains/{domain_id}/inventories/tree/pim/fusion/chm"

    def base_body(self, pim_grid, chm_grid):
        return {
            "source_pim_grid_id": pim_grid["id"],
            "source_chm_grid_id": chm_grid["id"],
        }

    def test_minimal_request_creates_inventory(
        self, client, domain_for_testing, pim_grid_for_fusion, chm_grid_for_fusion
    ):
        """Minimal request creates an inventory with the default reimputation method."""
        response = client.post(
            self.route(domain_for_testing["id"]),
            json=self.base_body(pim_grid_for_fusion, chm_grid_for_fusion),
        )

        assert response.status_code == 201

        data = response.json()
        assert len(data["id"]) == 32
        assert data["domain_id"] == domain_for_testing["id"]
        assert data["status"] == "pending"
        assert data["type"] == "tree"

        source = data["source"]
        assert source["name"] == "pim"
        assert source["fusion"] == ["chm"]
        assert source["source_pim_grid_id"] == pim_grid_for_fusion["id"]
        assert source["source_chm_grid_id"] == chm_grid_for_fusion["id"]
        assert source["source_pim_grid_checksum"] == pim_grid_for_fusion.get("checksum")
        assert source["source_chm_grid_checksum"] == chm_grid_for_fusion.get("checksum")
        assert source["method"]["name"] == "reimputation"
        assert source["method"]["resolution"] == 7.5
        assert source["method"]["min_height"] == 2.0
        assert source["method"]["cover_threshold"] == 0.2

        # Reimputation is a conditioned tree/pim expansion: full PIM column set.
        assert [c["key"] for c in data["columns"]] == [
            "x",
            "y",
            "fia_species_code",
            "fia_status_code",
            "dbh",
            "height",
            "crown_ratio",
        ]

    def test_request_with_method_and_metadata(
        self, client, domain_for_testing, pim_grid_for_fusion, chm_grid_for_fusion
    ):
        """Custom reimputation knobs, name, description, and tags are stored."""
        body = {
            **self.base_body(pim_grid_for_fusion, chm_grid_for_fusion),
            "method": {
                "name": "reimputation",
                "resolution": 10.0,
                "min_height": 1.0,
                "cover_threshold": 0.25,
            },
            "name": "Custom Fusion Inventory",
            "description": "Testing custom reimputation knobs",
            "tags": ["fusion", "test"],
        }

        response = client.post(self.route(domain_for_testing["id"]), json=body)

        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "Custom Fusion Inventory"
        assert data["tags"] == ["fusion", "test"]
        assert data["source"]["method"]["resolution"] == 10.0
        assert data["source"]["method"]["min_height"] == 1.0
        assert data["source"]["method"]["cover_threshold"] == 0.25

    def test_georeference_is_null_on_creation(
        self, client, domain_for_testing, pim_grid_for_fusion, chm_grid_for_fusion
    ):
        """Georeference is null until the backend populates it."""
        response = client.post(
            self.route(domain_for_testing["id"]),
            json=self.base_body(pim_grid_for_fusion, chm_grid_for_fusion),
        )
        assert response.status_code == 201
        assert response.json()["georeference"] is None

    def test_response_excludes_owner_id(
        self, client, domain_for_testing, pim_grid_for_fusion, chm_grid_for_fusion
    ):
        """Response should not expose the owner_id field."""
        response = client.post(
            self.route(domain_for_testing["id"]),
            json=self.base_body(pim_grid_for_fusion, chm_grid_for_fusion),
        )
        assert response.status_code == 201
        assert "owner_id" not in response.json()

    @pytest.mark.parametrize(
        "example_name,example_value", ALL_PIM_CHM_FUSION_EXAMPLE_VALUES
    )
    def test_documented_example_creates_inventory(
        self,
        client,
        domain_for_testing,
        pim_grid_for_fusion,
        chm_grid_for_fusion,
        example_name,
        example_value,
    ):
        """Each documented example should successfully create an inventory."""
        body = {**example_value}
        if body.get("source_pim_grid_id") == "PLACEHOLDER_PIM_GRID_ID":
            body["source_pim_grid_id"] = pim_grid_for_fusion["id"]
        if body.get("source_chm_grid_id") == "PLACEHOLDER_CHM_GRID_ID":
            body["source_chm_grid_id"] = chm_grid_for_fusion["id"]

        response = client.post(self.route(domain_for_testing["id"]), json=body)

        assert response.status_code == 201, (
            f"Example '{example_name}' failed with status {response.status_code}: "
            f"{response.json()}"
        )
        data = response.json()
        assert data["source"]["name"] == "pim"
        assert data["source"]["fusion"] == ["chm"]

    # --- Domain / ownership validation ---

    def test_invalid_domain_returns_404(
        self, client, pim_grid_for_fusion, chm_grid_for_fusion
    ):
        """Non-existent domain_id returns 404."""
        response = client.post(
            self.route("00000000000000000000000000000000"),
            json=self.base_body(pim_grid_for_fusion, chm_grid_for_fusion),
        )
        assert response.status_code == 404

    def test_wrong_owner_domain_returns_404(
        self,
        client,
        domain_with_different_owner,
        pim_grid_for_fusion,
        chm_grid_for_fusion,
    ):
        """Domain owned by another user returns 404."""
        response = client.post(
            self.route(domain_with_different_owner["id"]),
            json=self.base_body(pim_grid_for_fusion, chm_grid_for_fusion),
        )
        assert response.status_code == 404

    # --- Source grid validation ---

    def test_nonexistent_pim_grid_returns_404(
        self, client, domain_for_testing, chm_grid_for_fusion
    ):
        """A non-existent PIM grid returns 404."""
        response = client.post(
            self.route(domain_for_testing["id"]),
            json={
                "source_pim_grid_id": "00000000000000000000000000000000",
                "source_chm_grid_id": chm_grid_for_fusion["id"],
            },
        )
        assert response.status_code == 404

    def test_nonexistent_chm_grid_returns_404(
        self, client, domain_for_testing, pim_grid_for_fusion
    ):
        """A non-existent CHM grid returns 404."""
        response = client.post(
            self.route(domain_for_testing["id"]),
            json={
                "source_pim_grid_id": pim_grid_for_fusion["id"],
                "source_chm_grid_id": "00000000000000000000000000000000",
            },
        )
        assert response.status_code == 404

    def test_pim_grid_in_different_domain_returns_404(
        self,
        client,
        domain_for_testing,
        pim_grid_in_different_domain,
        chm_grid_for_fusion,
    ):
        """A PIM grid in a different domain returns 404."""
        response = client.post(
            self.route(domain_for_testing["id"]),
            json={
                "source_pim_grid_id": pim_grid_in_different_domain["id"],
                "source_chm_grid_id": chm_grid_for_fusion["id"],
            },
        )
        assert response.status_code == 404

    def test_chm_grid_in_different_domain_returns_404(
        self,
        client,
        domain_for_testing,
        pim_grid_for_fusion,
        chm_grid_in_different_domain,
    ):
        """A CHM grid in a different domain returns 404."""
        response = client.post(
            self.route(domain_for_testing["id"]),
            json={
                "source_pim_grid_id": pim_grid_for_fusion["id"],
                "source_chm_grid_id": chm_grid_in_different_domain["id"],
            },
        )
        assert response.status_code == 404

    def test_pim_grid_not_completed_returns_422(
        self, client, firestore_client, domain_for_testing, chm_grid_for_fusion
    ):
        """A PIM grid that is still pending returns 422."""
        grid_data = make_grid_data(
            domain_id=domain_for_testing["id"],
            name="Pending PIM Grid",
            status="pending",
            source={"name": "pim", "product": "treemap"},
            bands=[{"key": "tm_id", "type": "categorical", "unit": None, "index": 0}],
            georeference=None,
        )
        doc_ref = firestore_client.collection(GRIDS_COLLECTION).document(
            grid_data["id"]
        )
        doc_ref.set(grid_data)
        try:
            response = client.post(
                self.route(domain_for_testing["id"]),
                json={
                    "source_pim_grid_id": grid_data["id"],
                    "source_chm_grid_id": chm_grid_for_fusion["id"],
                },
            )
            assert response.status_code == 422
        finally:
            doc_ref.delete()

    def test_pim_grid_missing_plot_id_band_returns_422(
        self, client, firestore_client, domain_for_testing, chm_grid_for_fusion
    ):
        """A PIM grid lacking the tm_id plot-id band returns 422."""
        grid_data = make_grid_data(
            domain_id=domain_for_testing["id"],
            name="PIM grid without tm_id",
            status="completed",
            source={"name": "pim", "product": "treemap"},
            bands=[{"key": "plt_cn", "type": "categorical", "unit": None, "index": 0}],
        )
        doc_ref = firestore_client.collection(GRIDS_COLLECTION).document(
            grid_data["id"]
        )
        doc_ref.set(grid_data)
        try:
            response = client.post(
                self.route(domain_for_testing["id"]),
                json={
                    "source_pim_grid_id": grid_data["id"],
                    "source_chm_grid_id": chm_grid_for_fusion["id"],
                },
            )
            assert response.status_code == 422
            assert "missing required bands" in response.json()["detail"].lower()
        finally:
            doc_ref.delete()

    def test_composed_grid_with_plot_id_band_is_accepted(
        self, client, firestore_client, domain_for_testing, chm_grid_for_fusion
    ):
        """A grid carrying tm_id but not made by the PIM pipeline is accepted —
        the endpoint gates on the band contract, not on provenance."""
        grid_data = make_grid_data(
            domain_id=domain_for_testing["id"],
            name="Composed grid with tm_id",
            status="completed",
            source={"name": "compose"},
            bands=[{"key": "tm_id", "type": "categorical", "unit": None, "index": 0}],
        )
        doc_ref = firestore_client.collection(GRIDS_COLLECTION).document(
            grid_data["id"]
        )
        doc_ref.set(grid_data)
        try:
            response = client.post(
                self.route(domain_for_testing["id"]),
                json={
                    "source_pim_grid_id": grid_data["id"],
                    "source_chm_grid_id": chm_grid_for_fusion["id"],
                },
            )
            assert response.status_code == 201
            assert response.json()["source"]["name"] == "pim"
        finally:
            doc_ref.delete()

    def test_chm_grid_without_chm_band_returns_422(
        self, client, firestore_client, domain_for_testing, pim_grid_for_fusion
    ):
        """A CHM source grid that lacks a 'chm' band returns 422."""
        grid_data = make_grid_data(
            domain_id=domain_for_testing["id"],
            name="Grid without chm band",
            status="completed",
            source={"name": "canopy", "product": "naip"},
            bands=[{"key": "cover", "type": "continuous", "unit": "%", "index": 0}],
            georeference=CHM_GEOREFERENCE_1M,
        )
        doc_ref = firestore_client.collection(GRIDS_COLLECTION).document(
            grid_data["id"]
        )
        doc_ref.set(grid_data)
        try:
            response = client.post(
                self.route(domain_for_testing["id"]),
                json={
                    "source_pim_grid_id": pim_grid_for_fusion["id"],
                    "source_chm_grid_id": grid_data["id"],
                },
            )
            assert response.status_code == 422
            assert "missing required bands" in response.json()["detail"].lower()
        finally:
            doc_ref.delete()

    def test_chm_grid_non_meter_unit_returns_422(
        self, client, firestore_client, domain_for_testing, pim_grid_for_fusion
    ):
        """A 'chm' band in a non-meter unit is rejected."""
        grid_data = make_grid_data(
            domain_id=domain_for_testing["id"],
            name="CHM in feet",
            status="completed",
            source={"name": "upload", "format": "geotiff"},
            bands=[{"key": "chm", "type": "continuous", "unit": "ft", "index": 0}],
            georeference=CHM_GEOREFERENCE_1M,
        )
        doc_ref = firestore_client.collection(GRIDS_COLLECTION).document(
            grid_data["id"]
        )
        doc_ref.set(grid_data)
        try:
            response = client.post(
                self.route(domain_for_testing["id"]),
                json={
                    "source_pim_grid_id": pim_grid_for_fusion["id"],
                    "source_chm_grid_id": grid_data["id"],
                },
            )
            assert response.status_code == 422
            detail = response.json()["detail"].lower()
            assert "'ft'" in detail and "'m'" in detail
        finally:
            doc_ref.delete()

    def test_three_dimensional_chm_grid_returns_422(
        self, client, firestore_client, domain_for_testing, pim_grid_for_fusion
    ):
        """A volumetric (3-D) CHM source grid is rejected — a CHM is a surface."""
        grid_data = make_grid_data(
            domain_id=domain_for_testing["id"],
            name="3D grid with chm band",
            status="completed",
            source={"name": "upload", "format": "zarr"},
            bands=[{"key": "chm", "type": "continuous", "unit": "m", "index": 0}],
            georeference={
                "crs": "EPSG:32611",
                "transform": (1.0, 0.0, 500000.0, 0.0, -1.0, 5201000.0),
                "shape": (10, 1024, 1024),
            },
        )
        doc_ref = firestore_client.collection(GRIDS_COLLECTION).document(
            grid_data["id"]
        )
        doc_ref.set(grid_data)
        try:
            response = client.post(
                self.route(domain_for_testing["id"]),
                json={
                    "source_pim_grid_id": pim_grid_for_fusion["id"],
                    "source_chm_grid_id": grid_data["id"],
                },
            )
            assert response.status_code == 422
            assert "3d" in response.json()["detail"].lower()
        finally:
            doc_ref.delete()

    # --- Reimputation cell-size ordering ---

    def test_resolution_finer_than_chm_cell_returns_422(
        self, client, domain_for_testing, pim_grid_for_fusion, chm_grid_for_fusion
    ):
        """A resolution finer than the 1 m CHM cell is rejected."""
        response = client.post(
            self.route(domain_for_testing["id"]),
            json={
                "source_pim_grid_id": pim_grid_for_fusion["id"],
                "source_chm_grid_id": chm_grid_for_fusion["id"],
                "method": {"name": "reimputation", "resolution": 0.5},
            },
        )
        assert response.status_code == 422
        assert "finer than the chm cell" in response.json()["detail"].lower()

    def test_resolution_coarser_than_pim_cell_returns_422(
        self, client, domain_for_testing, pim_grid_for_fusion, chm_grid_for_fusion
    ):
        """A resolution coarser than the 30 m PIM cell is rejected."""
        response = client.post(
            self.route(domain_for_testing["id"]),
            json={
                "source_pim_grid_id": pim_grid_for_fusion["id"],
                "source_chm_grid_id": chm_grid_for_fusion["id"],
                "method": {"name": "reimputation", "resolution": 40.0},
            },
        )
        assert response.status_code == 422
        assert "coarser than the pim cell" in response.json()["detail"].lower()

    # --- Method discriminator ---

    def test_unknown_method_name_returns_422(
        self, client, domain_for_testing, pim_grid_for_fusion, chm_grid_for_fusion
    ):
        """An unsupported method name triggers a Pydantic discriminator 422."""
        response = client.post(
            self.route(domain_for_testing["id"]),
            json={
                "source_pim_grid_id": pim_grid_for_fusion["id"],
                "source_chm_grid_id": chm_grid_for_fusion["id"],
                "method": {"name": "surface_matching"},
            },
        )
        assert response.status_code == 422
        detail = response.json()["detail"]
        assert any(
            "discriminator" in str(error).lower() or "tag" in str(error).lower()
            for error in detail
        )
