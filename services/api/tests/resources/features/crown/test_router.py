"""
Integration tests for api/v2/resources/features/crown/router.py

Tests the POST /domains/{domain_id}/features/crown/chm endpoint.
These tests make real HTTP requests to the API and interact with Firestore.
"""

import uuid

import pytest
from api.resources.features.crown.examples import CROWN_EXAMPLE_VALUES

from lib.config import GRIDS_COLLECTION, INVENTORIES_COLLECTION
from tests.fixtures import make_grid_data, make_inventory_data

TREE_COLUMNS = [
    {"key": "tree_id", "type": "categorical", "unit": None},
    {"key": "x", "type": "continuous", "unit": "m"},
    {"key": "y", "type": "continuous", "unit": "m"},
    {"key": "height", "type": "continuous", "unit": "m"},
]
CHM_BAND = {"key": "chm", "type": "continuous", "unit": "m", "index": 0}


def georeference(cell: float) -> dict:
    return {
        "crs": "EPSG:32611",
        "transform": (cell, 0.0, 500000.0, 0.0, -cell, 5201000.0),
        "shape": (100, 100),
    }


@pytest.fixture(scope="module")
def seed(firestore_client):
    """Write Firestore docs for the module and delete them afterwards."""
    created = []

    def _seed(collection: str, data: dict) -> dict:
        firestore_client.collection(collection).document(data["id"]).set(data)
        created.append((collection, data["id"]))
        return data

    yield _seed
    for collection, doc_id in created:
        firestore_client.collection(collection).document(doc_id).delete()


@pytest.fixture(scope="module")
def domain_id(domain_for_testing):
    return domain_for_testing["id"]


def inventory(domain_id, **kwargs) -> dict:
    data = make_inventory_data(
        domain_id=domain_id,
        status=kwargs.pop("status", "completed"),
        columns=kwargs.pop("columns", TREE_COLUMNS),
        **kwargs,
    )
    data["checksum"] = uuid.uuid4().hex
    return data


def chm_grid(domain_id, cell=1.0, bands=None, **kwargs) -> dict:
    data = make_grid_data(
        domain_id=domain_id,
        status=kwargs.pop("status", "completed"),
        source={"name": "canopy", "product": "meta", "description": "Meta CHM"},
        bands=bands or [CHM_BAND],
        georeference=georeference(cell),
        **kwargs,
    )
    data["checksum"] = uuid.uuid4().hex
    return data


@pytest.fixture(scope="module")
def tree_inventory(seed, domain_id):
    return seed(INVENTORIES_COLLECTION, inventory(domain_id))


@pytest.fixture(scope="module")
def chm(seed, domain_id):
    return seed(GRIDS_COLLECTION, chm_grid(domain_id))


class TestCreateChmCrownFeature:
    def route(self, domain_id):
        return f"/domains/{domain_id}/features/crown/chm"

    def body(self, inventory_doc, grid_doc, **extra):
        return {
            "source_inventory_id": inventory_doc["id"],
            "source_chm_grid_id": grid_doc["id"],
            **extra,
        }

    def test_minimal_request_creates_feature(
        self, client, domain_id, tree_inventory, chm
    ):
        response = client.post(
            self.route(domain_id), json=self.body(tree_inventory, chm)
        )
        assert response.status_code == 201, response.json()

        data = response.json()
        assert len(data["id"]) == 32
        assert data["domain_id"] == domain_id
        assert data["type"] == "crown"
        assert data["status"] == "pending"
        assert data["georeference"] is None
        assert "owner_id" not in data
        assert data["source"] == {
            "product": "chm",
            "source_inventory_id": tree_inventory["id"],
            "source_inventory_checksum": tree_inventory["checksum"],
            "source_chm_grid_id": chm["id"],
            "source_chm_grid_checksum": chm["checksum"],
            "crown_segmentation": {
                "method": "dalponte2016",
                "min_relative_height": 0.45,
                "min_relative_crown_height": 0.55,
                "max_crown_radius": 10.0,
                "min_height": 2.0,
                "max_height": 120.0,
            },
        }

        fetched = client.get(f"/domains/{domain_id}/features/{data['id']}")
        assert fetched.status_code == 200
        assert fetched.json()["source"] == data["source"]

    def test_partial_settings_resolve_defaults(
        self, client, domain_id, tree_inventory, chm
    ):
        body = self.body(
            tree_inventory,
            chm,
            crown_segmentation={"max_crown_radius": 6.5, "max_height": None},
        )
        response = client.post(self.route(domain_id), json=body)
        assert response.status_code == 201, response.json()
        settings = response.json()["source"]["crown_segmentation"]
        assert settings["max_crown_radius"] == 6.5
        assert settings["max_height"] is None
        assert settings["min_height"] == 2.0

    @pytest.mark.parametrize("example_name,example_value", CROWN_EXAMPLE_VALUES)
    def test_documented_example_creates_feature(
        self, client, domain_id, tree_inventory, chm, example_name, example_value
    ):
        body = {
            **example_value,
            "source_inventory_id": tree_inventory["id"],
            "source_chm_grid_id": chm["id"],
        }
        response = client.post(self.route(domain_id), json=body)
        assert response.status_code == 201, (example_name, response.json())
        assert response.json()["type"] == "crown"

    @pytest.mark.parametrize(
        "settings",
        [
            {"min_height": -1},
            {"min_height": 10, "max_height": 5},
            {"min_relative_height": 1.0},
            {"min_relative_crown_height": -0.1},
            {"max_crown_radius": 0},
            {"max_crown_radius": 50},
            {"method": "watershed"},
        ],
    )
    def test_out_of_range_settings_rejected(
        self, client, domain_id, tree_inventory, chm, settings
    ):
        body = self.body(tree_inventory, chm, crown_segmentation=settings)
        assert client.post(self.route(domain_id), json=body).status_code == 422

    def test_pending_inventory_rejected(self, client, seed, domain_id, chm):
        doc = seed(INVENTORIES_COLLECTION, inventory(domain_id, status="pending"))
        response = client.post(self.route(domain_id), json=self.body(doc, chm))
        assert response.status_code == 422

    def test_non_tree_inventory_rejected(self, client, seed, domain_id, chm):
        doc = seed(INVENTORIES_COLLECTION, inventory(domain_id, inventory_type="shrub"))
        response = client.post(self.route(domain_id), json=self.body(doc, chm))
        assert response.status_code == 422
        assert "tree inventory" in response.json()["detail"]

    def test_inventory_without_tree_id_rejected(self, client, seed, domain_id, chm):
        doc = seed(
            INVENTORIES_COLLECTION, inventory(domain_id, columns=TREE_COLUMNS[1:])
        )
        response = client.post(self.route(domain_id), json=self.body(doc, chm))
        assert response.status_code == 422
        assert "tree_id" in response.json()["detail"]

    def test_pending_grid_rejected(self, client, seed, domain_id, tree_inventory):
        doc = seed(GRIDS_COLLECTION, chm_grid(domain_id, status="pending"))
        response = client.post(
            self.route(domain_id), json=self.body(tree_inventory, doc)
        )
        assert response.status_code == 422

    def test_grid_without_chm_band_rejected(
        self, client, seed, domain_id, tree_inventory
    ):
        band = {"key": "fbfm", "type": "categorical", "unit": None, "index": 0}
        doc = seed(GRIDS_COLLECTION, chm_grid(domain_id, bands=[band]))
        response = client.post(
            self.route(domain_id), json=self.body(tree_inventory, doc)
        )
        assert response.status_code == 422
        assert "chm" in response.json()["detail"]

    def test_chm_not_in_metres_rejected(self, client, seed, domain_id, tree_inventory):
        doc = seed(
            GRIDS_COLLECTION, chm_grid(domain_id, bands=[{**CHM_BAND, "unit": "ft"}])
        )
        response = client.post(
            self.route(domain_id), json=self.body(tree_inventory, doc)
        )
        assert response.status_code == 422
        assert "'m'" in response.json()["detail"]

    def test_coarse_chm_rejected(self, client, seed, domain_id, tree_inventory):
        doc = seed(GRIDS_COLLECTION, chm_grid(domain_id, cell=3.0))
        response = client.post(
            self.route(domain_id), json=self.body(tree_inventory, doc)
        )
        assert response.status_code == 422
        assert "2 m or finer" in response.json()["detail"]

    def test_two_metre_chm_accepted(self, client, seed, domain_id, tree_inventory):
        doc = seed(GRIDS_COLLECTION, chm_grid(domain_id, cell=2.0))
        response = client.post(
            self.route(domain_id), json=self.body(tree_inventory, doc)
        )
        assert response.status_code == 201, response.json()

    def test_missing_sources_return_404(self, client, domain_id, tree_inventory, chm):
        missing = "0" * 32
        for body in (
            self.body({"id": missing}, chm),
            self.body(tree_inventory, {"id": missing}),
        ):
            assert client.post(self.route(domain_id), json=body).status_code == 404

    def test_wrong_owner_domain_returns_404(
        self, client, domain_with_different_owner, tree_inventory, chm
    ):
        response = client.post(
            self.route(domain_with_different_owner["id"]),
            json=self.body(tree_inventory, chm),
        )
        assert response.status_code == 404
