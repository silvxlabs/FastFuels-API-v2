"""
Inventories inherit the attributions of the resources they are built from (#775).

Each test seeds parent documents carrying ``attributions`` directly in
Firestore, calls the live API, and checks the child's response and stored
document. Every created or seeded document is deleted on teardown.
"""

import uuid

import pytest

from lib.config import FEATURES_COLLECTION, GRIDS_COLLECTION, INVENTORIES_COLLECTION
from tests.fixtures import (
    make_feature_data,
    make_grid_data,
    make_inventory_data,
    make_osm_attribution,
)

META_ATTRIBUTION = {
    "dataset": "Meta CHM",
    "license_name": "CC-BY-4.0",
    "license_url": "https://creativecommons.org/licenses/by/4.0/",
    "citation": "Meta and World Resources Institute",
    "access_url": "https://registry.opendata.aws/dataforgood-fb-forests/",
    "accessed_on": "2026-10-01",
}
OSM_ATTRIBUTION = make_osm_attribution()


@pytest.fixture
def seed(firestore_client):
    """Write Firestore docs for one test and delete them afterwards. Returns
    ``(seed, track)``: ``track`` registers an API-created doc for cleanup."""
    created: list[tuple[str, str]] = []

    def _seed(collection: str, data: dict) -> dict:
        firestore_client.collection(collection).document(data["id"]).set(data)
        created.append((collection, data["id"]))
        return data

    def _track(collection: str, doc_id: str) -> None:
        created.append((collection, doc_id))

    yield _seed, _track
    for collection, doc_id in created:
        firestore_client.collection(collection).document(doc_id).delete()


def _stored(firestore_client, collection: str, doc_id: str) -> dict:
    return firestore_client.collection(collection).document(doc_id).get().to_dict()


def _osm_feature(domain_id: str) -> dict:
    feature = make_feature_data(domain_id=domain_id)
    feature["attributions"] = [OSM_ATTRIBUTION]
    return feature


def _feature_condition(feature_id: str) -> dict:
    return {"source": "feature", "operator": "intersects", "feature_id": feature_id}


def _chm_grid(domain_id: str, attributions: list[dict]) -> dict:
    grid = make_grid_data(
        domain_id=domain_id,
        status="completed",
        source={"name": "canopy", "product": "meta"},
        bands=[{"key": "chm", "type": "continuous", "unit": "m", "index": 0}],
    )
    grid["attributions"] = attributions
    return grid


def _pim_grid(domain_id: str, attributions: list[dict]) -> dict:
    grid = make_grid_data(
        domain_id=domain_id,
        status="completed",
        source={"name": "pim", "product": "treemap", "version": "2022"},
        bands=[{"key": "tm_id", "type": "categorical", "unit": None, "index": 0}],
    )
    grid["attributions"] = attributions
    return grid


def _completed_inventory(domain_id: str, attributions: list[dict]) -> dict:
    inventory = make_inventory_data(domain_id=domain_id, status="completed")
    inventory["checksum"] = uuid.uuid4().hex
    inventory["treatments"] = []
    inventory["attributions"] = attributions
    return inventory


class TestCreateInheritsAttributions:
    def test_chm_inherits_grid_and_condition_feature(
        self, client, firestore_client, domain_for_testing, seed
    ):
        _seed, track = seed
        domain_id = domain_for_testing["id"]
        grid = _seed(GRIDS_COLLECTION, _chm_grid(domain_id, [META_ATTRIBUTION]))
        feature = _seed(FEATURES_COLLECTION, _osm_feature(domain_id))

        response = client.post(
            f"/domains/{domain_id}/inventories/tree/chm",
            json={
                "source_chm_grid_id": grid["id"],
                "modifications": [
                    {
                        "conditions": [_feature_condition(feature["id"])],
                        "actions": [{"modifier": "remove"}],
                    }
                ],
            },
        )
        assert response.status_code == 201, response.text
        inventory_id = response.json()["id"]
        track(INVENTORIES_COLLECTION, inventory_id)

        expected = [META_ATTRIBUTION, OSM_ATTRIBUTION]
        assert response.json()["attributions"] == expected
        stored = _stored(firestore_client, INVENTORIES_COLLECTION, inventory_id)
        assert stored["attributions"] == expected

    def test_pim_inherits_grid(
        self, client, firestore_client, domain_for_testing, seed
    ):
        _seed, track = seed
        domain_id = domain_for_testing["id"]
        grid = _seed(GRIDS_COLLECTION, _pim_grid(domain_id, [OSM_ATTRIBUTION]))

        response = client.post(
            f"/domains/{domain_id}/inventories/tree/pim",
            json={"source_pim_grid_id": grid["id"], "seed": 42},
        )
        assert response.status_code == 201, response.text
        inventory_id = response.json()["id"]
        track(INVENTORIES_COLLECTION, inventory_id)

        assert response.json()["attributions"] == [OSM_ATTRIBUTION]
        stored = _stored(firestore_client, INVENTORIES_COLLECTION, inventory_id)
        assert stored["attributions"] == [OSM_ATTRIBUTION]

    def test_pim_chm_fusion_inherits_both_grids(
        self, client, firestore_client, domain_for_testing, seed
    ):
        _seed, track = seed
        domain_id = domain_for_testing["id"]
        pim = _seed(GRIDS_COLLECTION, _pim_grid(domain_id, [OSM_ATTRIBUTION]))
        chm = _chm_grid(domain_id, [META_ATTRIBUTION])
        # Finer than the default 7.5 m reimputation resolution.
        chm["georeference"] = {
            "crs": "EPSG:32611",
            "transform": (1.0, 0.0, 500000.0, 0.0, -1.0, 5201000.0),
            "shape": (1024, 1024),
        }
        _seed(GRIDS_COLLECTION, chm)

        response = client.post(
            f"/domains/{domain_id}/inventories/tree/pim/fusion/chm",
            json={"source_pim_grid_id": pim["id"], "source_chm_grid_id": chm["id"]},
        )
        assert response.status_code == 201, response.text
        inventory_id = response.json()["id"]
        track(INVENTORIES_COLLECTION, inventory_id)

        expected = [META_ATTRIBUTION, OSM_ATTRIBUTION]
        assert response.json()["attributions"] == expected
        stored = _stored(firestore_client, INVENTORIES_COLLECTION, inventory_id)
        assert stored["attributions"] == expected

    def test_crown_base_height_inherits_source_inventory(
        self, client, firestore_client, domain_for_testing, seed
    ):
        _seed, track = seed
        domain_id = domain_for_testing["id"]
        source = _seed(
            INVENTORIES_COLLECTION, _completed_inventory(domain_id, [OSM_ATTRIBUTION])
        )

        response = client.post(
            f"/domains/{domain_id}/inventories/tree/allometry/crown_base_height",
            json={"source_tree_inventory_id": source["id"]},
        )
        assert response.status_code == 201, response.text
        inventory_id = response.json()["id"]
        track(INVENTORIES_COLLECTION, inventory_id)

        assert response.json()["attributions"] == [OSM_ATTRIBUTION]
        stored = _stored(firestore_client, INVENTORIES_COLLECTION, inventory_id)
        assert stored["attributions"] == [OSM_ATTRIBUTION]

    def test_duplicate_copies_attributions(
        self, client, firestore_client, domain_for_testing, seed
    ):
        _seed, track = seed
        domain_id = domain_for_testing["id"]
        source = _seed(
            INVENTORIES_COLLECTION, _completed_inventory(domain_id, [OSM_ATTRIBUTION])
        )

        response = client.post(
            f"/domains/{domain_id}/inventories/{source['id']}/duplicate"
        )
        assert response.status_code == 201, response.text
        inventory_id = response.json()["id"]
        track(INVENTORIES_COLLECTION, inventory_id)

        assert response.json()["attributions"] == [OSM_ATTRIBUTION]
        stored = _stored(firestore_client, INVENTORIES_COLLECTION, inventory_id)
        assert stored["attributions"] == [OSM_ATTRIBUTION]


class TestInPlaceMergesAttributions:
    """In-place modifications and treatments add the referenced features'
    attributions to the ones the inventory already carries."""

    @pytest.mark.parametrize(
        ("path", "body_key", "item"),
        [
            ("modifications", "modifications", {"actions": [{"modifier": "remove"}]}),
            (
                "treatments",
                "treatments",
                {"metric": "diameter", "method": "from_below", "value": 5.0},
            ),
        ],
    )
    def test_merges_feature_attributions(
        self, client, firestore_client, domain_for_testing, seed, path, body_key, item
    ):
        _seed, _ = seed
        domain_id = domain_for_testing["id"]
        inventory = _seed(
            INVENTORIES_COLLECTION, _completed_inventory(domain_id, [META_ATTRIBUTION])
        )
        feature = _seed(FEATURES_COLLECTION, _osm_feature(domain_id))

        response = client.post(
            f"/domains/{domain_id}/inventories/{inventory['id']}/{path}",
            json={
                body_key: [{**item, "conditions": [_feature_condition(feature["id"])]}]
            },
        )
        assert response.status_code == 200, response.text

        expected = [META_ATTRIBUTION, OSM_ATTRIBUTION]
        assert response.json()["attributions"] == expected
        stored = _stored(firestore_client, INVENTORIES_COLLECTION, inventory["id"])
        assert stored["attributions"] == expected
