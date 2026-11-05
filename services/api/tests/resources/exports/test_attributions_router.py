"""
Exports inherit the attributions of the grids or inventory they read (#775).

Each test seeds parent documents carrying ``attributions`` directly in
Firestore, calls the live API, and checks the export's response, stored
document, and GET. Every created or seeded document is deleted on teardown.
"""

import pytest

from lib.config import EXPORTS_COLLECTION, GRIDS_COLLECTION, INVENTORIES_COLLECTION
from tests.fixtures import make_grid_data, make_inventory_data, make_osm_attribution
from tests.resources.grids.exports.landscape.test_router import (
    _CANOPY_BANDS,
    _FBFM_BANDS,
    _TOPO_BANDS,
    GEOREF_30M,
)
from tests.resources.grids.exports.landscape.test_router import (
    _minimal_body as landscape_body,
)
from tests.resources.grids.exports.quicfire.test_router import (
    GEOREF_2D,
    GEOREF_3D,
)
from tests.resources.grids.exports.quicfire.test_router import (
    _minimal_body as quicfire_body,
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


def _grid(domain_id: str, bands: list[dict], georeference: dict, attributions):
    grid = make_grid_data(
        domain_id=domain_id,
        status="completed",
        bands=bands,
        georeference=georeference,
    )
    grid["attributions"] = attributions
    return grid


def _band(key: str, unit: str | None, index: int) -> dict:
    return {"key": key, "type": "continuous", "unit": unit, "index": index}


def _assert_export_attributions(
    client, firestore_client, response, track, expected: list[dict]
) -> None:
    assert response.status_code == 201, response.text
    export_id = response.json()["id"]
    track(EXPORTS_COLLECTION, export_id)

    assert response.json()["attributions"] == expected
    stored = (
        firestore_client.collection(EXPORTS_COLLECTION).document(export_id).get()
    ).to_dict()
    assert stored["attributions"] == expected
    get = client.get(f"/exports/{export_id}")
    assert get.status_code == 200, get.text
    assert get.json()["attributions"] == expected


class TestExportsInheritAttributions:
    def test_grid_export(self, client, firestore_client, domain_for_testing, seed):
        _seed, track = seed
        domain_id = domain_for_testing["id"]
        grid = _seed(
            GRIDS_COLLECTION,
            _grid(
                domain_id,
                [_band("fuel_load.1hr", "kg/m**2", 0)],
                GEOREF_2D,
                [OSM_ATTRIBUTION],
            ),
        )

        response = client.post(
            f"/domains/{domain_id}/grids/{grid['id']}/exports/netcdf", json={}
        )
        _assert_export_attributions(
            client, firestore_client, response, track, [OSM_ATTRIBUTION]
        )

    def test_grid_export_without_attributions(
        self, client, firestore_client, domain_for_testing, seed
    ):
        _seed, track = seed
        domain_id = domain_for_testing["id"]
        grid = make_grid_data(domain_id=domain_id, status="completed")
        grid.pop("attributions", None)
        _seed(GRIDS_COLLECTION, grid)

        response = client.post(
            f"/domains/{domain_id}/grids/{grid['id']}/exports/geotiff", json={}
        )
        _assert_export_attributions(client, firestore_client, response, track, [])

    def test_inventory_export(self, client, firestore_client, domain_for_testing, seed):
        _seed, track = seed
        domain_id = domain_for_testing["id"]
        inventory = make_inventory_data(domain_id=domain_id, status="completed")
        inventory["attributions"] = [META_ATTRIBUTION, OSM_ATTRIBUTION]
        _seed(INVENTORIES_COLLECTION, inventory)

        response = client.post(
            f"/domains/{domain_id}/inventories/{inventory['id']}/exports/csv",
            json={},
        )
        _assert_export_attributions(
            client,
            firestore_client,
            response,
            track,
            [META_ATTRIBUTION, OSM_ATTRIBUTION],
        )

    def test_quicfire_export_merges_role_grids(
        self, client, firestore_client, domain_for_testing, seed
    ):
        _seed, track = seed
        domain_id = domain_for_testing["id"]
        canopy = _seed(
            GRIDS_COLLECTION,
            _grid(
                domain_id,
                [
                    _band("bulk_density.foliage.live", "kg/m**3", 0),
                    _band("fuel_moisture.live", "%", 1),
                ],
                GEOREF_3D,
                [META_ATTRIBUTION],
            ),
        )
        surface = _seed(
            GRIDS_COLLECTION,
            _grid(
                domain_id,
                [_band("fuel_load.1hr", "kg/m**2", 0), _band("fuel_depth", "m", 1)],
                GEOREF_2D,
                [OSM_ATTRIBUTION],
            ),
        )
        moisture = _seed(
            GRIDS_COLLECTION,
            _grid(domain_id, [_band("fuel_moisture.1hr", "%", 0)], GEOREF_2D, []),
        )

        response = client.post(
            f"/domains/{domain_id}/grids/exports/quicfire",
            json=quicfire_body(canopy["id"], surface["id"], moisture["id"]),
        )
        _assert_export_attributions(
            client,
            firestore_client,
            response,
            track,
            [META_ATTRIBUTION, OSM_ATTRIBUTION],
        )

    def test_landscape_export_merges_role_grids(
        self, client, firestore_client, domain_for_testing, seed
    ):
        _seed, track = seed
        domain_id = domain_for_testing["id"]
        topo = _seed(GRIDS_COLLECTION, _grid(domain_id, _TOPO_BANDS, GEOREF_30M, []))
        fbfm = _seed(
            GRIDS_COLLECTION,
            _grid(domain_id, _FBFM_BANDS, GEOREF_30M, [OSM_ATTRIBUTION]),
        )
        canopy = _seed(
            GRIDS_COLLECTION,
            _grid(domain_id, _CANOPY_BANDS, GEOREF_30M, [META_ATTRIBUTION]),
        )

        response = client.post(
            f"/domains/{domain_id}/grids/exports/landscape",
            json=landscape_body(topo["id"], fbfm["id"], canopy["id"]),
        )
        _assert_export_attributions(
            client,
            firestore_client,
            response,
            track,
            [META_ATTRIBUTION, OSM_ATTRIBUTION],
        )
