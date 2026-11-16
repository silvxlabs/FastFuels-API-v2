"""
Integration tests for POST /domains/{domain_id}/features/{road,water,building}/concatenate

These tests make real HTTP requests to the API and interact with Firestore and
GCS. Input features are created through the synchronous GeoJSON upload, so
each has a real GeoParquet blob.
"""

import pytest
from api.resources.features.building.examples import EXAMPLE_BUILDING_GEOJSON
from api.resources.features.road.examples import EXAMPLE_ROAD_GEOJSON

from lib.config import FEATURES_COLLECTION
from tests.fixtures import make_feature_data

OSM_ATTRIBUTION = {
    "dataset": "OpenStreetMap",
    "license_name": "ODbL-1.0",
    "license_url": "https://opendatacommons.org/licenses/odbl/1-0/",
    "citation": "© OpenStreetMap contributors",
    "access_url": "https://www.openstreetmap.org/copyright",
    "accessed_on": "2026-11-05",
}


@pytest.fixture
def created_features(firestore_client):
    """Collects created feature ids and deletes them on teardown."""
    ids = []
    yield ids
    for feature_id in ids:
        firestore_client.collection(FEATURES_COLLECTION).document(feature_id).delete()


def create(client, path, body, created_features):
    response = client.post(path, json=body)
    if response.status_code == 201:
        created_features.append(response.json()["id"])
    return response


@pytest.fixture
def road_inputs(client, blackfoot_domain, created_features):
    """Two uploaded road features with different columns, as response dicts.

    The first has the example's `type` column; the second has only `name`
    and a 30 m extent buffer.
    """
    domain_id = blackfoot_domain["id"]
    path = f"/domains/{domain_id}/features/road/geojson"
    first = create(client, path, EXAMPLE_ROAD_GEOJSON, created_features)
    lane = {
        **EXAMPLE_ROAD_GEOJSON,
        "extent_buffer_m": 30,
        "features": [
            {**EXAMPLE_ROAD_GEOJSON["features"][0], "properties": {"name": "Lane"}}
        ],
    }
    second = create(client, path, lane, created_features)
    assert first.status_code == 201, first.json()
    assert second.status_code == 201, second.json()
    return first.json(), second.json()


class TestConcatenateFeatures:
    def route(self, domain_id, feature_type="road"):
        return f"/domains/{domain_id}/features/{feature_type}/concatenate"

    def test_appends_rows_in_order(
        self, client, blackfoot_domain, road_inputs, created_features
    ):
        domain_id = blackfoot_domain["id"]
        first, second = road_inputs
        body = {
            "source_feature_ids": [first["id"], second["id"]],
            "name": "Roads",
            "tags": ["roads"],
        }
        response = create(client, self.route(domain_id), body, created_features)
        assert response.status_code == 201, response.json()

        data = response.json()
        assert data["type"] == "road"
        assert data["status"] == "completed"
        assert data["name"] == "Roads"
        assert data["tags"] == ["roads"]
        assert data["source"]["product"] == "concatenate"
        assert data["source"]["source_feature_ids"] == [first["id"], second["id"]]
        assert data["georeference"]["crs"] == "EPSG:32612"
        # Union of the inputs' bounds: the second input's buffer is wider.
        assert data["georeference"]["bounds"] == second["georeference"]["bounds"]

        rows = client.get(f"/domains/{domain_id}/features/{data['id']}/data/0")
        assert rows.status_code == 200
        properties = [f["properties"] for f in rows.json()["features"]]
        assert properties == [
            {"type": "driveway", "name": None, "source_feature_id": first["id"]},
            {"type": "driveway", "name": None, "source_feature_id": first["id"]},
            {"type": None, "name": "Lane", "source_feature_id": second["id"]},
        ]

    def test_merges_input_attributions(
        self,
        client,
        firestore_client,
        blackfoot_domain,
        road_inputs,
        created_features,
    ):
        first, second = road_inputs
        firestore_client.collection(FEATURES_COLLECTION).document(first["id"]).update(
            {"attributions": [OSM_ATTRIBUTION]}
        )
        body = {"source_feature_ids": [first["id"], second["id"]]}
        response = create(
            client, self.route(blackfoot_domain["id"]), body, created_features
        )
        assert response.status_code == 201, response.json()
        assert response.json()["attributions"] == [OSM_ATTRIBUTION]

    def test_other_type_returns_422(
        self, client, blackfoot_domain, road_inputs, created_features
    ):
        domain_id = blackfoot_domain["id"]
        building = create(
            client,
            f"/domains/{domain_id}/features/building/geojson",
            EXAMPLE_BUILDING_GEOJSON,
            created_features,
        )
        assert building.status_code == 201
        body = {"source_feature_ids": [road_inputs[0]["id"], building.json()["id"]]}
        response = create(client, self.route(domain_id), body, created_features)
        assert response.status_code == 422
        assert "is a building feature, not a road feature" in response.json()["detail"]

    def test_missing_feature_returns_422(
        self, client, blackfoot_domain, road_inputs, created_features
    ):
        body = {"source_feature_ids": [road_inputs[0]["id"], "0" * 32]}
        response = create(
            client, self.route(blackfoot_domain["id"]), body, created_features
        )
        assert response.status_code == 422
        assert "does not exist in this domain" in response.json()["detail"]

    def test_pending_feature_returns_422(
        self,
        client,
        firestore_client,
        blackfoot_domain,
        road_inputs,
        created_features,
    ):
        pending = make_feature_data(blackfoot_domain["id"], status="pending")
        firestore_client.collection(FEATURES_COLLECTION).document(pending["id"]).set(
            pending
        )
        created_features.append(pending["id"])
        body = {"source_feature_ids": [road_inputs[0]["id"], pending["id"]]}
        response = create(
            client, self.route(blackfoot_domain["id"]), body, created_features
        )
        assert response.status_code == 422
        assert "expected 'completed'" in response.json()["detail"]

    def test_feature_in_other_domain_returns_422(
        self,
        client,
        firestore_client,
        blackfoot_domain,
        domain_for_testing,
        road_inputs,
        created_features,
    ):
        elsewhere = make_feature_data(domain_for_testing["id"])
        firestore_client.collection(FEATURES_COLLECTION).document(elsewhere["id"]).set(
            elsewhere
        )
        created_features.append(elsewhere["id"])
        body = {"source_feature_ids": [road_inputs[0]["id"], elsewhere["id"]]}
        response = create(
            client, self.route(blackfoot_domain["id"]), body, created_features
        )
        assert response.status_code == 422

    def test_duplicate_ids_return_422(
        self, client, blackfoot_domain, road_inputs, created_features
    ):
        fid = road_inputs[0]["id"]
        body = {"source_feature_ids": [fid, fid]}
        response = create(
            client, self.route(blackfoot_domain["id"]), body, created_features
        )
        assert response.status_code == 422

    def test_crown_has_no_concatenate(self, client, blackfoot_domain):
        response = client.post(
            self.route(blackfoot_domain["id"], "crown"),
            json={"source_feature_ids": ["a", "b"]},
        )
        assert response.status_code in (404, 405)
