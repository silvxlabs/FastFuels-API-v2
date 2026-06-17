"""
Integration tests for the in-place grid modifications endpoint.

POST /domains/{domain_id}/grids/{grid_id}/modifications

These tests make real HTTP requests to the running API and interact with
Firestore. They assert on the synchronous response and the queued Firestore
state (pending_modifications, checksum rotation, status transition); the
actual data mutation is exercised end-to-end by the griddle integration
tests (services/griddle/tests/integration/test_modifications.py).

Note: the POST enqueues a real griddle Cloud Task, which will eventually fail
the test grid (there is no backing zarr). All assertions run on the immediate
post-request state, mirroring the inventory modifications tests.
"""

import json
import threading
import uuid

import pytest
from api.resources.grids.modifications.examples import (
    EXAMPLE_REPLACE_GR1_WITH_GR2,
    EXAMPLE_REPLACE_GR1_WITH_GR2_IN_POLYGON,
    EXAMPLE_ZERO_FUEL_IN_POLYGON,
    EXAMPLE_ZERO_HEAVY_FUEL_NEAR_ROAD,
)

from lib.config import FEATURES_COLLECTION, GRIDS_COLLECTION
from tests.fixtures import make_grid_data, make_layerset_feature_data


@pytest.fixture(scope="function")
def completed_grid(firestore_client, domain_for_testing):
    """A completed 2D grid with an fbfm band and a checksum."""
    grid_data = make_grid_data(
        domain_id=domain_for_testing["id"],
        name="Modifiable grid",
        status="completed",
    )
    grid_data["checksum"] = uuid.uuid4().hex
    doc_ref = firestore_client.collection(GRIDS_COLLECTION).document(grid_data["id"])
    doc_ref.set(grid_data)
    yield grid_data
    doc_ref.delete()


def route(domain_id, grid_id):
    return f"/domains/{domain_id}/grids/{grid_id}/modifications"


class TestApplyGridModifications:
    """Synchronous behavior of the in-place modifications endpoint."""

    def test_apply_returns_pending_grid_with_rotated_checksum(
        self, client, domain_for_testing, completed_grid
    ):
        """A valid submit returns this grid (same ID) as pending with a fresh
        checksum; the visible ledger grows only on completion."""
        response = client.post(
            route(domain_for_testing["id"], completed_grid["id"]),
            json=EXAMPLE_REPLACE_GR1_WITH_GR2,
        )
        assert response.status_code == 200, response.text
        data = response.json()

        assert data["id"] == completed_grid["id"]
        assert data["status"] == "pending"
        assert data["checksum"] != completed_grid["checksum"]
        assert data["error"] is None
        assert data["progress"] is None
        # Ledger == applied data: the submitted rules are queued, not yet
        # part of the visible modifications list.
        assert data["modifications"] == []

    def test_replace_gr1_with_gr2_example_queues_delta(
        self, client, firestore_client, domain_for_testing, completed_grid
    ):
        """The documented GR1→GR2 reclassification example is accepted and
        queued verbatim in pending_modifications."""
        response = client.post(
            route(domain_for_testing["id"], completed_grid["id"]),
            json=EXAMPLE_REPLACE_GR1_WITH_GR2,
        )
        assert response.status_code == 200, response.text

        doc = (
            firestore_client.collection(GRIDS_COLLECTION)
            .document(completed_grid["id"])
            .get()
            .to_dict()
        )
        assert doc["status"] == "pending"
        assert doc["modifications"] == []
        pending = doc["pending_modifications"]
        assert len(pending) == 1
        assert pending[0]["conditions"] == [
            {"band": "fbfm", "operator": "eq", "value": 101}
        ]
        assert pending[0]["actions"] == [
            {"band": "fbfm", "modifier": "replace", "value": 102}
        ]

    def test_replace_gr1_with_gr2_in_polygon_example_queues_both_conditions(
        self, client, firestore_client, domain_for_testing, completed_grid
    ):
        """The polygon-scoped reclassification example is accepted: the rule
        keeps both its attribute condition and its spatial condition (ANDed),
        with the polygon coordinates stringified for Firestore."""
        response = client.post(
            route(domain_for_testing["id"], completed_grid["id"]),
            json=EXAMPLE_REPLACE_GR1_WITH_GR2_IN_POLYGON,
        )
        assert response.status_code == 200, response.text

        doc = (
            firestore_client.collection(GRIDS_COLLECTION)
            .document(completed_grid["id"])
            .get()
            .to_dict()
        )
        rule = doc["pending_modifications"][0]
        assert len(rule["conditions"]) == 2
        assert rule["conditions"][0] == {
            "band": "fbfm",
            "operator": "eq",
            "value": 101,
        }
        spatial = rule["conditions"][1]
        assert spatial["source"] == "geometry"
        assert spatial["operator"] == "within"
        assert isinstance(spatial["geometry"]["coordinates"], str)
        assert rule["actions"] == [
            {"band": "fbfm", "modifier": "replace", "value": 102}
        ]

    def test_geometry_condition_coordinates_stringified_in_firestore(
        self, client, firestore_client, domain_for_testing, completed_grid
    ):
        """Inline-geometry coordinates are JSON-encoded in the stored delta
        (Firestore rejects nested arrays) but nested in the response."""
        response = client.post(
            route(domain_for_testing["id"], completed_grid["id"]),
            json=EXAMPLE_ZERO_FUEL_IN_POLYGON,
        )
        assert response.status_code == 200, response.text

        doc = (
            firestore_client.collection(GRIDS_COLLECTION)
            .document(completed_grid["id"])
            .get()
            .to_dict()
        )
        stored = doc["pending_modifications"][0]["conditions"][0]["geometry"][
            "coordinates"
        ]
        assert isinstance(stored, str)
        assert json.loads(stored)[0][0] == [-120.0, 38.0]

    def test_feature_condition_with_valid_feature_accepted(
        self, client, firestore_client, domain_for_testing, completed_grid
    ):
        """A feature-based condition referencing a completed Feature in the
        same domain passes the up-front validation."""
        feature = make_layerset_feature_data(domain_id=domain_for_testing["id"])
        feature_ref = firestore_client.collection(FEATURES_COLLECTION).document(
            feature["id"]
        )
        feature_ref.set(feature)
        try:
            response = client.post(
                route(domain_for_testing["id"], completed_grid["id"]),
                json={
                    "modifications": [
                        {
                            "conditions": [
                                {
                                    "source": "feature",
                                    "operator": "intersects",
                                    "feature_id": feature["id"],
                                    "target": "cell",
                                }
                            ],
                            "actions": [
                                {
                                    "band": "fuel_load.1hr",
                                    "modifier": "replace",
                                    "value": 0,
                                }
                            ],
                        }
                    ]
                },
            )
            assert response.status_code == 200, response.text
        finally:
            feature_ref.delete()

    def test_compound_feature_and_attribute_conditions_anded_in_one_rule(
        self, client, firestore_client, domain_for_testing
    ):
        """The zero-heavy-fuel-near-road example: a single rule with a feature
        condition AND an attribute condition. Both conditions are queued
        together in one rule (intersection semantics), demonstrating that a
        rule's condition list accepts multiple entries."""
        # This example references the fuel_load.100hr band, so seed a grid
        # that carries it (the default fixture grid does not).
        grid_data = make_grid_data(
            domain_id=domain_for_testing["id"],
            status="completed",
            bands=[
                {
                    "key": "fuel_load.100hr",
                    "type": "continuous",
                    "unit": "kg/m**2",
                    "index": 0,
                },
            ],
        )
        grid_data["checksum"] = uuid.uuid4().hex
        grid_ref = firestore_client.collection(GRIDS_COLLECTION).document(
            grid_data["id"]
        )
        grid_ref.set(grid_data)

        feature = make_layerset_feature_data(domain_id=domain_for_testing["id"])
        feature_ref = firestore_client.collection(FEATURES_COLLECTION).document(
            feature["id"]
        )
        feature_ref.set(feature)
        # Point the example's placeholder feature_id at the seeded feature.
        body = {
            "modifications": [
                {
                    "conditions": [
                        {
                            **EXAMPLE_ZERO_HEAVY_FUEL_NEAR_ROAD["modifications"][0][
                                "conditions"
                            ][0],
                            "feature_id": feature["id"],
                        },
                        EXAMPLE_ZERO_HEAVY_FUEL_NEAR_ROAD["modifications"][0][
                            "conditions"
                        ][1],
                    ],
                    "actions": EXAMPLE_ZERO_HEAVY_FUEL_NEAR_ROAD["modifications"][0][
                        "actions"
                    ],
                }
            ]
        }
        try:
            response = client.post(
                route(domain_for_testing["id"], grid_data["id"]),
                json=body,
            )
            assert response.status_code == 200, response.text

            doc = grid_ref.get().to_dict()
            pending = doc["pending_modifications"]
            # One rule carrying both conditions (feature + attribute).
            assert len(pending) == 1
            conditions = pending[0]["conditions"]
            assert len(conditions) == 2
            assert conditions[0]["source"] == "feature"
            assert conditions[0]["buffer_m"] == 10
            assert conditions[1] == {
                "band": "fuel_load.100hr",
                "operator": "gt",
                "value": 2.0,
            }
        finally:
            feature_ref.delete()
            grid_ref.delete()

    def test_retry_post_on_failed_grid_with_pending_appends(
        self, client, firestore_client, domain_for_testing
    ):
        """A failed in-place modification is not dead-ended: another POST is
        accepted and its delta joins the retained queue."""
        grid_data = make_grid_data(
            domain_id=domain_for_testing["id"],
            name="Failed modify",
            status="failed",
        )
        grid_data["checksum"] = uuid.uuid4().hex
        grid_data["pending_modifications"] = [
            {
                "conditions": [{"band": "fbfm", "operator": "eq", "value": 101}],
                "actions": [{"band": "fbfm", "modifier": "replace", "value": 102}],
            }
        ]
        doc_ref = firestore_client.collection(GRIDS_COLLECTION).document(
            grid_data["id"]
        )
        doc_ref.set(grid_data)
        try:
            response = client.post(
                route(domain_for_testing["id"], grid_data["id"]),
                json=EXAMPLE_REPLACE_GR1_WITH_GR2,
            )
            assert response.status_code == 200, response.text
            data = response.json()
            assert data["status"] == "pending"
            assert data["error"] is None

            doc = doc_ref.get().to_dict()
            assert len(doc["pending_modifications"]) == 2
        finally:
            doc_ref.delete()

    def test_concurrent_posts_exactly_one_wins(
        self, client, firestore_client, domain_for_testing, completed_grid
    ):
        """Two racing POSTs cannot drop a rule: the transactional append lets
        exactly one through; the loser sees the pending status and is
        rejected."""
        results = []

        def _post():
            r = client.post(
                route(domain_for_testing["id"], completed_grid["id"]),
                json=EXAMPLE_REPLACE_GR1_WITH_GR2,
            )
            results.append(r.status_code)

        threads = [threading.Thread(target=_post) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert sorted(results) == [200, 422], results
        doc = (
            firestore_client.collection(GRIDS_COLLECTION)
            .document(completed_grid["id"])
            .get()
            .to_dict()
        )
        assert len(doc["pending_modifications"]) == 1


class TestApplyGridModificationsErrors:
    """Documented error responses."""

    def test_nonexistent_grid_returns_404(self, client, domain_for_testing):
        response = client.post(
            route(domain_for_testing["id"], "00000000000000000000000000000000"),
            json=EXAMPLE_REPLACE_GR1_WITH_GR2,
        )
        assert response.status_code == 404

    def test_wrong_owner_returns_404(
        self, client, firestore_client, domain_with_different_owner
    ):
        grid_data = make_grid_data(
            domain_id=domain_with_different_owner["id"],
            owner_id="different-owner",
            status="completed",
        )
        doc_ref = firestore_client.collection(GRIDS_COLLECTION).document(
            grid_data["id"]
        )
        doc_ref.set(grid_data)
        try:
            response = client.post(
                route(domain_with_different_owner["id"], grid_data["id"]),
                json=EXAMPLE_REPLACE_GR1_WITH_GR2,
            )
            assert response.status_code == 404
        finally:
            doc_ref.delete()

    def test_grid_in_other_domain_returns_404(
        self, client, second_domain, completed_grid
    ):
        response = client.post(
            route(second_domain["id"], completed_grid["id"]),
            json=EXAMPLE_REPLACE_GR1_WITH_GR2,
        )
        assert response.status_code == 404

    def test_pending_grid_returns_422(
        self, client, firestore_client, domain_for_testing
    ):
        grid_data = make_grid_data(domain_id=domain_for_testing["id"], status="pending")
        doc_ref = firestore_client.collection(GRIDS_COLLECTION).document(
            grid_data["id"]
        )
        doc_ref.set(grid_data)
        try:
            response = client.post(
                route(domain_for_testing["id"], grid_data["id"]),
                json=EXAMPLE_REPLACE_GR1_WITH_GR2,
            )
            assert response.status_code == 422
        finally:
            doc_ref.delete()

    def test_failed_grid_without_pending_returns_422(
        self, client, firestore_client, domain_for_testing
    ):
        """A failed initial build has no data to modify — not retryable here."""
        grid_data = make_grid_data(domain_id=domain_for_testing["id"], status="failed")
        doc_ref = firestore_client.collection(GRIDS_COLLECTION).document(
            grid_data["id"]
        )
        doc_ref.set(grid_data)
        try:
            response = client.post(
                route(domain_for_testing["id"], grid_data["id"]),
                json=EXAMPLE_REPLACE_GR1_WITH_GR2,
            )
            assert response.status_code == 422
        finally:
            doc_ref.delete()

    def test_3d_voxel_grid_returns_422(
        self, client, firestore_client, domain_for_testing
    ):
        """3D voxel grids reject modifications — modify the source inventory
        and re-voxelize instead (#153)."""
        grid_data = make_grid_data(
            domain_id=domain_for_testing["id"],
            status="completed",
            bands=[
                {
                    "key": "bulk_density.foliage.live",
                    "type": "continuous",
                    "unit": "kg/m**3",
                    "index": 0,
                }
            ],
            georeference={
                "crs": "EPSG:32611",
                "transform": (2.0, 0.0, 500000.0, 0.0, -2.0, 5201000.0),
                "shape": (50, 34, 34),
                "z_resolution": 1.0,
                "z_origin": 0.0,
            },
        )
        doc_ref = firestore_client.collection(GRIDS_COLLECTION).document(
            grid_data["id"]
        )
        doc_ref.set(grid_data)
        try:
            response = client.post(
                route(domain_for_testing["id"], grid_data["id"]),
                json={
                    "modifications": [
                        {
                            "conditions": [
                                {
                                    "band": "bulk_density.foliage.live",
                                    "operator": "gt",
                                    "value": 0,
                                }
                            ],
                            "actions": [
                                {
                                    "band": "bulk_density.foliage.live",
                                    "modifier": "multiply",
                                    "value": 0.5,
                                }
                            ],
                        }
                    ]
                },
            )
            assert response.status_code == 422
            assert "3D" in response.json()["detail"]
        finally:
            doc_ref.delete()

    def test_unknown_band_returns_422(self, client, domain_for_testing, completed_grid):
        response = client.post(
            route(domain_for_testing["id"], completed_grid["id"]),
            json={
                "modifications": [
                    {
                        "conditions": [
                            {"band": "not_a_band", "operator": "eq", "value": 1}
                        ],
                        "actions": [
                            {"band": "not_a_band", "modifier": "replace", "value": 0}
                        ],
                    }
                ]
            },
        )
        assert response.status_code == 422

    def test_unknown_feature_returns_422(
        self, client, domain_for_testing, completed_grid
    ):
        response = client.post(
            route(domain_for_testing["id"], completed_grid["id"]),
            json={
                "modifications": [
                    {
                        "conditions": [
                            {
                                "source": "feature",
                                "operator": "intersects",
                                "feature_id": "00000000000000000000000000000000",
                            }
                        ],
                        "actions": [
                            {
                                "band": "fuel_load.1hr",
                                "modifier": "replace",
                                "value": 0,
                            }
                        ],
                    }
                ]
            },
        )
        assert response.status_code == 422

    def test_empty_modifications_returns_422(
        self, client, domain_for_testing, completed_grid
    ):
        response = client.post(
            route(domain_for_testing["id"], completed_grid["id"]),
            json={"modifications": []},
        )
        assert response.status_code == 422
