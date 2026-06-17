"""
Integration tests for the in-place inventory treatments endpoint.

Tests POST /domains/{domain_id}/inventories/{inventory_id}/treatments

The endpoint mutates the inventory **in place** (same ID): it queues the
submitted treatments in ``pending_treatments``, re-assigns its ``checksum``,
flips status to ``pending``, and enqueues standgen to re-derive the data. The
visible ``treatments`` ledger grows only on completion (standgen merges the
pending delta into the ledger atomically with status=completed), so the stored
ledger always equals the applied data. The read-validate-append runs in a
Firestore transaction so concurrent POSTs cannot drop a treatment. These tests
make real HTTP requests and interact with Firestore.
"""

import threading
import uuid

import pytest
from api.resources.inventories.schema import CHM_INVENTORY_COLUMNS
from api.resources.inventories.treatments.examples import (
    ALL_TREATMENTS_EXAMPLE_VALUES,
)
from api.resources.inventories.treatments.schema import ApplyTreatmentsRequest

from lib.config import DOMAINS_COLLECTION, FEATURES_COLLECTION, INVENTORIES_COLLECTION
from tests.fixtures import make_domain_data, make_feature_data, make_inventory_data

# Columns the CHM create endpoint stores: height and position only, no dbh.
# Shared with the router so the fixture cannot drift from what production
# documents actually contain.
CHM_COLUMNS = [c.model_dump() for c in CHM_INVENTORY_COLUMNS]


@pytest.fixture
def cleanup_inventories(firestore_client):
    """Collect inventory IDs created during a test; delete their Firestore docs
    on teardown."""
    created: list[str] = []
    yield created
    for inv_id in created:
        firestore_client.collection(INVENTORIES_COLLECTION).document(inv_id).delete()


@pytest.fixture
def source_inventory(firestore_client, domain_for_testing, cleanup_inventories):
    """A fresh completed inventory to treat in place.

    Function-scoped: the endpoint mutates this document (status -> pending), so
    each test gets its own. Carries a checksum and one existing treatment so
    tests can assert the checksum changes and the existing ledger is left
    untouched (the new delta is queued, not appended to the ledger).
    """
    inv = make_inventory_data(
        domain_id=domain_for_testing["id"],
        name="Source Inventory for Treatments",
        status="completed",
        georeference={
            "crs": "EPSG:32611",
            "bounds": [500000.0, 5200000.0, 501000.0, 5201000.0],
        },
    )
    inv["checksum"] = uuid.uuid4().hex
    inv["treatments"] = [{"metric": "diameter", "method": "from_below", "value": 10.0}]
    firestore_client.collection(INVENTORIES_COLLECTION).document(inv["id"]).set(inv)
    cleanup_inventories.append(inv["id"])
    return inv


@pytest.fixture(scope="session")
def pending_inventory(firestore_client, domain_for_testing):
    """A pending (not completed) inventory for status validation tests."""
    inv_data = make_inventory_data(
        domain_id=domain_for_testing["id"],
        name="Pending Inventory",
        status="pending",
    )
    doc_ref = firestore_client.collection(INVENTORIES_COLLECTION).document(
        inv_data["id"]
    )
    doc_ref.set(inv_data)
    yield inv_data
    doc_ref.delete()


@pytest.fixture(scope="session")
def completed_inventory_different_owner(firestore_client, domain_with_different_owner):
    """A completed inventory owned by a different user."""
    inv_data = make_inventory_data(
        domain_id=domain_with_different_owner["id"],
        owner_id="different-owner",
        name="Other Owner's Completed Inventory",
        status="completed",
        georeference={
            "crs": "EPSG:32611",
            "bounds": [500000.0, 5200000.0, 501000.0, 5201000.0],
        },
    )
    doc_ref = firestore_client.collection(INVENTORIES_COLLECTION).document(
        inv_data["id"]
    )
    doc_ref.set(inv_data)
    yield inv_data
    doc_ref.delete()


@pytest.fixture(scope="session")
def chm_inventory(firestore_client, domain_for_testing):
    """A completed CHM-derived inventory (no dbh column) for the dbh guard test."""
    inv_data = make_inventory_data(
        domain_id=domain_for_testing["id"],
        name="CHM Inventory (no dbh)",
        status="completed",
        georeference={
            "crs": "EPSG:32611",
            "bounds": [500000.0, 5200000.0, 501000.0, 5201000.0],
        },
    )
    inv_data["columns"] = CHM_COLUMNS
    doc_ref = firestore_client.collection(INVENTORIES_COLLECTION).document(
        inv_data["id"]
    )
    doc_ref.set(inv_data)
    yield inv_data
    doc_ref.delete()


@pytest.fixture(scope="session")
def large_domain(firestore_client, test_owner_id):
    """A domain whose bounding box exceeds the inventory-wide basal-area area
    limit (16 km²). 6 km × 6 km = 36 km²."""
    domain_data = make_domain_data(name="Large Domain for Treatment Area Limit")
    domain_data["bbox"] = [500000.0, 5200000.0, 506000.0, 5206000.0]
    doc_ref = firestore_client.collection(DOMAINS_COLLECTION).document(
        domain_data["id"]
    )
    doc_ref.set(domain_data)
    yield domain_data
    doc_ref.delete()


@pytest.fixture(scope="session")
def large_domain_inventory(firestore_client, large_domain):
    """A completed inventory in the large domain (with a dbh column)."""
    inv_data = make_inventory_data(
        domain_id=large_domain["id"],
        name="Inventory in Large Domain",
        status="completed",
        georeference={
            "crs": "EPSG:32611",
            "bounds": [500000.0, 5200000.0, 506000.0, 5206000.0],
        },
    )
    doc_ref = firestore_client.collection(INVENTORIES_COLLECTION).document(
        inv_data["id"]
    )
    doc_ref.set(inv_data)
    yield inv_data
    doc_ref.delete()


MINIMAL_TREATMENTS_BODY = {
    "treatments": [{"metric": "diameter", "method": "from_below", "value": 5.0}],
}


class TestApplyTreatmentsInPlace:
    """POST /domains/{domain_id}/inventories/{inventory_id}/treatments."""

    def route(self, domain_id, inventory_id):
        return f"/domains/{domain_id}/inventories/{inventory_id}/treatments"

    def test_applies_in_place_same_id(
        self, client, domain_for_testing, source_inventory, firestore_client
    ):
        """Mutates the same inventory: same ID, status pending, checksum
        changes, and the delta is queued in pending_treatments while the
        visible ledger is left untouched (it grows only on completion)."""
        source_id = source_inventory["id"]
        response = client.post(
            self.route(domain_for_testing["id"], source_id),
            json=MINIMAL_TREATMENTS_BODY,
        )

        assert response.status_code == 200, response.json()
        data = response.json()

        # Same inventory — not a new resource.
        assert data["id"] == source_id
        assert data["domain_id"] == domain_for_testing["id"]
        assert data["status"] == "pending"
        assert data["error"] is None

        # Ledger == applied data: the submitted treatment is queued, not yet
        # part of the visible treatments list, so the existing ledger is
        # unchanged.
        assert len(data["treatments"]) == 1
        assert data["treatments"][0]["value"] == 10.0

        # checksum is re-assigned so derivatives become detectably stale.
        assert data["checksum"] != source_inventory["checksum"]

        # The source is unchanged (still the root pim source).
        assert data["source"]["name"] == source_inventory["source"]["name"]

        # Firestore: only the new delta is queued for standgen to apply; the
        # ledger still holds just the pre-existing treatment. The job is in
        # flight (a real standgen task was enqueued), so its status may have
        # advanced past pending — but it cannot be completed (these test docs
        # have no backing parquet), so the queue is never cleared/merged here.
        doc = (
            firestore_client.collection(INVENTORIES_COLLECTION)
            .document(source_id)
            .get()
            .to_dict()
        )
        assert doc["status"] != "completed"
        assert len(doc["pending_treatments"]) == 1
        assert doc["pending_treatments"][0]["value"] == 5.0
        assert len(doc["treatments"]) == 1

    def test_response_excludes_owner_id(
        self, client, domain_for_testing, source_inventory
    ):
        """Response should not expose the owner_id field."""
        response = client.post(
            self.route(domain_for_testing["id"], source_inventory["id"]),
            json=MINIMAL_TREATMENTS_BODY,
        )
        assert response.status_code == 200
        assert "owner_id" not in response.json()
        assert "pending_treatments" not in response.json()

    def test_preserves_columns_and_georeference(
        self, client, domain_for_testing, source_inventory
    ):
        """In-place edit keeps the inventory's columns and georeference."""
        response = client.post(
            self.route(domain_for_testing["id"], source_inventory["id"]),
            json=MINIMAL_TREATMENTS_BODY,
        )
        assert response.status_code == 200
        data = response.json()
        expected_columns = [
            {**col, "summary": None} for col in source_inventory["columns"]
        ]
        assert data["columns"] == expected_columns
        assert data["georeference"] == source_inventory["georeference"]

    def test_queues_multiple_treatments(
        self, client, domain_for_testing, source_inventory, firestore_client
    ):
        """Multiple treatments in one request are all queued in the delta; the
        visible ledger stays at its pre-existing size until completion."""
        body = {
            "treatments": [
                {"metric": "diameter", "method": "from_below", "value": 2.54},
                {"metric": "basal_area", "method": "proportional", "value": 18.0},
            ]
        }
        response = client.post(
            self.route(domain_for_testing["id"], source_inventory["id"]),
            json=body,
        )
        assert response.status_code == 200, response.json()
        # Ledger unchanged (1 pre-existing treatment); both submitted queued.
        assert len(response.json()["treatments"]) == 1
        doc = (
            firestore_client.collection(INVENTORIES_COLLECTION)
            .document(source_inventory["id"])
            .get()
            .to_dict()
        )
        assert len(doc["pending_treatments"]) == 2

    def test_inventory_wide_basal_area_on_small_domain_succeeds(
        self, client, domain_for_testing, source_inventory, firestore_client
    ):
        """An inventory-wide basal-area treatment on a small domain is allowed."""
        body = {
            "treatments": [
                {"metric": "basal_area", "method": "from_below", "value": 25.0}
            ]
        }
        response = client.post(
            self.route(domain_for_testing["id"], source_inventory["id"]),
            json=body,
        )
        assert response.status_code == 200, response.json()
        doc = (
            firestore_client.collection(INVENTORIES_COLLECTION)
            .document(source_inventory["id"])
            .get()
            .to_dict()
        )
        assert doc["pending_treatments"][-1]["metric"] == "basal_area"

    def test_unit_conversion_preserved(
        self, client, domain_for_testing, source_inventory, firestore_client
    ):
        """A unit field on a treatment round-trips into the queued delta."""
        body = {
            "treatments": [
                {
                    "metric": "diameter",
                    "method": "from_above",
                    "value": 16.0,
                    "unit": "in",
                }
            ]
        }
        response = client.post(
            self.route(domain_for_testing["id"], source_inventory["id"]),
            json=body,
        )
        assert response.status_code == 200
        doc = (
            firestore_client.collection(INVENTORIES_COLLECTION)
            .document(source_inventory["id"])
            .get()
            .to_dict()
        )
        assert doc["pending_treatments"][-1]["unit"] == "in"

    # Error cases

    def test_nonexistent_inventory_returns_404(self, client, domain_for_testing):
        response = client.post(
            self.route(domain_for_testing["id"], "00000000000000000000000000000000"),
            json=MINIMAL_TREATMENTS_BODY,
        )
        assert response.status_code == 404

    def test_inventory_not_completed_returns_422(
        self, client, domain_for_testing, pending_inventory
    ):
        response = client.post(
            self.route(domain_for_testing["id"], pending_inventory["id"]),
            json=MINIMAL_TREATMENTS_BODY,
        )
        assert response.status_code == 422

    def test_retry_post_on_failed_inventory_with_pending_appends(
        self, client, firestore_client, domain_for_testing, cleanup_inventories
    ):
        """A failed in-place treatment is not dead-ended: another POST is
        accepted and its delta joins the retained queue."""
        inv = make_inventory_data(
            domain_id=domain_for_testing["id"],
            name="Failed treat",
            status="failed",
            georeference={
                "crs": "EPSG:32611",
                "bounds": [500000.0, 5200000.0, 501000.0, 5201000.0],
            },
        )
        inv["checksum"] = uuid.uuid4().hex
        inv["pending_treatments"] = [
            {"metric": "diameter", "method": "from_below", "value": 10.0}
        ]
        firestore_client.collection(INVENTORIES_COLLECTION).document(inv["id"]).set(inv)
        cleanup_inventories.append(inv["id"])

        response = client.post(
            self.route(domain_for_testing["id"], inv["id"]),
            json=MINIMAL_TREATMENTS_BODY,
        )
        assert response.status_code == 200, response.json()
        assert response.json()["status"] == "pending"

        doc = (
            firestore_client.collection(INVENTORIES_COLLECTION)
            .document(inv["id"])
            .get()
            .to_dict()
        )
        assert len(doc["pending_treatments"]) == 2

    def test_failed_inventory_without_pending_returns_422(
        self, client, firestore_client, domain_for_testing, cleanup_inventories
    ):
        """A failed initial build has no queued delta to retry — not retryable
        here, so a fresh treatment is rejected until the build succeeds."""
        inv = make_inventory_data(
            domain_id=domain_for_testing["id"],
            name="Failed build",
            status="failed",
        )
        firestore_client.collection(INVENTORIES_COLLECTION).document(inv["id"]).set(inv)
        cleanup_inventories.append(inv["id"])

        response = client.post(
            self.route(domain_for_testing["id"], inv["id"]),
            json=MINIMAL_TREATMENTS_BODY,
        )
        assert response.status_code == 422

    def test_concurrent_posts_exactly_one_wins(
        self, client, firestore_client, domain_for_testing, source_inventory
    ):
        """Two racing POSTs cannot drop a treatment: the transactional append
        lets exactly one through; the loser sees the pending status and is
        rejected."""
        results = []

        def _post():
            r = client.post(
                self.route(domain_for_testing["id"], source_inventory["id"]),
                json=MINIMAL_TREATMENTS_BODY,
            )
            results.append(r.status_code)

        threads = [threading.Thread(target=_post) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert sorted(results) == [200, 422], results
        doc = (
            firestore_client.collection(INVENTORIES_COLLECTION)
            .document(source_inventory["id"])
            .get()
            .to_dict()
        )
        assert len(doc["pending_treatments"]) == 1

    def test_wrong_owner_returns_404(
        self, client, domain_with_different_owner, completed_inventory_different_owner
    ):
        response = client.post(
            self.route(
                domain_with_different_owner["id"],
                completed_inventory_different_owner["id"],
            ),
            json=MINIMAL_TREATMENTS_BODY,
        )
        assert response.status_code == 404

    def test_invalid_domain_returns_404(self, client):
        response = client.post(
            self.route(
                "00000000000000000000000000000000",
                "00000000000000000000000000000000",
            ),
            json=MINIMAL_TREATMENTS_BODY,
        )
        assert response.status_code == 404

    def test_empty_treatments_returns_422(
        self, client, domain_for_testing, source_inventory
    ):
        response = client.post(
            self.route(domain_for_testing["id"], source_inventory["id"]),
            json={"treatments": []},
        )
        assert response.status_code == 422

    def test_proportional_diameter_returns_422(
        self, client, domain_for_testing, source_inventory
    ):
        """proportional is only valid for a basal-area target, not diameter."""
        response = client.post(
            self.route(domain_for_testing["id"], source_inventory["id"]),
            json={
                "treatments": [
                    {"metric": "diameter", "method": "proportional", "value": 10.0}
                ]
            },
        )
        assert response.status_code == 422

    def test_incompatible_unit_returns_422(
        self, client, domain_for_testing, source_inventory
    ):
        response = client.post(
            self.route(domain_for_testing["id"], source_inventory["id"]),
            json={
                "treatments": [
                    {
                        "metric": "diameter",
                        "method": "from_below",
                        "value": 10.0,
                        "unit": "kg",
                    }
                ]
            },
        )
        assert response.status_code == 422

    def test_chm_inventory_without_dbh_returns_422(
        self, client, domain_for_testing, chm_inventory
    ):
        """Treatments require a dbh column; a CHM-derived inventory has none."""
        response = client.post(
            self.route(domain_for_testing["id"], chm_inventory["id"]),
            json=MINIMAL_TREATMENTS_BODY,
        )
        assert response.status_code == 422
        assert "dbh" in response.json()["detail"]

    def test_inventory_wide_basal_area_over_area_limit_returns_422(
        self, client, large_domain, large_domain_inventory
    ):
        """An inventory-wide basal-area treatment over a too-large domain is
        rejected (it would hold the whole stand in memory)."""
        response = client.post(
            self.route(large_domain["id"], large_domain_inventory["id"]),
            json={
                "treatments": [
                    {"metric": "basal_area", "method": "from_below", "value": 25.0}
                ]
            },
        )
        assert response.status_code == 422
        assert "km" in response.json()["detail"]


# Feature-based spatial conditions on treatments (validated at request time,
# #282). A spatially scoped treatment is exempt from the inventory-wide area
# limit, so these also exercise the scoped basal-area path.


@pytest.fixture(scope="session")
def completed_feature(firestore_client, domain_for_testing):
    data = make_feature_data(domain_id=domain_for_testing["id"], status="completed")
    ref = firestore_client.collection(FEATURES_COLLECTION).document(data["id"])
    ref.set(data)
    yield data
    ref.delete()


@pytest.fixture(scope="session")
def pending_feature(firestore_client, domain_for_testing):
    data = make_feature_data(domain_id=domain_for_testing["id"], status="pending")
    ref = firestore_client.collection(FEATURES_COLLECTION).document(data["id"])
    ref.set(data)
    yield data
    ref.delete()


@pytest.fixture(scope="session")
def feature_in_different_domain(firestore_client, second_domain):
    data = make_feature_data(domain_id=second_domain["id"], status="completed")
    ref = firestore_client.collection(FEATURES_COLLECTION).document(data["id"])
    ref.set(data)
    yield data
    ref.delete()


class TestFeatureConditions:
    """Feature-source spatial conditions are validated at request time (#282)."""

    def route(self, domain_id, inventory_id):
        return f"/domains/{domain_id}/inventories/{inventory_id}/treatments"

    def _body(self, feature_id):
        return {
            "treatments": [
                {
                    "metric": "basal_area",
                    "method": "from_below",
                    "value": 20.0,
                    "conditions": [
                        {
                            "source": "feature",
                            "operator": "within",
                            "feature_id": feature_id,
                            "buffer_m": 30,
                        }
                    ],
                }
            ]
        }

    def test_unknown_feature_id_returns_422(
        self, client, domain_for_testing, source_inventory
    ):
        unknown = "00000000000000000000000000000000"
        response = client.post(
            self.route(domain_for_testing["id"], source_inventory["id"]),
            json=self._body(unknown),
        )
        assert response.status_code == 422
        assert unknown in response.json()["detail"]

    def test_feature_in_different_domain_returns_422(
        self, client, domain_for_testing, source_inventory, feature_in_different_domain
    ):
        feature_id = feature_in_different_domain["id"]
        response = client.post(
            self.route(domain_for_testing["id"], source_inventory["id"]),
            json=self._body(feature_id),
        )
        assert response.status_code == 422
        assert feature_id in response.json()["detail"]

    def test_pending_feature_returns_422(
        self, client, domain_for_testing, source_inventory, pending_feature
    ):
        feature_id = pending_feature["id"]
        response = client.post(
            self.route(domain_for_testing["id"], source_inventory["id"]),
            json=self._body(feature_id),
        )
        assert response.status_code == 422
        assert feature_id in response.json()["detail"]

    def test_completed_feature_succeeds(
        self,
        client,
        domain_for_testing,
        source_inventory,
        completed_feature,
        firestore_client,
    ):
        feature_id = completed_feature["id"]
        response = client.post(
            self.route(domain_for_testing["id"], source_inventory["id"]),
            json=self._body(feature_id),
        )
        assert response.status_code == 200, response.json()
        doc = (
            firestore_client.collection(INVENTORIES_COLLECTION)
            .document(source_inventory["id"])
            .get()
            .to_dict()
        )
        condition = doc["pending_treatments"][-1]["conditions"][0]
        assert condition["feature_id"] == feature_id


class TestOpenApiExamples:
    """Every OpenAPI example body round-trips through ApplyTreatmentsRequest."""

    @pytest.mark.parametrize(
        "example", ALL_TREATMENTS_EXAMPLE_VALUES, ids=lambda e: e[0]
    )
    def test_example_validates(self, example):
        _name, value = example
        request = ApplyTreatmentsRequest(**value)
        assert len(request.treatments) >= 1
