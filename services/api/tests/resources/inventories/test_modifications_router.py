"""
Integration tests for the in-place inventory modifications endpoint.

Tests POST /domains/{domain_id}/inventories/{inventory_id}/modifications

The endpoint mutates the inventory **in place** (same ID): it appends the
submitted rules to the inventory's ``modifications`` ledger, queues that delta
in ``pending_modifications``, re-assigns its ``checksum``, flips status to
``pending``, and enqueues standgen to re-derive the data. These tests make real
HTTP requests and interact with Firestore.
"""

import uuid

import pytest
from api.resources.inventories.modifications.examples import (
    ALL_MODIFICATIONS_EXAMPLE_VALUES,
)
from api.resources.inventories.modifications.schema import ApplyModificationsRequest
from api.resources.inventories.schema import CHM_INVENTORY_COLUMNS

from lib.config import FEATURES_COLLECTION, INVENTORIES_COLLECTION
from tests.fixtures import make_feature_data, make_inventory_data


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
    """A fresh completed inventory to modify in place.

    Function-scoped: the endpoint mutates this document (status -> pending), so
    each test gets its own. Carries a checksum and one existing modification so
    tests can assert the checksum changes and the ledger grows.
    """
    inv = make_inventory_data(
        domain_id=domain_for_testing["id"],
        name="Source Inventory for Modifications",
        status="completed",
        georeference={
            "crs": "EPSG:32611",
            "bounds": [500000.0, 5200000.0, 501000.0, 5201000.0],
        },
    )
    inv["checksum"] = uuid.uuid4().hex
    inv["modifications"] = [
        {
            "conditions": [{"attribute": "height", "operator": "gt", "value": 40.0}],
            "actions": [{"attribute": "height", "modifier": "multiply", "value": 0.9}],
        }
    ]
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


@pytest.fixture
def no_dbh_inventory(firestore_client, domain_for_testing, cleanup_inventories):
    """A completed CHM-derived inventory: position and height only, no dbh.

    Function-scoped because the success-path test mutates it (status -> pending).
    """
    inv = make_inventory_data(
        domain_id=domain_for_testing["id"],
        name="CHM Inventory (no dbh) for modifications guard",
        status="completed",
        source={
            "name": "chm",
            "source_chm_grid_id": f"test-{uuid.uuid4().hex}",
            "algorithm": {"name": "lmf"},
        },
        georeference={
            "crs": "EPSG:32611",
            "bounds": [500000.0, 5200000.0, 501000.0, 5201000.0],
        },
    )
    inv["columns"] = [c.model_dump() for c in CHM_INVENTORY_COLUMNS]
    firestore_client.collection(INVENTORIES_COLLECTION).document(inv["id"]).set(inv)
    cleanup_inventories.append(inv["id"])
    return inv


MINIMAL_MODIFICATIONS_BODY = {
    "modifications": [
        {
            "conditions": {"attribute": "dbh", "operator": "lt", "value": 5.0},
            "actions": {"modifier": "remove"},
        }
    ],
}


class TestApplyModificationsInPlace:
    """POST /domains/{domain_id}/inventories/{inventory_id}/modifications."""

    def route(self, domain_id, inventory_id):
        return f"/domains/{domain_id}/inventories/{inventory_id}/modifications"

    def test_applies_in_place_same_id(
        self, client, domain_for_testing, source_inventory, firestore_client
    ):
        """Mutates the same inventory: same ID, status pending, ledger grows,
        checksum changes, and the delta is queued in pending_modifications."""
        source_id = source_inventory["id"]
        response = client.post(
            self.route(domain_for_testing["id"], source_id),
            json=MINIMAL_MODIFICATIONS_BODY,
        )

        assert response.status_code == 200, response.json()
        data = response.json()

        # Same inventory — not a new resource.
        assert data["id"] == source_id
        assert data["domain_id"] == domain_for_testing["id"]
        assert data["status"] == "pending"
        assert data["error"] is None

        # The submitted rule is appended to the existing ledger (1 + 1 = 2).
        assert len(data["modifications"]) == 2
        assert data["modifications"][-1]["conditions"][0]["attribute"] == "dbh"

        # checksum is re-assigned so derivatives become detectably stale.
        assert data["checksum"] != source_inventory["checksum"]

        # The source is unchanged (still the root pim source, not "modifications").
        assert data["source"]["name"] == source_inventory["source"]["name"]

        # Firestore: only the new delta is queued for standgen to apply.
        doc = (
            firestore_client.collection(INVENTORIES_COLLECTION)
            .document(source_id)
            .get()
            .to_dict()
        )
        assert doc["status"] == "pending"
        assert len(doc["pending_modifications"]) == 1
        assert len(doc["modifications"]) == 2

    def test_response_excludes_owner_id(
        self, client, domain_for_testing, source_inventory
    ):
        """Response should not expose the owner_id field."""
        response = client.post(
            self.route(domain_for_testing["id"], source_inventory["id"]),
            json=MINIMAL_MODIFICATIONS_BODY,
        )
        assert response.status_code == 200
        assert "owner_id" not in response.json()

    def test_preserves_columns_and_georeference(
        self, client, domain_for_testing, source_inventory
    ):
        """In-place edit keeps the inventory's columns and georeference."""
        response = client.post(
            self.route(domain_for_testing["id"], source_inventory["id"]),
            json=MINIMAL_MODIFICATIONS_BODY,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["columns"] == source_inventory["columns"]
        assert data["georeference"] == source_inventory["georeference"]

    def test_appends_multiple_modifications(
        self, client, domain_for_testing, source_inventory
    ):
        """Multiple rules in one request all append to the ledger."""
        body = {
            "modifications": [
                {
                    "conditions": {"attribute": "dbh", "operator": "lt", "value": 2.54},
                    "actions": {"modifier": "remove"},
                },
                {
                    "conditions": {
                        "attribute": "height",
                        "operator": "gt",
                        "value": 50,
                    },
                    "actions": {
                        "attribute": "height",
                        "modifier": "multiply",
                        "value": 0.9,
                    },
                },
            ]
        }
        response = client.post(
            self.route(domain_for_testing["id"], source_inventory["id"]),
            json=body,
        )
        assert response.status_code == 200
        # 1 existing + 2 submitted.
        assert len(response.json()["modifications"]) == 3

    def test_unit_conversion_preserved(
        self, client, domain_for_testing, source_inventory
    ):
        """A unit field on a condition round-trips into the ledger."""
        body = {
            "modifications": [
                {
                    "conditions": {
                        "attribute": "dbh",
                        "operator": "lt",
                        "value": 1.0,
                        "unit": "in",
                    },
                    "actions": {"modifier": "remove"},
                }
            ]
        }
        response = client.post(
            self.route(domain_for_testing["id"], source_inventory["id"]),
            json=body,
        )
        assert response.status_code == 200
        assert response.json()["modifications"][-1]["conditions"][0]["unit"] == "in"

    def test_expression_condition_accepted(
        self, client, domain_for_testing, source_inventory
    ):
        """Expression conditions are accepted."""
        body = {
            "modifications": [
                {
                    "conditions": {"expression": "height * crown_ratio < 1.0"},
                    "actions": {"modifier": "remove"},
                }
            ]
        }
        response = client.post(
            self.route(domain_for_testing["id"], source_inventory["id"]),
            json=body,
        )
        assert response.status_code == 200

    # Error cases

    def test_nonexistent_inventory_returns_404(self, client, domain_for_testing):
        response = client.post(
            self.route(domain_for_testing["id"], "00000000000000000000000000000000"),
            json=MINIMAL_MODIFICATIONS_BODY,
        )
        assert response.status_code == 404

    def test_inventory_not_completed_returns_422(
        self, client, domain_for_testing, pending_inventory
    ):
        response = client.post(
            self.route(domain_for_testing["id"], pending_inventory["id"]),
            json=MINIMAL_MODIFICATIONS_BODY,
        )
        assert response.status_code == 422

    def test_wrong_owner_returns_404(
        self, client, domain_with_different_owner, completed_inventory_different_owner
    ):
        response = client.post(
            self.route(
                domain_with_different_owner["id"],
                completed_inventory_different_owner["id"],
            ),
            json=MINIMAL_MODIFICATIONS_BODY,
        )
        assert response.status_code == 404

    def test_invalid_domain_returns_404(self, client):
        response = client.post(
            self.route(
                "00000000000000000000000000000000",
                "00000000000000000000000000000000",
            ),
            json=MINIMAL_MODIFICATIONS_BODY,
        )
        assert response.status_code == 404

    def test_empty_modifications_returns_422(
        self, client, domain_for_testing, source_inventory
    ):
        response = client.post(
            self.route(domain_for_testing["id"], source_inventory["id"]),
            json={"modifications": []},
        )
        assert response.status_code == 422

    def test_invalid_operator_for_species_returns_422(
        self, client, domain_for_testing, source_inventory
    ):
        response = client.post(
            self.route(domain_for_testing["id"], source_inventory["id"]),
            json={
                "modifications": [
                    {
                        "conditions": {
                            "attribute": "fia_species_code",
                            "operator": "gt",
                            "value": 100,
                        },
                        "actions": {"modifier": "remove"},
                    }
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
                "modifications": [
                    {
                        "conditions": {
                            "attribute": "dbh",
                            "operator": "lt",
                            "value": 5.0,
                            "unit": "kg",
                        },
                        "actions": {"modifier": "remove"},
                    }
                ]
            },
        )
        assert response.status_code == 422

    def test_invalid_expression_returns_422(
        self, client, domain_for_testing, source_inventory
    ):
        response = client.post(
            self.route(domain_for_testing["id"], source_inventory["id"]),
            json={
                "modifications": [
                    {
                        "conditions": {"expression": "abs(dbh) < 5"},
                        "actions": {"modifier": "remove"},
                    }
                ]
            },
        )
        assert response.status_code == 422

    def test_divide_by_zero_returns_422(
        self, client, domain_for_testing, source_inventory
    ):
        response = client.post(
            self.route(domain_for_testing["id"], source_inventory["id"]),
            json={
                "modifications": [
                    {
                        "conditions": {
                            "attribute": "height",
                            "operator": "gt",
                            "value": 0,
                        },
                        "actions": {
                            "attribute": "height",
                            "modifier": "divide",
                            "value": 0,
                        },
                    }
                ]
            },
        )
        assert response.status_code == 422

    def test_modification_referencing_missing_column_returns_422(
        self, client, domain_for_testing, no_dbh_inventory
    ):
        """A rule that filters on dbh can't apply to a CHM inventory with no dbh.
        The error names what was required versus what the inventory provides."""
        response = client.post(
            self.route(domain_for_testing["id"], no_dbh_inventory["id"]),
            json={
                "modifications": [
                    {
                        "conditions": {
                            "attribute": "dbh",
                            "operator": "lt",
                            "value": 5.0,
                        },
                        "actions": {"modifier": "remove"},
                    }
                ]
            },
        )
        assert response.status_code == 422
        detail = response.json()["detail"]
        assert "Required column(s)" in detail
        assert "Available column(s)" in detail
        # dbh is the referenced-but-missing column: present in the required
        # portion of the message but absent from the inventory's available set.
        required_part, available_part = detail.split("Available column(s)")
        assert "dbh" in required_part
        assert "dbh" not in available_part

    def test_modification_referencing_present_column_succeeds(
        self, client, domain_for_testing, no_dbh_inventory
    ):
        """The guard doesn't over-reject: a rule that only references height
        applies fine to a position-and-height-only inventory."""
        response = client.post(
            self.route(domain_for_testing["id"], no_dbh_inventory["id"]),
            json={
                "modifications": [
                    {
                        "conditions": {
                            "attribute": "height",
                            "operator": "gt",
                            "value": 40.0,
                        },
                        "actions": {
                            "attribute": "height",
                            "modifier": "multiply",
                            "value": 0.9,
                        },
                    }
                ]
            },
        )
        assert response.status_code == 200, response.json()


# Feature-based spatial conditions (issue #276 — end-to-end through the
# in-place modifications endpoint).


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
    """Feature-source spatial conditions are validated at request time (#276/#282)."""

    def route(self, domain_id, inventory_id):
        return f"/domains/{domain_id}/inventories/{inventory_id}/modifications"

    def _body(self, feature_id):
        return {
            "modifications": [
                {
                    "conditions": [
                        {
                            "source": "feature",
                            "operator": "within",
                            "feature_id": feature_id,
                            "buffer_m": 3,
                        }
                    ],
                    "actions": [{"modifier": "remove"}],
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
        self, client, domain_for_testing, source_inventory, completed_feature
    ):
        feature_id = completed_feature["id"]
        response = client.post(
            self.route(domain_for_testing["id"], source_inventory["id"]),
            json=self._body(feature_id),
        )
        assert response.status_code == 200, response.json()
        condition = response.json()["modifications"][-1]["conditions"][0]
        assert condition["feature_id"] == feature_id


class TestOpenApiExamples:
    """Every OpenAPI example body round-trips through ApplyModificationsRequest
    (issue #276, item 2)."""

    @pytest.mark.parametrize(
        "example", ALL_MODIFICATIONS_EXAMPLE_VALUES, ids=lambda e: e[0]
    )
    def test_example_validates(self, example):
        _name, value = example
        request = ApplyModificationsRequest(**value)
        assert len(request.modifications) >= 1
