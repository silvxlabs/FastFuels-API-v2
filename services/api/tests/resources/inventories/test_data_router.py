"""
Integration tests for inventory data streaming endpoints.

Tests the inventory data streaming endpoints:
  GET /domains/{domain_id}/inventories/{inventory_id}/data/metadata
  GET /domains/{domain_id}/inventories/{inventory_id}/data/{partition_index}

These tests use the static-test-blue-mtn-pim-inventory fixture which has
real partitioned Parquet data on GCS.
"""

import json

import pytest

from lib.config import INVENTORIES_COLLECTION
from lib.testing import SHARED_TEST_INVENTORIES_DIR
from tests.fixtures import make_inventory_data

STATIC_PIM_INVENTORY = "static-test-blue-mtn-pim-inventory"
STATIC_CHM_INVENTORY = "static-test-blue-mtn-chm-inventory"


def _load_static_template(static_name: str) -> dict:
    path = SHARED_TEST_INVENTORIES_DIR / f"{static_name}.json"
    with open(path) as f:
        return json.load(f)


# Fixtures


@pytest.fixture(scope="session")
def static_inventory_in_firestore(firestore_client, test_owner_id, domain_for_testing):
    """Register the static PIM inventory fixture as a Firestore doc."""
    template = _load_static_template(STATIC_PIM_INVENTORY)
    template["id"] = STATIC_PIM_INVENTORY
    template["owner_id"] = test_owner_id
    template["domain_id"] = domain_for_testing["id"]
    doc_ref = firestore_client.collection(INVENTORIES_COLLECTION).document(
        STATIC_PIM_INVENTORY
    )
    doc_ref.set(template)
    yield template
    doc_ref.delete()


@pytest.fixture(scope="session")
def static_chm_inventory_in_firestore(
    firestore_client, test_owner_id, domain_for_testing
):
    """Register the static CHM inventory fixture as a Firestore doc."""
    template = _load_static_template(STATIC_CHM_INVENTORY)
    template["id"] = STATIC_CHM_INVENTORY
    template["owner_id"] = test_owner_id
    template["domain_id"] = domain_for_testing["id"]
    doc_ref = firestore_client.collection(INVENTORIES_COLLECTION).document(
        STATIC_CHM_INVENTORY
    )
    doc_ref.set(template)
    yield template
    doc_ref.delete()


@pytest.fixture(scope="session")
def pending_inventory_in_firestore(firestore_client, domain_for_testing):
    """A pending inventory (not completed) for validation tests."""
    inv_data = make_inventory_data(
        domain_id=domain_for_testing["id"],
        name="Pending inventory for data tests",
        status="pending",
    )
    doc_ref = firestore_client.collection(INVENTORIES_COLLECTION).document(
        inv_data["id"]
    )
    doc_ref.set(inv_data)
    yield inv_data
    doc_ref.delete()


@pytest.fixture(scope="session")
def inventory_with_different_owner(firestore_client, domain_with_different_owner):
    """An inventory owned by a different user."""
    inv_data = make_inventory_data(
        domain_id=domain_with_different_owner["id"],
        owner_id="different-owner",
        name="Other User's Inventory",
        status="completed",
    )
    doc_ref = firestore_client.collection(INVENTORIES_COLLECTION).document(
        inv_data["id"]
    )
    doc_ref.set(inv_data)
    yield inv_data
    doc_ref.delete()


# Helpers


def metadata_route(domain_id, inventory_id):
    return f"/domains/{domain_id}/inventories/{inventory_id}/data/metadata"


def data_route(domain_id, inventory_id, partition_index, **params):
    return (
        f"/domains/{domain_id}/inventories/{inventory_id}/data/{partition_index}",
        params,
    )


# GET /domains/{domain_id}/inventories/{inventory_id}/data/metadata


class TestGetInventoryDataMetadata:
    def test_returns_200(
        self, client, domain_for_testing, static_inventory_in_firestore
    ):
        response = client.get(
            metadata_route(domain_for_testing["id"], STATIC_PIM_INVENTORY)
        )
        assert response.status_code == 200

        data = response.json()
        assert data["inventory_id"] == STATIC_PIM_INVENTORY
        assert data["num_partitions"] >= 1
        assert data["total_rows"] > 0
        assert isinstance(data["columns"], list)
        assert len(data["columns"]) > 0
        assert len(data["partitions"]) == data["num_partitions"]
        for p in data["partitions"]:
            assert "index" in p
            assert "num_rows" in p
            assert p["num_rows"] > 0

    def test_inventory_not_completed_returns_422(
        self, client, domain_for_testing, pending_inventory_in_firestore
    ):
        response = client.get(
            metadata_route(
                domain_for_testing["id"],
                pending_inventory_in_firestore["id"],
            )
        )
        assert response.status_code == 422

    def test_inventory_not_found_returns_404(self, client, domain_for_testing):
        response = client.get(
            metadata_route(
                domain_for_testing["id"],
                "00000000000000000000000000000000",
            )
        )
        assert response.status_code == 404

    def test_inventory_wrong_owner_returns_404(
        self, client, domain_for_testing, inventory_with_different_owner
    ):
        response = client.get(
            metadata_route(
                domain_for_testing["id"],
                inventory_with_different_owner["id"],
            )
        )
        assert response.status_code == 404

    def test_inventory_wrong_domain_returns_404(
        self, client, domain_with_different_owner, static_inventory_in_firestore
    ):
        response = client.get(
            metadata_route(domain_with_different_owner["id"], STATIC_PIM_INVENTORY)
        )
        assert response.status_code == 404


# GET /domains/{domain_id}/inventories/{inventory_id}/data/{partition_index}


class TestGetInventoryData:
    def test_json_split_default_returns_200(
        self, client, domain_for_testing, static_inventory_in_firestore
    ):
        """Default JSON split response."""
        url, params = data_route(
            domain_for_testing["id"], STATIC_PIM_INVENTORY, 0, format="json"
        )
        response = client.get(url, params=params)
        assert response.status_code == 200

        data = response.json()
        assert data["partition"] == 0
        assert data["num_rows"] > 0
        assert isinstance(data["columns"], list)
        assert isinstance(data["data"], list)
        assert isinstance(data["data"][0], list)

    def test_json_records_returns_200(
        self, client, domain_for_testing, static_inventory_in_firestore
    ):
        """JSON records orientation."""
        url, params = data_route(
            domain_for_testing["id"],
            STATIC_PIM_INVENTORY,
            0,
            format="json",
            json_orientation="records",
        )
        response = client.get(url, params=params)
        assert response.status_code == 200

        data = response.json()
        assert data["num_rows"] > 0
        assert isinstance(data["data"][0], dict)

    def test_csv_returns_200(
        self, client, domain_for_testing, static_inventory_in_firestore
    ):
        """CSV response with metadata headers."""
        url, params = data_route(
            domain_for_testing["id"], STATIC_PIM_INVENTORY, 0, format="csv"
        )
        response = client.get(url, params=params)
        assert response.status_code == 200
        assert "text/csv" in response.headers["content-type"]
        assert "X-Partition-Index" in response.headers
        assert "X-Row-Count" in response.headers
        assert "X-Total-Rows" in response.headers
        assert "X-Num-Partitions" in response.headers

    def test_column_subset(
        self, client, domain_for_testing, static_inventory_in_firestore
    ):
        """Request only specific columns."""
        url, params = data_route(
            domain_for_testing["id"],
            STATIC_PIM_INVENTORY,
            0,
            format="json",
            columns="x,y,dbh",
        )
        response = client.get(url, params=params)
        assert response.status_code == 200

        data = response.json()
        assert data["columns"] == ["x", "y", "dbh"]
        assert len(data["data"][0]) == 3

    def test_partition_out_of_range_returns_422(
        self, client, domain_for_testing, static_inventory_in_firestore
    ):
        url, params = data_route(domain_for_testing["id"], STATIC_PIM_INVENTORY, 9999)
        response = client.get(url, params=params)
        assert response.status_code == 422

    def test_invalid_column_returns_422(
        self, client, domain_for_testing, static_inventory_in_firestore
    ):
        url, params = data_route(
            domain_for_testing["id"],
            STATIC_PIM_INVENTORY,
            0,
            columns="x,nonexistent_col",
        )
        response = client.get(url, params=params)
        assert response.status_code == 422

    def test_inventory_not_completed_returns_422(
        self, client, domain_for_testing, pending_inventory_in_firestore
    ):
        url, params = data_route(
            domain_for_testing["id"],
            pending_inventory_in_firestore["id"],
            0,
        )
        response = client.get(url, params=params)
        assert response.status_code == 422

    def test_inventory_not_found_returns_404(self, client, domain_for_testing):
        url, params = data_route(
            domain_for_testing["id"],
            "00000000000000000000000000000000",
            0,
        )
        response = client.get(url, params=params)
        assert response.status_code == 404

    def test_inventory_wrong_owner_returns_404(
        self, client, domain_for_testing, inventory_with_different_owner
    ):
        url, params = data_route(
            domain_for_testing["id"],
            inventory_with_different_owner["id"],
            0,
        )
        response = client.get(url, params=params)
        assert response.status_code == 404

    def test_inventory_wrong_domain_returns_404(
        self, client, domain_with_different_owner, static_inventory_in_firestore
    ):
        url, params = data_route(
            domain_with_different_owner["id"],
            STATIC_PIM_INVENTORY,
            0,
        )
        response = client.get(url, params=params)
        assert response.status_code == 404


class TestGetChmInventoryData:
    """Tests for CHM inventory data serialization.

    CHM inventories created via ITD may contain nullable string columns
    with pd.NA values. These tests verify the data endpoint correctly
    serializes that data without errors.
    """

    def test_json_split_returns_200(
        self, client, domain_for_testing, static_chm_inventory_in_firestore
    ):
        """Default split orientation serializes without pd.NA errors."""
        url, params = data_route(
            domain_for_testing["id"], STATIC_CHM_INVENTORY, 0, format="json"
        )
        response = client.get(url, params=params)
        assert response.status_code == 200

        data = response.json()
        assert data["num_rows"] > 0
        assert isinstance(data["data"], list)
        assert isinstance(data["data"][0], list)

    def test_json_records_returns_200(
        self, client, domain_for_testing, static_chm_inventory_in_firestore
    ):
        """Records orientation serializes without pd.NA errors."""
        url, params = data_route(
            domain_for_testing["id"],
            STATIC_CHM_INVENTORY,
            0,
            format="json",
            json_orientation="records",
        )
        response = client.get(url, params=params)
        assert response.status_code == 200

        data = response.json()
        assert data["num_rows"] > 0
        assert isinstance(data["data"][0], dict)
