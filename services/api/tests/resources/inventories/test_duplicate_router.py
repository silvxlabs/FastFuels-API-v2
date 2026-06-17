"""
Integration tests for the inventory duplicate endpoint.

POST /domains/{domain_id}/inventories/{inventory_id}/duplicate

These tests make real HTTP requests to the running API and interact with
Firestore. The router-level tests in TestDuplicateInventory assert on the
synchronous 201 response and the cloned Firestore document; the background GCS
parquet copy is exercised end-to-end by TestDuplicateInventoryCopiesData, which
stages real parquet on GCS.
"""

import time
import uuid

import dask.dataframe as dd
import pandas as pd
import pytest

from lib.config import INVENTORIES_BUCKET, INVENTORIES_COLLECTION
from lib.gcs import delete_directory, exists
from tests.fixtures import make_inventory_data

# A completed source inventory carrying every field a duplicate should clone.
EXAMPLE_SOURCE = {
    "name": "pim",
    "source_pim_grid_id": "src-grid-123",
    "point_process": "inhomogeneous_poisson",
    "seed": 7,
}
EXAMPLE_GEOREF = {"crs": "EPSG:5070", "bounds": [0.0, 0.0, 100.0, 100.0]}
EXAMPLE_MODIFICATION = {
    "conditions": [{"attribute": "dbh", "operator": "lt", "value": 12.7}],
    "actions": [{"modifier": "remove"}],
}


@pytest.fixture(scope="function")
def cleanup_inventories(firestore_client):
    """Collect inventory IDs created by a test; delete their Firestore docs and
    any GCS data on teardown."""
    created_ids: list[str] = []
    yield created_ids
    for inv_id in created_ids:
        firestore_client.collection(INVENTORIES_COLLECTION).document(inv_id).delete()
        gcs_path = f"gs://{INVENTORIES_BUCKET}/{inv_id}"
        if exists(gcs_path):
            delete_directory(gcs_path)


@pytest.fixture(scope="function")
def source_inventory(firestore_client, domain_for_testing):
    """A completed source inventory with a full set of carry-over fields."""
    inv_data = make_inventory_data(
        domain_id=domain_for_testing["id"],
        name="Original Inventory",
        description="The original",
        status="completed",
        tags=["original"],
        source=dict(EXAMPLE_SOURCE),
        georeference=dict(EXAMPLE_GEOREF),
    )
    inv_data["checksum"] = uuid.uuid4().hex
    inv_data["modifications"] = [dict(EXAMPLE_MODIFICATION)]
    inv_data["treatments"] = []
    doc_ref = firestore_client.collection(INVENTORIES_COLLECTION).document(
        inv_data["id"]
    )
    doc_ref.set(inv_data)
    yield inv_data
    doc_ref.delete()


class TestDuplicateInventory:
    """Router-level tests for the duplicate endpoint (synchronous response)."""

    def route(self, domain_id, inventory_id):
        return f"/domains/{domain_id}/inventories/{inventory_id}/duplicate"

    def test_duplicate_creates_new_pending_inventory(
        self, client, domain_for_testing, source_inventory, cleanup_inventories
    ):
        """A duplicate is a fresh, distinct, pending inventory."""
        response = client.post(
            self.route(domain_for_testing["id"], source_inventory["id"])
        )
        assert response.status_code == 201, response.text
        data = response.json()
        cleanup_inventories.append(data["id"])

        assert len(data["id"]) == 32
        assert data["id"] != source_inventory["id"]
        assert data["domain_id"] == domain_for_testing["id"]
        assert data["status"] == "pending"
        assert data["type"] == "tree"
        assert data["error"] is None
        assert "owner_id" not in data

    def test_duplicate_carries_over_clone_fields(
        self, client, domain_for_testing, source_inventory, cleanup_inventories
    ):
        """source, checksum, columns, and georeference are carried over verbatim."""
        response = client.post(
            self.route(domain_for_testing["id"], source_inventory["id"])
        )
        assert response.status_code == 201, response.text
        data = response.json()
        cleanup_inventories.append(data["id"])

        assert data["source"] == source_inventory["source"]
        assert data["checksum"] == source_inventory["checksum"]
        assert data["columns"] == source_inventory["columns"]
        assert data["georeference"] == EXAMPLE_GEOREF

    def test_duplicate_clones_firestore_document(
        self,
        client,
        firestore_client,
        domain_for_testing,
        source_inventory,
        cleanup_inventories,
    ):
        """The new Firestore doc clones the transform list, checksum, source, and
        owner verbatim, with fresh identity and equal create/modify timestamps."""
        response = client.post(
            self.route(domain_for_testing["id"], source_inventory["id"])
        )
        assert response.status_code == 201, response.text
        new_id = response.json()["id"]
        cleanup_inventories.append(new_id)

        new_doc = (
            firestore_client.collection(INVENTORIES_COLLECTION)
            .document(new_id)
            .get()
            .to_dict()
        )
        assert new_doc["id"] == new_id
        assert new_doc["checksum"] == source_inventory["checksum"]
        assert new_doc["source"] == source_inventory["source"]
        assert new_doc["modifications"] == source_inventory["modifications"]
        assert new_doc["treatments"] == source_inventory["treatments"]
        assert new_doc["columns"] == source_inventory["columns"]
        assert new_doc["owner_id"] == source_inventory["owner_id"]
        # Fresh, equal create/modify timestamps at write time. Assert on the
        # response — not the read-back doc — because the background copy bumps
        # modified_on when it completes, which races this read.
        assert response.json()["created_on"] == response.json()["modified_on"]

    def test_duplicate_name_override(
        self, client, domain_for_testing, source_inventory, cleanup_inventories
    ):
        """A supplied name overrides; omitted metadata is carried over."""
        response = client.post(
            self.route(domain_for_testing["id"], source_inventory["id"]),
            json={"name": "Scenario B"},
        )
        assert response.status_code == 201, response.text
        data = response.json()
        cleanup_inventories.append(data["id"])

        assert data["name"] == "Scenario B"
        assert data["description"] == source_inventory["description"]
        assert data["tags"] == source_inventory["tags"]

    def test_duplicate_full_metadata_override(
        self, client, domain_for_testing, source_inventory, cleanup_inventories
    ):
        """All three metadata fields can be overridden at once."""
        response = client.post(
            self.route(domain_for_testing["id"], source_inventory["id"]),
            json={
                "name": "Copy",
                "description": "A copy",
                "tags": ["copy", "branch"],
            },
        )
        assert response.status_code == 201, response.text
        data = response.json()
        cleanup_inventories.append(data["id"])

        assert data["name"] == "Copy"
        assert data["description"] == "A copy"
        assert data["tags"] == ["copy", "branch"]

    def test_duplicate_no_body_carries_over_metadata(
        self, client, domain_for_testing, source_inventory, cleanup_inventories
    ):
        """With no request body, name/description/tags are carried over verbatim."""
        response = client.post(
            self.route(domain_for_testing["id"], source_inventory["id"])
        )
        assert response.status_code == 201, response.text
        data = response.json()
        cleanup_inventories.append(data["id"])

        assert data["name"] == source_inventory["name"]
        assert data["description"] == source_inventory["description"]
        assert data["tags"] == source_inventory["tags"]

    def test_duplicate_nonexistent_source_returns_404(self, client, domain_for_testing):
        """Duplicating a non-existent inventory returns 404."""
        response = client.post(
            self.route(domain_for_testing["id"], "00000000000000000000000000000000")
        )
        assert response.status_code == 404

    def test_duplicate_pending_source_returns_422(
        self, client, firestore_client, domain_for_testing
    ):
        """A source that exists but is not completed cannot be duplicated (422)."""
        inv_data = make_inventory_data(
            domain_id=domain_for_testing["id"],
            name="Pending source",
            status="pending",
        )
        doc_ref = firestore_client.collection(INVENTORIES_COLLECTION).document(
            inv_data["id"]
        )
        doc_ref.set(inv_data)
        try:
            response = client.post(self.route(domain_for_testing["id"], inv_data["id"]))
            assert response.status_code == 422
        finally:
            doc_ref.delete()

    def test_duplicate_wrong_owner_returns_404(
        self, client, firestore_client, domain_with_different_owner
    ):
        """A source owned by another user cannot be duplicated (404)."""
        inv_data = make_inventory_data(
            domain_id=domain_with_different_owner["id"],
            owner_id="different-owner",
            name="Other user's inventory",
            status="completed",
        )
        doc_ref = firestore_client.collection(INVENTORIES_COLLECTION).document(
            inv_data["id"]
        )
        doc_ref.set(inv_data)
        try:
            response = client.post(
                self.route(domain_with_different_owner["id"], inv_data["id"])
            )
            assert response.status_code == 404
        finally:
            doc_ref.delete()

    def test_duplicate_source_in_other_domain_returns_404(
        self, client, second_domain, source_inventory
    ):
        """A source in a different domain than the path domain returns 404."""
        response = client.post(self.route(second_domain["id"], source_inventory["id"]))
        assert response.status_code == 404


def _poll_inventory(client, domain_id, inventory_id, timeout=60) -> dict:
    """Poll GET inventory until terminal status; return the final doc."""
    url = f"/domains/{domain_id}/inventories/{inventory_id}"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        r = client.get(url)
        assert r.status_code == 200, r.text
        doc = r.json()
        if doc["status"] in ("completed", "failed"):
            return doc
        time.sleep(1)
    pytest.fail(f"Duplicate did not reach a terminal status within {timeout}s")


class TestDuplicateInventoryCopiesData:
    """End-to-end test of the background GCS parquet copy. Requires GCS access."""

    def route(self, domain_id, inventory_id):
        return f"/domains/{domain_id}/inventories/{inventory_id}/duplicate"

    def test_duplicate_copies_parquet_and_completes(
        self, client, firestore_client, domain_for_testing
    ):
        """Staged source parquet is byte-copied to the new path; status -> completed."""
        source = make_inventory_data(
            domain_id=domain_for_testing["id"],
            name="Source with data",
            status="completed",
            source=dict(EXAMPLE_SOURCE),
            georeference=dict(EXAMPLE_GEOREF),
        )
        source["checksum"] = uuid.uuid4().hex
        source_id = source["id"]
        new_id = None

        try:
            # Stage a small parquet directory (with _metadata) at the source path.
            pdf = pd.DataFrame(
                {
                    "x": [1.0, 2.0, 3.0],
                    "y": [4.0, 5.0, 6.0],
                    "height": [10.0, 15.0, 20.0],
                }
            )
            ddf = dd.from_pandas(pdf, npartitions=1)
            ddf.to_parquet(
                f"gs://{INVENTORIES_BUCKET}/{source_id}", write_metadata_file=True
            )

            firestore_client.collection(INVENTORIES_COLLECTION).document(source_id).set(
                source
            )

            response = client.post(self.route(domain_for_testing["id"], source_id))
            assert response.status_code == 201, response.text
            new_id = response.json()["id"]
            assert response.json()["status"] == "pending"

            completed = _poll_inventory(client, domain_for_testing["id"], new_id)
            assert completed["status"] == "completed", completed.get("error")

            # The copied parquet exists at the new path and matches the source.
            copied = dd.read_parquet(f"gs://{INVENTORIES_BUCKET}/{new_id}").compute()
            assert sorted(copied["height"].tolist()) == [10.0, 15.0, 20.0]

            # Data endpoints work on the copy.
            meta = client.get(
                f"/domains/{domain_for_testing['id']}/inventories/{new_id}/data/metadata"
            )
            assert meta.status_code == 200, meta.text
            assert meta.json()["total_rows"] == 3
        finally:
            for inv_id in (source_id, new_id):
                if inv_id is None:
                    continue
                gcs_path = f"gs://{INVENTORIES_BUCKET}/{inv_id}"
                if exists(gcs_path):
                    delete_directory(gcs_path)
                firestore_client.collection(INVENTORIES_COLLECTION).document(
                    inv_id
                ).delete()

    def test_duplicate_without_source_data_fails(
        self, client, firestore_client, domain_for_testing, cleanup_inventories
    ):
        """A completed doc with no backing GCS data fails the copy verification:
        the duplicate transitions to failed with a structured error instead of
        completing as an empty clone."""
        source = make_inventory_data(
            domain_id=domain_for_testing["id"],
            name="Source without data",
            status="completed",
            source=dict(EXAMPLE_SOURCE),
            georeference=dict(EXAMPLE_GEOREF),
        )
        source["checksum"] = uuid.uuid4().hex
        source_id = source["id"]
        doc_ref = firestore_client.collection(INVENTORIES_COLLECTION).document(
            source_id
        )
        doc_ref.set(source)

        try:
            response = client.post(self.route(domain_for_testing["id"], source_id))
            assert response.status_code == 201, response.text
            new_id = response.json()["id"]
            cleanup_inventories.append(new_id)

            final = _poll_inventory(client, domain_for_testing["id"], new_id)
            assert final["status"] == "failed"
            assert final["error"]["code"] == "INVENTORY_DUPLICATE_COPY_FAILED"
        finally:
            doc_ref.delete()
