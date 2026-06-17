"""
Integration tests for the grid duplicate endpoint.

POST /domains/{domain_id}/grids/{grid_id}/duplicate

These tests make real HTTP requests to the running API and interact with
Firestore. The router-level tests in TestDuplicateGrid assert on the
synchronous 201 response and the cloned Firestore document; the background GCS
zarr copy (and its listing verification) is exercised end-to-end by
TestDuplicateGridCopiesData, which stages a real zarr on GCS.
"""

import json
import time
import uuid

import numpy as np
import pytest
import xarray as xr

from lib.config import GRIDS_BUCKET, GRIDS_COLLECTION
from lib.gcs import delete_directory, exists
from tests.fixtures import make_grid_data

# A completed source grid carrying every field a duplicate should clone. The
# inline-geometry modification is stored with stringified coordinates, exactly
# as the create routers write it to Firestore.
EXAMPLE_MODIFICATION = {
    "conditions": [
        {"band": "fbfm", "operator": "eq", "value": 91},
        {
            "source": "geometry",
            "operator": "within",
            "geometry": {
                "type": "Polygon",
                "coordinates": json.dumps(
                    [
                        [
                            [500000.0, 5200000.0],
                            [500100.0, 5200000.0],
                            [500100.0, 5200100.0],
                            [500000.0, 5200000.0],
                        ]
                    ]
                ),
            },
            "crs": None,
            "buffer_m": None,
            "target": "centroid",
        },
    ],
    "actions": [{"band": "fuel_load.1hr", "modifier": "multiply", "value": 0.5}],
}
EXAMPLE_CHUNKS = {"shape": [34, 34], "count": 1, "count_by_axis": {"y": 1, "x": 1}}


@pytest.fixture(scope="function")
def cleanup_grids(firestore_client):
    """Collect grid IDs created by a test; delete their Firestore docs and any
    GCS data on teardown."""
    created_ids: list[str] = []
    yield created_ids
    for grid_id in created_ids:
        firestore_client.collection(GRIDS_COLLECTION).document(grid_id).delete()
        gcs_path = f"gs://{GRIDS_BUCKET}/{grid_id}"
        if exists(gcs_path):
            delete_directory(gcs_path)


@pytest.fixture(scope="function")
def source_grid(firestore_client, domain_for_testing):
    """A completed source grid with a full set of carry-over fields."""
    grid_data = make_grid_data(
        domain_id=domain_for_testing["id"],
        name="Original Grid",
        description="The original",
        status="completed",
        tags=["original"],
        chunks=dict(EXAMPLE_CHUNKS),
    )
    grid_data["checksum"] = uuid.uuid4().hex
    grid_data["modifications"] = [json.loads(json.dumps(EXAMPLE_MODIFICATION))]
    doc_ref = firestore_client.collection(GRIDS_COLLECTION).document(grid_data["id"])
    doc_ref.set(grid_data)
    yield grid_data
    doc_ref.delete()


class TestDuplicateGrid:
    """Router-level tests for the duplicate endpoint (synchronous response)."""

    def route(self, domain_id, grid_id):
        return f"/domains/{domain_id}/grids/{grid_id}/duplicate"

    def test_duplicate_creates_new_pending_grid(
        self, client, domain_for_testing, source_grid, cleanup_grids
    ):
        """A duplicate is a fresh, distinct, pending grid."""
        response = client.post(self.route(domain_for_testing["id"], source_grid["id"]))
        assert response.status_code == 201, response.text
        data = response.json()
        cleanup_grids.append(data["id"])

        assert len(data["id"]) == 32
        assert data["id"] != source_grid["id"]
        assert data["domain_id"] == domain_for_testing["id"]
        assert data["status"] == "pending"
        assert data["error"] is None
        assert "owner_id" not in data

    def test_duplicate_carries_over_clone_fields(
        self, client, domain_for_testing, source_grid, cleanup_grids
    ):
        """source, checksum, bands, georeference, and chunks are carried over
        verbatim; stringified modification coordinates come back decoded."""
        response = client.post(self.route(domain_for_testing["id"], source_grid["id"]))
        assert response.status_code == 201, response.text
        data = response.json()
        cleanup_grids.append(data["id"])

        assert data["source"] == source_grid["source"]
        assert data["checksum"] == source_grid["checksum"]
        # The Band response model serializes optional fields as explicit Nones;
        # compare only the fields the minimal fixture doc carries.
        response_bands = [
            {k: band[k] for k in ("key", "type", "unit", "index")}
            for band in data["bands"]
        ]
        assert response_bands == source_grid["bands"]
        assert data["georeference"]["crs"] == source_grid["georeference"]["crs"]
        assert data["chunks"]["count"] == EXAMPLE_CHUNKS["count"]
        # The response decodes stringified inline-geometry coordinates.
        coords = data["modifications"][0]["conditions"][1]["geometry"]["coordinates"]
        assert isinstance(coords, list)
        assert coords[0][0] == [500000.0, 5200000.0]

    def test_duplicate_clones_firestore_document(
        self,
        client,
        firestore_client,
        domain_for_testing,
        source_grid,
        cleanup_grids,
    ):
        """The new Firestore doc clones source, checksum, owner, and the
        modifications ledger verbatim (coordinates still stringified), with
        fresh identity and equal create/modify timestamps."""
        response = client.post(self.route(domain_for_testing["id"], source_grid["id"]))
        assert response.status_code == 201, response.text
        data = response.json()
        new_id = data["id"]
        cleanup_grids.append(new_id)

        # Fresh, equal create/modify timestamps (both set to the request time).
        # Assert this on the synchronous response: the background copy task
        # updates the persisted doc's modified_on once it runs, so re-reading
        # the doc here would race that update.
        assert data["created_on"] == data["modified_on"]

        new_doc = (
            firestore_client.collection(GRIDS_COLLECTION)
            .document(new_id)
            .get()
            .to_dict()
        )
        assert new_doc["id"] == new_id
        assert new_doc["checksum"] == source_grid["checksum"]
        assert new_doc["source"] == source_grid["source"]
        assert new_doc["modifications"] == source_grid["modifications"]
        # Stored coordinates remain stringified, exactly as the source stored them.
        stored_coords = new_doc["modifications"][0]["conditions"][1]["geometry"][
            "coordinates"
        ]
        assert isinstance(stored_coords, str)
        assert new_doc["owner_id"] == source_grid["owner_id"]

    def test_duplicate_name_override(
        self, client, domain_for_testing, source_grid, cleanup_grids
    ):
        """A supplied name overrides; omitted metadata is carried over."""
        response = client.post(
            self.route(domain_for_testing["id"], source_grid["id"]),
            json={"name": "Scenario B"},
        )
        assert response.status_code == 201, response.text
        data = response.json()
        cleanup_grids.append(data["id"])

        assert data["name"] == "Scenario B"
        assert data["description"] == source_grid["description"]
        assert data["tags"] == source_grid["tags"]

    def test_duplicate_full_metadata_override(
        self, client, domain_for_testing, source_grid, cleanup_grids
    ):
        """All three metadata fields can be overridden at once."""
        response = client.post(
            self.route(domain_for_testing["id"], source_grid["id"]),
            json={
                "name": "Copy",
                "description": "A copy",
                "tags": ["copy", "branch"],
            },
        )
        assert response.status_code == 201, response.text
        data = response.json()
        cleanup_grids.append(data["id"])

        assert data["name"] == "Copy"
        assert data["description"] == "A copy"
        assert data["tags"] == ["copy", "branch"]

    def test_duplicate_no_body_carries_over_metadata(
        self, client, domain_for_testing, source_grid, cleanup_grids
    ):
        """With no request body, name/description/tags are carried over verbatim."""
        response = client.post(self.route(domain_for_testing["id"], source_grid["id"]))
        assert response.status_code == 201, response.text
        data = response.json()
        cleanup_grids.append(data["id"])

        assert data["name"] == source_grid["name"]
        assert data["description"] == source_grid["description"]
        assert data["tags"] == source_grid["tags"]

    def test_duplicate_nonexistent_source_returns_404(self, client, domain_for_testing):
        """Duplicating a non-existent grid returns 404."""
        response = client.post(
            self.route(domain_for_testing["id"], "00000000000000000000000000000000")
        )
        assert response.status_code == 404

    def test_duplicate_pending_source_returns_422(
        self, client, firestore_client, domain_for_testing
    ):
        """A source that exists but is not completed cannot be duplicated (422)."""
        grid_data = make_grid_data(
            domain_id=domain_for_testing["id"],
            name="Pending source",
            status="pending",
        )
        doc_ref = firestore_client.collection(GRIDS_COLLECTION).document(
            grid_data["id"]
        )
        doc_ref.set(grid_data)
        try:
            response = client.post(
                self.route(domain_for_testing["id"], grid_data["id"])
            )
            assert response.status_code == 422
        finally:
            doc_ref.delete()

    def test_duplicate_wrong_owner_returns_404(
        self, client, firestore_client, domain_with_different_owner
    ):
        """A source owned by another user cannot be duplicated (404)."""
        grid_data = make_grid_data(
            domain_id=domain_with_different_owner["id"],
            owner_id="different-owner",
            name="Other user's grid",
            status="completed",
        )
        doc_ref = firestore_client.collection(GRIDS_COLLECTION).document(
            grid_data["id"]
        )
        doc_ref.set(grid_data)
        try:
            response = client.post(
                self.route(domain_with_different_owner["id"], grid_data["id"])
            )
            assert response.status_code == 404
        finally:
            doc_ref.delete()

    def test_duplicate_source_in_other_domain_returns_404(
        self, client, second_domain, source_grid
    ):
        """A source in a different domain than the path domain returns 404."""
        response = client.post(self.route(second_domain["id"], source_grid["id"]))
        assert response.status_code == 404


def _poll_grid(client, domain_id, grid_id, timeout=60) -> dict:
    """Poll GET grid until terminal status; return the final doc."""
    url = f"/domains/{domain_id}/grids/{grid_id}"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        r = client.get(url)
        assert r.status_code == 200, r.text
        doc = r.json()
        if doc["status"] in ("completed", "failed"):
            return doc
        time.sleep(1)
    pytest.fail(f"Duplicate did not reach a terminal status within {timeout}s")


class TestDuplicateGridCopiesData:
    """End-to-end tests of the background GCS zarr copy. Requires GCS access."""

    def route(self, domain_id, grid_id):
        return f"/domains/{domain_id}/grids/{grid_id}/duplicate"

    def test_duplicate_copies_zarr_and_completes(
        self, client, firestore_client, domain_for_testing
    ):
        """Staged source zarr is byte-copied to the new path; status -> completed."""
        source = make_grid_data(
            domain_id=domain_for_testing["id"],
            name="Source with data",
            status="completed",
        )
        source["checksum"] = uuid.uuid4().hex
        source_id = source["id"]
        new_id = None

        try:
            # Stage a small zarr at the source path.
            values = np.arange(16, dtype=np.float32).reshape(4, 4)
            ds = xr.Dataset({"fbfm": (("y", "x"), values)})
            ds.to_zarr(f"gs://{GRIDS_BUCKET}/{source_id}", mode="w")

            firestore_client.collection(GRIDS_COLLECTION).document(source_id).set(
                source
            )

            response = client.post(self.route(domain_for_testing["id"], source_id))
            assert response.status_code == 201, response.text
            new_id = response.json()["id"]
            assert response.json()["status"] == "pending"

            final = _poll_grid(client, domain_for_testing["id"], new_id)
            assert final["status"] == "completed", final.get("error")

            # The copied zarr exists at the new path and matches the source.
            copied = xr.open_zarr(f"gs://{GRIDS_BUCKET}/{new_id}")
            np.testing.assert_array_equal(copied["fbfm"].values, values)
        finally:
            for grid_id in (source_id, new_id):
                if grid_id is None:
                    continue
                gcs_path = f"gs://{GRIDS_BUCKET}/{grid_id}"
                if exists(gcs_path):
                    delete_directory(gcs_path)
                firestore_client.collection(GRIDS_COLLECTION).document(grid_id).delete()

    def test_duplicate_without_source_data_fails(
        self, client, firestore_client, domain_for_testing, cleanup_grids
    ):
        """A completed doc with no backing GCS data fails the copy verification:
        the duplicate transitions to failed with a structured error instead of
        completing as an empty clone."""
        source = make_grid_data(
            domain_id=domain_for_testing["id"],
            name="Source without data",
            status="completed",
        )
        source_id = source["id"]
        doc_ref = firestore_client.collection(GRIDS_COLLECTION).document(source_id)
        doc_ref.set(source)

        try:
            response = client.post(self.route(domain_for_testing["id"], source_id))
            assert response.status_code == 201, response.text
            new_id = response.json()["id"]
            cleanup_grids.append(new_id)

            final = _poll_grid(client, domain_for_testing["id"], new_id)
            assert final["status"] == "failed"
            assert final["error"]["code"] == "GRID_DUPLICATE_COPY_FAILED"
        finally:
            doc_ref.delete()
