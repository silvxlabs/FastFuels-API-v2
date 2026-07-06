"""Integration tests against live Firestore + GCS.

Scoped to freshly-seeded ``test-{uuid}`` resources — these exercise walle's
real-IO primitives (GCS artifact deletion, owner-doc TTL resolution, the
projected scan) without running the full bucket-wide ``run()`` pass, which would
reconcile real project data.
"""

import uuid

import pytest
from walle import cleanup
from walle.layouts import RESOURCE_LAYOUTS, delete_artifact, list_artifact_ids

from lib.config import APPLICATIONS_COLLECTION, GRIDS_BUCKET, GRIDS_COLLECTION
from lib.firestore.documents import firestore_client
from lib.gcs.blobs import exists, get_gcsfs_client


@pytest.fixture
def grid_layout():
    return next(x for x in RESOURCE_LAYOUTS if x.name == "grids")


def test_delete_artifact_removes_prefix():
    grid_id = f"test-{uuid.uuid4().hex}"
    prefix = f"{GRIDS_BUCKET}/{grid_id}"
    fs = get_gcsfs_client()
    fs.pipe(f"{prefix}/part-0", b"hello")
    fs.pipe(f"{prefix}/part-1", b"world")
    fs.invalidate_cache(prefix)
    assert exists(prefix)

    delete_artifact(prefix)

    fs.invalidate_cache(prefix)
    assert not exists(prefix)


def test_delete_artifact_missing_is_noop():
    # Idempotency: reaping an already-gone artifact must not raise.
    delete_artifact(f"{GRIDS_BUCKET}/test-{uuid.uuid4().hex}")


def test_list_artifact_ids_finds_seeded_prefix(grid_layout):
    grid_id = f"test-{uuid.uuid4().hex}"
    prefix = f"{GRIDS_BUCKET}/{grid_id}"
    fs = get_gcsfs_client()
    fs.pipe(f"{prefix}/chunk", b"x")
    try:
        ids = list_artifact_ids(grid_layout)
        assert grid_id in ids
        assert ids[grid_id].rstrip("/").endswith(grid_id)
    finally:
        fs.rm(prefix, recursive=True)


def test_resolve_owner_ttls_reads_application_doc():
    owner_id = f"test-{uuid.uuid4().hex}"
    ref = firestore_client.collection(APPLICATIONS_COLLECTION).document(owner_id)
    ref.set({"tier": "application"})
    try:
        cleanup.resolve_owner_ttls.cache_clear()
        assert cleanup.resolve_owner_ttls(owner_id) == (None, 14)
    finally:
        ref.delete()
        cleanup.resolve_owner_ttls.cache_clear()


def test_resolve_owner_ttls_absent_owner_is_standard():
    cleanup.resolve_owner_ttls.cache_clear()
    assert cleanup.resolve_owner_ttls(f"test-{uuid.uuid4().hex}") == (180, 14)


def test_scan_collection_projects_seeded_doc(grid_layout):
    grid_id = f"test-{uuid.uuid4().hex}"
    ref = firestore_client.collection(GRIDS_COLLECTION).document(grid_id)
    ref.set(
        {
            "domain_id": "test-domain",
            "owner_id": "test-owner",
            "status": "completed",
            "size_bytes": 4242,
        }
    )
    try:
        record = next(
            r for r in cleanup.scan_collection(grid_layout) if r.doc_id == grid_id
        )
        assert record.domain_id == "test-domain"
        assert record.owner_id == "test-owner"
        assert record.status == "completed"
        assert record.size_bytes == 4242
    finally:
        ref.delete()
