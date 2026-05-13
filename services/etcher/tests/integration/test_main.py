"""
Integration tests for the feature orchestrator (main.py).

Tests the full process_feature_request path end-to-end with real
Firestore and GCS. Verifies status transitions, error handling, and
cleanup when using real infrastructure.
"""

from uuid import uuid4

import pytest

from lib.config import (
    DOMAINS_COLLECTION,
    FEATURES_BUCKET,
    FEATURES_COLLECTION,
)
from lib.firestore.documents import delete_document, get_document, set_document
from lib.gcs.blobs import delete_file, exists

from .conftest import (
    DOMAINS_DIR,
    FEATURES_DIR,
    MockRequest,
    _stringify_coordinates,
    load_json,
)


@pytest.fixture
def firestore_domain():
    """Create a domain in Firestore with cleanup."""
    created_ids = []

    def _create(domain_file: str = "naip_chm_2_tiles.json") -> str:
        domain_data = load_json(DOMAINS_DIR / domain_file)
        domain_id = f"test-{uuid4().hex}"
        data = _stringify_coordinates(domain_data)
        data["id"] = domain_id
        set_document(DOMAINS_COLLECTION, domain_id, data)
        created_ids.append(domain_id)
        return domain_id

    yield _create

    for domain_id in created_ids:
        delete_document(DOMAINS_COLLECTION, domain_id)


@pytest.fixture
def firestore_feature():
    """Create a feature in Firestore with cleanup (GCS + Firestore)."""
    created_ids = []

    def _create(domain_id: str, feature_file: str = "road_osm.json") -> str:
        feature_id = f"test-{uuid4().hex}"
        data = load_json(FEATURES_DIR / feature_file)
        data["id"] = feature_id
        data["domain_id"] = domain_id
        set_document(FEATURES_COLLECTION, feature_id, data)
        created_ids.append((domain_id, feature_id))
        return feature_id

    yield _create

    for domain_id, feature_id in created_ids:
        # Clean up GCS GeoJSON if it exists
        gcs_path = f"gs://{FEATURES_BUCKET}/{domain_id}/{feature_id}.geojson"
        if exists(gcs_path):
            delete_file(gcs_path)
        delete_document(FEATURES_COLLECTION, feature_id)


class TestProcessFeatureRequest:
    """End-to-end tests for the full orchestrator."""

    def test_status_transitions(self, firestore_domain, firestore_feature):
        """Feature status transitions: pending -> running -> completed."""
        from etcher.main import process_feature_request

        domain_id = firestore_domain()
        feature_id = firestore_feature(domain_id)

        # Verify initial status
        _, snapshot = get_document(FEATURES_COLLECTION, feature_id)
        assert snapshot.to_dict()["status"] == "pending"

        # Run feature job
        request = MockRequest(data={"id": feature_id})
        response, status_code = process_feature_request(request)

        assert status_code == 200
        assert response == "OK"

        # Verify final status
        _, snapshot = get_document(FEATURES_COLLECTION, feature_id)
        feature = snapshot.to_dict()
        assert feature["status"] == "completed"
        assert feature["georeference"] is not None

    def test_geojson_written_to_gcs(self, firestore_domain, firestore_feature):
        """Processing should write a GeoJSON file to GCS."""
        from etcher.main import process_feature_request

        domain_id = firestore_domain()
        feature_id = firestore_feature(domain_id)

        request = MockRequest(data={"id": feature_id})
        process_feature_request(request)

        gcs_path = f"gs://{FEATURES_BUCKET}/{domain_id}/{feature_id}.geojson"
        assert exists(gcs_path), f"Expected GeoJSON data at {gcs_path}"

    def test_missing_domain_marks_failed(self, firestore_feature):
        """Referencing a nonexistent domain should mark feature as failed."""
        from etcher.main import process_feature_request

        # Create feature pointing to a domain that doesn't exist
        fake_domain_id = f"test-nonexistent-{uuid4().hex}"
        feature_id = firestore_feature(fake_domain_id)

        request = MockRequest(data={"id": feature_id})
        response, status_code = process_feature_request(request)

        assert status_code == 200

        _, snapshot = get_document(FEATURES_COLLECTION, feature_id)
        feature = snapshot.to_dict()
        assert feature["status"] == "failed"
        assert feature["error"]["code"] == "DOMAIN_NOT_FOUND"

    def test_deleted_feature_returns_ok(self):
        """Processing a deleted feature should return 200 gracefully."""
        from etcher.main import process_feature_request

        request = MockRequest(data={"id": f"test-nonexistent-{uuid4().hex}"})
        response, status_code = process_feature_request(request)

        assert status_code == 200

    def test_progress_updates(self, firestore_domain, firestore_feature):
        """After completion, progress should show 100%."""
        from etcher.main import process_feature_request

        domain_id = firestore_domain()
        feature_id = firestore_feature(domain_id)

        request = MockRequest(data={"id": feature_id})
        process_feature_request(request)

        _, snapshot = get_document(FEATURES_COLLECTION, feature_id)
        feature = snapshot.to_dict()
        assert feature["progress"]["percent"] == 100
        assert feature["progress"]["message"] == "Complete"
