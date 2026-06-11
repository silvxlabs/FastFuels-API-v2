"""
Integration tests for api/v2/resources/point_clouds/upload/router.py

POSTs to the upload endpoint create a pending point cloud and return a signed PUT
URL. These make real HTTP requests to the API and hit Firestore + the GCS signer;
each created document is cleaned up.
"""

import pytest
from api.resources.point_clouds.upload.examples import ALL_UPLOAD_EXAMPLE_VALUES

from lib.config import POINT_CLOUDS_COLLECTION


class TestCreatePointCloudUpload:
    def route(self, domain_id):
        return f"/domains/{domain_id}/pointclouds/upload"

    def test_creates_pending_and_returns_signed_url(
        self, client, firestore_client, domain_for_testing
    ):
        body = {
            "type": "tls",
            "name": "Plot 3 TLS",
            "tags": ["plot-3"],
        }
        response = client.post(self.route(domain_for_testing["id"]), json=body)
        assert response.status_code == 201, response.text
        data = response.json()
        pc = data["point_cloud"]
        upload = data["upload"]

        try:
            assert pc["status"] == "pending"
            assert pc["type"] == "tls"
            assert pc["name"] == "Plot 3 TLS"
            assert pc["georeference"] is None
            assert pc["summary"] is None
            assert pc["source"]["name"] == "upload"
            assert pc["source"]["object_name"] == f"pointclouds/{pc['id']}/upload"

            assert upload["method"] == "PUT"
            assert upload["content_type"] == "application/octet-stream"
            assert upload["url"].startswith("https://")
            assert upload["max_size_bytes"] > 0
            assert "expires_at" in upload

            doc = (
                firestore_client.collection(POINT_CLOUDS_COLLECTION)
                .document(pc["id"])
                .get()
            )
            assert doc.exists
            assert doc.to_dict()["status"] == "pending"
        finally:
            firestore_client.collection(POINT_CLOUDS_COLLECTION).document(
                pc["id"]
            ).delete()

    @pytest.mark.parametrize("example_name,body", ALL_UPLOAD_EXAMPLE_VALUES)
    def test_documented_examples_are_accepted(
        self, client, firestore_client, domain_for_testing, example_name, body
    ):
        response = client.post(self.route(domain_for_testing["id"]), json=body)
        assert response.status_code == 201, f"{example_name}: {response.text}"
        pc_id = response.json()["point_cloud"]["id"]
        firestore_client.collection(POINT_CLOUDS_COLLECTION).document(pc_id).delete()

    def test_invalid_type_rejected(self, client, domain_for_testing):
        response = client.post(
            self.route(domain_for_testing["id"]),
            json={"type": "mls", "name": "Mobile scan"},
        )
        assert response.status_code == 422

    def test_missing_type_rejected(self, client, domain_for_testing):
        response = client.post(
            self.route(domain_for_testing["id"]), json={"name": "No type"}
        )
        assert response.status_code == 422
