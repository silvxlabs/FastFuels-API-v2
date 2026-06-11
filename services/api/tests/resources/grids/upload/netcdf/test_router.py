"""
Integration tests for api/v2/resources/grids/upload/netcdf/router.py

Tests the netCDF upload grid creation endpoint. The pending-grid creation
path (POST → 201 with signed URL) does not require a netCDF file — only the
full PUT-and-process flow does. Full-flow tests live alongside the uploader
integration suite.
"""

from datetime import UTC, datetime, timedelta

import pytest
from api.resources.grids.upload.netcdf.examples import (
    ALL_NETCDF_UPLOAD_EXAMPLE_VALUES,
)

from lib.config import GRIDS_COLLECTION


class TestCreateNetcdfUpload:
    """Test the POST /domains/{domain_id}/grids/upload/netcdf endpoint."""

    def route(self, domain_id):
        return f"/domains/{domain_id}/grids/upload/netcdf"

    def test_minimal_request_returns_201(
        self, client, domain_for_testing, firestore_client
    ):
        response = client.post(self.route(domain_for_testing["id"]), json={})
        assert response.status_code == 201, response.text

        data = response.json()
        grid = data["grid"]
        assert grid["domain_id"] == domain_for_testing["id"]
        assert grid["status"] == "pending"
        assert grid["bands"] == []
        assert grid["georeference"] is None
        assert grid["chunks"] is None
        assert grid["source"]["name"] == "upload"
        assert grid["source"]["format"] == "netcdf"
        assert grid["source"]["object_name"].endswith("upload.nc")
        assert "bands" not in grid["source"]

        up = data["upload"]
        assert up["method"] == "PUT"
        assert up["content_type"] == "application/x-netcdf"
        assert up["max_size_bytes"] == 1_073_741_824
        assert up["headers"] == {
            "Content-Type": up["content_type"],
            "x-goog-content-length-range": f"0,{up['max_size_bytes']}",
        }
        # Every header the spec asks the client to send is one the URL signed.
        signed = up["url"].split("X-Goog-SignedHeaders=")[1].split("&")[0]
        for header_name in up["headers"]:
            assert header_name.lower() in signed.replace("%3B", ";").split(";")
        expires_at = datetime.fromisoformat(up["expires_at"])
        assert expires_at > datetime.now(UTC) + timedelta(minutes=50)

        firestore_client.collection(GRIDS_COLLECTION).document(grid["id"]).delete()

    def test_metadata_stored(self, client, domain_for_testing, firestore_client):
        response = client.post(
            self.route(domain_for_testing["id"]),
            json={
                "name": "Custom 3D grid",
                "description": "Voxelized bulk density",
                "tags": ["lidar"],
                "num_buffer_cells": 2,
            },
        )
        assert response.status_code == 201

        grid = response.json()["grid"]
        assert grid["name"] == "Custom 3D grid"
        assert grid["description"] == "Voxelized bulk density"
        assert grid["tags"] == ["lidar"]
        assert grid["source"]["num_buffer_cells"] == 2

        firestore_client.collection(GRIDS_COLLECTION).document(grid["id"]).delete()

    def test_response_excludes_owner_id(
        self, client, domain_for_testing, firestore_client
    ):
        response = client.post(self.route(domain_for_testing["id"]), json={})
        assert response.status_code == 201
        assert "owner_id" not in response.json()["grid"]

        firestore_client.collection(GRIDS_COLLECTION).document(
            response.json()["grid"]["id"]
        ).delete()

    def test_invalid_domain_returns_404(self, client):
        response = client.post(self.route("00000000000000000000000000000000"), json={})
        assert response.status_code == 404

    def test_wrong_owner_domain_returns_404(self, client, domain_with_different_owner):
        response = client.post(self.route(domain_with_different_owner["id"]), json={})
        assert response.status_code == 404

    def test_negative_buffer_returns_422(self, client, domain_for_testing):
        response = client.post(
            self.route(domain_for_testing["id"]),
            json={"num_buffer_cells": -1},
        )
        assert response.status_code == 422

    @pytest.mark.parametrize(
        "example_name,example_value", ALL_NETCDF_UPLOAD_EXAMPLE_VALUES
    )
    def test_openapi_examples_return_201(
        self,
        example_name,
        example_value,
        client,
        domain_for_testing,
        firestore_client,
    ):
        response = client.post(
            self.route(domain_for_testing["id"]),
            json=example_value,
        )
        assert response.status_code == 201, f"{example_name}: {response.text}"

        grid_id = response.json()["grid"]["id"]
        firestore_client.collection(GRIDS_COLLECTION).document(grid_id).delete()
