"""
Integration tests for api/v2/resources/grids/upload/geotiff/router.py

Tests the GeoTIFF upload grid creation endpoint.
These tests make real HTTP requests to the API and interact with Firestore.
"""

import tempfile
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
import rasterio
import requests
from api.resources.grids.upload.geotiff.examples import (
    ALL_GRID_UPLOAD_EXAMPLE_VALUES,
)
from rasterio.crs import CRS
from rasterio.transform import from_bounds

from lib.config import GRIDS_BUCKET, GRIDS_COLLECTION, UPLOADS_BUCKET
from lib.gcs import exists

SINGLE_BAND = [{"key": "fbfm", "type": "categorical"}]
MULTI_BAND = [
    {"key": "bulk_density.foliage", "type": "continuous", "unit": "kg/m**3"},
    {"key": "bulk_density.branchwood", "type": "continuous", "unit": "kg/m**3"},
]

# GeoTIFF bounds inside domain_for_testing extent (x=[500000,501000], y=[5200000,5201000])
_TIFF_XMIN, _TIFF_YMIN = 500100.0, 5200100.0
_TIFF_XMAX, _TIFF_YMAX = 500900.0, 5200900.0


class TestCreateGridUpload:
    """Test the POST /domains/{domain_id}/grids/upload/geotiff endpoint."""

    def route(self, domain_id):
        return f"/domains/{domain_id}/grids/upload/geotiff"

    def test_minimal_request_returns_201(
        self, client, domain_for_testing, firestore_client
    ):
        """Minimal request returns 201 with grid and upload spec."""
        response = client.post(
            self.route(domain_for_testing["id"]),
            json={"bands": SINGLE_BAND},
        )
        assert response.status_code == 201

        data = response.json()
        assert "grid" in data
        assert "upload" in data

        grid = data["grid"]
        assert len(grid["id"]) == 32
        assert grid["domain_id"] == domain_for_testing["id"]
        assert grid["status"] == "pending"
        assert grid["georeference"] is None
        assert grid["chunks"] is None
        assert grid["source"]["name"] == "upload"
        assert grid["source"]["format"] == "geotiff"
        assert "upload.tif" in grid["source"]["object_name"]

        assert len(grid["bands"]) == 1
        assert grid["bands"][0]["key"] == "fbfm"
        assert grid["bands"][0]["type"] == "categorical"
        assert grid["bands"][0]["index"] == 0

        up = data["upload"]
        assert up["method"] == "PUT"
        assert up["content_type"] == "image/tiff"
        assert len(up["url"]) > 0
        assert up["max_size_bytes"] == 1_073_741_824
        expires_at = datetime.fromisoformat(up["expires_at"])
        assert expires_at > datetime.now(UTC) + timedelta(minutes=50)

        firestore_client.collection(GRIDS_COLLECTION).document(grid["id"]).delete()

    def test_multi_band_request(self, client, domain_for_testing, firestore_client):
        """Multi-band request stores all bands with correct indices."""
        response = client.post(
            self.route(domain_for_testing["id"]),
            json={"bands": MULTI_BAND},
        )
        assert response.status_code == 201

        grid = response.json()["grid"]
        assert len(grid["bands"]) == 2
        assert grid["bands"][0]["key"] == "bulk_density.foliage"
        assert grid["bands"][0]["index"] == 0
        assert grid["bands"][1]["key"] == "bulk_density.branchwood"
        assert grid["bands"][1]["index"] == 1

        firestore_client.collection(GRIDS_COLLECTION).document(grid["id"]).delete()

    def test_metadata_stored(self, client, domain_for_testing, firestore_client):
        """Name, description, and tags are stored in the grid."""
        response = client.post(
            self.route(domain_for_testing["id"]),
            json={
                "bands": SINGLE_BAND,
                "name": "Custom FBFM40",
                "description": "Derived from LiDAR",
                "tags": ["lidar", "2024"],
            },
        )
        assert response.status_code == 201

        grid = response.json()["grid"]
        assert grid["name"] == "Custom FBFM40"
        assert grid["description"] == "Derived from LiDAR"
        assert grid["tags"] == ["lidar", "2024"]

        firestore_client.collection(GRIDS_COLLECTION).document(grid["id"]).delete()

    def test_response_excludes_owner_id(
        self, client, domain_for_testing, firestore_client
    ):
        """owner_id must not appear in the response."""
        response = client.post(
            self.route(domain_for_testing["id"]),
            json={"bands": SINGLE_BAND},
        )
        assert response.status_code == 201
        assert "owner_id" not in response.json()["grid"]

        firestore_client.collection(GRIDS_COLLECTION).document(
            response.json()["grid"]["id"]
        ).delete()

    def test_invalid_domain_returns_404(self, client):
        """Non-existent domain returns 404."""
        response = client.post(
            self.route("00000000000000000000000000000000"),
            json={"bands": SINGLE_BAND},
        )
        assert response.status_code == 404

    def test_wrong_owner_domain_returns_404(self, client, domain_with_different_owner):
        """Domain owned by a different user returns 404."""
        response = client.post(
            self.route(domain_with_different_owner["id"]),
            json={"bands": SINGLE_BAND},
        )
        assert response.status_code == 404

    def test_empty_bands_returns_422(self, client, domain_for_testing):
        """Empty bands list returns 422."""
        response = client.post(
            self.route(domain_for_testing["id"]),
            json={"bands": []},
        )
        assert response.status_code == 422

    def test_missing_bands_returns_422(self, client, domain_for_testing):
        """Missing bands field returns 422."""
        response = client.post(
            self.route(domain_for_testing["id"]),
            json={},
        )
        assert response.status_code == 422

    def test_invalid_band_type_returns_422(self, client, domain_for_testing):
        """Invalid band type returns 422."""
        response = client.post(
            self.route(domain_for_testing["id"]),
            json={"bands": [{"key": "fbfm", "type": "nominal"}]},
        )
        assert response.status_code == 422

    @pytest.mark.parametrize(
        "example_name,example_value", ALL_GRID_UPLOAD_EXAMPLE_VALUES
    )
    def test_openapi_examples_return_201(
        self, example_name, example_value, client, domain_for_testing, firestore_client
    ):
        """Every OpenAPI example must be accepted by the endpoint."""
        response = client.post(
            self.route(domain_for_testing["id"]),
            json=example_value,
        )
        assert response.status_code == 201, f"{example_name}: {response.text}"

        grid_id = response.json()["grid"]["id"]
        firestore_client.collection(GRIDS_COLLECTION).document(grid_id).delete()


def _poll_grid(client, domain_id, grid_id, timeout=120) -> dict:
    """Poll GET grid until terminal status, return the final doc."""
    url = f"/domains/{domain_id}/grids/{grid_id}"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        r = client.get(url)
        assert r.status_code == 200, r.text
        doc = r.json()
        if doc["status"] == "completed":
            return doc
        if doc["status"] == "failed":
            pytest.fail(f"Grid failed: {doc.get('error')}")
        time.sleep(1)
    pytest.fail(f"Grid did not complete within {timeout}s")


def _make_geotiff_file(domain_crs: str, n_bands: int = 1) -> Path:
    """Write a small GeoTIFF in domain_crs to a temp file and return the path."""
    width, height = 20, 20
    transform = from_bounds(
        _TIFF_XMIN, _TIFF_YMIN, _TIFF_XMAX, _TIFF_YMAX, width, height
    )
    epsg = int(domain_crs.split(":")[1])
    f = tempfile.NamedTemporaryFile(suffix=".tif", delete=False)
    f.close()
    with rasterio.open(
        f.name,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=n_bands,
        crs=CRS.from_epsg(epsg),
        transform=transform,
        dtype="float32",
    ) as dst:
        for b in range(1, n_bands + 1):
            dst.write(np.full((height, width), float(b), dtype="float32"), b)
    return Path(f.name)


class TestUploadFullFlow:
    """Full upload flow: POST → PUT file → Eventarc → uploader → completed.

    Requires the uploader Cloud Run service to be deployed and wired to Eventarc.
    """

    def test_single_band_upload_completes(
        self, client, domain_for_testing, firestore_client
    ):
        """POST → PUT GeoTIFF to signed URL → poll → status=completed, Zarr exists."""
        domain_id = domain_for_testing["id"]
        domain_crs = domain_for_testing["crs"]["properties"]["name"]
        grid_id = None

        try:
            response = client.post(
                f"/domains/{domain_id}/grids/upload/geotiff",
                json={"bands": SINGLE_BAND, "name": "E2E Test Grid"},
            )
            assert response.status_code == 201, response.text
            data = response.json()
            grid_id = data["grid"]["id"]
            object_name = data["grid"]["source"]["object_name"]
            signed_url = data["upload"]["url"]
            content_type = data["upload"]["content_type"]
            max_size_bytes = data["upload"]["max_size_bytes"]

            tiff_path = _make_geotiff_file(domain_crs, n_bands=1)
            try:
                with open(tiff_path, "rb") as f:
                    put_response = requests.put(
                        signed_url,
                        data=f,
                        headers={
                            "Content-Type": content_type,
                            "x-goog-content-length-range": f"0,{max_size_bytes}",
                        },
                        timeout=30,
                    )
                assert put_response.status_code == 200, put_response.text
            finally:
                tiff_path.unlink(missing_ok=True)

            assert exists(f"gs://{UPLOADS_BUCKET}/{object_name}")

            completed = _poll_grid(client, domain_id, grid_id)

            assert not exists(f"gs://{UPLOADS_BUCKET}/{object_name}")
            assert completed["georeference"]["crs"] == domain_crs
            assert len(completed["georeference"]["transform"]) == 6
            assert len(completed["georeference"]["shape"]) == 2
            assert completed["chunks"] is not None
            assert completed["progress"]["percent"] == 100
            assert exists(f"gs://{GRIDS_BUCKET}/{grid_id}")

        finally:
            if grid_id:
                from lib.gcs import delete_directory

                gcs_path = f"gs://{GRIDS_BUCKET}/{grid_id}"
                if exists(gcs_path):
                    delete_directory(gcs_path)
                firestore_client.collection(GRIDS_COLLECTION).document(grid_id).delete()
