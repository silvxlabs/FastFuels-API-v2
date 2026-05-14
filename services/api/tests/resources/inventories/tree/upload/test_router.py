"""
Integration tests for api/v2/resources/inventories/tree/upload/router.py

Tests the upload inventory creation endpoint.
These tests make real HTTP requests to the API and interact with Firestore.
"""

import tempfile
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import dask.dataframe as dd
import geopandas as gpd
import pandas as pd
import pytest
import requests
from shapely.geometry import Point

from lib.config import INVENTORIES_BUCKET, INVENTORIES_COLLECTION, UPLOADS_BUCKET
from lib.gcs import delete_directory, exists


class TestCreateInventoryUpload:
    """Test the POST /domains/{domain_id}/inventories/tree/upload endpoint."""

    def route(self, domain_id):
        return f"/domains/{domain_id}/inventories/tree/upload"

    def test_minimal_csv_request_returns_201(
        self, client, domain_for_testing, firestore_client
    ):
        """Minimal CSV request returns 201 with inventory and upload spec."""
        response = client.post(
            self.route(domain_for_testing["id"]),
            json={"format": "csv"},
        )
        assert response.status_code == 201

        data = response.json()
        assert "inventory" in data
        assert "upload" in data

        inv = data["inventory"]
        assert len(inv["id"]) == 32
        assert inv["domain_id"] == domain_for_testing["id"]
        assert inv["type"] == "tree"
        assert inv["status"] == "pending"
        assert inv["georeference"] is None
        assert inv["source"]["name"] == "upload"
        assert inv["source"]["format"] == "csv"
        assert "object_name" in inv["source"]
        assert inv["source"]["object_name"].endswith(".csv")

        up = data["upload"]
        assert up["method"] == "PUT"
        assert up["content_type"] == "text/csv"
        assert len(up["url"]) > 0
        assert up["max_size_bytes"] == 524_288_000
        expires_at = datetime.fromisoformat(up["expires_at"])
        assert expires_at > datetime.now(UTC) + timedelta(minutes=50)

        # Teardown
        firestore_client.collection(INVENTORIES_COLLECTION).document(inv["id"]).delete()

    def test_geojson_format_sets_content_type(
        self, client, domain_for_testing, firestore_client
    ):
        """GeoJSON format returns correct content type and file extension."""
        response = client.post(
            self.route(domain_for_testing["id"]),
            json={"format": "geojson"},
        )
        assert response.status_code == 201

        data = response.json()
        assert data["upload"]["content_type"] == "application/geo+json"
        assert data["inventory"]["source"]["object_name"].endswith(".geojson")

        firestore_client.collection(INVENTORIES_COLLECTION).document(
            data["inventory"]["id"]
        ).delete()

    def test_geopackage_format_sets_content_type(
        self, client, domain_for_testing, firestore_client
    ):
        """GeoPackage format returns correct content type and file extension."""
        response = client.post(
            self.route(domain_for_testing["id"]),
            json={"format": "geopackage"},
        )
        assert response.status_code == 201

        data = response.json()
        assert data["upload"]["content_type"] == "application/geopackage+sqlite3"
        assert data["inventory"]["source"]["object_name"].endswith(".gpkg")

        firestore_client.collection(INVENTORIES_COLLECTION).document(
            data["inventory"]["id"]
        ).delete()

    def test_column_mapping_stored_in_source(
        self, client, domain_for_testing, firestore_client
    ):
        """Column mapping is stored in source.columns."""
        response = client.post(
            self.route(domain_for_testing["id"]),
            json={
                "format": "csv",
                "columns": {"height": "HT", "fia_species_code": "SPCD"},
            },
        )
        assert response.status_code == 201

        source = response.json()["inventory"]["source"]
        assert source["columns"]["height"] == "HT"
        assert source["columns"]["fia_species_code"] == "SPCD"

        firestore_client.collection(INVENTORIES_COLLECTION).document(
            response.json()["inventory"]["id"]
        ).delete()

    def test_metadata_stored(self, client, domain_for_testing, firestore_client):
        """Name, description, and tags are stored in the inventory."""
        response = client.post(
            self.route(domain_for_testing["id"]),
            json={
                "format": "csv",
                "name": "Survey 2024",
                "description": "Plot measurements",
                "tags": ["field", "2024"],
            },
        )
        assert response.status_code == 201

        inv = response.json()["inventory"]
        assert inv["name"] == "Survey 2024"
        assert inv["description"] == "Plot measurements"
        assert inv["tags"] == ["field", "2024"]

        firestore_client.collection(INVENTORIES_COLLECTION).document(inv["id"]).delete()

    def test_response_excludes_owner_id(
        self, client, domain_for_testing, firestore_client
    ):
        """owner_id must not appear in the response."""
        response = client.post(
            self.route(domain_for_testing["id"]),
            json={"format": "csv"},
        )
        assert response.status_code == 201

        data = response.json()
        assert "owner_id" not in data["inventory"]

        firestore_client.collection(INVENTORIES_COLLECTION).document(
            data["inventory"]["id"]
        ).delete()

    def test_invalid_domain_returns_404(self, client):
        """Non-existent domain returns 404."""
        response = client.post(
            self.route("00000000000000000000000000000000"),
            json={"format": "csv"},
        )
        assert response.status_code == 404

    def test_wrong_owner_domain_returns_404(self, client, domain_with_different_owner):
        """Domain owned by a different user returns 404."""
        response = client.post(
            self.route(domain_with_different_owner["id"]),
            json={"format": "csv"},
        )
        assert response.status_code == 404

    def test_invalid_format_returns_422(self, client, domain_for_testing):
        """Unrecognized format value returns 422."""
        response = client.post(
            self.route(domain_for_testing["id"]),
            json={"format": "shapefile"},
        )
        assert response.status_code == 422

    def test_missing_format_returns_422(self, client, domain_for_testing):
        """Missing required format field returns 422."""
        response = client.post(
            self.route(domain_for_testing["id"]),
            json={"name": "no format"},
        )
        assert response.status_code == 422

    def test_unknown_column_key_returns_422(self, client, domain_for_testing):
        """Unknown key in columns mapping returns 422."""
        response = client.post(
            self.route(domain_for_testing["id"]),
            json={"format": "csv", "columns": {"weight": "weight_col"}},
        )
        assert response.status_code == 422


def _poll_inventory(client, domain_id, inventory_id, timeout=120) -> dict:
    """Poll GET inventory until terminal status, return the final doc."""
    url = f"/domains/{domain_id}/inventories/{inventory_id}"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        r = client.get(url)
        assert r.status_code == 200, r.text
        doc = r.json()
        if doc["status"] == "completed":
            return doc
        if doc["status"] == "failed":
            pytest.fail(f"Inventory failed: {doc.get('error')}")
        time.sleep(1)
    pytest.fail(f"Inventory did not complete within {timeout}s")


def _make_upload_file(fmt: str, domain_crs: str) -> Path:
    """Write a small upload file to a temp path and return it."""
    # Coordinates inside domain_for_testing bounds (x=[500000,501000], y=[5200000,5201000])
    x = [500200.0, 500400.0, 500600.0]
    y = [5200200.0, 5200400.0, 5200600.0]
    height = [10.0, 15.0, 20.0]

    if fmt == "csv":
        f = tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w")
        pd.DataFrame({"x": x, "y": y, "height": height}).to_csv(f, index=False)
        f.close()
        return Path(f.name)

    # GeoJSON: create in domain CRS then reproject to WGS84 (required by spec)
    gdf = gpd.GeoDataFrame(
        {"height": height},
        geometry=[Point(xi, yi) for xi, yi in zip(x, y)],
        crs=domain_crs,
    ).to_crs("EPSG:4326")
    f = tempfile.NamedTemporaryFile(suffix=".geojson", delete=False, mode="w")
    f.write(gdf.to_json())
    f.close()
    return Path(f.name)


class TestUploadFullFlow:
    """Full upload flow: POST → PUT file → Eventarc → uploader → completed.

    Requires the uploader Cloud Run service to be deployed and wired to Eventarc.
    """

    @pytest.mark.parametrize("fmt", ["csv", "geojson"])
    def test_upload_completes(self, fmt, client, domain_for_testing, firestore_client):
        """POST → PUT file to signed URL → poll → status=completed, Parquet matches upload."""
        domain_id = domain_for_testing["id"]
        domain_crs = domain_for_testing["crs"]["properties"]["name"]
        inventory_id = None

        try:
            response = client.post(
                f"/domains/{domain_id}/inventories/tree/upload",
                json={"format": fmt},
            )
            assert response.status_code == 201, response.text
            data = response.json()
            inventory_id = data["inventory"]["id"]
            object_name = data["inventory"]["source"]["object_name"]
            signed_url = data["upload"]["url"]
            content_type = data["upload"]["content_type"]
            max_size_bytes = data["upload"]["max_size_bytes"]

            file_path = _make_upload_file(fmt, domain_crs)
            with open(file_path, "rb") as f:
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

            assert exists(f"gs://{UPLOADS_BUCKET}/{object_name}")

            completed = _poll_inventory(client, domain_id, inventory_id)

            assert not exists(f"gs://{UPLOADS_BUCKET}/{object_name}")
            assert completed["georeference"]["crs"] == domain_crs

            parquet_df = dd.read_parquet(
                f"gs://{INVENTORIES_BUCKET}/{inventory_id}"
            ).compute()
            assert len(parquet_df) == 3
            assert list(parquet_df["height"]) == [10.0, 15.0, 20.0]

        finally:
            if inventory_id:
                gcs_path = f"gs://{INVENTORIES_BUCKET}/{inventory_id}"
                if exists(gcs_path):
                    delete_directory(gcs_path)
                firestore_client.collection(INVENTORIES_COLLECTION).document(
                    inventory_id
                ).delete()
