"""
Integration tests for api/v2/resources/inventories/tree/upload/router.py

Tests the upload inventory creation endpoint.
These tests make real HTTP requests to the API and interact with Firestore.
"""

from datetime import UTC, datetime, timedelta

from lib.config import INVENTORIES_COLLECTION


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
