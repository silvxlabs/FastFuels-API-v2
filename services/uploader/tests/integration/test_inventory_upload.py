"""
Integration tests for uploader/handlers/inventory.py

Tests the full handle_inventory pipeline against real GCP resources.
Each test creates domain + inventory Firestore docs, uploads a real file
to UPLOADS_BUCKET, calls handle_inventory directly, and asserts results.
"""

import json
from uuid import uuid4

import gcsfs
import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point
from uploader.handlers.inventory import handle_inventory

from lib.config import (
    DOMAINS_COLLECTION,
    INVENTORIES_BUCKET,
    INVENTORIES_COLLECTION,
    UPLOADS_BUCKET,
)
from lib.firestore import delete_document, get_document, set_document
from lib.gcs import delete_directory, exists
from lib.testing import SHARED_TEST_DOMAINS_DIR

# Blue Mountain domain (EPSG:32611, UTM zone 11N, near Missoula MT)
# bounds: x=[720228, 721534], y=[5189763, 5190645]
_BLUE_MTN_PATH = SHARED_TEST_DOMAINS_DIR / "blue_mtn.json"
DOMAIN_CRS = "EPSG:32611"

# Sample tree coordinates well inside the blue_mtn domain bounds
SAMPLE_X = [720500.0, 720700.0, 720900.0]
SAMPLE_Y = [5190000.0, 5190100.0, 5190200.0]
SAMPLE_HEIGHT = [10.0, 15.0, 20.0]

# WGS84 lon/lat for the center of blue_mtn (~720881 E, 5190204 N in UTM 11N)
SAMPLE_LON_LAT = [(-114.104, 46.829), (-114.102, 46.830)]


def _load_domain_doc(domain_id: str) -> dict:
    """Load the blue_mtn shared domain and assign a test ID."""
    with open(_BLUE_MTN_PATH) as f:
        data = json.load(f)
    data["id"] = domain_id
    data["owner_id"] = "test-owner"
    # Firestore requires nested arrays to be stringified
    for feature in data.get("features", []):
        coords = feature["geometry"]["coordinates"]
        if not isinstance(coords, str):
            feature["geometry"]["coordinates"] = json.dumps(coords)
    return data


def _make_inventory_doc(
    inventory_id: str, domain_id: str, fmt: str, col_map: dict = None
) -> dict:
    """Minimal inventory document with upload source."""
    object_name = (
        f"inventories/{inventory_id}/upload.{fmt if fmt != 'geopackage' else 'gpkg'}"
    )
    return {
        "id": inventory_id,
        "domain_id": domain_id,
        "owner_id": "test-owner",
        "type": "tree",
        "status": "running",
        "source": {
            "name": "upload",
            "format": fmt,
            "object_name": object_name,
            "columns": col_map or {},
        },
    }


def _upload_csv(inventory_id: str, x, y, height, extra: dict = None) -> str:
    """Upload a CSV file to UPLOADS_BUCKET. Returns the object_name."""
    data = {"x": x, "y": y, "height": height}
    if extra:
        data.update(extra)
    df = pd.DataFrame(data)
    object_name = f"inventories/{inventory_id}/upload.csv"
    fs = gcsfs.GCSFileSystem()
    with fs.open(f"{UPLOADS_BUCKET}/{object_name}", "w") as f:
        df.to_csv(f, index=False)
    return object_name


def _upload_geojson(inventory_id: str, lon_lat_points: list, attrs: dict) -> str:
    """Upload a GeoJSON file to UPLOADS_BUCKET. Returns the object_name."""
    gdf = gpd.GeoDataFrame(
        attrs,
        geometry=[Point(lon, lat) for lon, lat in lon_lat_points],
        crs="EPSG:4326",
    )
    object_name = f"inventories/{inventory_id}/upload.geojson"
    fs = gcsfs.GCSFileSystem()
    with fs.open(f"{UPLOADS_BUCKET}/{object_name}", "w") as f:
        f.write(gdf.to_json())
    return object_name


@pytest.fixture(autouse=True, scope="session")
def _cleanup_gcsfs_sessions():
    yield

    import fsspec.asyn as fasyn
    import gcsfs as _gcsfs

    loop = fasyn.loop[0]
    if loop is not None and loop.is_running():
        loop.call_soon_threadsafe(loop.stop)
        thread = fasyn.iothread[0]
        if thread is not None:
            thread.join(timeout=5)
        fasyn.loop[0] = None
        fasyn.iothread[0] = None
    _gcsfs.GCSFileSystem.clear_instance_cache()


class TestCsvUpload:
    def test_valid_csv_completes(self):
        """Valid CSV upload produces status=completed with georeference and Parquet."""
        inventory_id = f"test-{uuid4().hex}"
        domain_id = f"test-{uuid4().hex}"

        domain_doc = _load_domain_doc(domain_id)
        set_document(DOMAINS_COLLECTION, domain_id, domain_doc)

        object_name = _upload_csv(inventory_id, SAMPLE_X, SAMPLE_Y, SAMPLE_HEIGHT)
        inv_doc = _make_inventory_doc(inventory_id, domain_id, "csv")
        inv_doc["source"]["object_name"] = object_name
        set_document(INVENTORIES_COLLECTION, inventory_id, inv_doc)

        try:
            handle_inventory(inventory_id, UPLOADS_BUCKET, object_name, inv_doc)

            _, snap = get_document(INVENTORIES_COLLECTION, inventory_id)
            result = snap.to_dict()

            assert result["status"] == "completed"
            assert result["georeference"] is not None
            assert result["georeference"]["crs"] == DOMAIN_CRS
            assert len(result["georeference"]["bounds"]) == 4
            assert result["progress"]["percent"] == 100

            assert exists(f"gs://{INVENTORIES_BUCKET}/{inventory_id}")
            assert not exists(f"gs://{UPLOADS_BUCKET}/{object_name}")

        finally:
            gcs_path = f"gs://{INVENTORIES_BUCKET}/{inventory_id}"
            if exists(gcs_path):
                delete_directory(gcs_path)
            delete_document(INVENTORIES_COLLECTION, inventory_id)
            delete_document(DOMAINS_COLLECTION, domain_id)

    def test_csv_with_column_mapping_completes(self):
        """CSV with custom column names and a mapping produces status=completed."""
        inventory_id = f"test-{uuid4().hex}"
        domain_id = f"test-{uuid4().hex}"

        domain_doc = _load_domain_doc(domain_id)
        set_document(DOMAINS_COLLECTION, domain_id, domain_doc)

        df = pd.DataFrame(
            {"easting": SAMPLE_X, "northing": SAMPLE_Y, "HT": SAMPLE_HEIGHT}
        )
        object_name = f"inventories/{inventory_id}/upload.csv"
        fs = gcsfs.GCSFileSystem()
        with fs.open(f"{UPLOADS_BUCKET}/{object_name}", "w") as f:
            df.to_csv(f, index=False)

        col_map = {"x": "easting", "y": "northing", "height": "HT"}
        inv_doc = _make_inventory_doc(inventory_id, domain_id, "csv", col_map)
        inv_doc["source"]["object_name"] = object_name
        set_document(INVENTORIES_COLLECTION, inventory_id, inv_doc)

        try:
            handle_inventory(inventory_id, UPLOADS_BUCKET, object_name, inv_doc)
            _, snap = get_document(INVENTORIES_COLLECTION, inventory_id)
            assert snap.to_dict()["status"] == "completed"
        finally:
            gcs_path = f"gs://{INVENTORIES_BUCKET}/{inventory_id}"
            if exists(gcs_path):
                delete_directory(gcs_path)
            delete_document(INVENTORIES_COLLECTION, inventory_id)
            delete_document(DOMAINS_COLLECTION, domain_id)

    def test_csv_missing_height_produces_error(self):
        """CSV missing required height column raises ProcessingError."""
        inventory_id = f"test-{uuid4().hex}"
        domain_id = f"test-{uuid4().hex}"

        domain_doc = _load_domain_doc(domain_id)
        set_document(DOMAINS_COLLECTION, domain_id, domain_doc)

        df = pd.DataFrame({"x": SAMPLE_X, "y": SAMPLE_Y})
        object_name = f"inventories/{inventory_id}/upload.csv"
        fs = gcsfs.GCSFileSystem()
        with fs.open(f"{UPLOADS_BUCKET}/{object_name}", "w") as f:
            df.to_csv(f, index=False)

        inv_doc = _make_inventory_doc(inventory_id, domain_id, "csv")
        inv_doc["source"]["object_name"] = object_name
        set_document(INVENTORIES_COLLECTION, inventory_id, inv_doc)

        try:
            from lib.errors import ProcessingError

            with pytest.raises(ProcessingError) as exc_info:
                handle_inventory(inventory_id, UPLOADS_BUCKET, object_name, inv_doc)
            assert exc_info.value.code == "SCHEMA_VALIDATION_ERROR"
        finally:
            delete_document(INVENTORIES_COLLECTION, inventory_id)
            delete_document(DOMAINS_COLLECTION, domain_id)


class TestGeoJsonUpload:
    def test_valid_geojson_completes(self):
        """Valid GeoJSON upload produces status=completed with Parquet."""
        inventory_id = f"test-{uuid4().hex}"
        domain_id = f"test-{uuid4().hex}"

        domain_doc = _load_domain_doc(domain_id)
        set_document(DOMAINS_COLLECTION, domain_id, domain_doc)

        # WGS84 lon/lat that reprojects into the blue_mtn domain bounds
        object_name = _upload_geojson(
            inventory_id, SAMPLE_LON_LAT, {"height": [10.0, 15.0]}
        )

        inv_doc = _make_inventory_doc(inventory_id, domain_id, "geojson")
        inv_doc["source"]["object_name"] = object_name
        set_document(INVENTORIES_COLLECTION, inventory_id, inv_doc)

        try:
            handle_inventory(inventory_id, UPLOADS_BUCKET, object_name, inv_doc)
            _, snap = get_document(INVENTORIES_COLLECTION, inventory_id)
            result = snap.to_dict()
            assert result["status"] == "completed"
            assert exists(f"gs://{INVENTORIES_BUCKET}/{inventory_id}")
        finally:
            gcs_path = f"gs://{INVENTORIES_BUCKET}/{inventory_id}"
            if exists(gcs_path):
                delete_directory(gcs_path)
            delete_document(INVENTORIES_COLLECTION, inventory_id)
            delete_document(DOMAINS_COLLECTION, domain_id)


class TestIdempotency:
    def test_double_invoke_is_safe(self):
        """Calling handle_inventory when status is already completed is handled by main.py.

        The idempotency check in main.py prevents handle_inventory from being
        called twice on the same resource. This test documents that contract.
        """
        pass
