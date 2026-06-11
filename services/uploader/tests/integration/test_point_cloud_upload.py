"""
Integration tests for uploader/handlers/point_cloud.py

Runs the full handle_point_cloud pipeline against real GCP resources: seeds a
domain + point cloud doc, uploads a synthesized LAS/LAZ to UPLOADS_BUCKET,
calls the handler directly, and asserts the stored cloud.laz, the completed
document, and cleanup. Covers the three ingest paths: passthrough copy (LAZ in
the domain CRS), recompression (LAS upload), and reprojection (LAZ in a
different CRS).
"""

import json
from uuid import uuid4

import gcsfs
import laspy
import pytest
from pyproj import CRS, Transformer
from uploader.handlers.point_cloud import handle_point_cloud

from lib.config import (
    DOMAINS_COLLECTION,
    POINT_CLOUDS_BUCKET,
    POINT_CLOUDS_COLLECTION,
    UPLOADS_BUCKET,
)
from lib.errors import ProcessingError
from lib.firestore import delete_document, get_document, set_document
from lib.gcs import delete_directory, delete_file, exists, get_gcsfs_client
from lib.testing import SHARED_TEST_DOMAINS_DIR
from tests.pointcloud_helpers import make_test_las

# Blackfoot domain is EPSG:32612 (UTM 12N) — matches the synthesizer default.
_DOMAIN_PATH = SHARED_TEST_DOMAINS_DIR / "blackfoot.json"
DOMAIN_CRS = "EPSG:32612"


def _load_domain_doc(domain_id: str) -> dict:
    with open(_DOMAIN_PATH) as f:
        data = json.load(f)
    data["id"] = domain_id
    data["owner_id"] = "test-owner"
    # Firestore rejects nested arrays — stringify domain coordinates.
    for feature in data.get("features", []):
        coords = feature["geometry"]["coordinates"]
        if not isinstance(coords, str):
            feature["geometry"]["coordinates"] = json.dumps(coords)
    return data


def _pc_doc(pc_id: str, domain_id: str, object_name: str) -> dict:
    return {
        "id": pc_id,
        "domain_id": domain_id,
        "owner_id": "test-owner",
        "type": "als",
        "status": "running",
        "source": {"name": "upload", "object_name": object_name},
    }


def _upload(local_path: str, object_name: str) -> None:
    gcsfs.GCSFileSystem().put(local_path, f"{UPLOADS_BUCKET}/{object_name}")


def _cleanup(pc_id: str, domain_id: str, object_name: str) -> None:
    out = f"gs://{POINT_CLOUDS_BUCKET}/{pc_id}"
    if exists(out):
        delete_directory(out)
    staged = f"gs://{UPLOADS_BUCKET}/{object_name}"
    if exists(staged):
        delete_file(staged)
    delete_document(POINT_CLOUDS_COLLECTION, pc_id)
    delete_document(DOMAINS_COLLECTION, domain_id)


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


def _read_stored_header(pc_id: str) -> laspy.LasHeader:
    with get_gcsfs_client().open(f"{POINT_CLOUDS_BUCKET}/{pc_id}/cloud.laz", "rb") as f:
        with laspy.open(f) as reader:
            return reader.header


class TestPointCloudUpload:
    def test_valid_laz_passthrough_completes(self, tmp_path):
        pc_id = f"test-{uuid4().hex}"
        domain_id = f"test-{uuid4().hex}"
        set_document(DOMAINS_COLLECTION, domain_id, _load_domain_doc(domain_id))

        local = tmp_path / "upload.laz"
        truth = make_test_las(str(local), n=100, epsg=32612, classes=(1, 2, 5))
        object_name = f"pointclouds/{pc_id}/upload"
        _upload(str(local), object_name)
        doc = _pc_doc(pc_id, domain_id, object_name)
        set_document(POINT_CLOUDS_COLLECTION, pc_id, doc)

        try:
            handle_point_cloud(pc_id, UPLOADS_BUCKET, object_name, doc)

            _, snap = get_document(POINT_CLOUDS_COLLECTION, pc_id)
            result = snap.to_dict()
            assert result["status"] == "completed"
            assert result["georeference"]["crs"] == DOMAIN_CRS
            assert len(result["georeference"]["bounds"]) == 6
            assert result["summary"]["point_count"] == 100
            assert result["summary"]["point_classes"] == [1, 2, 5]
            assert result["summary"]["density"] == pytest.approx(
                100 / truth["xy_area"], rel=1e-6
            )
            assert exists(f"gs://{POINT_CLOUDS_BUCKET}/{pc_id}/cloud.laz")
            assert not exists(f"gs://{UPLOADS_BUCKET}/{object_name}")
        finally:
            _cleanup(pc_id, domain_id, object_name)

    def test_las_upload_is_compressed(self, tmp_path):
        pc_id = f"test-{uuid4().hex}"
        domain_id = f"test-{uuid4().hex}"
        set_document(DOMAINS_COLLECTION, domain_id, _load_domain_doc(domain_id))

        local = tmp_path / "upload.las"
        make_test_las(str(local), n=80, epsg=32612, classes=(2, 5))
        object_name = f"pointclouds/{pc_id}/upload"
        _upload(str(local), object_name)
        doc = _pc_doc(pc_id, domain_id, object_name)
        set_document(POINT_CLOUDS_COLLECTION, pc_id, doc)

        try:
            handle_point_cloud(pc_id, UPLOADS_BUCKET, object_name, doc)

            _, snap = get_document(POINT_CLOUDS_COLLECTION, pc_id)
            result = snap.to_dict()
            assert result["status"] == "completed"
            assert result["summary"]["point_count"] == 80
            assert result["summary"]["point_classes"] == [2, 5]

            header = _read_stored_header(pc_id)
            assert header.are_points_compressed
            assert header.point_count == 80
            assert not exists(f"gs://{UPLOADS_BUCKET}/{object_name}")
        finally:
            _cleanup(pc_id, domain_id, object_name)

    def test_reprojects_to_domain_crs(self, tmp_path):
        pc_id = f"test-{uuid4().hex}"
        domain_id = f"test-{uuid4().hex}"
        set_document(DOMAINS_COLLECTION, domain_id, _load_domain_doc(domain_id))

        local = tmp_path / "upload.laz"
        # UTM 13N cloud into a UTM 12N domain: must be reprojected, not rejected.
        truth = make_test_las(str(local), n=60, epsg=32613, classes=(2,))
        object_name = f"pointclouds/{pc_id}/upload"
        _upload(str(local), object_name)
        doc = _pc_doc(pc_id, domain_id, object_name)
        set_document(POINT_CLOUDS_COLLECTION, pc_id, doc)

        transformer = Transformer.from_crs(
            CRS.from_epsg(32613), CRS.from_epsg(32612), always_xy=True
        )
        expected_x, expected_y = transformer.transform(truth["x"], truth["y"])

        try:
            handle_point_cloud(pc_id, UPLOADS_BUCKET, object_name, doc)

            _, snap = get_document(POINT_CLOUDS_COLLECTION, pc_id)
            result = snap.to_dict()
            assert result["status"] == "completed"
            assert result["georeference"]["crs"] == DOMAIN_CRS
            bounds = result["georeference"]["bounds"]
            assert bounds[0] == pytest.approx(expected_x.min(), abs=0.011)
            assert bounds[3] == pytest.approx(expected_x.max(), abs=0.011)
            assert bounds[1] == pytest.approx(expected_y.min(), abs=0.011)
            assert bounds[4] == pytest.approx(expected_y.max(), abs=0.011)

            header = _read_stored_header(pc_id)
            assert header.parse_crs().to_epsg() == 32612
            assert header.point_count == 60
            assert not exists(f"gs://{UPLOADS_BUCKET}/{object_name}")
        finally:
            _cleanup(pc_id, domain_id, object_name)

    def test_missing_crs_fails_and_cleans_up(self, tmp_path):
        pc_id = f"test-{uuid4().hex}"
        domain_id = f"test-{uuid4().hex}"
        set_document(DOMAINS_COLLECTION, domain_id, _load_domain_doc(domain_id))

        local = tmp_path / "upload.laz"
        make_test_las(str(local), n=50, with_srs=False)
        object_name = f"pointclouds/{pc_id}/upload"
        _upload(str(local), object_name)
        doc = _pc_doc(pc_id, domain_id, object_name)
        set_document(POINT_CLOUDS_COLLECTION, pc_id, doc)

        try:
            with pytest.raises(ProcessingError) as exc:
                handle_point_cloud(pc_id, UPLOADS_BUCKET, object_name, doc)
            assert exc.value.code == "MISSING_CRS"
            # Staged upload is cleaned up even on failure; nothing stored.
            assert not exists(f"gs://{UPLOADS_BUCKET}/{object_name}")
            assert not exists(f"gs://{POINT_CLOUDS_BUCKET}/{pc_id}/cloud.laz")
        finally:
            _cleanup(pc_id, domain_id, object_name)
