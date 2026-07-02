"""
Integration tests for uploader/handlers/grid.py — GeoTIFF handler.

Tests the full handle_grid_geotiff pipeline against real GCP resources.
Each test creates domain + grid Firestore docs, uploads a real GeoTIFF to
UPLOADS_BUCKET, calls handle_grid_geotiff directly, and asserts results.
"""

import json
import os
import tempfile
from uuid import uuid4

import fsspec.asyn as fasyn
import gcsfs
import numpy as np
import pytest
import rasterio
from google.cloud import storage
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from uploader.handlers.grid import handle_grid_geotiff

import lib.gcs.blobs as _gcs_blobs
from lib.config import (
    DOMAINS_COLLECTION,
    GRIDS_BUCKET,
    GRIDS_COLLECTION,
    UPLOADS_BUCKET,
)
from lib.firestore import delete_document, get_document, set_document
from lib.gcs import delete_directory, exists
from lib.testing import SHARED_TEST_DOMAINS_DIR

# Blue Mountain domain (EPSG:32611, UTM zone 11N, near Missoula MT)
# bounds: x=[720228, 721534], y=[5189763, 5190645]
_BLUE_MTN_PATH = SHARED_TEST_DOMAINS_DIR / "blue_mtn.json"
DOMAIN_CRS = "EPSG:32611"

# GeoTIFF bounds well inside blue_mtn domain (EPSG:32611)
TIFF_XMIN, TIFF_YMIN = 720400.0, 5190000.0
TIFF_XMAX, TIFF_YMAX = 721200.0, 5190400.0

# Same area in WGS84 (EPSG:4326) — used for CRS mismatch tests
TIFF_WGS84_XMIN, TIFF_WGS84_YMIN = -114.11, 46.825
TIFF_WGS84_XMAX, TIFF_WGS84_YMAX = -114.07, 46.845


def _load_domain_doc(domain_id: str) -> dict:
    """Load the blue_mtn shared domain and assign a test ID."""
    with open(_BLUE_MTN_PATH) as f:
        data = json.load(f)
    data["id"] = domain_id
    data["owner_id"] = "test-owner"
    for feature in data.get("features", []):
        coords = feature["geometry"]["coordinates"]
        if not isinstance(coords, str):
            feature["geometry"]["coordinates"] = json.dumps(coords)
    return data


_UPLOAD_FILENAME = "upload.tif"


def _teardown_gcsfs() -> None:
    """Stop the shared gcsfs async event loop and create a fresh instance.

    After each test, multiple gcsfs operations (zarr writes, deletes) leave the
    async event loop in a partially-used state. Concurrent zarr writes in the next
    test can fail silently if the loop is not reset. Stopping and recreating the
    loop between tests gives each test a clean async environment.
    """
    loop = fasyn.loop[0]
    if loop is not None and loop.is_running():
        loop.call_soon_threadsafe(loop.stop)
        thread = fasyn.iothread[0]
        if thread is not None:
            thread.join(timeout=3)
        fasyn.loop[0] = None
        fasyn.iothread[0] = None
    gcsfs.GCSFileSystem.clear_instance_cache()
    # Drop the lib accessor's cached client so the next get_gcsfs_client()
    # rebuilds against the fresh loop instead of the one we just stopped.
    _gcs_blobs.get_gcsfs_client.cache_clear()


@pytest.fixture(autouse=True)
def _reset_gcsfs():
    yield
    _teardown_gcsfs()


def _make_grid_doc(grid_id: str, domain_id: str, bands_spec: list[dict]) -> dict:
    """Minimal grid document with upload source."""
    object_name = f"grids/{grid_id}/{_UPLOAD_FILENAME}"
    return {
        "id": grid_id,
        "domain_id": domain_id,
        "owner_id": "test-owner",
        "status": "running",
        "source": {
            "name": "upload",
            "format": "geotiff",
            "object_name": object_name,
            "bands": bands_spec,
        },
    }


def _upload_geotiff(
    grid_id: str,
    n_bands: int = 1,
    crs: str = DOMAIN_CRS,
    set_crs: bool = True,
    width: int = 40,
    height: int = 20,
    tiff_xmin: float = TIFF_XMIN,
    tiff_ymin: float = TIFF_YMIN,
    tiff_xmax: float = TIFF_XMAX,
    tiff_ymax: float = TIFF_YMAX,
) -> str:
    """Write a GeoTIFF and upload it to UPLOADS_BUCKET. Returns the object_name.

    Defaults produce square pixels: default TIFF bounds are 800m x 400m
    (UTM) and WGS84 mismatch bounds are 0.04deg x 0.02deg — both yield
    20m / 0.001deg square pixels at width=40, height=20. The
    NON_SQUARE_PIXELS validator rejects anything else.
    """
    object_name = f"grids/{grid_id}/{_UPLOAD_FILENAME}"
    transform = from_bounds(tiff_xmin, tiff_ymin, tiff_xmax, tiff_ymax, width, height)

    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
        tmp_path = f.name

    try:
        with rasterio.open(
            tmp_path,
            "w",
            driver="GTiff",
            width=width,
            height=height,
            count=n_bands,
            crs=CRS.from_epsg(int(crs.split(":")[1])) if set_crs else None,
            transform=transform,
            dtype="float32",
        ) as dst:
            for b in range(1, n_bands + 1):
                dst.write(np.full((height, width), float(b), dtype="float32"), b)

        gcs = storage.Client()
        bucket = gcs.bucket(UPLOADS_BUCKET)
        blob = bucket.blob(object_name)
        blob.upload_from_filename(tmp_path)
    finally:
        os.unlink(tmp_path)

    return object_name


class TestSingleBandUpload:
    def test_valid_geotiff_completes(self):
        """Valid single-band GeoTIFF upload produces status=completed with Zarr."""
        grid_id = f"test-{uuid4().hex}"
        domain_id = f"test-{uuid4().hex}"
        bands_spec = [{"key": "fbfm", "type": "categorical", "unit": None}]

        domain_doc = _load_domain_doc(domain_id)
        set_document(DOMAINS_COLLECTION, domain_id, domain_doc)

        object_name = _upload_geotiff(grid_id, n_bands=1)
        grid_doc = _make_grid_doc(grid_id, domain_id, bands_spec)
        set_document(GRIDS_COLLECTION, grid_id, grid_doc)

        try:
            handle_grid_geotiff(grid_id, UPLOADS_BUCKET, object_name, grid_doc)

            _, snap = get_document(GRIDS_COLLECTION, grid_id)
            result = snap.to_dict()

            assert result["status"] == "completed"
            # The written zarr store's footprint is recorded on completion (#342).
            assert result["size_bytes"] > 0
            assert result["georeference"] is not None
            assert result["georeference"]["crs"] == DOMAIN_CRS
            assert len(result["georeference"]["transform"]) == 6
            assert len(result["georeference"]["shape"]) == 2
            assert result["chunks"] is not None
            assert result["progress"]["percent"] == 100

            zarr_path = f"gs://{GRIDS_BUCKET}/{grid_id}"
            assert exists(zarr_path)

            import xarray as xr

            ds = xr.open_zarr(zarr_path, decode_coords="all")
            assert "fbfm" in ds.data_vars

        finally:
            gcs_path = f"gs://{GRIDS_BUCKET}/{grid_id}"
            if exists(gcs_path):
                delete_directory(gcs_path)
            delete_document(GRIDS_COLLECTION, grid_id)
            delete_document(DOMAINS_COLLECTION, domain_id)


class TestMultiBandUpload:
    def test_multi_band_geotiff_completes(self):
        """Two-band GeoTIFF produces Zarr with two variables."""
        grid_id = f"test-{uuid4().hex}"
        domain_id = f"test-{uuid4().hex}"
        bands_spec = [
            {"key": "bulk_density.foliage", "type": "continuous", "unit": "kg/m**3"},
            {"key": "bulk_density.branchwood", "type": "continuous", "unit": "kg/m**3"},
        ]

        domain_doc = _load_domain_doc(domain_id)
        set_document(DOMAINS_COLLECTION, domain_id, domain_doc)

        object_name = _upload_geotiff(grid_id, n_bands=2)
        grid_doc = _make_grid_doc(grid_id, domain_id, bands_spec)
        set_document(GRIDS_COLLECTION, grid_id, grid_doc)

        try:
            handle_grid_geotiff(grid_id, UPLOADS_BUCKET, object_name, grid_doc)

            _, snap = get_document(GRIDS_COLLECTION, grid_id)
            result = snap.to_dict()
            assert result["status"] == "completed"

            import xarray as xr

            ds = xr.open_zarr(f"gs://{GRIDS_BUCKET}/{grid_id}", decode_coords="all")
            assert "bulk_density.foliage" in ds.data_vars
            assert "bulk_density.branchwood" in ds.data_vars

        finally:
            gcs_path = f"gs://{GRIDS_BUCKET}/{grid_id}"
            if exists(gcs_path):
                delete_directory(gcs_path)
            delete_document(GRIDS_COLLECTION, grid_id)
            delete_document(DOMAINS_COLLECTION, domain_id)


class TestErrorCases:
    def test_band_count_mismatch_fails(self):
        """1-band GeoTIFF with 2-band spec sets status=failed with BAND_COUNT_MISMATCH."""
        grid_id = f"test-{uuid4().hex}"
        domain_id = f"test-{uuid4().hex}"
        bands_spec = [
            {"key": "bulk_density.foliage", "type": "continuous", "unit": None},
            {"key": "bulk_density.branchwood", "type": "continuous", "unit": None},
        ]

        domain_doc = _load_domain_doc(domain_id)
        set_document(DOMAINS_COLLECTION, domain_id, domain_doc)

        object_name = _upload_geotiff(grid_id, n_bands=1)
        grid_doc = _make_grid_doc(grid_id, domain_id, bands_spec)
        set_document(GRIDS_COLLECTION, grid_id, grid_doc)

        try:
            from lib.errors import ProcessingError

            with pytest.raises(ProcessingError) as exc_info:
                handle_grid_geotiff(grid_id, UPLOADS_BUCKET, object_name, grid_doc)

            assert exc_info.value.code == "BAND_COUNT_MISMATCH"

        finally:
            gcs_path = f"gs://{GRIDS_BUCKET}/{grid_id}"
            if exists(gcs_path):
                delete_directory(gcs_path)
            delete_document(GRIDS_COLLECTION, grid_id)
            delete_document(DOMAINS_COLLECTION, domain_id)

    def test_missing_crs_fails(self):
        """GeoTIFF without CRS sets status=failed with MISSING_CRS."""
        grid_id = f"test-{uuid4().hex}"
        domain_id = f"test-{uuid4().hex}"
        bands_spec = [{"key": "fbfm", "type": "categorical", "unit": None}]

        domain_doc = _load_domain_doc(domain_id)
        set_document(DOMAINS_COLLECTION, domain_id, domain_doc)

        object_name = _upload_geotiff(grid_id, n_bands=1, set_crs=False)
        grid_doc = _make_grid_doc(grid_id, domain_id, bands_spec)
        set_document(GRIDS_COLLECTION, grid_id, grid_doc)

        try:
            from lib.errors import ProcessingError

            with pytest.raises(ProcessingError) as exc_info:
                handle_grid_geotiff(grid_id, UPLOADS_BUCKET, object_name, grid_doc)

            assert exc_info.value.code == "MISSING_CRS"

        finally:
            gcs_path = f"gs://{GRIDS_BUCKET}/{grid_id}"
            if exists(gcs_path):
                delete_directory(gcs_path)
            delete_document(GRIDS_COLLECTION, grid_id)
            delete_document(DOMAINS_COLLECTION, domain_id)

    def test_crs_mismatch_fails(self):
        """GeoTIFF in wrong CRS raises CRS_MISMATCH and staged file is deleted."""
        grid_id = f"test-{uuid4().hex}"
        domain_id = f"test-{uuid4().hex}"
        bands_spec = [{"key": "fbfm", "type": "categorical", "unit": None}]

        domain_doc = _load_domain_doc(domain_id)
        set_document(DOMAINS_COLLECTION, domain_id, domain_doc)

        # Upload a GeoTIFF in WGS84 (EPSG:4326) but domain expects EPSG:32611
        object_name = _upload_geotiff(
            grid_id,
            n_bands=1,
            crs="EPSG:4326",
            tiff_xmin=TIFF_WGS84_XMIN,
            tiff_ymin=TIFF_WGS84_YMIN,
            tiff_xmax=TIFF_WGS84_XMAX,
            tiff_ymax=TIFF_WGS84_YMAX,
        )
        grid_doc = _make_grid_doc(grid_id, domain_id, bands_spec)
        set_document(GRIDS_COLLECTION, grid_id, grid_doc)

        try:
            from lib.errors import ProcessingError

            with pytest.raises(ProcessingError) as exc_info:
                handle_grid_geotiff(grid_id, UPLOADS_BUCKET, object_name, grid_doc)

            assert exc_info.value.code == "CRS_MISMATCH"

        finally:
            gcs_path = f"gs://{GRIDS_BUCKET}/{grid_id}"
            if exists(gcs_path):
                delete_directory(gcs_path)
            delete_document(GRIDS_COLLECTION, grid_id)
            delete_document(DOMAINS_COLLECTION, domain_id)

    def test_no_overlap_fails(self):
        """GeoTIFF outside domain bounds raises NO_OVERLAP and staged file is deleted."""
        grid_id = f"test-{uuid4().hex}"
        domain_id = f"test-{uuid4().hex}"
        bands_spec = [{"key": "fbfm", "type": "categorical", "unit": None}]

        domain_doc = _load_domain_doc(domain_id)
        set_document(DOMAINS_COLLECTION, domain_id, domain_doc)

        # Upload a GeoTIFF far north of the blue_mtn domain bounds
        object_name = _upload_geotiff(
            grid_id,
            n_bands=1,
            tiff_xmin=720400.0,
            tiff_ymin=5300000.0,
            tiff_xmax=721200.0,
            tiff_ymax=5300400.0,
        )
        grid_doc = _make_grid_doc(grid_id, domain_id, bands_spec)
        set_document(GRIDS_COLLECTION, grid_id, grid_doc)

        try:
            from lib.errors import ProcessingError

            with pytest.raises(ProcessingError) as exc_info:
                handle_grid_geotiff(grid_id, UPLOADS_BUCKET, object_name, grid_doc)

            assert exc_info.value.code == "NO_OVERLAP"

        finally:
            gcs_path = f"gs://{GRIDS_BUCKET}/{grid_id}"
            if exists(gcs_path):
                delete_directory(gcs_path)
            delete_document(GRIDS_COLLECTION, grid_id)
            delete_document(DOMAINS_COLLECTION, domain_id)
