"""
Integration tests for uploader/handlers/grid.py — netCDF handler.

Tests the full handle_grid_netcdf pipeline against real GCP resources.
Each test creates domain + grid Firestore docs, uploads a real netCDF to
UPLOADS_BUCKET, calls handle_grid_netcdf directly, and asserts results.
"""

import json
import os
import tempfile
from uuid import uuid4

import fsspec.asyn as fasyn
import gcsfs
import numpy as np
import pytest
import rioxarray  # noqa: F401  registers .rio accessor
import xarray as xr
from google.cloud import storage
from uploader.handlers.grid import handle_grid_netcdf

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
from tests.integration.staging import staged_object_name

_BLUE_MTN_PATH = SHARED_TEST_DOMAINS_DIR / "blue_mtn.json"
DOMAIN_CRS = "EPSG:32611"

# Window inside the blue_mtn domain.
NC_XMIN, NC_YMIN = 720400.0, 5190000.0
NC_XMAX, NC_YMAX = 721200.0, 5190400.0


def _load_domain_doc(domain_id: str) -> dict:
    with open(_BLUE_MTN_PATH) as f:
        data = json.load(f)
    data["id"] = domain_id
    data["owner_id"] = "test-owner"
    for feature in data.get("features", []):
        coords = feature["geometry"]["coordinates"]
        if not isinstance(coords, str):
            feature["geometry"]["coordinates"] = json.dumps(coords)
    return data


_UPLOAD_FILENAME = "upload.nc"


def _teardown_gcsfs() -> None:
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


def _make_grid_doc(grid_id: str, domain_id: str) -> dict:
    object_name = staged_object_name(grid_id, _UPLOAD_FILENAME)
    return {
        "id": grid_id,
        "domain_id": domain_id,
        "owner_id": "test-owner",
        "status": "running",
        "source": {
            "name": "upload",
            "format": "netcdf",
            "object_name": object_name,
            "num_buffer_cells": 0,
        },
    }


def _upload_netcdf_2d(grid_id: str) -> str:
    """Write a 2D CF netCDF and upload it to UPLOADS_BUCKET."""
    object_name = staged_object_name(grid_id, _UPLOAD_FILENAME)
    nx, ny = 40, 20
    x = (
        NC_XMIN
        + (NC_XMAX - NC_XMIN) / nx / 2
        + np.arange(nx) * (NC_XMAX - NC_XMIN) / nx
    )
    y = (
        NC_YMAX
        - (NC_YMAX - NC_YMIN) / ny / 2
        - np.arange(ny) * (NC_YMAX - NC_YMIN) / ny
    )

    ds = xr.Dataset(
        {
            "fbfm": xr.DataArray(
                np.full((ny, nx), 101, dtype=np.int32),
                dims=("y", "x"),
                coords={"y": y, "x": x},
            ),
        }
    ).rio.write_crs(DOMAIN_CRS)

    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as f:
        tmp_path = f.name
    try:
        ds.to_netcdf(tmp_path, engine="h5netcdf")
        gcs = storage.Client()
        blob = gcs.bucket(UPLOADS_BUCKET).blob(object_name)
        blob.upload_from_filename(tmp_path)
    finally:
        os.unlink(tmp_path)
    return object_name


def _upload_netcdf_3d(grid_id: str, nz: int = 5) -> str:
    """Write a 3D CF netCDF (z up, uniform spacing) and upload it."""
    object_name = staged_object_name(grid_id, _UPLOAD_FILENAME)
    nx, ny = 40, 20
    x = (
        NC_XMIN
        + (NC_XMAX - NC_XMIN) / nx / 2
        + np.arange(nx) * (NC_XMAX - NC_XMIN) / nx
    )
    y = (
        NC_YMAX
        - (NC_YMAX - NC_YMIN) / ny / 2
        - np.arange(ny) * (NC_YMAX - NC_YMIN) / ny
    )
    z = np.arange(nz, dtype=np.float64)

    da = xr.DataArray(
        np.random.rand(nz, ny, nx).astype(np.float32),
        dims=("z", "y", "x"),
        coords={"z": z, "y": y, "x": x},
    )
    ds = xr.Dataset({"bulk_density.foliage": da})
    ds["bulk_density.foliage"].attrs["units"] = "kg/m**3"
    ds["z"].attrs["positive"] = "up"
    ds = ds.rio.write_crs(DOMAIN_CRS)

    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as f:
        tmp_path = f.name
    try:
        ds.to_netcdf(tmp_path, engine="h5netcdf")
        gcs = storage.Client()
        blob = gcs.bucket(UPLOADS_BUCKET).blob(object_name)
        blob.upload_from_filename(tmp_path)
    finally:
        os.unlink(tmp_path)
    return object_name


class TestNetcdf2DUpload:
    def test_valid_2d_completes(self):
        """Valid 2D netCDF upload produces status=completed with 2D georeference."""
        grid_id = f"test-{uuid4().hex}"
        domain_id = f"test-{uuid4().hex}"

        domain_doc = _load_domain_doc(domain_id)
        set_document(DOMAINS_COLLECTION, domain_id, domain_doc)

        object_name = _upload_netcdf_2d(grid_id)
        grid_doc = _make_grid_doc(grid_id, domain_id)
        set_document(GRIDS_COLLECTION, grid_id, grid_doc)

        try:
            handle_grid_netcdf(grid_id, UPLOADS_BUCKET, object_name, grid_doc)

            _, snap = get_document(GRIDS_COLLECTION, grid_id)
            result = snap.to_dict()

            assert result["status"] == "completed"
            # The written zarr store's footprint is recorded on completion (#342).
            assert result["size_bytes"] > 0
            assert result["georeference"]["crs"] == DOMAIN_CRS
            assert len(result["georeference"]["shape"]) == 2
            assert "z_resolution" not in result["georeference"]
            assert result["chunks"]["count_by_axis"].keys() == {"y", "x"}
            assert result["progress"]["percent"] == 100

            # bands derived from data variable
            assert len(result["bands"]) == 1
            assert result["bands"][0]["key"] == "fbfm"
            assert result["bands"][0]["type"] == "categorical"

            zarr_path = f"gs://{GRIDS_BUCKET}/{grid_id}"
            assert exists(zarr_path)

            ds = xr.open_zarr(zarr_path, decode_coords="all")
            assert "fbfm" in ds.data_vars

        finally:
            gcs_path = f"gs://{GRIDS_BUCKET}/{grid_id}"
            if exists(gcs_path):
                delete_directory(gcs_path)
            delete_document(GRIDS_COLLECTION, grid_id)
            delete_document(DOMAINS_COLLECTION, domain_id)


class TestNetcdf3DUpload:
    def test_valid_3d_completes(self):
        """Valid 3D netCDF upload produces status=completed with 3D georeference."""
        grid_id = f"test-{uuid4().hex}"
        domain_id = f"test-{uuid4().hex}"

        domain_doc = _load_domain_doc(domain_id)
        set_document(DOMAINS_COLLECTION, domain_id, domain_doc)

        nz = 5
        object_name = _upload_netcdf_3d(grid_id, nz=nz)
        grid_doc = _make_grid_doc(grid_id, domain_id)
        set_document(GRIDS_COLLECTION, grid_id, grid_doc)

        try:
            handle_grid_netcdf(grid_id, UPLOADS_BUCKET, object_name, grid_doc)

            _, snap = get_document(GRIDS_COLLECTION, grid_id)
            result = snap.to_dict()

            assert result["status"] == "completed"
            assert len(result["georeference"]["shape"]) == 3
            assert result["georeference"]["shape"][0] == nz
            assert result["georeference"]["z_resolution"] == pytest.approx(1.0)
            assert result["georeference"]["z_origin"] == pytest.approx(-0.5)
            assert result["chunks"]["count_by_axis"].keys() == {"z", "y", "x"}
            assert result["chunks"]["count_by_axis"]["z"] == 1

            assert len(result["bands"]) == 1
            assert result["bands"][0]["key"] == "bulk_density.foliage"
            assert result["bands"][0]["unit"] == "kg/m**3"
            assert result["bands"][0]["type"] == "continuous"

            zarr_path = f"gs://{GRIDS_BUCKET}/{grid_id}"
            assert exists(zarr_path)

            ds = xr.open_zarr(zarr_path, decode_coords="all")
            assert "bulk_density.foliage" in ds.data_vars
            assert ds["bulk_density.foliage"].dims == ("z", "y", "x")

        finally:
            gcs_path = f"gs://{GRIDS_BUCKET}/{grid_id}"
            if exists(gcs_path):
                delete_directory(gcs_path)
            delete_document(GRIDS_COLLECTION, grid_id)
            delete_document(DOMAINS_COLLECTION, domain_id)


class TestDispatch:
    def test_unknown_format_raises(self):
        """A grid doc with an unrecognized source.format raises UNKNOWN_GRID_FORMAT."""
        from uploader.dispatch import dispatch_handler

        from lib.errors import ProcessingError

        grid_id = f"test-{uuid4().hex}"
        doc = {
            "id": grid_id,
            "source": {"name": "upload", "format": "shapefile", "object_name": "x"},
        }
        with pytest.raises(ProcessingError) as exc:
            dispatch_handler("grids", grid_id, UPLOADS_BUCKET, "x", doc)
        assert exc.value.code == "UNKNOWN_GRID_FORMAT"
