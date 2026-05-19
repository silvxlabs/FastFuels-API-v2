"""
Round-trip integration tests for the netCDF upload handler — issue #242
acceptance criteria.

For each rank (2D and 3D):
1. Synthesize a source Grid (zarr in GRIDS_BUCKET + Firestore doc).
2. Run the real exporter handler (services/exporter, #241) to produce a
   CF-conformant netCDF in EXPORTS_BUCKET.
3. Stage that .nc into UPLOADS_BUCKET as if a user uploaded it.
4. Run the upload handler.
5. Assert the destination Grid's values, dims, CRS, transform, and
   (3D-only) z_resolution/z_origin equal the source's.

The source Grid is sized to fit inside the blue_mtn domain so the upload
handler's clip-to-domain step is a no-op and values can be compared element
by element.
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
from exporter.handlers.netcdf import export_netcdf
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
from lib.gcs import delete_directory, delete_file, exists
from lib.testing import SHARED_TEST_DOMAINS_DIR
from lib.zarr_utils import save_zarr

_BLUE_MTN_PATH = SHARED_TEST_DOMAINS_DIR / "blue_mtn.json"
DOMAIN_CRS = "EPSG:32611"

# Source grid extent — strictly inside blue_mtn domain
# (domain bounds: x=[720228, 721534], y=[5189763, 5190645])
SRC_XMIN, SRC_YMIN = 720400.0, 5190000.0
SRC_XMAX, SRC_YMAX = 721200.0, 5190400.0


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
    _gcs_blobs.gcsfs_client = gcsfs.GCSFileSystem()


@pytest.fixture(autouse=True)
def _reset_gcsfs():
    yield
    _teardown_gcsfs()


def _build_2d_source() -> tuple[xr.Dataset, list[dict]]:
    """Build a 2D source dataset with a single categorical band."""
    nx, ny = 16, 8
    dx = (SRC_XMAX - SRC_XMIN) / nx
    dy = (SRC_YMAX - SRC_YMIN) / ny
    x = SRC_XMIN + dx / 2 + np.arange(nx) * dx
    y = SRC_YMAX - dy / 2 - np.arange(ny) * dy

    values = np.arange(ny * nx, dtype=np.int32).reshape(ny, nx) + 101
    ds = xr.Dataset(
        {"fbfm": xr.DataArray(values, dims=("y", "x"), coords={"y": y, "x": x})}
    ).rio.write_crs(DOMAIN_CRS)
    bands = [{"key": "fbfm", "type": "categorical", "unit": None, "index": 0}]
    return ds, bands


def _build_3d_source() -> tuple[xr.Dataset, list[dict]]:
    """Build a 3D source dataset with a single continuous band, z up + uniform."""
    nx, ny, nz = 16, 8, 4
    dx = (SRC_XMAX - SRC_XMIN) / nx
    dy = (SRC_YMAX - SRC_YMIN) / ny
    x = SRC_XMIN + dx / 2 + np.arange(nx) * dx
    y = SRC_YMAX - dy / 2 - np.arange(ny) * dy
    dz = 1.0
    z = np.arange(nz, dtype=np.float64) * dz + dz / 2  # cell centers above 0

    rng = np.random.default_rng(seed=42)
    data = rng.random((nz, ny, nx)).astype(np.float32)
    da = xr.DataArray(data, dims=("z", "y", "x"), coords={"z": z, "y": y, "x": x})
    ds = xr.Dataset({"bulk_density.foliage": da}).rio.write_crs(DOMAIN_CRS)
    ds["z"].attrs["positive"] = "up"
    bands = [
        {
            "key": "bulk_density.foliage",
            "type": "continuous",
            "unit": "kg/m**3",
            "index": 0,
        }
    ]
    return ds, bands


def _stage_export_to_uploads(export_gcs_path: str, dest_object_name: str) -> None:
    """Download from EXPORTS_BUCKET and re-upload to UPLOADS_BUCKET."""
    without_scheme = export_gcs_path.removeprefix("gs://")
    src_bucket_name, src_blob_path = without_scheme.split("/", 1)
    client = storage.Client()
    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as f:
        tmp_path = f.name
    try:
        client.bucket(src_bucket_name).blob(src_blob_path).download_to_filename(
            tmp_path
        )
        client.bucket(UPLOADS_BUCKET).blob(dest_object_name).upload_from_filename(
            tmp_path
        )
    finally:
        os.unlink(tmp_path)


def _run_roundtrip(
    ds_source: xr.Dataset,
    bands: list[dict],
) -> tuple[xr.Dataset, xr.Dataset, dict]:
    """Persist source, export to netCDF, upload back, return (src, dst, dst_doc).

    Cleans up its own GCS / Firestore artifacts in a finally block on the caller.
    """
    src_grid_id = f"test-src-{uuid4().hex}"
    dst_grid_id = f"test-dst-{uuid4().hex}"
    export_id = f"test-exp-{uuid4().hex}"
    domain_id = f"test-{uuid4().hex}"

    upload_object_name = f"grids/{dst_grid_id}/upload.nc"

    src_zarr = f"gs://{GRIDS_BUCKET}/{src_grid_id}"
    dst_zarr = f"gs://{GRIDS_BUCKET}/{dst_grid_id}"
    export_gcs_path = None
    upload_path = f"gs://{UPLOADS_BUCKET}/{upload_object_name}"

    try:
        # Domain doc
        set_document(DOMAINS_COLLECTION, domain_id, _load_domain_doc(domain_id))

        # 1. Persist source grid (zarr + Firestore doc)
        save_zarr(src_zarr, ds_source, chunk_shape=(512, 512))
        set_document(
            GRIDS_COLLECTION,
            src_grid_id,
            {
                "id": src_grid_id,
                "domain_id": domain_id,
                "owner_id": "test-owner",
                "status": "completed",
                "bands": bands,
            },
        )

        # 2. Run the real netCDF exporter
        export_doc = {"id": export_id, "name": "roundtrip"}
        source_arg = {"grid_id": src_grid_id, "bands": None}
        export_gcs_path = export_netcdf(export_doc, source_arg, lambda *a, **k: None)
        assert exists(export_gcs_path), f"exporter did not produce {export_gcs_path}"

        # 3. Stage the .nc into UPLOADS_BUCKET
        _stage_export_to_uploads(export_gcs_path, upload_object_name)

        # 4. Persist destination grid doc (mirrors what the upload route writes)
        dst_doc = {
            "id": dst_grid_id,
            "domain_id": domain_id,
            "owner_id": "test-owner",
            "status": "running",
            "source": {
                "name": "upload",
                "format": "netcdf",
                "object_name": upload_object_name,
                "num_buffer_cells": 0,
            },
        }
        set_document(GRIDS_COLLECTION, dst_grid_id, dst_doc)

        # 5. Run the upload handler
        handle_grid_netcdf(dst_grid_id, UPLOADS_BUCKET, upload_object_name, dst_doc)

        # 6. Load both zarrs for comparison
        src_reload = xr.open_zarr(src_zarr, decode_coords="all")
        dst_reload = xr.open_zarr(dst_zarr, decode_coords="all")

        _, dst_snap = get_document(GRIDS_COLLECTION, dst_grid_id)
        dst_doc_final = dst_snap.to_dict()

        return src_reload, dst_reload, dst_doc_final

    finally:
        for path in (src_zarr, dst_zarr):
            if exists(path):
                delete_directory(path)
        if export_gcs_path:
            try:
                delete_file(export_gcs_path)
            except Exception:
                pass
        if exists(upload_path):
            try:
                delete_file(upload_path)
            except Exception:
                pass
        delete_document(GRIDS_COLLECTION, src_grid_id)
        delete_document(GRIDS_COLLECTION, dst_grid_id)
        delete_document(DOMAINS_COLLECTION, domain_id)


class TestRoundtrip2D:
    def test_2d_roundtrip_preserves_values_and_georeference(self):
        ds_source, bands = _build_2d_source()
        src, dst, dst_doc = _run_roundtrip(ds_source, bands)

        # Same data vars in the same order
        assert list(dst.data_vars) == list(src.data_vars)

        for var in src.data_vars:
            assert src[var].dims == dst[var].dims
            np.testing.assert_array_equal(src[var].values, dst[var].values)

        assert str(dst.rio.crs) == str(src.rio.crs)
        assert dst.rio.transform() == src.rio.transform()
        assert dst.rio.height == src.rio.height
        assert dst.rio.width == src.rio.width

        # 2D georeference doc
        gref = dst_doc["georeference"]
        assert len(gref["shape"]) == 2
        assert gref["crs"] == DOMAIN_CRS
        assert "z_resolution" not in gref

        # Derived band metadata round-trips
        assert len(dst_doc["bands"]) == 1
        assert dst_doc["bands"][0]["key"] == "fbfm"
        assert dst_doc["bands"][0]["type"] == "categorical"


class TestRoundtrip3D:
    def test_3d_roundtrip_preserves_values_and_georeference(self):
        ds_source, bands = _build_3d_source()
        src, dst, dst_doc = _run_roundtrip(ds_source, bands)

        assert list(dst.data_vars) == list(src.data_vars)

        for var in src.data_vars:
            assert src[var].dims == dst[var].dims
            np.testing.assert_allclose(
                src[var].values, dst[var].values, rtol=1e-6, atol=1e-6
            )

        assert str(dst.rio.crs) == str(src.rio.crs)
        assert dst.rio.transform() == src.rio.transform()

        np.testing.assert_allclose(dst["z"].values, src["z"].values)

        gref = dst_doc["georeference"]
        assert len(gref["shape"]) == 3
        assert gref["shape"][0] == src.sizes["z"]
        assert gref["z_resolution"] == pytest.approx(1.0)
        # z_origin is the bottom edge of the first cell. Source z[0] is the
        # center of the first cell at dz/2, so z_origin = 0.0.
        assert gref["z_origin"] == pytest.approx(0.0)

        assert dst_doc["bands"][0]["key"] == "bulk_density.foliage"
        assert dst_doc["bands"][0]["unit"] == "kg/m**3"
        assert dst_doc["bands"][0]["type"] == "continuous"
