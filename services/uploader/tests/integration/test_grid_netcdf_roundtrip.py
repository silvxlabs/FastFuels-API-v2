"""
Round-trip integration tests for the netCDF upload handler — issue #242
acceptance criteria.

For each rank (2D and 3D):
1. Synthesize a source Grid (zarr in GRIDS_BUCKET + Firestore doc).
2. Build a CF-conformant netCDF inline (mirroring the logic in
   services/exporter/exporter/handlers/netcdf.py::export_netcdf) and
   write it directly to UPLOADS_BUCKET. Inlined rather than imported
   so the uploader service does not need an editable dep on exporter
   (which fails in the production Docker image — the exporter source
   is not copied into the uploader's container).
3. Run the upload handler.
4. Assert the destination Grid's values, dims, CRS, transform, and
   (3D-only) z_resolution/z_origin equal the source's.

The source Grid is sized to fit inside the blue_mtn domain so the upload
handler's clip-to-domain step is a no-op and values can be compared element
by element.
"""

import json
import os
import tempfile
from datetime import datetime
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
from lib.cf_utils import stamp_cf
from lib.config import (
    DOMAINS_COLLECTION,
    GRIDS_BUCKET,
    GRIDS_COLLECTION,
    UPLOADS_BUCKET,
)
from lib.firestore import delete_document, get_document, set_document
from lib.gcs import delete_directory, delete_file, exists
from lib.testing import SHARED_TEST_DOMAINS_DIR, load_json
from lib.zarr_utils import save_zarr
from tests.integration.staging import staged_object_name

# Internal attrs that the exporter strips before CF stamping — same list
# as services/exporter/exporter/handlers/netcdf.py::_INTERNAL_ATTRS_TO_STRIP.
_INTERNAL_ATTRS_TO_STRIP = ("transform", "z_origin", "z_resolution")

_BLUE_MTN_PATH = SHARED_TEST_DOMAINS_DIR / "blue_mtn.json"
DOMAIN_CRS = "EPSG:32611"

# Source grid extent — strictly inside blue_mtn domain
# (domain bounds: x=[720228, 721534], y=[5189763, 5190645])
SRC_XMIN, SRC_YMIN = 720400.0, 5190000.0
SRC_XMAX, SRC_YMAX = 721200.0, 5190400.0


def _load_domain_doc(domain_id: str) -> dict:
    data = load_json(_BLUE_MTN_PATH)
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
    # Drop the lib accessor's cached client so the next get_gcsfs_client()
    # rebuilds against the fresh loop instead of the one we just stopped.
    _gcs_blobs.get_gcsfs_client.cache_clear()


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


def _export_netcdf_to_uploads(
    src_zarr_path: str, bands: list[dict], dest_object_name: str
) -> None:
    """Build a CF netCDF from a source zarr and write it directly to UPLOADS_BUCKET.

    Mirrors services/exporter/exporter/handlers/netcdf.py::export_netcdf
    (read zarr → strip internal attrs → stamp_cf → set zlib encoding →
    write h5netcdf → upload), but inline to avoid a cross-service dep
    on `exporter` from the uploader's pyproject.toml.

    Note: .load() forces an eager read of the zarr into memory before
    handing the Dataset to h5netcdf. The lazy-dask write path through
    h5netcdf can silently write zeros for int data on some Linux x86_64
    HDF5 / numpy wheel combinations (CI hits this; macOS arm64 does not).
    The roundtrip-test datasets are tiny (kilobytes), so eager load is
    cheap; the real production exporter on Cloud Run has the same
    vulnerability for int-dtype grids and tracks #__ separately.
    """
    ds = xr.open_zarr(src_zarr_path, decode_coords="all").load()
    for k in _INTERNAL_ATTRS_TO_STRIP:
        ds.attrs.pop(k, None)
    stamp_cf(ds, bands=bands, vertical=("z" in ds.dims))
    # Per cf_utils convention: mutate var.encoding in place. Passing
    # encoding= kwarg to to_netcdf would wipe grid_mapping.
    for var in ds.data_vars:
        ds[var].encoding["zlib"] = True
        ds[var].encoding["complevel"] = 4

    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as f:
        tmp_path = f.name
    try:
        ds.to_netcdf(tmp_path, engine="h5netcdf")
        storage.Client().bucket(UPLOADS_BUCKET).blob(
            dest_object_name
        ).upload_from_filename(tmp_path)
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
    domain_id = f"test-{uuid4().hex}"

    upload_object_name = staged_object_name(dst_grid_id, "upload.nc")

    src_zarr = f"gs://{GRIDS_BUCKET}/{src_grid_id}"
    dst_zarr = f"gs://{GRIDS_BUCKET}/{dst_grid_id}"
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
                "created_on": datetime.now(),
                "modified_on": datetime.now(),
                "bands": bands,
            },
        )

        # 2. Build CF netCDF inline and stage it into UPLOADS_BUCKET.
        _export_netcdf_to_uploads(src_zarr, bands, upload_object_name)
        assert exists(upload_path), f"inline export did not produce {upload_path}"

        # 3. Persist destination grid doc (mirrors what the upload route writes)
        dst_doc = {
            "id": dst_grid_id,
            "domain_id": domain_id,
            "owner_id": "test-owner",
            "status": "running",
            "created_on": datetime.now(),
            "modified_on": datetime.now(),
            "source": {
                "name": "upload",
                "format": "netcdf",
                "object_name": upload_object_name,
                "num_buffer_cells": 0,
            },
        }
        set_document(GRIDS_COLLECTION, dst_grid_id, dst_doc)

        # 4. Run the upload handler
        handle_grid_netcdf(dst_grid_id, UPLOADS_BUCKET, upload_object_name, dst_doc)

        # 5. Load both zarrs for comparison. Eager: the finally below deletes
        # both zarrs before the caller asserts on them, so a lazy Dataset would
        # be reading chunks out of a deleted prefix.
        src_reload = xr.open_zarr(src_zarr, decode_coords="all").load()
        dst_reload = xr.open_zarr(dst_zarr, decode_coords="all").load()

        _, dst_snap = get_document(GRIDS_COLLECTION, dst_grid_id)
        dst_doc_final = dst_snap.to_dict()

        return src_reload, dst_reload, dst_doc_final

    finally:
        for path in (src_zarr, dst_zarr):
            if exists(path):
                delete_directory(path)
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
