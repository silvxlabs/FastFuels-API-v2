"""
Tests for grid data streaming endpoints.

Endpoints under test:
  GET /domains/{domain_id}/grids/{grid_id}/chunks/{chunk_index}
  GET /domains/{domain_id}/grids/{grid_id}/data/{band}/{chunk_index}
  GET /domains/{domain_id}/grids/{grid_id}/data/{band}/{chunk_index}/binary

Each endpoint has its own test class. Endpoint classes end with 2D and 3D
integration tests that pull data through the API and compare against the
source zarr.
"""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr
from api.resources.grids import router as grids_router
from api.resources.grids.schema import (
    GridDataArrayFormat,
    GridDataOrder,
)
from fastapi import HTTPException

from lib.config import GRIDS_BUCKET, GRIDS_COLLECTION
from lib.testing import SHARED_TEST_GRIDS_DIR
from tests.fixtures import make_grid_data

# Static-fixture identifiers (created by tests/e2e/test_create_fixtures.py).
STATIC_NAME = "static-test-blue-mtn-landfire-fbfm40"
STATIC_BAND = "fbfm"
STATIC_3D_NAME = "static-test-blue-mtn-tree-inventory-voxels"
STATIC_3D_BAND = "bulk_density.foliage.live"


# Fixture loaders


def _load_static_template(static_name: str) -> dict:
    path = SHARED_TEST_GRIDS_DIR / f"{static_name}.json"
    with open(path) as f:
        return json.load(f)


def _load_static_template_or_skip(static_name: str) -> dict:
    path = SHARED_TEST_GRIDS_DIR / f"{static_name}.json"
    if not path.exists():
        pytest.skip(
            f"{path} not found. Run the e2e fixture generator for {static_name}."
        )
    with open(path) as f:
        return json.load(f)


def _read_static_zarr_chunk(static_name: str, band: str, metadata: dict) -> np.ndarray:
    """Read the slice corresponding to a chunk's metadata from the source zarr."""
    offset = metadata["offset"]
    shape = metadata["shape"]
    slices = tuple(slice(start, start + length) for start, length in zip(offset, shape))

    ds = xr.open_zarr(
        f"gs://{GRIDS_BUCKET}/{static_name}",
        consolidated=True,
        storage_options={"token": "google_default"},
    )
    try:
        data_array = ds[band]
        if len(slices) == 3:
            return data_array.isel(z=slices[0], y=slices[1], x=slices[2]).values
        return data_array.isel(y=slices[0], x=slices[1]).values
    finally:
        ds.close()


# Firestore fixtures


@pytest.fixture(scope="session")
def static_grid_in_firestore(firestore_client, test_owner_id, domain_for_testing):
    """Register the static FBFM40 fixture as a Firestore grid doc."""
    template = _load_static_template(STATIC_NAME)
    template["id"] = STATIC_NAME
    template["owner_id"] = test_owner_id
    template["domain_id"] = domain_for_testing["id"]
    doc_ref = firestore_client.collection(GRIDS_COLLECTION).document(STATIC_NAME)
    doc_ref.set(template)
    yield template
    doc_ref.delete()


@pytest.fixture(scope="session")
def static_3d_grid_in_firestore(firestore_client, test_owner_id, domain_for_testing):
    """Register the static 3D tree voxel fixture as a Firestore grid doc."""
    template = _load_static_template_or_skip(STATIC_3D_NAME)
    template["id"] = STATIC_3D_NAME
    template["owner_id"] = test_owner_id
    template["domain_id"] = domain_for_testing["id"]
    doc_ref = firestore_client.collection(GRIDS_COLLECTION).document(STATIC_3D_NAME)
    doc_ref.set(template)
    yield template
    doc_ref.delete()


@pytest.fixture(scope="session")
def pending_grid_in_firestore(firestore_client, domain_for_testing):
    """A pending grid (not completed) for validation tests."""
    grid_data = make_grid_data(
        domain_id=domain_for_testing["id"],
        name="Pending grid for data tests",
        status="pending",
        georeference=None,
    )
    doc_ref = firestore_client.collection(GRIDS_COLLECTION).document(grid_data["id"])
    doc_ref.set(grid_data)
    yield grid_data
    doc_ref.delete()


@pytest.fixture(scope="session")
def grid_with_different_owner(firestore_client, domain_with_different_owner):
    """A grid owned by a different user."""
    grid_data = make_grid_data(
        domain_id=domain_with_different_owner["id"],
        owner_id="different-owner",
        name="Other User's Grid",
        status="completed",
    )
    doc_ref = firestore_client.collection(GRIDS_COLLECTION).document(grid_data["id"])
    doc_ref.set(grid_data)
    yield grid_data
    doc_ref.delete()


@pytest.fixture(scope="session")
def completed_3d_grid_in_firestore(firestore_client, domain_for_testing):
    """A completed 3D tree grid document without backing zarr data."""
    grid_data = make_grid_data(
        domain_id=domain_for_testing["id"],
        name="Completed 3D grid for data endpoint tests",
        status="completed",
        source={
            "name": "inventory",
            "product": "tree",
            "description": "3D tree fuel grid from tree inventory voxelization",
            "source_inventory_id": "test-source-inv",
            "resolution": {"horizontal": 2.0, "vertical": 1.0},
            "bands": ["bulk_density.foliage.live"],
            "crown_profile_model": "purves",
            "biomass_source": {
                "type": "allometry",
                "equations": "nsvb",
                "components": ["foliage"],
                "component_states": {"foliage": {"live": 1.0, "dead": 0.0}},
            },
            "moisture_model": None,
            "seed": 42,
        },
        bands=[
            {
                "key": "bulk_density.foliage.live",
                "type": "continuous",
                "unit": "kg/m³",
                "index": 0,
            },
        ],
        georeference={
            "crs": "EPSG:32611",
            "transform": (2.0, 0.0, 500000.0, 0.0, -2.0, 5201000.0),
            "shape": [5, 1000, 800],
            "z_resolution": 0.5,
            "z_origin": 10.0,
        },
    )
    grid_data["chunk_shape"] = [2, 512, 512]
    doc_ref = firestore_client.collection(GRIDS_COLLECTION).document(grid_data["id"])
    doc_ref.set(grid_data)
    yield grid_data
    doc_ref.delete()


# URL builders


def chunk_route(domain_id, grid_id, chunk_index):
    return f"/domains/{domain_id}/grids/{grid_id}/chunks/{chunk_index}"


def data_route(domain_id, grid_id, band, chunk_index=0, **params):
    return f"/domains/{domain_id}/grids/{grid_id}/data/{band}/{chunk_index}", params


def binary_data_route(domain_id, grid_id, band, chunk_index=0, **params):
    return (
        f"/domains/{domain_id}/grids/{grid_id}/data/{band}/{chunk_index}/binary",
        params,
    )


# Fakes and helpers for unit-style handler tests


class FakeGridArray:
    def __init__(self, data, fill_value=0):
        self.data = data
        if fill_value is not None:
            self.metadata = SimpleNamespace(fill_value=fill_value)
        self.selection = None

    async def getitem(self, selection):
        self.selection = selection
        return self.data


async def _call_handler(
    monkeypatch,
    fake_array,
    handler,
    *,
    band="bulk_density.foliage.live",
    array_format=GridDataArrayFormat.dense,
    order=GridDataOrder.C,
):
    """Invoke a handler directly with a fake zarr array (no HTTP, no GCS)."""
    shape = list(fake_array.data.shape)
    georeference = {
        "crs": "EPSG:32611",
        "transform": (2.0, 0.0, 500000.0, 0.0, -2.0, 5201000.0),
        "shape": shape,
    }
    if len(shape) == 3:
        georeference["z_resolution"] = 0.5
        georeference["z_origin"] = 10.0

    grid_data = {
        "id": "grid-fake",
        "domain_id": "domain-1",
        "owner_id": "owner-1",
        "status": "completed",
        "bands": [
            {"key": band, "type": "continuous", "unit": None, "index": 0},
        ],
        "georeference": georeference,
        "chunk_shape": shape,
    }

    async def fake_get_document_async(*args, **kwargs):
        return None, SimpleNamespace(to_dict=lambda: grid_data)

    async def fake_get_grid_array(grid_id, requested_band):
        assert grid_id == "grid-fake"
        assert requested_band == band
        return fake_array

    monkeypatch.setattr(grids_router, "get_document_async", fake_get_document_async)
    monkeypatch.setattr(grids_router, "get_grid_array", fake_get_grid_array)

    return await handler(
        request=SimpleNamespace(state=SimpleNamespace(id="owner-1")),
        domain={"id": "domain-1"},
        grid_id="grid-fake",
        chunk_index=0,
        band=band,
        array_format=array_format,
        order=order,
    )


# Static-zarr assertion helpers


def _assert_dense_json_matches_zarr(payload: dict, expected: np.ndarray) -> None:
    assert payload["data"]["format"] == "dense"
    actual = np.asarray(payload["data"]["values"], dtype=expected.dtype).reshape(
        payload["shape"], order=payload["order"]
    )
    np.testing.assert_array_equal(actual, expected)


def _assert_sparse_json_matches_zarr(payload: dict, expected: np.ndarray) -> None:
    sparse = payload["data"]
    assert sparse["format"] == "sparse"
    if sparse["fill_value"] is None:
        actual = np.asarray(sparse["values"], dtype=expected.dtype).reshape(
            payload["shape"], order=payload["order"]
        )
    else:
        actual = np.full(payload["shape"], sparse["fill_value"], dtype=expected.dtype)
        indices = np.asarray(sparse["indices"], dtype=np.int32)
        values = np.asarray(sparse["values"], dtype=expected.dtype)
        actual.ravel(order=payload["order"])[indices] = values
    np.testing.assert_array_equal(actual, expected)


def _assert_dense_binary_matches_zarr(response, expected: np.ndarray) -> None:
    assert response.headers["content-type"] == "application/octet-stream"
    assert response.headers["X-Data-Format"] == "dense"
    shape = tuple(int(s) for s in response.headers["X-Data-Shape"].split(","))
    dtype = np.dtype(response.headers["X-Data-Dtype"])
    actual = np.frombuffer(response.content, dtype=dtype).reshape(
        shape, order=response.headers["X-Data-Order"]
    )
    np.testing.assert_array_equal(actual, expected)


def _assert_sparse_binary_matches_zarr(response, expected: np.ndarray) -> None:
    assert response.headers["content-type"] == "application/octet-stream"
    assert response.headers["X-Data-Format"] == "sparse"
    nnz = int(response.headers["X-Data-NNZ"])
    index_dtype = np.dtype(response.headers["X-Data-Index-Dtype"])
    value_dtype = np.dtype(response.headers["X-Data-Value-Dtype"])
    index_bytes = nnz * index_dtype.itemsize
    indices = np.frombuffer(response.content[:index_bytes], dtype=index_dtype)
    values = np.frombuffer(response.content[index_bytes:], dtype=value_dtype)
    shape = tuple(int(s) for s in response.headers["X-Data-Shape"].split(","))
    fill_header = response.headers.get("X-Data-Fill-Value")
    if fill_header is None:
        actual = np.asarray(values, dtype=expected.dtype).reshape(
            shape, order=response.headers["X-Data-Order"]
        )
    else:
        fill_value = (
            float(fill_header) if expected.dtype.kind == "f" else int(fill_header)
        )
        actual = np.full(shape, fill_value, dtype=expected.dtype)
        actual.ravel(order=response.headers["X-Data-Order"])[indices] = values
    np.testing.assert_array_equal(actual, expected)


# ============================================================================
# GET /domains/{domain_id}/grids/{grid_id}/chunks/{chunk_index}
# ============================================================================


class TestGetChunkMetadata:
    def test_grid_not_found_returns_404(self, client, domain_for_testing):
        response = client.get(
            chunk_route(domain_for_testing["id"], "00000000000000000000000000000000", 0)
        )
        assert response.status_code == 404

    def test_grid_not_completed_returns_422(
        self, client, domain_for_testing, pending_grid_in_firestore
    ):
        response = client.get(
            chunk_route(domain_for_testing["id"], pending_grid_in_firestore["id"], 0)
        )
        assert response.status_code == 422

    def test_grid_wrong_owner_returns_404(
        self, client, domain_for_testing, grid_with_different_owner
    ):
        response = client.get(
            chunk_route(domain_for_testing["id"], grid_with_different_owner["id"], 0)
        )
        assert response.status_code == 404

    def test_grid_wrong_domain_returns_404(
        self, client, domain_with_different_owner, static_grid_in_firestore
    ):
        response = client.get(
            chunk_route(domain_with_different_owner["id"], STATIC_NAME, 0)
        )
        assert response.status_code == 404

    def test_chunk_out_of_range_returns_422(
        self, client, domain_for_testing, static_grid_in_firestore
    ):
        response = client.get(chunk_route(domain_for_testing["id"], STATIC_NAME, 1))
        assert response.status_code == 422

    def test_3d_chunk_out_of_range_returns_422(
        self, client, domain_for_testing, completed_3d_grid_in_firestore
    ):
        response = client.get(
            chunk_route(
                domain_for_testing["id"], completed_3d_grid_in_firestore["id"], 12
            )
        )
        assert response.status_code == 422

    def test_3d_offset_chunk_returns_correct_metadata(
        self, client, domain_for_testing, completed_3d_grid_in_firestore
    ):
        response = client.get(
            chunk_route(
                domain_for_testing["id"], completed_3d_grid_in_firestore["id"], 11
            )
        )
        assert response.status_code == 200

        data = response.json()
        assert data["index"] == 11
        assert data["shape"] == [1, 488, 288]
        assert data["offset"] == [4, 512, 512]
        assert data["transform"][2] == pytest.approx(500000.0 + 2.0 * 512)
        assert data["transform"][5] == pytest.approx(5201000.0 + (-2.0) * 512)
        assert data["z_origin"] == pytest.approx(12.0)
        assert data["z_resolution"] == pytest.approx(0.5)

    # Static-fixture integration

    def test_2d_chunk_0_matches_static_fixture(
        self, client, domain_for_testing, static_grid_in_firestore
    ):
        response = client.get(chunk_route(domain_for_testing["id"], STATIC_NAME, 0))
        assert response.status_code == 200

        data = response.json()
        georef = static_grid_in_firestore["georeference"]
        assert data["index"] == 0
        assert data["shape"] == list(georef["shape"])
        assert data["offset"] == [0, 0]
        assert "z_origin" not in data
        assert "z_resolution" not in data
        assert len(data["transform"]) == 6
        for i in range(6):
            assert data["transform"][i] == pytest.approx(georef["transform"][i])

    def test_3d_chunk_0_matches_static_fixture(
        self, client, domain_for_testing, static_3d_grid_in_firestore
    ):
        response = client.get(chunk_route(domain_for_testing["id"], STATIC_3D_NAME, 0))
        assert response.status_code == 200

        data = response.json()
        georef = static_3d_grid_in_firestore["georeference"]
        chunk_shape = static_3d_grid_in_firestore["chunk_shape"]
        expected_shape = [
            min(chunk, dimension)
            for chunk, dimension in zip(chunk_shape, georef["shape"])
        ]

        assert data["index"] == 0
        assert data["shape"] == expected_shape
        assert data["offset"] == [0, 0, 0]
        assert data["z_origin"] == pytest.approx(georef["z_origin"])
        assert data["z_resolution"] == pytest.approx(georef["z_resolution"])
        assert len(data["transform"]) == 6
        for i in range(6):
            assert data["transform"][i] == pytest.approx(georef["transform"][i])


# ============================================================================
# GET /domains/{domain_id}/grids/{grid_id}/data/{band}/{chunk_index}
# ============================================================================


class TestGetGridDataJson:
    # Validation

    def test_grid_not_found_returns_404(self, client, domain_for_testing):
        url, params = data_route(
            domain_for_testing["id"], "00000000000000000000000000000000", band="fbfm"
        )
        response = client.get(url, params=params)
        assert response.status_code == 404

    def test_grid_not_completed_returns_422(
        self, client, domain_for_testing, pending_grid_in_firestore
    ):
        url, params = data_route(
            domain_for_testing["id"], pending_grid_in_firestore["id"], band="fbfm"
        )
        response = client.get(url, params=params)
        assert response.status_code == 422

    def test_grid_wrong_owner_returns_404(
        self, client, domain_for_testing, grid_with_different_owner
    ):
        url, params = data_route(
            domain_for_testing["id"], grid_with_different_owner["id"], band="fbfm"
        )
        response = client.get(url, params=params)
        assert response.status_code == 404

    def test_grid_wrong_domain_returns_404(
        self, client, domain_with_different_owner, static_grid_in_firestore
    ):
        url, params = data_route(
            domain_with_different_owner["id"], STATIC_NAME, band="fbfm"
        )
        response = client.get(url, params=params)
        assert response.status_code == 404

    def test_invalid_band_returns_422(
        self, client, domain_for_testing, static_grid_in_firestore
    ):
        url, params = data_route(
            domain_for_testing["id"], STATIC_NAME, band="nonexistent_band"
        )
        response = client.get(url, params=params)
        assert response.status_code == 422

    def test_chunk_out_of_range_returns_422(
        self, client, domain_for_testing, static_grid_in_firestore
    ):
        url, params = data_route(
            domain_for_testing["id"], STATIC_NAME, chunk_index=99, band="fbfm"
        )
        response = client.get(url, params=params)
        assert response.status_code == 422

    def test_missing_band_path_segment_returns_404(
        self, client, domain_for_testing, static_grid_in_firestore
    ):
        response = client.get(
            f"/domains/{domain_for_testing['id']}/grids/{STATIC_NAME}/data"
        )
        assert response.status_code == 404

    # Default behavior

    def test_default_params_are_dense_C_order(
        self, client, domain_for_testing, static_grid_in_firestore
    ):
        url, params = data_route(
            domain_for_testing["id"], STATIC_NAME, band=STATIC_BAND
        )
        response = client.get(url, params=params)
        assert response.status_code == 200

        payload = response.json()
        assert payload["order"] == "C"
        assert payload["data"]["format"] == "dense"
        assert payload["shape"] == list(
            static_grid_in_firestore["georeference"]["shape"]
        )

    # Dense unit tests

    @pytest.mark.anyio
    async def test_3d_chunk_reads_with_z_y_x_slices(self, monkeypatch):
        # Verifies _read_grid_chunk slices a 3D grid in (z, y, x) order.
        grid_data = {
            "id": "grid-3d",
            "domain_id": "domain-1",
            "owner_id": "owner-1",
            "status": "completed",
            "bands": [
                {
                    "key": "bulk_density.foliage.live",
                    "type": "continuous",
                    "unit": "kg/m³",
                    "index": 0,
                }
            ],
            "georeference": {
                "crs": "EPSG:32611",
                "transform": (2.0, 0.0, 500000.0, 0.0, -2.0, 5201000.0),
                "shape": [5, 4, 6],
                "z_resolution": 0.5,
                "z_origin": 10.0,
            },
            "chunk_shape": [2, 2, 3],
        }

        fake_array = FakeGridArray(np.arange(12, dtype=np.float32).reshape((2, 2, 3)))

        async def fake_get_document_async(*args, **kwargs):
            return None, SimpleNamespace(to_dict=lambda: grid_data)

        async def fake_get_grid_array(grid_id, band):
            assert grid_id == "grid-3d"
            assert band == "bulk_density.foliage.live"
            return fake_array

        monkeypatch.setattr(grids_router, "get_document_async", fake_get_document_async)
        monkeypatch.setattr(grids_router, "get_grid_array", fake_get_grid_array)

        response = await grids_router.get_grid_data_json(
            request=SimpleNamespace(state=SimpleNamespace(id="owner-1")),
            domain={"id": "domain-1"},
            grid_id="grid-3d",
            chunk_index=5,
            band="bulk_density.foliage.live",
            array_format=GridDataArrayFormat.dense,
            order=GridDataOrder.C,
        )

        assert fake_array.selection == (slice(2, 4), slice(0, 2), slice(3, 6))
        assert response.shape == [2, 2, 3]
        assert response.data.format == "dense"
        assert response.data.values == list(range(12))

    @pytest.mark.anyio
    async def test_dense_above_scalar_limit_returns_413(self, monkeypatch):
        rows = 1001
        cols = grids_router.MAX_JSON_SCALARS // rows + 1
        fake_array = FakeGridArray(np.zeros((rows, cols), dtype=np.float32))
        assert fake_array.data.size > grids_router.MAX_JSON_SCALARS

        with pytest.raises(HTTPException) as exc:
            await _call_handler(
                monkeypatch,
                fake_array,
                grids_router.get_grid_data_json,
                array_format=GridDataArrayFormat.dense,
            )

        assert exc.value.status_code == 413
        assert "sparse" in exc.value.detail

    # Sparse unit tests

    @pytest.mark.anyio
    async def test_sparse_empty_chunk_returns_empty_indices(self, monkeypatch):
        fake_array = FakeGridArray(np.zeros((2, 3), dtype=np.float32), fill_value=0.0)

        response = await _call_handler(
            monkeypatch,
            fake_array,
            grids_router.get_grid_data_json,
            array_format=GridDataArrayFormat.sparse,
        )

        assert response.shape == [2, 3]
        assert response.data.format == "sparse"
        assert response.data.fill_value == 0.0
        assert response.data.indices == []
        assert response.data.values == []

    @pytest.mark.anyio
    async def test_sparse_uses_nonzero_fill_value(self, monkeypatch):
        fake_array = FakeGridArray(
            np.array([[-1, 0, 7]], dtype=np.int32), fill_value=-1
        )

        response = await _call_handler(
            monkeypatch,
            fake_array,
            grids_router.get_grid_data_json,
            band="tree_id",
            array_format=GridDataArrayFormat.sparse,
        )

        assert response.data.fill_value == -1
        assert response.data.indices == [1, 2]
        assert response.data.values == [0, 7]

    @pytest.mark.anyio
    async def test_sparse_no_fill_value_returns_all_cells_with_null_fill(
        self, monkeypatch
    ):
        # Band has no fill value defined → sparse returns every cell with
        # fill_value=null (no compression applied).
        fake_array = FakeGridArray(
            np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32), fill_value=None
        )

        response = await _call_handler(
            monkeypatch,
            fake_array,
            grids_router.get_grid_data_json,
            array_format=GridDataArrayFormat.sparse,
        )

        assert response.data.format == "sparse"
        assert response.data.fill_value is None
        assert response.data.indices == [0, 1, 2, 3, 4, 5]
        assert response.data.values == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

    @pytest.mark.anyio
    async def test_sparse_order_f_uses_fortran_flat_indices(self, monkeypatch):
        fake_array = FakeGridArray(
            np.array([[0, 1, 0], [2, 0, 3]], dtype=np.float32), fill_value=0.0
        )

        response = await _call_handler(
            monkeypatch,
            fake_array,
            grids_router.get_grid_data_json,
            array_format=GridDataArrayFormat.sparse,
            order=GridDataOrder.F,
        )

        assert response.order == "F"
        assert response.data.indices == [1, 2, 5]
        assert response.data.values == [2.0, 1.0, 3.0]

    @pytest.mark.anyio
    async def test_sparse_above_scalar_limit_returns_413(self, monkeypatch):
        # Fully-dense data (no fill values) maximizes nnz.
        rows = 1001
        cols = grids_router.MAX_JSON_SCALARS // rows + 1
        fake_array = FakeGridArray(
            np.ones((rows, cols), dtype=np.float32), fill_value=0.0
        )
        assert 2 * fake_array.data.size > grids_router.MAX_JSON_SCALARS

        with pytest.raises(HTTPException) as exc:
            await _call_handler(
                monkeypatch,
                fake_array,
                grids_router.get_grid_data_json,
                array_format=GridDataArrayFormat.sparse,
            )

        assert exc.value.status_code == 413
        assert "binary" in exc.value.detail

    @pytest.mark.anyio
    async def test_sparse_chunk_above_int32_index_limit_returns_413(self, monkeypatch):
        # Patch the int32 limit down so the guard fires without allocating
        # a 2.1B-element array.
        monkeypatch.setattr(grids_router, "MAX_SPARSE_INDEX", 5)
        fake_array = FakeGridArray(np.zeros((3, 3), dtype=np.float32), fill_value=0.0)
        assert fake_array.data.size > 5

        with pytest.raises(HTTPException) as exc:
            await _call_handler(
                monkeypatch,
                fake_array,
                grids_router.get_grid_data_json,
                array_format=GridDataArrayFormat.sparse,
            )

        assert exc.value.status_code == 413
        assert "int32" in exc.value.detail

    # Static-fixture integration

    def test_2d_dense_matches_static_zarr(
        self, client, domain_for_testing, static_grid_in_firestore
    ):
        metadata_response = client.get(
            chunk_route(domain_for_testing["id"], STATIC_NAME, 0)
        )
        assert metadata_response.status_code == 200
        metadata = metadata_response.json()

        url, params = data_route(
            domain_for_testing["id"], STATIC_NAME, band=STATIC_BAND
        )
        response = client.get(url, params=params)
        assert response.status_code == 200

        expected = _read_static_zarr_chunk(STATIC_NAME, STATIC_BAND, metadata)
        _assert_dense_json_matches_zarr(response.json(), expected)

    def test_2d_sparse_matches_static_zarr(
        self, client, domain_for_testing, static_grid_in_firestore
    ):
        metadata_response = client.get(
            chunk_route(domain_for_testing["id"], STATIC_NAME, 0)
        )
        assert metadata_response.status_code == 200
        metadata = metadata_response.json()

        url, params = data_route(
            domain_for_testing["id"],
            STATIC_NAME,
            band=STATIC_BAND,
            array_format="sparse",
        )
        response = client.get(url, params=params)
        assert response.status_code == 200

        expected = _read_static_zarr_chunk(STATIC_NAME, STATIC_BAND, metadata)
        _assert_sparse_json_matches_zarr(response.json(), expected)

    def test_3d_dense_returns_413_for_full_static_chunk(
        self, client, domain_for_testing, static_3d_grid_in_firestore
    ):
        # The static 3D fixture's chunk 0 covers the whole grid (~7M cells),
        # well above MAX_JSON_SCALARS. Dense JSON should refuse with a hint
        # pointing the client at sparse or binary. Data correctness for 3D
        # dense is verified via the binary route.
        url, params = data_route(
            domain_for_testing["id"], STATIC_3D_NAME, band=STATIC_3D_BAND
        )
        response = client.get(url, params=params)
        assert response.status_code == 413
        detail = response.json()["detail"]
        assert "sparse" in detail or "binary" in detail

    def test_3d_sparse_matches_static_zarr(
        self, client, domain_for_testing, static_3d_grid_in_firestore
    ):
        metadata_response = client.get(
            chunk_route(domain_for_testing["id"], STATIC_3D_NAME, 0)
        )
        assert metadata_response.status_code == 200
        metadata = metadata_response.json()

        url, params = data_route(
            domain_for_testing["id"],
            STATIC_3D_NAME,
            band=STATIC_3D_BAND,
            array_format="sparse",
        )
        response = client.get(url, params=params)
        assert response.status_code == 200

        expected = _read_static_zarr_chunk(STATIC_3D_NAME, STATIC_3D_BAND, metadata)
        _assert_sparse_json_matches_zarr(response.json(), expected)


# ============================================================================
# GET /domains/{domain_id}/grids/{grid_id}/data/{band}/{chunk_index}/binary
# ============================================================================


class TestGetGridDataBinary:
    # Validation

    def test_grid_not_found_returns_404(self, client, domain_for_testing):
        url, params = binary_data_route(
            domain_for_testing["id"], "00000000000000000000000000000000", band="fbfm"
        )
        response = client.get(url, params=params)
        assert response.status_code == 404

    def test_grid_not_completed_returns_422(
        self, client, domain_for_testing, pending_grid_in_firestore
    ):
        url, params = binary_data_route(
            domain_for_testing["id"], pending_grid_in_firestore["id"], band="fbfm"
        )
        response = client.get(url, params=params)
        assert response.status_code == 422

    def test_grid_wrong_owner_returns_404(
        self, client, domain_for_testing, grid_with_different_owner
    ):
        url, params = binary_data_route(
            domain_for_testing["id"], grid_with_different_owner["id"], band="fbfm"
        )
        response = client.get(url, params=params)
        assert response.status_code == 404

    def test_grid_wrong_domain_returns_404(
        self, client, domain_with_different_owner, static_grid_in_firestore
    ):
        url, params = binary_data_route(
            domain_with_different_owner["id"], STATIC_NAME, band="fbfm"
        )
        response = client.get(url, params=params)
        assert response.status_code == 404

    def test_invalid_band_returns_422(
        self, client, domain_for_testing, static_grid_in_firestore
    ):
        url, params = binary_data_route(
            domain_for_testing["id"], STATIC_NAME, band="nonexistent_band"
        )
        response = client.get(url, params=params)
        assert response.status_code == 422

    def test_chunk_out_of_range_returns_422(
        self, client, domain_for_testing, static_grid_in_firestore
    ):
        url, params = binary_data_route(
            domain_for_testing["id"], STATIC_NAME, chunk_index=99, band="fbfm"
        )
        response = client.get(url, params=params)
        assert response.status_code == 422

    # Default behavior

    def test_default_params_are_dense_C_order(
        self, client, domain_for_testing, static_grid_in_firestore
    ):
        url, params = binary_data_route(
            domain_for_testing["id"], STATIC_NAME, band=STATIC_BAND
        )
        response = client.get(url, params=params)
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/octet-stream"
        assert response.headers["X-Data-Order"] == "C"
        assert response.headers["X-Data-Format"] == "dense"
        assert "X-Data-Dtype" in response.headers
        expected_shape = static_grid_in_firestore["georeference"]["shape"]
        assert response.headers["X-Data-Shape"] == ",".join(
            str(s) for s in expected_shape
        )

    # Dense unit tests

    @pytest.mark.anyio
    async def test_dense_returns_correct_headers_and_bytes(self, monkeypatch):
        data = np.arange(6, dtype=np.float32).reshape((2, 3))
        fake_array = FakeGridArray(data)

        response = await _call_handler(
            monkeypatch,
            fake_array,
            grids_router.get_grid_data_binary,
            array_format=GridDataArrayFormat.dense,
        )

        assert response.headers["X-Data-Shape"] == "2,3"
        assert response.headers["X-Data-Dtype"] == "float32"
        assert response.headers["X-Data-Order"] == "C"
        assert response.headers["X-Data-Format"] == "dense"
        assert response.body == data.ravel(order="C").tobytes()

    @pytest.mark.anyio
    async def test_dense_above_byte_limit_returns_413(self, monkeypatch):
        # float64 8 bytes/cell → just over MAX_BINARY_BYTES.
        cells_needed = grids_router.MAX_BINARY_BYTES // 8 + 1
        cols = 4096
        rows = cells_needed // cols + 1
        fake_array = FakeGridArray(np.zeros((rows, cols), dtype=np.float64))
        assert fake_array.data.size * 8 > grids_router.MAX_BINARY_BYTES

        with pytest.raises(HTTPException) as exc:
            await _call_handler(
                monkeypatch,
                fake_array,
                grids_router.get_grid_data_binary,
                array_format=GridDataArrayFormat.dense,
            )

        assert exc.value.status_code == 413
        assert "sparse" in exc.value.detail

    # Sparse unit tests

    @pytest.mark.anyio
    async def test_sparse_returns_correct_headers_and_bytes(self, monkeypatch):
        data = np.array([[0, 1, 0], [2, 0, 3]], dtype=np.float32)
        fake_array = FakeGridArray(data, fill_value=0.0)

        response = await _call_handler(
            monkeypatch,
            fake_array,
            grids_router.get_grid_data_binary,
            array_format=GridDataArrayFormat.sparse,
        )

        assert response.headers["X-Data-Shape"] == "2,3"
        assert response.headers["X-Data-Order"] == "C"
        assert response.headers["X-Data-Format"] == "sparse"
        assert response.headers["X-Data-Fill-Value"] == "0.0"
        assert response.headers["X-Data-NNZ"] == "3"
        assert response.headers["X-Data-Index-Dtype"] == "int32"
        assert response.headers["X-Data-Value-Dtype"] == "float32"

        nnz = 3
        index_bytes = nnz * np.dtype(np.int32).itemsize
        indices = np.frombuffer(response.body[:index_bytes], dtype=np.int32)
        values = np.frombuffer(response.body[index_bytes:], dtype=np.float32)
        assert indices.tolist() == [1, 3, 5]
        assert values.tolist() == [1.0, 2.0, 3.0]

    @pytest.mark.anyio
    async def test_sparse_empty_chunk_returns_zero_nnz(self, monkeypatch):
        fake_array = FakeGridArray(np.zeros((2, 3), dtype=np.float32), fill_value=0.0)

        response = await _call_handler(
            monkeypatch,
            fake_array,
            grids_router.get_grid_data_binary,
            array_format=GridDataArrayFormat.sparse,
        )

        assert response.headers["X-Data-NNZ"] == "0"
        assert response.body == b""

    @pytest.mark.anyio
    async def test_sparse_no_fill_value_omits_fill_header_and_lists_all_cells(
        self, monkeypatch
    ):
        data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        fake_array = FakeGridArray(data, fill_value=None)

        response = await _call_handler(
            monkeypatch,
            fake_array,
            grids_router.get_grid_data_binary,
            array_format=GridDataArrayFormat.sparse,
        )

        assert response.headers["X-Data-Format"] == "sparse"
        assert "X-Data-Fill-Value" not in response.headers
        assert response.headers["X-Data-NNZ"] == "6"

        index_bytes = 6 * np.dtype(np.int32).itemsize
        indices = np.frombuffer(response.body[:index_bytes], dtype=np.int32)
        values = np.frombuffer(response.body[index_bytes:], dtype=np.float32)
        assert indices.tolist() == [0, 1, 2, 3, 4, 5]
        assert values.tolist() == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

    @pytest.mark.anyio
    async def test_sparse_order_f_uses_fortran_flat_indices(self, monkeypatch):
        data = np.array([[0, 1, 0], [2, 0, 3]], dtype=np.float32)
        fake_array = FakeGridArray(data, fill_value=0.0)

        response = await _call_handler(
            monkeypatch,
            fake_array,
            grids_router.get_grid_data_binary,
            array_format=GridDataArrayFormat.sparse,
            order=GridDataOrder.F,
        )

        assert response.headers["X-Data-Order"] == "F"
        nnz = int(response.headers["X-Data-NNZ"])
        index_bytes = nnz * np.dtype(np.int32).itemsize
        indices = np.frombuffer(response.body[:index_bytes], dtype=np.int32)
        values = np.frombuffer(response.body[index_bytes:], dtype=np.float32)
        assert indices.tolist() == [1, 2, 5]
        assert values.tolist() == [2.0, 1.0, 3.0]

    @pytest.mark.anyio
    async def test_sparse_above_byte_limit_returns_413(self, monkeypatch):
        # 4 (int32 index) + 8 (float64 value) = 12 bytes per nnz.
        target_bytes = grids_router.MAX_BINARY_BYTES + 1024
        nnz_needed = target_bytes // 12 + 1
        cols = 4096
        rows = nnz_needed // cols + 1
        fake_array = FakeGridArray(
            np.ones((rows, cols), dtype=np.float64), fill_value=0.0
        )
        assert fake_array.data.size * 12 > grids_router.MAX_BINARY_BYTES

        with pytest.raises(HTTPException) as exc:
            await _call_handler(
                monkeypatch,
                fake_array,
                grids_router.get_grid_data_binary,
                array_format=GridDataArrayFormat.sparse,
            )

        assert exc.value.status_code == 413
        assert "smaller chunk" in exc.value.detail

    @pytest.mark.anyio
    async def test_sparse_chunk_above_int32_index_limit_returns_413(self, monkeypatch):
        monkeypatch.setattr(grids_router, "MAX_SPARSE_INDEX", 5)
        fake_array = FakeGridArray(np.zeros((3, 3), dtype=np.float32), fill_value=0.0)
        assert fake_array.data.size > 5

        with pytest.raises(HTTPException) as exc:
            await _call_handler(
                monkeypatch,
                fake_array,
                grids_router.get_grid_data_binary,
                array_format=GridDataArrayFormat.sparse,
            )

        assert exc.value.status_code == 413
        assert "int32" in exc.value.detail

    # Static-fixture integration

    def test_2d_dense_matches_static_zarr(
        self, client, domain_for_testing, static_grid_in_firestore
    ):
        metadata_response = client.get(
            chunk_route(domain_for_testing["id"], STATIC_NAME, 0)
        )
        assert metadata_response.status_code == 200
        metadata = metadata_response.json()

        url, params = binary_data_route(
            domain_for_testing["id"], STATIC_NAME, band=STATIC_BAND
        )
        response = client.get(url, params=params)
        assert response.status_code == 200

        expected = _read_static_zarr_chunk(STATIC_NAME, STATIC_BAND, metadata)
        _assert_dense_binary_matches_zarr(response, expected)

    def test_2d_sparse_matches_static_zarr(
        self, client, domain_for_testing, static_grid_in_firestore
    ):
        metadata_response = client.get(
            chunk_route(domain_for_testing["id"], STATIC_NAME, 0)
        )
        assert metadata_response.status_code == 200
        metadata = metadata_response.json()

        url, params = binary_data_route(
            domain_for_testing["id"],
            STATIC_NAME,
            band=STATIC_BAND,
            array_format="sparse",
        )
        response = client.get(url, params=params)
        assert response.status_code == 200

        expected = _read_static_zarr_chunk(STATIC_NAME, STATIC_BAND, metadata)
        _assert_sparse_binary_matches_zarr(response, expected)

    def test_3d_dense_matches_static_zarr(
        self, client, domain_for_testing, static_3d_grid_in_firestore
    ):
        metadata_response = client.get(
            chunk_route(domain_for_testing["id"], STATIC_3D_NAME, 0)
        )
        assert metadata_response.status_code == 200
        metadata = metadata_response.json()

        url, params = binary_data_route(
            domain_for_testing["id"], STATIC_3D_NAME, band=STATIC_3D_BAND
        )
        response = client.get(url, params=params)
        assert response.status_code == 200

        expected = _read_static_zarr_chunk(STATIC_3D_NAME, STATIC_3D_BAND, metadata)
        _assert_dense_binary_matches_zarr(response, expected)

    def test_3d_sparse_matches_static_zarr(
        self, client, domain_for_testing, static_3d_grid_in_firestore
    ):
        metadata_response = client.get(
            chunk_route(domain_for_testing["id"], STATIC_3D_NAME, 0)
        )
        assert metadata_response.status_code == 200
        metadata = metadata_response.json()

        url, params = binary_data_route(
            domain_for_testing["id"],
            STATIC_3D_NAME,
            band=STATIC_3D_BAND,
            array_format="sparse",
        )
        response = client.get(url, params=params)
        assert response.status_code == 200

        expected = _read_static_zarr_chunk(STATIC_3D_NAME, STATIC_3D_BAND, metadata)
        _assert_sparse_binary_matches_zarr(response, expected)
