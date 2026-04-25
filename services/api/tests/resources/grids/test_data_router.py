"""
Integration tests for grid data streaming endpoints.

Tests the grid data streaming endpoints:
  GET /domains/{domain_id}/grids/{grid_id}/chunks/{chunk_index}
  GET /domains/{domain_id}/grids/{grid_id}/data/{band}/{chunk_index}

These tests use static grid fixtures which have real zarr data on GCS.
"""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr
from api.resources.grids import router as grids_router
from api.resources.grids.schema import GridDataFormat, GridDataOrder

from lib.config import GRIDS_BUCKET, GRIDS_COLLECTION
from lib.testing import SHARED_TEST_GRIDS_DIR
from tests.fixtures import make_grid_data

STATIC_NAME = "static-test-blue-mtn-landfire-fbfm40"
STATIC_3D_NAME = "static-test-blue-mtn-tree-inventory-voxels"
STATIC_3D_BAND = "bulk_density.foliage.live"


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
        assert tuple(data_array.dims) == ("z", "y", "x")
        return data_array.isel(z=slices[0], y=slices[1], x=slices[2]).values
    finally:
        ds.close()


# Fixtures


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
            "resolution": [2.0, 2.0, 1.0],
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


# Helpers


def chunk_route(domain_id, grid_id, chunk_index):
    return f"/domains/{domain_id}/grids/{grid_id}/chunks/{chunk_index}"


def data_route(domain_id, grid_id, band, chunk_index=0, **params):
    return f"/domains/{domain_id}/grids/{grid_id}/data/{band}/{chunk_index}", params


# GET /domains/{domain_id}/grids/{grid_id}/chunks/{chunk_index}


class TestGetGridDataChunkMetadata:
    def test_chunk_0_returns_200(
        self, client, domain_for_testing, static_grid_in_firestore
    ):
        """Chunk 0 of static fixture returns correct metadata."""
        response = client.get(chunk_route(domain_for_testing["id"], STATIC_NAME, 0))
        assert response.status_code == 200

        data = response.json()
        georef = static_grid_in_firestore["georeference"]
        # Static fixture fits in a single chunk (chunk_shape is 512×512), so
        # chunk 0 covers the entire grid.
        assert data["index"] == 0
        assert data["shape"] == list(georef["shape"])
        assert data["offset"] == [0, 0]
        assert "z_origin" not in data
        assert "z_resolution" not in data
        assert len(data["transform"]) == 6
        for i in range(6):
            assert data["transform"][i] == pytest.approx(georef["transform"][i])

    def test_3d_chunk_returns_200(
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

    def test_static_3d_chunk_returns_200(
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

    def test_3d_chunk_out_of_range_returns_422(
        self, client, domain_for_testing, completed_3d_grid_in_firestore
    ):
        response = client.get(
            chunk_route(
                domain_for_testing["id"], completed_3d_grid_in_firestore["id"], 12
            )
        )
        assert response.status_code == 422

    def test_chunk_out_of_range_returns_422(
        self, client, domain_for_testing, static_grid_in_firestore
    ):
        """Chunk index beyond total chunks returns 422."""
        response = client.get(chunk_route(domain_for_testing["id"], STATIC_NAME, 1))
        assert response.status_code == 422

    def test_grid_not_completed_returns_422(
        self, client, domain_for_testing, pending_grid_in_firestore
    ):
        response = client.get(
            chunk_route(domain_for_testing["id"], pending_grid_in_firestore["id"], 0)
        )
        assert response.status_code == 422

    def test_grid_not_found_returns_404(self, client, domain_for_testing):
        response = client.get(
            chunk_route(
                domain_for_testing["id"],
                "00000000000000000000000000000000",
                0,
            )
        )
        assert response.status_code == 404

    def test_grid_wrong_owner_returns_404(
        self, client, domain_for_testing, grid_with_different_owner
    ):
        response = client.get(
            chunk_route(
                domain_for_testing["id"],
                grid_with_different_owner["id"],
                0,
            )
        )
        assert response.status_code == 404

    def test_grid_wrong_domain_returns_404(
        self, client, domain_with_different_owner, static_grid_in_firestore
    ):
        response = client.get(
            chunk_route(domain_with_different_owner["id"], STATIC_NAME, 0)
        )
        assert response.status_code == 404


# GET /domains/{domain_id}/grids/{grid_id}/data/{band}/{chunk_index}


class TestGetGridData:
    def test_json_format_returns_200(
        self, client, domain_for_testing, static_grid_in_firestore
    ):
        """JSON response with band data."""
        url, params = data_route(
            domain_for_testing["id"],
            STATIC_NAME,
            band="fbfm",
            format="json",
        )
        response = client.get(url, params=params)
        assert response.status_code == 200

        data = response.json()
        expected_shape = list(static_grid_in_firestore["georeference"]["shape"])
        assert data["shape"] == expected_shape
        assert data["order"] == "C"
        assert isinstance(data["data"], list)
        assert len(data["data"]) == expected_shape[0] * expected_shape[1]

    def test_3d_json_format_matches_static_zarr_chunk(
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
            format="json",
            order="C",
        )
        response = client.get(url, params=params)
        assert response.status_code == 200

        payload = response.json()
        expected = _read_static_zarr_chunk(STATIC_3D_NAME, STATIC_3D_BAND, metadata)
        actual = np.asarray(payload["data"], dtype=expected.dtype).reshape(
            payload["shape"],
            order=payload["order"],
        )

        assert payload["shape"] == metadata["shape"]
        np.testing.assert_allclose(actual, expected)

    def test_binary_format_returns_200(
        self, client, domain_for_testing, static_grid_in_firestore
    ):
        """Binary response has correct headers."""
        url, params = data_route(
            domain_for_testing["id"],
            STATIC_NAME,
            band="fbfm",
            format="binary",
        )
        response = client.get(url, params=params)
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/octet-stream"
        assert "X-Data-Shape" in response.headers
        assert "X-Data-Dtype" in response.headers
        assert response.headers["X-Data-Order"] == "C"
        expected_shape = static_grid_in_firestore["georeference"]["shape"]
        assert response.headers["X-Data-Shape"] == ",".join(
            str(s) for s in expected_shape
        )

    def test_default_params(self, client, domain_for_testing, static_grid_in_firestore):
        """Defaults: format=json, order=C."""
        url, params = data_route(domain_for_testing["id"], STATIC_NAME, band="fbfm")
        response = client.get(url, params=params)
        assert response.status_code == 200

        data = response.json()
        assert data["order"] == "C"
        assert data["shape"] == list(static_grid_in_firestore["georeference"]["shape"])

    def test_missing_band_returns_404(
        self, client, domain_for_testing, static_grid_in_firestore
    ):
        """Omitting band from the path returns 404 (no matching route)."""
        response = client.get(
            f"/domains/{domain_for_testing['id']}/grids/{STATIC_NAME}/data"
        )
        assert response.status_code == 404

    def test_invalid_band_returns_422(
        self, client, domain_for_testing, static_grid_in_firestore
    ):
        """Non-existent band returns 422."""
        url, params = data_route(
            domain_for_testing["id"],
            STATIC_NAME,
            band="nonexistent_band",
        )
        response = client.get(url, params=params)
        assert response.status_code == 422

    def test_chunk_out_of_range_returns_422(
        self, client, domain_for_testing, static_grid_in_firestore
    ):
        url, params = data_route(
            domain_for_testing["id"],
            STATIC_NAME,
            chunk_index=99,
            band="fbfm",
        )
        response = client.get(url, params=params)
        assert response.status_code == 422

    def test_grid_not_completed_returns_422(
        self, client, domain_for_testing, pending_grid_in_firestore
    ):
        url, params = data_route(
            domain_for_testing["id"],
            pending_grid_in_firestore["id"],
            band="fbfm",
        )
        response = client.get(url, params=params)
        assert response.status_code == 422

    def test_grid_not_found_returns_404(self, client, domain_for_testing):
        url, params = data_route(
            domain_for_testing["id"],
            "00000000000000000000000000000000",
            band="fbfm",
        )
        response = client.get(url, params=params)
        assert response.status_code == 404

    def test_grid_wrong_owner_returns_404(
        self, client, domain_for_testing, grid_with_different_owner
    ):
        url, params = data_route(
            domain_for_testing["id"],
            grid_with_different_owner["id"],
            band="fbfm",
        )
        response = client.get(url, params=params)
        assert response.status_code == 404

    def test_grid_wrong_domain_returns_404(
        self, client, domain_with_different_owner, static_grid_in_firestore
    ):
        url, params = data_route(
            domain_with_different_owner["id"],
            STATIC_NAME,
            band="fbfm",
        )
        response = client.get(url, params=params)
        assert response.status_code == 404

    @pytest.mark.anyio
    async def test_3d_data_route_reads_with_z_y_x_slices(self, monkeypatch):
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

        class FakeArray:
            selection = None

            async def getitem(self, selection):
                self.selection = selection
                return np.arange(12, dtype=np.float32).reshape((2, 2, 3))

        fake_array = FakeArray()

        async def fake_get_document_async(*args, **kwargs):
            return None, SimpleNamespace(to_dict=lambda: grid_data)

        async def fake_get_grid_array(grid_id, band):
            assert grid_id == "grid-3d"
            assert band == "bulk_density.foliage.live"
            return fake_array

        monkeypatch.setattr(grids_router, "get_document_async", fake_get_document_async)
        monkeypatch.setattr(grids_router, "get_grid_array", fake_get_grid_array)

        response = await grids_router.get_grid_data(
            request=SimpleNamespace(state=SimpleNamespace(id="owner-1")),
            domain={"id": "domain-1"},
            grid_id="grid-3d",
            chunk_index=5,
            band="bulk_density.foliage.live",
            data_format=GridDataFormat.json,
            order=GridDataOrder.C,
        )

        assert fake_array.selection == (
            slice(2, 4),
            slice(0, 2),
            slice(3, 6),
        )
        assert response.shape == [2, 2, 3]
        assert response.data == list(range(12))
