"""Unit tests for point-cloud tile metadata and streaming handlers."""

from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pytest
from api.app import app
from api.dependencies import get_verified_domain
from api.resources.point_clouds import router as point_clouds_router
from api.resources.point_clouds import utils as point_cloud_utils
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from lib.errors import ProcessingError
from lib.pointcloud.reader import PointCloudStorage

MANIFEST = {
    "tiles": 1,
    "tile_m": 500.0,
    "points": 4,
    "lod_levels": 6,
    "scales": [0.01, 0.01, 0.01],
    "offsets": [271000.0, 5185000.0, 0.0],
    "mins": [271000.0, 5185000.0, 1000.0],
    "maxs": [275000.0, 5189000.0, 2000.0],
}
COLUMNS = (
    {"name": "X", "dtype": "int32"},
    {"name": "Y", "dtype": "int32"},
    {"name": "Z", "dtype": "int32"},
    {"name": "intensity", "dtype": "uint16"},
    {"name": "classification", "dtype": "uint8"},
)
TILES = (
    {
        "tile_x": -1,
        "tile_y": 2,
        "points": 4,
        "points_by_lod": [1, 1, 1, 1, 1, 4],
    },
)
STORAGE = PointCloudStorage(
    prefix="bucket/cloud-1/cloud.parquet",
    manifest=MANIFEST,
    dataset=None,
    columns=COLUMNS,
    tiles=TILES,
)

RESOURCE = {
    "id": "cloud-1",
    "domain_id": "domain-1",
    "owner_id": "owner-1",
    "type": "als",
    "source": {"name": "3dep"},
    "status": "completed",
    "checksum": "checksum-1",
    "summary": {"point_count": 4, "point_classes": [2, 5], "density": 1.0},
    "georeference": {
        "crs": "EPSG:32612",
        "bounds": [270500.0, 5186000.0, 1000.0, 271000.0, 5186500.0, 2000.0],
    },
}


def _request():
    return SimpleNamespace(state=SimpleNamespace(id="owner-1"))


def _install_storage(monkeypatch, *, data=None):
    calls = {}

    async def get_document(*args, **kwargs):
        calls["document"] = (args, kwargs)
        return None, SimpleNamespace(to_dict=lambda: RESOURCE)

    async def get_storage(point_cloud_id, checksum):
        calls["storage"] = (point_cloud_id, checksum)
        return STORAGE

    async def get_tile(storage, tile_x, tile_y, columns, classes, lod):
        calls["tile"] = (storage, tile_x, tile_y, columns, classes, lod)
        if data is not None:
            return pa.table(data)
        tile = next(
            tile
            for tile in storage.tiles
            if (tile["tile_x"], tile["tile_y"]) == (tile_x, tile_y)
        )
        count = tile["points_by_lod"][lod]
        return pa.table(
            {
                name: np.arange(
                    count,
                    dtype=np.dtype(
                        next(
                            column["dtype"]
                            for column in storage.columns
                            if column["name"] == name
                        )
                    ),
                )
                for name in columns
            }
        )

    monkeypatch.setattr(point_cloud_utils, "get_document_async", get_document)
    monkeypatch.setattr(point_cloud_utils, "get_point_cloud_storage", get_storage)
    monkeypatch.setattr(point_cloud_utils, "get_point_cloud_tile", get_tile)
    return calls


def _http_client():
    test_app = FastAPI()

    @test_app.middleware("http")
    async def add_owner(request, call_next):
        request.state.id = "owner-1"
        return await call_next(request)

    test_app.dependency_overrides[get_verified_domain] = lambda: {"id": "domain-1"}
    test_app.include_router(
        point_clouds_router.router,
        prefix="/domains/{domain_id}/pointclouds",
    )
    return TestClient(test_app)


class TestMetadata:
    @pytest.mark.asyncio
    async def test_returns_public_catalogue_without_internal_parts(self, monkeypatch):
        calls = _install_storage(monkeypatch)

        response = await point_clouds_router.get_point_cloud_data_metadata(
            request=_request(), domain={"id": "domain-1"}, point_cloud_id="cloud-1"
        )
        payload = response.model_dump()

        assert payload["bounds"] == (270500.0, 5186000.0, 271000.0, 5186500.0)
        assert payload["tiles"][0] == {
            "tile_x": -1,
            "tile_y": 2,
            "bounds": (270500.0, 5186000.0, 271000.0, 5186500.0),
            "points_by_lod": [1, 1, 1, 1, 1, 4],
        }
        assert "parts" not in payload["tiles"][0]
        assert calls["storage"] == ("cloud-1", "checksum-1")
        _, kwargs = calls["document"]
        assert kwargs == {
            "owner_id": "owner-1",
            "domain_id": "domain-1",
            "document_status": "completed",
        }

    @pytest.mark.asyncio
    async def test_invalid_manifest_maps_to_422(self, monkeypatch):
        async def get_document(*args, **kwargs):
            return None, SimpleNamespace(to_dict=lambda: RESOURCE)

        async def get_storage(*args, **kwargs):
            raise ProcessingError(
                code="POINT_CLOUD_UNREADABLE",
                message="Stored data is invalid.",
                suggestion="Recreate it.",
            )

        monkeypatch.setattr(point_cloud_utils, "get_document_async", get_document)
        monkeypatch.setattr(point_cloud_utils, "get_point_cloud_storage", get_storage)

        with pytest.raises(HTTPException) as error:
            await point_clouds_router.get_point_cloud_data_metadata(
                request=_request(),
                domain={"id": "domain-1"},
                point_cloud_id="cloud-1",
            )

        assert error.value.status_code == 422
        assert error.value.detail == "Stored data is invalid. Recreate it."


def test_openapi_documents_routes_parameters_workflows_and_response_fields():
    openapi = app.openapi()
    paths = openapi["paths"]
    prefix = "/domains/{domain_id}/pointclouds/{point_cloud_id}/data"

    metadata = paths[f"{prefix}/metadata"]["get"]
    json_data = paths[f"{prefix}/{{tile_x}}/{{tile_y}}"]["get"]
    binary = paths[f"{prefix}/{{tile_x}}/{{tile_y}}/binary"]["get"]

    assert "# Get Point-Cloud Data Metadata" in metadata["description"]
    assert "## Response" in metadata["description"]
    assert "points_by_lod" in metadata["description"]
    assert "# Get Point-Cloud Tile Data as JSON" in json_data["description"]
    assert "## Query Parameters" in json_data["description"]
    assert "# Get Point-Cloud Tile Data as Binary" in binary["description"]
    assert "## Response Headers" in binary["description"]

    parameters = {parameter["name"]: parameter for parameter in json_data["parameters"]}
    assert set(parameters) == {
        "domain_id",
        "point_cloud_id",
        "tile_x",
        "tile_y",
        "lod",
        "classes",
        "columns",
    }
    for name in ("tile_x", "tile_y", "lod", "classes", "columns"):
        assert parameters[name]["description"]

    assert set(binary["responses"]["200"]["content"]) == {"application/octet-stream"}

    schemas = openapi["components"]["schemas"]
    for schema_name in ("PointCloudDataMetadata", "PointCloudTileDataResponse"):
        schema = schemas[schema_name]
        assert schema.get("examples")
        assert all(
            property_schema.get("description")
            for property_schema in schema["properties"].values()
        )

    example = schemas["PointCloudTileDataResponse"]["examples"][0]
    assert example["classes"] == [1, 2]
    assert list(example["columns"]) == ["X", "Y", "Z", "classification"]
    assert list(example["data"]) == list(example["columns"])
    assert {len(values) for values in example["data"].values()} == {2}


def test_http_request_and_response_bodies(monkeypatch):
    """Exercise FastAPI's real path/query parsing and response serialization."""
    _install_storage(monkeypatch)
    with _http_client() as client:
        response = client.get(
            "/domains/domain-1/pointclouds/cloud-1/data/-1/2",
            params={
                "lod": "4",
                "classes": "5,2,5",
                "columns": "Z,classification",
            },
        )
        binary = client.get(
            "/domains/domain-1/pointclouds/cloud-1/data/-1/2/binary",
            params={"lod": "0", "columns": "X,classification"},
        )

    assert response.status_code == 200
    assert response.json() == {
        "tile_x": -1,
        "tile_y": 2,
        "bounds": [270500.0, 5186000.0, 271000.0, 5186500.0],
        "lod": 4,
        "classes": [2, 5],
        "scales": [0.01, 0.01, 0.01],
        "offsets": [271000.0, 5185000.0, 0.0],
        "columns": {"Z": "int32", "classification": "uint8"},
        "data": {"Z": [0], "classification": [0]},
    }
    assert binary.status_code == 200
    assert binary.headers["content-type"] == "application/octet-stream"
    assert binary.headers["X-Data-Columns"] == "X,classification"
    assert len(binary.content) == (
        np.dtype("int32").itemsize + np.dtype("uint8").itemsize
    )


class TestJsonTile:
    @pytest.mark.asyncio
    async def test_projects_filters_and_reports_the_selection(self, monkeypatch):
        data = {
            "Z": np.array([100, 200], dtype=np.int32),
            "classification": np.array([2, 5], dtype=np.uint8),
        }
        calls = _install_storage(monkeypatch, data=data)

        response = await point_clouds_router.get_point_cloud_data_json(
            request=_request(),
            domain={"id": "domain-1"},
            point_cloud_id="cloud-1",
            tile_x=-1,
            tile_y=2,
            lod=4,
            classes="5,2,5",
            columns="Z,classification",
        )

        assert response.lod == 4
        assert response.classes == [2, 5]
        assert response.data == {"Z": [100, 200], "classification": [2, 5]}
        assert response.columns == {"Z": "int32", "classification": "uint8"}
        assert calls["tile"][3:] == (["Z", "classification"], [2, 5], 4)

    @pytest.mark.asyncio
    async def test_unknown_tile_returns_422_without_reading_data(self, monkeypatch):
        calls = _install_storage(monkeypatch)

        with pytest.raises(HTTPException) as error:
            await point_clouds_router.get_point_cloud_data_json(
                request=_request(),
                domain={"id": "domain-1"},
                point_cloud_id="cloud-1",
                tile_x=99,
                tile_y=99,
                lod=0,
                classes=None,
                columns="X",
            )

        assert error.value.status_code == 422
        assert "no tile" in error.value.detail
        assert "tile" not in calls

    @pytest.mark.parametrize(
        "params,detail",
        [
            ([("lod", "6"), ("columns", "X")], "lod must be"),
            ([("classes", "256"), ("columns", "X")], "classes"),
            ([("classes", "two"), ("columns", "X")], "classes"),
            ([("columns", "unknown")], "Unknown point-cloud columns"),
            ([("columns", "X,X")], "duplicates"),
        ],
    )
    def test_rejects_invalid_selection(self, monkeypatch, params, detail):
        _install_storage(monkeypatch)
        with _http_client() as client:
            response = client.get(
                "/domains/domain-1/pointclouds/cloud-1/data/-1/2", params=params
            )

        assert response.status_code == 422
        assert detail in response.text

    @pytest.mark.asyncio
    async def test_catalogue_count_prevents_oversized_read(self, monkeypatch):
        calls = _install_storage(monkeypatch)
        monkeypatch.setattr(point_cloud_utils, "MAX_JSON_SCALARS", 3)

        with pytest.raises(HTTPException) as error:
            await point_clouds_router.get_point_cloud_data_json(
                request=_request(),
                domain={"id": "domain-1"},
                point_cloud_id="cloud-1",
                tile_x=-1,
                tile_y=2,
                lod=5,
                classes=None,
                columns="X",
            )

        assert error.value.status_code == 413
        assert "tile" not in calls


class TestBinaryTile:
    @pytest.mark.asyncio
    async def test_concatenates_typed_column_blocks_and_headers(self, monkeypatch):
        data = {
            "X": np.array([1, 2, 3, 4], dtype=np.int32),
            "classification": np.array([2, 5, 2, 5], dtype=np.uint8),
        }
        _install_storage(monkeypatch, data=data)

        response = await point_clouds_router.get_point_cloud_data_binary(
            request=_request(),
            domain={"id": "domain-1"},
            point_cloud_id="cloud-1",
            tile_x=-1,
            tile_y=2,
            lod=5,
            classes=None,
            columns="X,classification",
        )

        assert response.media_type == "application/octet-stream"
        assert response.headers["X-Data-Columns"] == "X,classification"
        assert response.headers["X-Data-Dtypes"] == "int32,uint8"
        assert response.headers["X-Data-Count"] == "4"
        assert response.headers["X-Data-Tile"] == "-1,2"
        assert response.headers["X-Data-LOD"] == "5"
        assert response.headers["X-Data-Classes"] == "all"
        x_bytes = 4 * np.dtype("int32").itemsize
        np.testing.assert_array_equal(
            np.frombuffer(response.body[:x_bytes], dtype="<i4"), [1, 2, 3, 4]
        )
        np.testing.assert_array_equal(
            np.frombuffer(response.body[x_bytes:], dtype="u1"), [2, 5, 2, 5]
        )

    @pytest.mark.asyncio
    async def test_binary_size_limit_is_enforced_before_read(self, monkeypatch):
        calls = _install_storage(monkeypatch)
        monkeypatch.setattr(point_cloud_utils, "MAX_BINARY_BYTES", 15)

        with pytest.raises(HTTPException) as error:
            await point_clouds_router.get_point_cloud_data_binary(
                request=_request(),
                domain={"id": "domain-1"},
                point_cloud_id="cloud-1",
                tile_x=-1,
                tile_y=2,
                lod=5,
                classes=None,
                columns="X",
            )

        assert error.value.status_code == 413
        assert "tile" not in calls
