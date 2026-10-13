"""
Parity of the off-loop point-cloud tile JSON body with response_model output.
"""

import json
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
from api.resources.point_clouds import utils
from api.resources.point_clouds.schema import PointCloudTileDataResponse
from fastapi import FastAPI
from fastapi.testclient import TestClient


def _selection():
    storage = SimpleNamespace(
        columns=[
            {"name": "X", "dtype": "int32"},
            {"name": "Y", "dtype": "int32"},
            {"name": "classification", "dtype": "uint8"},
        ],
        manifest={
            "tile_m": 100.0,
            "mins": [1000.0, 2000.0],
            "scales": [0.001, 0.001, 0.001],
            "offsets": [1000.0, 2000.0, 0.0],
        },
    )
    return utils.TileSelection(
        point_cloud_id="pc",
        storage=storage,
        tile={"tile_x": -1, "tile_y": 0},
        columns=["X", "classification"],
        classes=[1, 2],
        lod=3,
    )


def test_json_body_matches_response_model():
    selection = _selection()
    table = pa.table(
        {
            "X": pa.array([5, -7, 1_000_000], type=pa.int32()),
            "Y": pa.array([1, 2, 3], type=pa.int32()),
            "classification": pa.array([1, 2, 2], type=pa.uint8()),
        }
    )
    app = FastAPI()

    @app.get("/", response_model=PointCloudTileDataResponse)
    def route():
        return utils.json_response(selection, table)

    reply = TestClient(app).get("/")
    assert reply.status_code == 200
    assert json.loads(utils.json_body(selection, table)) == reply.json()


def _all_columns_selection():
    selection = _selection()
    selection.storage.columns = [
        {"name": "X", "dtype": "int32"},
        {"name": "intensity", "dtype": "uint16"},
        {"name": "classification", "dtype": "uint8"},
        {"name": "lod", "dtype": "uint8"},
    ]
    selection.columns = ["X", "intensity", "classification", "lod"]
    return selection


def test_json_body_exact_for_every_column_dtype_and_chunked_table():
    selection = _all_columns_selection()
    i32 = np.iinfo(np.int32)
    part = pa.table(
        {
            "X": pa.array([i32.min, i32.max, 0], type=pa.int32()),
            "intensity": pa.array([0, 65535, 1], type=pa.uint16()),
            "classification": pa.array([0, 255, 2], type=pa.uint8()),
            "lod": pa.array([0, 5, 3], type=pa.uint8()),
        }
    )
    table = pa.concat_tables([part, part.slice(1)])
    assert table.column("X").num_chunks == 2
    expected = utils.json_response(selection, table).model_dump_json()
    body = utils.json_body(selection, table)
    # Integers only: the parsed body must match the Pydantic encoding exactly.
    assert json.dumps(json.loads(body)) == json.dumps(json.loads(expected))
    assert json.loads(body)["data"]["X"] == [i32.min, i32.max, 0, i32.max, 0]


def test_json_body_empty_tile():
    selection = _all_columns_selection()
    table = pa.table(
        {
            "X": pa.array([], type=pa.int32()),
            "intensity": pa.array([], type=pa.uint16()),
            "classification": pa.array([], type=pa.uint8()),
            "lod": pa.array([], type=pa.uint8()),
        }
    )
    body = json.loads(utils.json_body(selection, table))
    assert body["data"] == {"X": [], "intensity": [], "classification": [], "lod": []}
    assert body["columns"] == {
        "X": "int32",
        "intensity": "uint16",
        "classification": "uint8",
        "lod": "uint8",
    }
