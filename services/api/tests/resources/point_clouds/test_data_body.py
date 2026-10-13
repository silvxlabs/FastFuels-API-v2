"""
Parity of the off-loop point-cloud tile JSON body with response_model output.
"""

import json
from types import SimpleNamespace

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
