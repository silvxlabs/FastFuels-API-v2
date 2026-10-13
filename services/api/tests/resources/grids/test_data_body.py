"""
Parity of the off-loop grid JSON body with FastAPI's response_model output.
"""

import json

import numpy as np
import pytest
from api.resources.grids.router import (
    _grid_data_json_body,
    _json_scalar,
    _json_values,
    _sparse_components,
)
from api.resources.grids.schema import (
    DenseGridData,
    GridDataArrayFormat,
    GridDataChunkMetadata,
    GridDataOrder,
    GridDataResponse,
    SparseGridData,
)
from fastapi import FastAPI
from fastapi.testclient import TestClient


def _meta_2d():
    return GridDataChunkMetadata(
        index=0,
        shape=(3, 4),
        offset=(0, 0),
        transform=(30.0, 0.0, 500000.0, 0.0, -30.0, 5200000.0),
    )


def _meta_3d():
    return GridDataChunkMetadata(
        index=1,
        shape=(2, 3, 4),
        offset=(0, 0, 0),
        transform=(30.0, 0.0, 500000.0, 0.0, -30.0, 5200000.0),
        z_origin=10.0,
        z_resolution=0.5,
    )


def _reference(response: GridDataResponse) -> dict:
    # The path the route used before: response_model validation + serialization.
    app = FastAPI()

    @app.get("/", response_model=GridDataResponse)
    def route():
        return response

    reply = TestClient(app).get("/")
    assert reply.status_code == 200
    return reply.json()


def _expected(data, meta, fill_value, array_format, order):
    if array_format == GridDataArrayFormat.sparse:
        indices, values = _sparse_components(data, fill_value, order)
        payload = SparseGridData(
            format="sparse",
            fill_value=_json_scalar(fill_value),
            indices=indices.tolist(),
            values=_json_values(values),
        )
    else:
        payload = DenseGridData(
            format="dense", values=_json_values(data.ravel(order=order.value))
        )
    return _reference(
        GridDataResponse(
            shape=list(data.shape), order=order.value, metadata=meta, data=payload
        )
    )


def _float_chunk_2d():
    data = np.arange(12, dtype=np.float32).reshape(3, 4) / 7
    data[0, 1] = np.nan
    data[2, 3] = np.inf
    return data


CASES = [
    pytest.param(_float_chunk_2d(), _meta_2d(), np.nan, id="float-nan-2d"),
    pytest.param(
        np.arange(12, dtype=np.int16).reshape(3, 4) % 3, _meta_2d(), 0, id="int-2d"
    ),
    pytest.param(
        np.arange(24, dtype=np.float32).reshape(2, 3, 4), _meta_3d(), -9999.0, id="3d"
    ),
]


@pytest.mark.parametrize("data,meta,fill", CASES)
@pytest.mark.parametrize("array_format", list(GridDataArrayFormat))
@pytest.mark.parametrize("order", list(GridDataOrder))
def test_body_matches_response_model(data, meta, fill, array_format, order):
    body = _grid_data_json_body(data, meta, fill, array_format, order)
    assert json.loads(body) == _expected(data, meta, fill, array_format, order)


def test_2d_omits_z_fields_and_3d_keeps_them():
    body = json.loads(
        _grid_data_json_body(
            np.zeros((3, 4), dtype=np.float32),
            _meta_2d(),
            np.nan,
            GridDataArrayFormat.dense,
            GridDataOrder.C,
        )
    )
    assert "z_origin" not in body["metadata"]
    assert "z_resolution" not in body["metadata"]
    body = json.loads(
        _grid_data_json_body(
            np.zeros((2, 3, 4), dtype=np.float32),
            _meta_3d(),
            np.nan,
            GridDataArrayFormat.dense,
            GridDataOrder.C,
        )
    )
    assert body["metadata"]["z_origin"] == 10.0
    assert body["metadata"]["z_resolution"] == 0.5


def test_sparse_null_fill_value_is_present():
    body = json.loads(
        _grid_data_json_body(
            _float_chunk_2d(),
            _meta_2d(),
            np.nan,
            GridDataArrayFormat.sparse,
            GridDataOrder.C,
        )
    )
    assert "fill_value" in body["data"]
    assert body["data"]["fill_value"] is None


def test_non_finite_becomes_null_in_dense_values():
    body = json.loads(
        _grid_data_json_body(
            _float_chunk_2d(),
            _meta_2d(),
            np.nan,
            GridDataArrayFormat.dense,
            GridDataOrder.C,
        )
    )
    assert body["data"]["values"][1] is None
    assert body["data"]["values"][11] is None
