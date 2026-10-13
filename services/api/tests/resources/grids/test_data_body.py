"""
Parity of the orjson grid JSON body with FastAPI's response_model output.

The reference is the pre-orjson path: Python lists validated and serialized
through the response model. Float values are compared after casting to the
band dtype (orjson prints float32 in shortest form); everything else must
match exactly, including JSON types.
"""

import json

import numpy as np
import pytest
from api.resources.grids.router import (
    _grid_data_json_body,
    _json_scalar,
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


def _legacy_values(flat: np.ndarray) -> list:
    # The pre-orjson value encoding: tolist() with non-finite floats as None.
    values = flat.tolist()
    if flat.dtype.kind == "f":
        for index in np.flatnonzero(~np.isfinite(flat)).tolist():
            values[index] = None
    return values


def _reference(response: GridDataResponse) -> dict:
    # The path the route used before #691: response_model validation +
    # serialization.
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
            values=_legacy_values(values),
        )
    else:
        payload = DenseGridData(
            format="dense", values=_legacy_values(data.ravel(order=order.value))
        )
    return _reference(
        GridDataResponse(
            shape=list(data.shape), order=order.value, metadata=meta, data=payload
        )
    )


def _parse(body: bytes) -> dict:
    def reject(token):
        raise ValueError(f"non-JSON token {token}")

    return json.loads(body, parse_constant=reject)


def _assert_parity(body: bytes, expected: dict, dtype: np.dtype) -> None:
    actual = _parse(body)
    actual_values = actual["data"].pop("values")
    expected_values = expected["data"].pop("values")
    assert json.dumps(actual, sort_keys=True) == json.dumps(expected, sort_keys=True)
    if np.dtype(dtype).kind != "f" or np.dtype(dtype) == np.float64:
        assert json.dumps(actual_values) == json.dumps(expected_values)
        return
    assert [v is None for v in actual_values] == [v is None for v in expected_values]
    as_band = [
        np.array([np.nan if v is None else v for v in values], dtype=dtype)
        for values in (actual_values, expected_values)
    ]
    np.testing.assert_array_equal(as_band[0], as_band[1])


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
        np.arange(24, dtype=np.float32).reshape(2, 3, 4) / 3,
        _meta_3d(),
        -9999.0,
        id="3d",
    ),
]


@pytest.mark.parametrize("data,meta,fill", CASES)
@pytest.mark.parametrize("array_format", list(GridDataArrayFormat))
@pytest.mark.parametrize("order", list(GridDataOrder))
def test_body_matches_response_model(data, meta, fill, array_format, order):
    body = _grid_data_json_body(data, meta, fill, array_format, order)
    _assert_parity(body, _expected(data, meta, fill, array_format, order), data.dtype)


def _dtype_chunk(dtype: str) -> np.ndarray:
    if dtype == "bool":
        flat = np.array([True, False] * 6)
    elif np.dtype(dtype).kind == "f":
        flat = np.array(
            [0.1, 7.1, -0.0, np.nan, np.inf, -np.inf, 1 / 3, 2.5, 0, 1e4, -1e-3, 65],
            dtype=dtype,
        )
    else:
        info = np.iinfo(dtype)
        flat = np.array(
            [info.min, info.max, 0, 1, info.max - 1, info.min + 1] * 2, dtype=dtype
        )
    return flat.reshape(3, 4)


DTYPES = [
    "float16",
    "float32",
    "float64",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "bool",
]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("array_format", list(GridDataArrayFormat))
@pytest.mark.parametrize("fill", [None, "zero"])
def test_every_dtype_matches_response_model(dtype, array_format, fill):
    data = _dtype_chunk(dtype)
    fill_value = None if fill is None else data.dtype.type(0).item()
    body = _grid_data_json_body(
        data, _meta_2d(), fill_value, array_format, GridDataOrder.C
    )
    expected = _expected(data, _meta_2d(), fill_value, array_format, GridDataOrder.C)
    _assert_parity(body, expected, data.dtype)


def test_int64_extremes_exact():
    data = np.array(
        [np.iinfo(np.int64).min, np.iinfo(np.int64).max] * 6, dtype=np.int64
    ).reshape(3, 4)
    body = _grid_data_json_body(
        data, _meta_2d(), None, GridDataArrayFormat.dense, GridDataOrder.C
    )
    assert _parse(body)["data"]["values"][:2] == [
        -9223372036854775808,
        9223372036854775807,
    ]


def test_float32_prints_shortest_form():
    data = np.full((3, 4), 7.1, dtype=np.float32)
    body = _grid_data_json_body(
        data, _meta_2d(), np.nan, GridDataArrayFormat.dense, GridDataOrder.C
    )
    assert b"7.1," in body
    assert b"7.099999904632568" not in body
    assert np.float32(_parse(body)["data"]["values"][0]) == np.float32(7.1)


def test_order_f_flattens_column_major():
    data = np.arange(12, dtype=np.int32).reshape(3, 4)
    body = _parse(
        _grid_data_json_body(
            data, _meta_2d(), None, GridDataArrayFormat.dense, GridDataOrder.F
        )
    )
    assert body["order"] == "F"
    assert body["data"]["values"] == [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]


def test_sparse_indices_and_values():
    data = np.zeros((3, 4), dtype=np.uint8)
    data[0, 2] = 5
    data[2, 1] = 7
    for order, indices in ((GridDataOrder.C, [2, 9]), (GridDataOrder.F, [5, 6])):
        body = _parse(
            _grid_data_json_body(data, _meta_2d(), 0, GridDataArrayFormat.sparse, order)
        )
        assert body["data"]["indices"] == indices
        expected_values = [5, 7] if order == GridDataOrder.C else [7, 5]
        assert body["data"]["values"] == expected_values
        assert body["data"]["fill_value"] == 0


@pytest.mark.parametrize(
    "transform",
    [
        pytest.param(lambda a: a.astype(a.dtype.newbyteorder(">")), id="big-endian"),
        pytest.param(lambda a: np.asfortranarray(a), id="fortran"),
        pytest.param(
            lambda a: np.repeat(a, 2, axis=1)[:, ::2], id="non-contiguous-view"
        ),
    ],
)
@pytest.mark.parametrize("dtype", ["float32", "float64", "int16", "uint32"])
@pytest.mark.parametrize("array_format", list(GridDataArrayFormat))
@pytest.mark.parametrize("order", list(GridDataOrder))
def test_unusual_layouts_match_native(transform, dtype, array_format, order):
    native = _dtype_chunk(dtype)
    unusual = transform(native)
    assert np.array_equal(unusual, native, equal_nan=True)
    fill = None if array_format == GridDataArrayFormat.dense else 0
    assert _grid_data_json_body(
        unusual, _meta_2d(), fill, array_format, order
    ) == _grid_data_json_body(native, _meta_2d(), fill, array_format, order)


def test_3d_chunk_values_and_metadata():
    data = np.arange(24, dtype=np.float32).reshape(2, 3, 4) / 4
    body = _parse(
        _grid_data_json_body(
            data, _meta_3d(), np.nan, GridDataArrayFormat.dense, GridDataOrder.C
        )
    )
    assert body["shape"] == [2, 3, 4]
    assert body["metadata"]["shape"] == [2, 3, 4]
    assert body["data"]["values"] == (np.arange(24) / 4).tolist()


def test_2d_omits_z_fields_and_3d_keeps_them():
    body = _parse(
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
    body = _parse(
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
    body = _parse(
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


def test_sparse_float32_fill_prints_like_values():
    data = np.array([[0.1, 0.2, 0.1], [0.3, 0.1, 0.1]], dtype=np.float32)
    raw = _grid_data_json_body(
        data,
        _meta_2d(),
        np.float32(0.1).item(),
        GridDataArrayFormat.sparse,
        GridDataOrder.C,
    )
    assert b'"fill_value":0.1,' in raw
    body = _parse(raw)
    assert body["data"]["values"] == [0.2, 0.3]
    assert np.float32(body["data"]["fill_value"]) == np.float32(0.1)


def test_non_finite_becomes_null_in_dense_values():
    data = _float_chunk_2d()
    data[1, 0] = -np.inf
    body = _parse(
        _grid_data_json_body(
            data,
            _meta_2d(),
            np.nan,
            GridDataArrayFormat.dense,
            GridDataOrder.C,
        )
    )
    assert body["data"]["values"][1] is None
    assert body["data"]["values"][4] is None
    assert body["data"]["values"][11] is None


def test_non_finite_becomes_null_in_sparse_values():
    body = _parse(
        _grid_data_json_body(
            _float_chunk_2d(),
            _meta_2d(),
            None,
            GridDataArrayFormat.sparse,
            GridDataOrder.C,
        )
    )
    assert body["data"]["indices"] == list(range(12))
    assert body["data"]["values"][1] is None
    assert body["data"]["values"][11] is None
