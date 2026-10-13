"""
Parity of the off-loop inventory bodies with FastAPI's response_model output.
"""

import json

import numpy as np
import pandas as pd
import pytest
from api.resources.inventories.router import (
    _inventory_csv_body,
    _inventory_json_body,
)
from api.resources.inventories.schema import (
    InventoryDataResponse,
    InventoryJsonOrientation,
)
from fastapi import FastAPI
from fastapi.testclient import TestClient


def _frame():
    return pd.DataFrame(
        {
            "height": [1.5, np.nan, 3.25],
            "species": [1, 2, 3],
            "name": ["a", None, "c"],
        }
    )


def _reference(df, orientation):
    app = FastAPI()

    @app.get("/", response_model=InventoryDataResponse)
    def route():
        if orientation == InventoryJsonOrientation.records:
            data = df.to_dict(orient="records")
        else:
            data = df.to_dict(orient="split")["data"]
        return InventoryDataResponse(
            partition=2, num_rows=len(df), columns=list(df.columns), data=data
        )

    reply = TestClient(app).get("/")
    assert reply.status_code == 200
    return reply.json()


@pytest.mark.parametrize("orientation", list(InventoryJsonOrientation))
def test_json_body_matches_response_model(orientation):
    df = _frame()
    body = _inventory_json_body(df, 2, orientation)
    assert json.loads(body) == _reference(df, orientation)


def test_nan_is_emitted_as_null():
    body = json.loads(_inventory_json_body(_frame(), 0, InventoryJsonOrientation.split))
    assert body["data"][1][0] is None


def test_csv_body_matches_to_csv():
    df = _frame()
    assert _inventory_csv_body(df) == df.to_csv(index=False)
