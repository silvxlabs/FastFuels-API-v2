"""Integration tests for the compose grid endpoint."""

import pytest

from lib.config import GRIDS_COLLECTION
from tests.fixtures import make_grid_data


@pytest.fixture(scope="session")
def complete_grid_b(firestore_client, domain_for_testing):
    grid_data = make_grid_data(
        domain_id=domain_for_testing["id"],
        name="Second source grid for compose tests",
        status="completed",
        bands=[
            {
                "key": "fuel_load.1hr",
                "type": "continuous",
                "unit": "kg/m**2",
                "index": 0,
            },
            {"key": "fuel_depth", "type": "continuous", "unit": "m", "index": 1},
        ],
    )
    grid_data["checksum"] = "checksum-b"
    doc_ref = firestore_client.collection(GRIDS_COLLECTION).document(grid_data["id"])
    doc_ref.set(grid_data)
    yield grid_data
    doc_ref.delete()


@pytest.fixture(scope="session")
def complete_unitless_grid(firestore_client, domain_for_testing):
    grid_data = make_grid_data(
        domain_id=domain_for_testing["id"],
        name="Unitless source grid for compose tests",
        status="completed",
        bands=[
            {"key": "ratio", "type": "continuous", "unit": None, "index": 0},
        ],
    )
    doc_ref = firestore_client.collection(GRIDS_COLLECTION).document(grid_data["id"])
    doc_ref.set(grid_data)
    yield grid_data
    doc_ref.delete()


@pytest.fixture(scope="session")
def complete_percent_grid(firestore_client, domain_for_testing):
    grid_data = make_grid_data(
        domain_id=domain_for_testing["id"],
        name="Percent source grid for compose tests",
        status="completed",
        bands=[
            {"key": "moisture", "type": "continuous", "unit": "%", "index": 0},
        ],
    )
    doc_ref = firestore_client.collection(GRIDS_COLLECTION).document(grid_data["id"])
    doc_ref.set(grid_data)
    yield grid_data
    doc_ref.delete()


def _route(domain_id: str) -> str:
    return f"/domains/{domain_id}/grids/compose"


class TestCreateCompose:
    def test_single_grid_compute_with_literal_creates_pending_grid(
        self, client, domain_for_testing, complete_grid
    ):
        body = {
            "inputs": [{"grid_id": complete_grid["id"], "alias": "a"}],
            "bands": [
                {"key": "fuel_load.1hr", "type": "continuous", "unit": "kg/m**2"}
            ],
            "compute": [
                {
                    "output": "fuel_load.1hr",
                    "operator": "multiply",
                    "operands": ["a.fuel_load.1hr", 0.5],
                }
            ],
            "name": "Half fuel load",
        }

        response = client.post(_route(domain_for_testing["id"]), json=body)

        assert response.status_code == 201, response.json()
        data = response.json()
        assert data["status"] == "pending"
        assert data["name"] == "Half fuel load"
        assert data["source"]["name"] == "compose"
        assert data["source"]["inputs"][0]["grid_id"] == complete_grid["id"]
        assert data["source"]["inputs"][0]["alias"] == "a"
        assert data["bands"][0]["key"] == "fuel_load.1hr"
        assert data["georeference"] is None

    def test_multi_grid_compute_records_source_checksums(
        self, client, domain_for_testing, complete_grid, complete_grid_b
    ):
        body = {
            "inputs": [
                {"grid_id": complete_grid["id"], "alias": "a"},
                {"grid_id": complete_grid_b["id"], "alias": "b"},
            ],
            "bands": [
                {"key": "fuel_load.1hr", "type": "continuous", "unit": "kg/m**2"}
            ],
            "compute": [
                {
                    "output": "fuel_load.1hr",
                    "operator": "add",
                    "operands": ["a.fuel_load.1hr", "b.fuel_load.1hr"],
                }
            ],
        }

        response = client.post(_route(domain_for_testing["id"]), json=body)

        assert response.status_code == 201, response.json()
        inputs = response.json()["source"]["inputs"]
        assert {inp["alias"] for inp in inputs} == {"a", "b"}
        assert all("source_grid_checksum" in inp for inp in inputs)

    def test_missing_band_returns_422(self, client, domain_for_testing, complete_grid):
        body = {
            "inputs": [{"grid_id": complete_grid["id"], "alias": "a"}],
            "bands": [{"key": "missing", "type": "continuous", "unit": "kg/m**2"}],
            "select": [{"output": "missing", "from": "a.missing"}],
        }

        response = client.post(_route(domain_for_testing["id"]), json=body)

        assert response.status_code == 422
        assert "missing" in response.json()["detail"].lower()

    def test_incomplete_source_grid_returns_422(
        self, client, domain_for_testing, pending_grid
    ):
        body = {
            "inputs": [{"grid_id": pending_grid["id"], "alias": "a"}],
            "bands": [
                {"key": "fuel_load.1hr", "type": "continuous", "unit": "kg/m**2"}
            ],
            "select": [{"output": "fuel_load.1hr", "from": "a.fuel_load.1hr"}],
        }

        response = client.post(_route(domain_for_testing["id"]), json=body)

        assert response.status_code == 422
        assert "status" in response.json()["detail"].lower()

    def test_cross_domain_source_grid_returns_404(
        self, client, domain_for_testing, grid_in_different_domain
    ):
        body = {
            "inputs": [{"grid_id": grid_in_different_domain["id"], "alias": "a"}],
            "bands": [
                {"key": "fuel_load.1hr", "type": "continuous", "unit": "kg/m**2"}
            ],
            "select": [{"output": "fuel_load.1hr", "from": "a.fuel_load.1hr"}],
        }

        response = client.post(_route(domain_for_testing["id"]), json=body)

        assert response.status_code == 404

    def test_unit_mismatch_returns_422(self, client, domain_for_testing, complete_grid):
        body = {
            "inputs": [{"grid_id": complete_grid["id"], "alias": "a"}],
            "bands": [{"key": "fuel_load.1hr", "type": "continuous", "unit": "m"}],
            "compute": [
                {
                    "output": "fuel_load.1hr",
                    "operator": "multiply",
                    "operands": ["a.fuel_load.1hr", 0.5],
                }
            ],
        }

        response = client.post(_route(domain_for_testing["id"]), json=body)

        assert response.status_code == 422
        assert "unit" in response.json()["detail"].lower()

    def test_categorical_condition_rejects_ordering_operator(
        self, client, domain_for_testing, complete_grid
    ):
        body = {
            "inputs": [{"grid_id": complete_grid["id"], "alias": "a"}],
            "bands": [
                {"key": "fuel_load.1hr", "type": "continuous", "unit": "kg/m**2"}
            ],
            "select": [
                {
                    "output": "fuel_load.1hr",
                    "from": "a.fuel_load.1hr",
                    "conditions": [{"band": "a.fbfm", "operator": "gt", "value": 90}],
                    "else": 0,
                }
            ],
        }

        response = client.post(_route(domain_for_testing["id"]), json=body)

        assert response.status_code == 422
        assert "categorical" in response.json()["detail"].lower()

    def test_categorical_condition_accepts_fbfm_label(
        self, client, domain_for_testing, complete_grid
    ):
        body = {
            "inputs": [{"grid_id": complete_grid["id"], "alias": "a"}],
            "bands": [
                {"key": "fuel_load.1hr", "type": "continuous", "unit": "kg/m**2"}
            ],
            "select": [
                {
                    "output": "fuel_load.1hr",
                    "from": "a.fuel_load.1hr",
                    "conditions": [
                        {"band": "a.fbfm", "operator": "eq", "value": "GR1"}
                    ],
                    "else": 0,
                }
            ],
        }

        response = client.post(_route(domain_for_testing["id"]), json=body)

        assert response.status_code == 201, response.json()
        stored = response.json()["source"]["select"][0]["conditions"][0]["value"]
        assert stored == 101

    def test_unknown_fbfm_label_returns_422(
        self, client, domain_for_testing, complete_grid
    ):
        body = {
            "inputs": [{"grid_id": complete_grid["id"], "alias": "a"}],
            "bands": [
                {"key": "fuel_load.1hr", "type": "continuous", "unit": "kg/m**2"}
            ],
            "select": [
                {
                    "output": "fuel_load.1hr",
                    "from": "a.fuel_load.1hr",
                    "conditions": [
                        {"band": "a.fbfm", "operator": "in", "value": ["GR1", "GRX"]}
                    ],
                    "else": 0,
                }
            ],
        }

        response = client.post(_route(domain_for_testing["id"]), json=body)

        assert response.status_code == 422
        assert "FBFM" in response.json()["detail"]

    def test_in_condition_requires_list_value(
        self, client, domain_for_testing, complete_grid
    ):
        body = {
            "inputs": [{"grid_id": complete_grid["id"], "alias": "a"}],
            "bands": [
                {"key": "fuel_load.1hr", "type": "continuous", "unit": "kg/m**2"}
            ],
            "select": [
                {
                    "output": "fuel_load.1hr",
                    "from": "a.fuel_load.1hr",
                    "conditions": [{"band": "a.fbfm", "operator": "in", "value": 101}],
                    "else": 0,
                }
            ],
        }

        response = client.post(_route(domain_for_testing["id"]), json=body)

        assert response.status_code == 422
        assert "requires a list" in response.json()["detail"]

    def test_label_fallback_resolves_to_code(
        self, client, domain_for_testing, complete_grid
    ):
        body = {
            "inputs": [{"grid_id": complete_grid["id"], "alias": "a"}],
            "bands": [{"key": "fbfm", "type": "categorical", "unit": None}],
            "select": [
                {
                    "output": "fbfm",
                    "from": "a.fbfm",
                    "conditions": [{"band": "a.fbfm", "operator": "eq", "value": 91}],
                    "else": "GR2",
                }
            ],
        }

        response = client.post(_route(domain_for_testing["id"]), json=body)

        assert response.status_code == 201, response.json()
        assert response.json()["source"]["select"][0]["else"] == 102

    def test_numeric_categorical_fallback_is_allowed(
        self, client, domain_for_testing, complete_grid
    ):
        body = {
            "inputs": [{"grid_id": complete_grid["id"], "alias": "a"}],
            "bands": [{"key": "fbfm", "type": "categorical", "unit": None}],
            "select": [
                {
                    "output": "fbfm",
                    "from": "a.fbfm",
                    "conditions": [{"band": "a.fbfm", "operator": "ne", "value": 91}],
                    "else": 101,
                }
            ],
        }

        response = client.post(_route(domain_for_testing["id"]), json=body)

        assert response.status_code == 201, response.json()

    def test_percent_multiply_preserves_percent_unit(
        self, client, domain_for_testing, complete_percent_grid
    ):
        body = {
            "inputs": [{"grid_id": complete_percent_grid["id"], "alias": "a"}],
            "bands": [{"key": "moisture", "type": "continuous", "unit": "%"}],
            "compute": [
                {
                    "output": "moisture",
                    "operator": "multiply",
                    "operands": ["a.moisture", 0.5],
                }
            ],
        }

        response = client.post(_route(domain_for_testing["id"]), json=body)

        assert response.status_code == 201, response.json()

    def test_add_rejects_unitless_and_unitful_raster_operands(
        self, client, domain_for_testing, complete_grid, complete_unitless_grid
    ):
        body = {
            "inputs": [
                {"grid_id": complete_grid["id"], "alias": "a"},
                {"grid_id": complete_unitless_grid["id"], "alias": "b"},
            ],
            "bands": [
                {"key": "fuel_load.1hr", "type": "continuous", "unit": "kg/m**2"}
            ],
            "compute": [
                {
                    "output": "fuel_load.1hr",
                    "operator": "add",
                    "operands": ["a.fuel_load.1hr", "b.ratio"],
                }
            ],
        }

        response = client.post(_route(domain_for_testing["id"]), json=body)

        assert response.status_code == 422
        assert "unitless and unitful" in response.json()["detail"]

    def test_missing_georeference_first_input_returns_422(
        self, client, domain_for_testing, complete_grid_no_georeference
    ):
        body = {
            "inputs": [{"grid_id": complete_grid_no_georeference["id"], "alias": "a"}],
            "bands": [
                {"key": "fuel_load.1hr", "type": "continuous", "unit": "kg/m**2"}
            ],
            "select": [{"output": "fuel_load.1hr", "from": "a.fuel_load.1hr"}],
        }

        response = client.post(_route(domain_for_testing["id"]), json=body)

        assert response.status_code == 422
        assert "georeference" in response.json()["detail"].lower()
