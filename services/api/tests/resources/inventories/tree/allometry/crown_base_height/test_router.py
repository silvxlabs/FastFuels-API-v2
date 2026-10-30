"""
Integration tests for api/v2/resources/inventories/tree/allometry/crown_base_height/router.py

These tests make real HTTP requests to the API and interact with Firestore and
Cloud Tasks.
"""

import json

import pytest
from api.resources.inventories.tree.allometry.crown_base_height.examples import (
    ALL_CROWN_BASE_HEIGHT_EXAMPLE_VALUES,
)

from lib.config import INVENTORIES_COLLECTION
from tests.fixtures import make_inventory_data

_SOURCE_CHECKSUM = "cbh-src-checksum-123"
_POSITION_HEIGHT_COLUMNS = [
    {"key": "x", "type": "continuous", "unit": "m"},
    {"key": "y", "type": "continuous", "unit": "m"},
    {"key": "height", "type": "continuous", "unit": "m"},
]
_FULL_MORPHOLOGY_COLUMNS = [
    *_POSITION_HEIGHT_COLUMNS,
    {"key": "dbh", "type": "continuous", "unit": "cm"},
    {"key": "crown_ratio", "type": "continuous"},
    {"key": "fia_species_code", "type": "categorical"},
]


def _summary(null_count):
    return {
        "type": "continuous",
        "count": 10,
        "null_count": null_count,
        "mean": 3.0,
        "std": 1.0,
        "min": 1.0,
        "max": 5.0,
    }


@pytest.fixture
def make_source(firestore_client, domain_for_testing):
    """Write completed tree inventories with the given columns; delete on teardown."""
    refs = []

    def _make(columns, inventory_type="tree", checksum=None):
        inventory_data = make_inventory_data(
            domain_id=domain_for_testing["id"],
            name="Source for crown base height tests",
            status="completed",
            inventory_type="tree",
        )
        inventory_data["columns"] = columns
        inventory_data["type"] = inventory_type
        if checksum is not None:
            inventory_data["checksum"] = checksum
        ref = firestore_client.collection(INVENTORIES_COLLECTION).document(
            inventory_data["id"]
        )
        ref.set(inventory_data)
        refs.append(ref)
        return inventory_data

    yield _make
    for ref in refs:
        ref.delete()


class TestCreateCrownBaseHeightInventory:
    def route(self, domain_id):
        return f"/domains/{domain_id}/inventories/tree/allometry/crown_base_height"

    def test_minimal_request_creates_inventory(
        self, client, domain_for_testing, make_source
    ):
        source = make_source(_FULL_MORPHOLOGY_COLUMNS, checksum=_SOURCE_CHECKSUM)
        response = client.post(
            self.route(domain_for_testing["id"]),
            json={"source_tree_inventory_id": source["id"]},
        )
        assert response.status_code == 201
        data = response.json()
        assert data["status"] == "pending"
        assert data["type"] == "tree"
        assert data["georeference"] is None
        assert isinstance(data["checksum"], str) and len(data["checksum"]) == 32
        assert data["source"] == {
            "name": "crown_base_height",
            "source_tree_inventory_id": source["id"],
            "source_tree_inventory_checksum": _SOURCE_CHECKSUM,
            "model": {"name": "fia_national_logit_2026"},
            "overwrite": False,
        }
        assert [c["key"] for c in data["columns"]] == [
            "x",
            "y",
            "height",
            "dbh",
            "crown_ratio",
            "fia_species_code",
            "crown_base_height",
        ]
        assert "owner_id" not in data

    def test_model_and_overwrite_persisted(
        self, client, domain_for_testing, make_source
    ):
        source = make_source(
            [
                *_FULL_MORPHOLOGY_COLUMNS,
                {
                    "key": "crown_base_height",
                    "type": "continuous",
                    "unit": "m",
                    "summary": _summary(0),
                },
            ]
        )
        response = client.post(
            self.route(domain_for_testing["id"]),
            json={
                "source_tree_inventory_id": source["id"],
                "model": {"name": "toney_reeves_2009"},
                "overwrite": True,
            },
        )
        assert response.status_code == 201
        data = response.json()
        assert data["source"]["model"] == {"name": "toney_reeves_2009"}
        assert data["source"]["overwrite"] is True
        # The existing column is not duplicated.
        keys = [c["key"] for c in data["columns"]]
        assert keys.count("crown_base_height") == 1

    def test_partially_filled_crown_base_height_creates_inventory(
        self, client, domain_for_testing, make_source
    ):
        source = make_source(
            [
                *_FULL_MORPHOLOGY_COLUMNS,
                {
                    "key": "crown_base_height",
                    "type": "continuous",
                    "unit": "m",
                    "summary": _summary(4),
                },
            ]
        )
        response = client.post(
            self.route(domain_for_testing["id"]),
            json={"source_tree_inventory_id": source["id"]},
        )
        assert response.status_code == 201

    def test_complete_crown_base_height_without_overwrite_returns_422(
        self, client, domain_for_testing, make_source
    ):
        source = make_source(
            [
                *_FULL_MORPHOLOGY_COLUMNS,
                {
                    "key": "crown_base_height",
                    "type": "continuous",
                    "unit": "m",
                    "summary": _summary(0),
                },
            ]
        )
        response = client.post(
            self.route(domain_for_testing["id"]),
            json={"source_tree_inventory_id": source["id"]},
        )
        assert response.status_code == 422
        assert "overwrite: true" in response.json()["detail"]

    def test_missing_morphology_names_one_gdam_request(
        self, client, domain_for_testing, make_source
    ):
        # Species missing, crown ratio present but with empty values.
        source = make_source(
            [
                *_POSITION_HEIGHT_COLUMNS,
                {"key": "dbh", "type": "continuous", "unit": "cm"},
                {
                    "key": "crown_ratio",
                    "type": "continuous",
                    "summary": _summary(2),
                },
            ]
        )
        domain_id = domain_for_testing["id"]
        response = client.post(
            self.route(domain_id),
            json={"source_tree_inventory_id": source["id"]},
        )
        assert response.status_code == 422
        detail = response.json()["detail"]
        assert f"POST /domains/{domain_id}/inventories/tree/allometry/gdam" in detail
        body = {
            "source_tree_inventory_id": source["id"],
            "impute_columns": ["crown_ratio", "fia_species_code", "crown_base_height"],
            "crown_base_height_model": {"name": "fia_national_logit_2026"},
        }
        assert json.dumps(body) in detail

    def test_position_height_source_names_full_gdam_request(
        self, client, domain_for_testing, make_source
    ):
        source = make_source(_POSITION_HEIGHT_COLUMNS)
        response = client.post(
            self.route(domain_for_testing["id"]),
            json={"source_tree_inventory_id": source["id"]},
        )
        assert response.status_code == 422
        assert (
            '"impute_columns": ["dbh", "crown_ratio", "fia_species_code", '
            '"crown_base_height"]'
        ) in response.json()["detail"]

    def test_empty_heights_return_422(self, client, domain_for_testing, make_source):
        source = make_source(
            [
                {"key": "x", "type": "continuous", "unit": "m"},
                {"key": "y", "type": "continuous", "unit": "m"},
                {
                    "key": "height",
                    "type": "continuous",
                    "unit": "m",
                    "summary": _summary(3),
                },
                *_FULL_MORPHOLOGY_COLUMNS[3:],
            ]
        )
        response = client.post(
            self.route(domain_for_testing["id"]),
            json={"source_tree_inventory_id": source["id"]},
        )
        assert response.status_code == 422
        detail = response.json()["detail"]
        assert "height" in detail
        assert "cannot be imputed" in detail
        assert "gdam" not in detail

    def test_non_tree_source_returns_422(self, client, domain_for_testing, make_source):
        source = make_source(_FULL_MORPHOLOGY_COLUMNS, inventory_type="shrub")
        response = client.post(
            self.route(domain_for_testing["id"]),
            json={"source_tree_inventory_id": source["id"]},
        )
        assert response.status_code == 422
        assert "tree" in response.json()["detail"].lower()

    def test_unknown_model_returns_422(self, client, domain_for_testing, make_source):
        source = make_source(_FULL_MORPHOLOGY_COLUMNS)
        response = client.post(
            self.route(domain_for_testing["id"]),
            json={"source_tree_inventory_id": source["id"], "model": {"name": "bogus"}},
        )
        assert response.status_code == 422

    def test_gdam_model_needs_only_position_and_height(
        self, client, domain_for_testing, make_source
    ):
        source = make_source(_POSITION_HEIGHT_COLUMNS)
        response = client.post(
            self.route(domain_for_testing["id"]),
            json={"source_tree_inventory_id": source["id"], "model": {"name": "gdam"}},
        )
        assert response.status_code == 201
        data = response.json()
        assert data["source"]["model"] == {"name": "gdam"}
        assert [c["key"] for c in data["columns"]] == [
            "x",
            "y",
            "height",
            "crown_base_height",
        ]

    def test_gdam_model_nothing_to_fill_returns_422(
        self, client, domain_for_testing, make_source
    ):
        source = make_source(
            [
                *_POSITION_HEIGHT_COLUMNS,
                {
                    "key": "crown_base_height",
                    "type": "continuous",
                    "unit": "m",
                    "summary": _summary(0),
                },
            ]
        )
        response = client.post(
            self.route(domain_for_testing["id"]),
            json={"source_tree_inventory_id": source["id"], "model": {"name": "gdam"}},
        )
        assert response.status_code == 422
        assert "overwrite: true" in response.json()["detail"]

    def test_nonexistent_source_inventory_returns_404(self, client, domain_for_testing):
        response = client.post(
            self.route(domain_for_testing["id"]),
            json={"source_tree_inventory_id": "00000000000000000000000000000000"},
        )
        assert response.status_code == 404

    def test_wrong_owner_domain_returns_404(
        self, client, domain_with_different_owner, make_source
    ):
        source = make_source(_FULL_MORPHOLOGY_COLUMNS)
        response = client.post(
            self.route(domain_with_different_owner["id"]),
            json={"source_tree_inventory_id": source["id"]},
        )
        assert response.status_code == 404

    @pytest.mark.parametrize(
        "example_name,example_value", ALL_CROWN_BASE_HEIGHT_EXAMPLE_VALUES
    )
    def test_documented_example_creates_inventory(
        self, client, domain_for_testing, make_source, example_name, example_value
    ):
        source = make_source(_FULL_MORPHOLOGY_COLUMNS)
        body = {**example_value, "source_tree_inventory_id": source["id"]}
        response = client.post(self.route(domain_for_testing["id"]), json=body)
        assert response.status_code == 201, (
            f"Example '{example_name}' failed with status {response.status_code}: "
            f"{response.text}"
        )
