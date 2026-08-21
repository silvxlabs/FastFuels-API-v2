"""
Integration tests for POST /domains/{domain_id}/grids/canopy/inventory.

These make real HTTP requests to the API and interact with Firestore and Cloud
Tasks. The endpoint's job is to validate the source tree inventory and the
resolved lattice before a grid document exists, and to persist a source that
records every resolved modeling choice, so the tests assert both the acceptance
path (the grid and its resolved source) and each rejection's status and
message.
"""

import uuid

import pytest
from api.resources.grids.canopy.inventory.examples import (
    EXAMPLE_INVENTORY_CANOPY_FUELCALC_COMPARISON,
    INVENTORY_CANOPY_EXAMPLE_VALUES,
)

from lib.config import DOMAINS_COLLECTION, INVENTORIES_COLLECTION
from tests.fixtures import make_domain_data, make_inventory_data

# Columns the documented examples read beyond the standard morphology set.
_EXTRA_COLUMNS = [
    {"key": "available_canopy_fuel_kg", "type": "continuous", "unit": "kg"},
    {"key": "crown_radius_m", "type": "continuous", "unit": "m"},
]


def _seed_inventory(firestore_client, **kwargs):
    """Write a tree inventory document with a checksum; delete on teardown."""
    kwargs.setdefault("status", "completed")
    kwargs.setdefault("inventory_type", "tree")
    data = make_inventory_data(**kwargs)
    data["checksum"] = uuid.uuid4().hex
    doc_ref = firestore_client.collection(INVENTORIES_COLLECTION).document(data["id"])
    doc_ref.set(data)
    yield data
    doc_ref.delete()


@pytest.fixture(scope="session")
def tree_inventory_for_canopy(firestore_client, domain_for_testing):
    """A completed tree inventory in domain_for_testing, standard columns."""
    yield from _seed_inventory(
        firestore_client,
        domain_id=domain_for_testing["id"],
        name="Tree inventory for canopy tests",
    )


@pytest.fixture(scope="session")
def rich_tree_inventory_for_canopy(firestore_client, domain_for_testing):
    """A completed tree inventory carrying every column the examples reference,
    so each documented example is accepted."""
    data = make_inventory_data(
        domain_id=domain_for_testing["id"],
        name="Rich tree inventory for canopy examples",
        status="completed",
        inventory_type="tree",
    )
    data["checksum"] = uuid.uuid4().hex
    data["columns"] = data["columns"] + _EXTRA_COLUMNS
    doc_ref = firestore_client.collection(INVENTORIES_COLLECTION).document(data["id"])
    doc_ref.set(data)
    yield data
    doc_ref.delete()


@pytest.fixture(scope="session")
def non_tree_inventory(firestore_client, domain_for_testing):
    """A completed inventory that is not a tree inventory."""
    yield from _seed_inventory(
        firestore_client,
        domain_id=domain_for_testing["id"],
        name="Surface inventory",
        inventory_type="surface",
    )


@pytest.fixture(scope="session")
def columnless_tree_inventory(firestore_client, domain_for_testing):
    """A CHM-only tree inventory: position and height, no morphology."""
    data = make_inventory_data(
        domain_id=domain_for_testing["id"],
        name="CHM-only tree inventory",
        status="completed",
        inventory_type="tree",
    )
    data["checksum"] = uuid.uuid4().hex
    data["columns"] = [
        {"key": "x", "type": "continuous", "unit": "m"},
        {"key": "y", "type": "continuous", "unit": "m"},
        {"key": "height", "type": "continuous", "unit": "m"},
    ]
    doc_ref = firestore_client.collection(INVENTORIES_COLLECTION).document(data["id"])
    doc_ref.set(data)
    yield data
    doc_ref.delete()


@pytest.fixture(scope="session")
def pending_tree_inventory(firestore_client, domain_for_testing):
    """A tree inventory still being produced (not completed)."""
    data = make_inventory_data(
        domain_id=domain_for_testing["id"],
        name="Pending tree inventory",
        status="pending",
        inventory_type="tree",
    )
    doc_ref = firestore_client.collection(INVENTORIES_COLLECTION).document(data["id"])
    doc_ref.set(data)
    yield data
    doc_ref.delete()


@pytest.fixture(scope="session")
def second_domain_for_canopy(firestore_client):
    """A second domain owned by test-owner, for cross-domain tests."""
    data = make_domain_data(name="Second Domain for Canopy Inventory")
    doc_ref = firestore_client.collection(DOMAINS_COLLECTION).document(data["id"])
    doc_ref.set(data)
    yield data
    doc_ref.delete()


@pytest.fixture(scope="session")
def tree_inventory_in_other_domain(firestore_client, second_domain_for_canopy):
    """A completed tree inventory in a different domain of the same owner."""
    yield from _seed_inventory(
        firestore_client,
        domain_id=second_domain_for_canopy["id"],
        name="Tree inventory in second domain",
    )


class TestCreateInventoryCanopyGrid:
    """POST /domains/{domain_id}/grids/canopy/inventory."""

    def route(self, domain_id):
        return f"/domains/{domain_id}/grids/canopy/inventory"

    def test_minimal_request_creates_grid_with_resolved_source(
        self, client, domain_for_testing, tree_inventory_for_canopy
    ):
        response = client.post(
            self.route(domain_for_testing["id"]),
            json={"source_inventory_id": tree_inventory_for_canopy["id"]},
        )

        assert response.status_code == 201
        grid = response.json()
        assert len(grid["id"]) == 32
        assert grid["domain_id"] == domain_for_testing["id"]
        assert grid["status"] == "pending"
        assert grid["georeference"] is None
        assert grid["modifications"] == []

        source = grid["source"]
        assert source["name"] == "canopy"
        assert source["product"] == "inventory"
        assert source["source_inventory_id"] == tree_inventory_for_canopy["id"]
        # Domain target resolves the 30 m default and persists it.
        assert source["alignment"]["target"] == "domain"
        assert source["alignment"]["resolution"] == 30.0
        assert source["bands"] == ["cbd", "cbh", "chm", "cc"]
        # FastFuels-native defaults, resolved onto the source. NSVB defaults to
        # the national `none` partition (prices every species) at 0.075.
        assert source["biomass_source"] == {"type": "allometry", "equations": "nsvb"}
        assert source["available_fuel"]["branchwood"] == {
            "size_partition": "none",
            "fraction": 0.075,
        }
        assert source["species_inclusion"] == "all_species"
        assert source["crown_class_adjustment"] == {"method": "none"}
        assert source["vertical_distribution"] == "reinhardt_2006"
        assert source["horizontal_distribution"] == "crown_projected"
        assert source["max_crown_radius_source"] == {
            "type": "allometry",
            "equations": "purves",
        }
        # Each requested band's method resolves to its documented default.
        assert source["cbd"]["method"] == "maximum_running_mean"
        assert source["cbh"]["method"] == "bulk_density_threshold"
        assert source["chm"]["method"] == "bulk_density_threshold"
        assert source["cc"]["method"] == "crown_union"

    def test_bands_carry_landfire_canopy_keys_and_units(
        self, client, domain_for_testing, tree_inventory_for_canopy
    ):
        response = client.post(
            self.route(domain_for_testing["id"]),
            json={"source_inventory_id": tree_inventory_for_canopy["id"]},
        )
        bands = {b["key"]: b for b in response.json()["bands"]}
        assert list(bands) == ["cbd", "cbh", "chm", "cc"]
        assert bands["cbd"]["unit"] == "kg/m**3"
        assert bands["cbh"]["unit"] == "m"
        assert bands["chm"]["unit"] == "m"
        assert bands["cc"]["unit"] == "%"

    def test_captures_source_checksum_for_staleness(
        self, client, domain_for_testing, tree_inventory_for_canopy
    ):
        response = client.post(
            self.route(domain_for_testing["id"]),
            json={"source_inventory_id": tree_inventory_for_canopy["id"]},
        )
        source = response.json()["source"]
        assert (
            source["source_inventory_checksum"] == tree_inventory_for_canopy["checksum"]
        )

    def test_cfl_band_is_opt_in(
        self, client, domain_for_testing, tree_inventory_for_canopy
    ):
        response = client.post(
            self.route(domain_for_testing["id"]),
            json={
                "source_inventory_id": tree_inventory_for_canopy["id"],
                "bands": ["cbd", "cbh", "chm", "cc", "cfl"],
            },
        )
        assert response.status_code == 201
        bands = {b["key"]: b for b in response.json()["bands"]}
        assert "cfl" in bands
        assert bands["cfl"]["unit"] == "kg/m**2"

    def test_custom_resolution_is_persisted(
        self, client, domain_for_testing, tree_inventory_for_canopy
    ):
        response = client.post(
            self.route(domain_for_testing["id"]),
            json={
                "source_inventory_id": tree_inventory_for_canopy["id"],
                "alignment": {"target": "domain", "resolution": 10.0},
            },
        )
        assert response.status_code == 201
        assert response.json()["source"]["alignment"]["resolution"] == 10.0

    def test_fuelcalc_comparison_example_is_accepted(
        self, client, domain_for_testing, tree_inventory_for_canopy
    ):
        body = {
            **EXAMPLE_INVENTORY_CANOPY_FUELCALC_COMPARISON,
            "source_inventory_id": tree_inventory_for_canopy["id"],
        }
        response = client.post(self.route(domain_for_testing["id"]), json=body)
        assert response.status_code == 201
        source = response.json()["source"]
        assert source["biomass_source"]["equations"] == "brown_1978"

    @pytest.mark.parametrize(
        "name,example",
        INVENTORY_CANOPY_EXAMPLE_VALUES,
    )
    def test_documented_examples_are_accepted(
        self, client, domain_for_testing, rich_tree_inventory_for_canopy, name, example
    ):
        """Every OpenAPI example creates a grid against an inventory carrying
        the columns it reads."""
        body = {**example, "source_inventory_id": rich_tree_inventory_for_canopy["id"]}
        response = client.post(self.route(domain_for_testing["id"]), json=body)
        assert response.status_code == 201, f"example {name!r}: {response.text}"


class TestInventoryCanopyValidation:
    """Rejections the endpoint owns, with their status and message."""

    def route(self, domain_id):
        return f"/domains/{domain_id}/grids/canopy/inventory"

    def test_rejects_non_tree_inventory(
        self, client, domain_for_testing, non_tree_inventory
    ):
        response = client.post(
            self.route(domain_for_testing["id"]),
            json={"source_inventory_id": non_tree_inventory["id"]},
        )
        assert response.status_code == 422
        assert "tree inventory" in response.json()["detail"]

    def test_rejects_inventory_missing_required_columns(
        self, client, domain_for_testing, columnless_tree_inventory
    ):
        response = client.post(
            self.route(domain_for_testing["id"]),
            json={"source_inventory_id": columnless_tree_inventory["id"]},
        )
        assert response.status_code == 422
        detail = response.json()["detail"]
        # The morphology columns the default path needs, and how to get them.
        assert "dbh" in detail
        assert "allometry" in detail

    def test_rejects_native_alignment(
        self, client, domain_for_testing, tree_inventory_for_canopy
    ):
        response = client.post(
            self.route(domain_for_testing["id"]),
            json={
                "source_inventory_id": tree_inventory_for_canopy["id"],
                "alignment": {"target": "native"},
            },
        )
        assert response.status_code == 422
        assert "native" in response.json()["detail"]

    def test_rejects_inventory_in_other_domain(
        self, client, domain_for_testing, tree_inventory_in_other_domain
    ):
        response = client.post(
            self.route(domain_for_testing["id"]),
            json={"source_inventory_id": tree_inventory_in_other_domain["id"]},
        )
        assert response.status_code == 404

    def test_pending_inventory_returns_422(
        self, client, domain_for_testing, pending_tree_inventory
    ):
        """An inventory still being produced is not yet a valid source."""
        response = client.post(
            self.route(domain_for_testing["id"]),
            json={"source_inventory_id": pending_tree_inventory["id"]},
        )
        assert response.status_code == 422

    def test_rejects_method_for_unrequested_band(
        self, client, domain_for_testing, tree_inventory_for_canopy
    ):
        response = client.post(
            self.route(domain_for_testing["id"]),
            json={
                "source_inventory_id": tree_inventory_for_canopy["id"],
                "bands": ["cc"],
                "cbd": {"method": "maximum_running_mean"},
            },
        )
        assert response.status_code == 422
