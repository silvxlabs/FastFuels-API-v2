"""
Unit tests for api/v2/resources/inventories/schema.py
and api/v2/resources/inventories/pim/schema.py

Tests the Inventory schema models, enums, and PIM source models.
These are pure unit tests with no external dependencies.
"""

import pytest
from api.resources.inventories.pim.schema import (
    CreatePimInventoryRequest,
    PimInventorySource,
)
from api.resources.inventories.schema import (
    CreateInventoryRequestBase,
    Inventory,
    InventorySortField,
    InventoryType,
    ListInventoriesResponse,
    PointProcess,
    UpdateInventoryRequestBody,
)
from pydantic import ValidationError


class TestInventoryType:
    """Tests for InventoryType enum."""

    def test_tree_value(self):
        assert InventoryType.tree == "tree"

    def test_tree_is_only_value(self):
        assert list(InventoryType) == [InventoryType.tree]


class TestPointProcess:
    """Tests for PointProcess enum."""

    def test_inhomogeneous_poisson_value(self):
        assert PointProcess.inhomogeneous_poisson == "inhomogeneous_poisson"

    def test_inhomogeneous_poisson_is_only_value(self):
        assert list(PointProcess) == [PointProcess.inhomogeneous_poisson]


class TestInventorySortField:
    """Tests for InventorySortField enum."""

    def test_all_values(self):
        assert InventorySortField.created_on == "created_on"
        assert InventorySortField.modified_on == "modified_on"
        assert InventorySortField.name == "name"


class TestCreateInventoryRequestBase:
    """Tests for CreateInventoryRequestBase model."""

    def test_defaults(self):
        request = CreateInventoryRequestBase()
        assert request.type == InventoryType.tree
        assert request.name == ""
        assert request.description == ""
        assert request.tags == []

    def test_custom_values(self):
        request = CreateInventoryRequestBase(
            type="tree",
            name="My Inventory",
            description="A test inventory",
            tags=["test"],
        )
        assert request.name == "My Inventory"
        assert request.description == "A test inventory"
        assert request.tags == ["test"]

    def test_name_max_length(self):
        with pytest.raises(ValidationError):
            CreateInventoryRequestBase(name="x" * 256)

    def test_description_max_length(self):
        with pytest.raises(ValidationError):
            CreateInventoryRequestBase(description="x" * 2001)


class TestUpdateInventoryRequestBody:
    """Tests for UpdateInventoryRequestBody model."""

    def test_all_fields_optional(self):
        body = UpdateInventoryRequestBody()
        assert body.name is None
        assert body.description is None
        assert body.tags is None

    def test_partial_update(self):
        body = UpdateInventoryRequestBody(name="Updated")
        assert body.name == "Updated"
        assert body.description is None
        assert body.tags is None

    def test_name_max_length(self):
        with pytest.raises(ValidationError):
            UpdateInventoryRequestBody(name="x" * 256)

    def test_description_max_length(self):
        with pytest.raises(ValidationError):
            UpdateInventoryRequestBody(description="x" * 2001)


class TestInventory:
    """Tests for Inventory model."""

    def _make_inventory(self, **overrides):
        defaults = {
            "id": "abc123",
            "domain_id": "domain456",
            "type": "tree",
            "status": "pending",
            "created_on": "2026-01-01T00:00:00",
            "modified_on": "2026-01-01T00:00:00",
            "source": {
                "name": "pim",
                "source_pim_grid_id": "grid789",
                "seed": 42,
                "point_process": "inhomogeneous_poisson",
            },
        }
        defaults.update(overrides)
        return Inventory(**defaults)

    def test_minimal_inventory(self):
        inv = self._make_inventory()
        assert inv.id == "abc123"
        assert inv.domain_id == "domain456"
        assert inv.type == InventoryType.tree
        assert inv.name == ""
        assert inv.description == ""
        assert inv.status == "pending"
        assert inv.modifications == []
        assert inv.georeference is None
        assert inv.error is None
        assert inv.tags == []
        assert inv.progress is None

    def test_inventory_with_georeference(self):
        geo = {
            "crs": "EPSG:32611",
            "bounds": (500000.0, 5200000.0, 501000.0, 5201000.0),
        }
        inv = self._make_inventory(georeference=geo)
        assert inv.georeference is not None
        assert inv.georeference.crs == "EPSG:32611"
        assert inv.georeference.bounds == (500000.0, 5200000.0, 501000.0, 5201000.0)

    def test_required_fields(self):
        with pytest.raises(ValidationError):
            Inventory(id="abc", domain_id="def")


class TestListInventoriesResponse:
    """Tests for ListInventoriesResponse model."""

    def test_empty_list(self):
        response = ListInventoriesResponse(
            inventories=[],
            current_page=0,
            page_size=100,
            total_items=0,
        )
        assert response.inventories == []
        assert response.current_page == 0
        assert response.total_items == 0

    def test_pagination_fields(self):
        response = ListInventoriesResponse(
            inventories=[],
            current_page=2,
            page_size=10,
            total_items=50,
        )
        assert response.current_page == 2
        assert response.page_size == 10
        assert response.total_items == 50


class TestPimInventorySource:
    """Tests for PimInventorySource model."""

    def test_name_is_always_pim(self):
        source = PimInventorySource(
            source_pim_grid_id="grid123",
            point_process="inhomogeneous_poisson",
            seed=42,
        )
        assert source.name == "pim"

    def test_name_cannot_be_overridden(self):
        with pytest.raises(ValidationError):
            PimInventorySource(
                name="other",
                source_pim_grid_id="grid123",
                point_process="inhomogeneous_poisson",
                seed=42,
            )

    def test_all_fields_required(self):
        with pytest.raises(ValidationError):
            PimInventorySource(source_pim_grid_id="grid123")

    def test_model_dump(self):
        source = PimInventorySource(
            source_pim_grid_id="grid123",
            point_process="inhomogeneous_poisson",
            seed=42,
        )
        data = source.model_dump()
        assert data["name"] == "pim"
        assert data["source_pim_grid_id"] == "grid123"
        assert data["point_process"] == "inhomogeneous_poisson"
        assert data["seed"] == 42


class TestCreatePimInventoryRequest:
    """Tests for CreatePimInventoryRequest model."""

    def test_all_defaults(self):
        """Only source_pim_grid_id is required — other fields have defaults."""
        request = CreatePimInventoryRequest(source_pim_grid_id="grid123")
        assert request.source_pim_grid_id == "grid123"
        assert isinstance(request.seed, int)
        assert request.point_process == PointProcess.inhomogeneous_poisson
        assert request.type == InventoryType.tree
        assert request.name == ""
        assert request.description == ""
        assert request.tags == []

    def test_source_pim_grid_id_required(self):
        """Omitting source_pim_grid_id raises ValidationError."""
        with pytest.raises(ValidationError):
            CreatePimInventoryRequest()

    def test_with_explicit_grid(self):
        request = CreatePimInventoryRequest(
            source_pim_grid_id="grid123",
            seed=42,
        )
        assert request.source_pim_grid_id == "grid123"
        assert request.seed == 42
        assert request.point_process == PointProcess.inhomogeneous_poisson

    def test_full_request(self):
        request = CreatePimInventoryRequest(
            source_pim_grid_id="grid123",
            seed=12345,
            point_process="inhomogeneous_poisson",
            type="tree",
            name="Test Inventory",
            description="A test inventory",
            tags=["test"],
        )
        assert request.name == "Test Inventory"
        assert request.seed == 12345

    def test_seed_is_randomly_generated_when_omitted(self):
        r1 = CreatePimInventoryRequest(source_pim_grid_id="grid123")
        r2 = CreatePimInventoryRequest(source_pim_grid_id="grid456")
        # Both should have int seeds (could theoretically collide but
        # range is 1..1B so practically never)
        assert isinstance(r1.seed, int)
        assert isinstance(r2.seed, int)

    def test_seed_range(self):
        request = CreatePimInventoryRequest(source_pim_grid_id="grid123")
        assert 1 <= request.seed <= 1_000_000_000
