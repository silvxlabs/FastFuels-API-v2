"""
Unit tests for api/v2/resources/inventories/schema.py
and api/v2/resources/inventories/tree/pim/schema.py

Tests the Inventory schema models, enums, and PIM source models.
These are pure unit tests with no external dependencies.
"""

import pytest
from api.resources.inventories.schema import (
    CategoricalColumnSummary,
    Column,
    ColumnType,
    ContinuousColumnSummary,
    CreateInventoryRequestBase,
    FIASpeciesGroupShare,
    Inventory,
    InventoryDataFormat,
    InventoryDataMetadata,
    InventoryDataResponse,
    InventoryJsonOrientation,
    InventoryPartitionInfo,
    InventorySortField,
    InventoryType,
    ListInventoriesResponse,
    PointProcess,
    TreeForestryMetrics,
    UpdateInventoryRequestBody,
)
from api.resources.inventories.tree.pim.schema import (
    CreatePimInventoryRequest,
    PimInventorySource,
)
from pydantic import ValidationError

from lib.units import validate_unit


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

    def test_checksum_defaults_to_none(self):
        """checksum defaults to None when absent (e.g. legacy documents)."""
        inv = self._make_inventory()
        assert inv.checksum is None

    def test_checksum_round_trips(self):
        """checksum is carried through the model and serialization."""
        inv = self._make_inventory(checksum="cafe" * 8)
        assert inv.checksum == "cafe" * 8
        assert inv.model_dump()["checksum"] == "cafe" * 8

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

    def test_source_pim_grid_checksum_defaults_to_none(self):
        source = PimInventorySource(
            source_pim_grid_id="grid123",
            point_process="inhomogeneous_poisson",
            seed=42,
        )
        assert source.source_pim_grid_checksum is None

    def test_source_pim_grid_checksum_round_trips(self):
        source = PimInventorySource(
            source_pim_grid_id="grid123",
            source_pim_grid_checksum="sum123",
            point_process="inhomogeneous_poisson",
            seed=42,
        )
        assert source.source_pim_grid_checksum == "sum123"
        assert source.model_dump()["source_pim_grid_checksum"] == "sum123"


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


class TestInventoryDataFormat:
    def test_json_value(self):
        assert InventoryDataFormat.json == "json"

    def test_csv_value(self):
        assert InventoryDataFormat.csv == "csv"

    def test_from_string(self):
        assert InventoryDataFormat("json") == InventoryDataFormat.json
        assert InventoryDataFormat("csv") == InventoryDataFormat.csv

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            InventoryDataFormat("parquet")


class TestInventoryJsonOrientation:
    def test_split_value(self):
        assert InventoryJsonOrientation.split == "split"

    def test_records_value(self):
        assert InventoryJsonOrientation.records == "records"

    def test_from_string(self):
        assert InventoryJsonOrientation("split") == InventoryJsonOrientation.split
        assert InventoryJsonOrientation("records") == InventoryJsonOrientation.records


class TestInventoryPartitionInfo:
    def test_basic(self):
        p = InventoryPartitionInfo(index=0, num_rows=1000)
        assert p.index == 0
        assert p.num_rows == 1000

    def test_serialization(self):
        p = InventoryPartitionInfo(index=2, num_rows=500)
        d = p.model_dump()
        assert d == {"index": 2, "num_rows": 500}


class TestInventoryDataMetadata:
    def test_basic(self):
        meta = InventoryDataMetadata(
            inventory_id="abc123",
            num_partitions=2,
            total_rows=2000,
            columns=["x", "y", "dbh"],
            partitions=[
                InventoryPartitionInfo(index=0, num_rows=1000),
                InventoryPartitionInfo(index=1, num_rows=1000),
            ],
        )
        assert meta.inventory_id == "abc123"
        assert meta.num_partitions == 2
        assert meta.total_rows == 2000
        assert meta.columns == ["x", "y", "dbh"]
        assert len(meta.partitions) == 2

    def test_serialization_round_trip(self):
        meta = InventoryDataMetadata(
            inventory_id="test",
            num_partitions=1,
            total_rows=100,
            columns=["x"],
            partitions=[InventoryPartitionInfo(index=0, num_rows=100)],
        )
        d = meta.model_dump()
        restored = InventoryDataMetadata(**d)
        assert restored == meta


class TestInventoryDataResponse:
    def test_split_format(self):
        resp = InventoryDataResponse(
            partition=0,
            num_rows=2,
            columns=["x", "y", "dbh"],
            data=[[1.0, 2.0, 25.3], [3.0, 4.0, 12.1]],
        )
        assert resp.partition == 0
        assert resp.num_rows == 2
        assert len(resp.data) == 2
        assert resp.data[0] == [1.0, 2.0, 25.3]

    def test_records_format(self):
        resp = InventoryDataResponse(
            partition=0,
            num_rows=2,
            columns=["x", "y", "dbh"],
            data=[
                {"x": 1.0, "y": 2.0, "dbh": 25.3},
                {"x": 3.0, "y": 4.0, "dbh": 12.1},
            ],
        )
        assert resp.num_rows == 2
        assert resp.data[0]["dbh"] == 25.3

    def test_empty_data(self):
        resp = InventoryDataResponse(
            partition=0,
            num_rows=0,
            columns=["x", "y"],
            data=[],
        )
        assert resp.num_rows == 0
        assert resp.data == []


class TestColumnSummary:
    """Tests for Column summary discriminated union."""

    def test_continuous_summary_round_trip(self):
        col = Column(
            key="dbh",
            type=ColumnType.continuous,
            unit="cm",
            summary={
                "type": "continuous",
                "count": 100,
                "null_count": 5,
                "min": 2.5,
                "max": 80.0,
                "mean": 25.3,
                "std": 10.1,
            },
        )
        assert isinstance(col.summary, ContinuousColumnSummary)
        assert col.summary.count == 100
        assert col.summary.min == 2.5
        d = col.model_dump()
        restored = Column(**d)
        assert restored.summary.mean == col.summary.mean

    def test_categorical_summary_round_trip(self):
        col = Column(
            key="fia_species_code",
            type=ColumnType.categorical,
            summary={
                "type": "categorical",
                "count": 200,
                "null_count": 0,
                "unique_count": 12,
            },
        )
        assert isinstance(col.summary, CategoricalColumnSummary)
        assert col.summary.unique_count == 12
        d = col.model_dump()
        restored = Column(**d)
        assert restored.summary.unique_count == 12

    def test_continuous_all_null_column(self):
        col = Column(
            key="dbh",
            type=ColumnType.continuous,
            summary={
                "type": "continuous",
                "count": 0,
                "null_count": 50,
                "min": None,
                "max": None,
                "mean": None,
                "std": None,
            },
        )
        assert col.summary.count == 0
        assert col.summary.min is None

    def test_summary_defaults_to_none(self):
        col = Column(key="dbh", type=ColumnType.continuous)
        assert col.summary is None

    def test_wrong_discriminator_raises(self):
        with pytest.raises(ValidationError):
            Column(
                key="dbh",
                type=ColumnType.continuous,
                summary={"type": "unknown", "count": 1, "null_count": 0},
            )


class TestForestryMetrics:
    """Tests for FIASpeciesGroupShare, TreeForestryMetrics, and ForestryMetrics."""

    def _make_tree_metrics(self, **overrides):
        defaults = {
            "type": "tree",
            "tree_count": 42,
            "basal_area_per_area": 120.5,
            "tree_density": 200.0,
            "quadratic_mean_diameter": 9.3,
            "dominant_species_groups": [
                {"spgrpcd": 10, "name": "Douglas-fir", "basal_area_share": 0.6},
                {"spgrpcd": 20, "name": "Ponderosa pine", "basal_area_share": 0.4},
            ],
        }
        defaults.update(overrides)
        return TreeForestryMetrics(**defaults)

    def test_round_trip(self):
        metrics = self._make_tree_metrics()
        d = metrics.model_dump()
        restored = TreeForestryMetrics(**d)
        assert restored == metrics
        assert restored.tree_count == 42
        assert restored.dominant_species_groups[0].name == "Douglas-fir"

    def test_inventory_with_forestry_metrics_round_trip(self):
        """ForestryMetrics survives a full Inventory serialize/deserialize cycle."""
        inv = Inventory(
            id="abc",
            domain_id="dom",
            type="tree",
            status="completed",
            source={
                "name": "pim",
                "source_pim_grid_id": "g1",
                "seed": 1,
                "point_process": "inhomogeneous_poisson",
            },
            forestry_metrics={
                "type": "tree",
                "tree_count": 10,
                "basal_area_per_area": 80.0,
                "tree_density": 150.0,
                "quadratic_mean_diameter": 7.5,
            },
        )
        d = inv.model_dump()
        restored = Inventory(**d)
        assert isinstance(restored.forestry_metrics, TreeForestryMetrics)
        assert restored.forestry_metrics.tree_count == 10

    def test_forestry_metrics_defaults_to_none(self):
        inv = Inventory(
            id="abc",
            domain_id="dom",
            type="tree",
            status="pending",
            source={
                "name": "pim",
                "source_pim_grid_id": "g1",
                "seed": 1,
                "point_process": "inhomogeneous_poisson",
            },
        )
        assert inv.forestry_metrics is None

    def test_basal_area_share_out_of_range_rejected(self):
        with pytest.raises(ValidationError):
            FIASpeciesGroupShare(spgrpcd=10, name="Douglas-fir", basal_area_share=1.5)
        with pytest.raises(ValidationError):
            FIASpeciesGroupShare(spgrpcd=10, name="Douglas-fir", basal_area_share=-0.1)

    def test_basal_area_share_boundary_values_accepted(self):
        FIASpeciesGroupShare(spgrpcd=10, name="x", basal_area_share=0.0)
        FIASpeciesGroupShare(spgrpcd=10, name="x", basal_area_share=1.0)

    def test_unit_string_compliance(self):
        validate_unit("ft**2/acre")
        validate_unit("1/acre")
        validate_unit("in")
