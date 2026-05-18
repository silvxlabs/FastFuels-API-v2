"""
Unit tests for api/v2/resources/grids/schema.py

Tests the core Grid schema models, enums, and base classes.
These are pure unit tests with no external dependencies.
"""

from datetime import datetime

import pytest
from api.resources.grids.schema import (
    CHUNK_SHAPE,
    Band,
    BandType,
    Chunks,
    CreateGridRequestBase,
    Georeference,
    Georeference3D,
    Grid,
    GridSortField,
    ListGridsResponse,
    UpdateGridRequestBody,
)
from api.schema import JobStatus
from pydantic import ValidationError


class TestChunkShape:
    """Tests for CHUNK_SHAPE constant."""

    def test_chunk_shape_is_512_512(self):
        """CHUNK_SHAPE is [512, 512]."""
        assert CHUNK_SHAPE == [512, 512]


class TestChunks:
    """Tests for the Chunks model."""

    def test_2d_shape_valid(self):
        c = Chunks(shape=(512, 512))
        assert c.shape == (512, 512)
        assert c.count is None
        assert c.count_by_axis is None

    def test_3d_shape_valid(self):
        c = Chunks(
            shape=(2, 512, 512), count=12, count_by_axis={"z": 3, "y": 2, "x": 2}
        )
        assert c.shape == (2, 512, 512)
        assert c.count == 12
        assert c.count_by_axis == {"z": 3, "y": 2, "x": 2}

    def test_shape_is_required(self):
        with pytest.raises(ValidationError):
            Chunks()


class TestBandType:
    """Tests for BandType enum."""

    def test_continuous_value(self):
        """continuous enum has correct value."""
        assert BandType.continuous.value == "continuous"

    def test_categorical_value(self):
        """categorical enum has correct value."""
        assert BandType.categorical.value == "categorical"

    def test_enum_count(self):
        """Enum has exactly 2 members."""
        assert len(BandType) == 2

    def test_can_create_from_string(self):
        """Enum can be created from string value."""
        assert BandType("continuous") == BandType.continuous
        assert BandType("categorical") == BandType.categorical

    def test_invalid_string_raises_valueerror(self):
        """Invalid string raises ValueError."""
        with pytest.raises(ValueError):
            BandType("invalid")


class TestGridSortField:
    """Tests for GridSortField enum."""

    def test_created_on_value(self):
        """created_on enum has correct value."""
        assert GridSortField.created_on.value == "created_on"

    def test_modified_on_value(self):
        """modified_on enum has correct value."""
        assert GridSortField.modified_on.value == "modified_on"

    def test_name_value(self):
        """name enum has correct value."""
        assert GridSortField.name.value == "name"

    def test_enum_count(self):
        """Enum has exactly 3 members."""
        assert len(GridSortField) == 3

    def test_can_create_from_string(self):
        """Enum can be created from string value."""
        assert GridSortField("created_on") == GridSortField.created_on
        assert GridSortField("modified_on") == GridSortField.modified_on
        assert GridSortField("name") == GridSortField.name


class TestBand:
    """Tests for Band model."""

    def test_minimal_valid_band(self):
        """Minimal band with required fields."""
        band = Band(key="fbfm", type=BandType.categorical, index=0)
        assert band.key == "fbfm"
        assert band.type == BandType.categorical
        assert band.index == 0
        assert band.unit is None

    def test_band_with_unit(self):
        """Band with unit specified."""
        band = Band(
            key="fuel_load.1hr", type=BandType.continuous, unit="kg/m**2", index=1
        )
        assert band.key == "fuel_load.1hr"
        assert band.type == BandType.continuous
        assert band.unit == "kg/m**2"
        assert band.index == 1

    def test_key_is_required(self):
        """key field is required."""
        with pytest.raises(ValidationError):
            Band(type=BandType.categorical, index=0)

    def test_type_is_required(self):
        """type field is required."""
        with pytest.raises(ValidationError):
            Band(key="fbfm", index=0)

    def test_index_is_required(self):
        """index field is required."""
        with pytest.raises(ValidationError):
            Band(key="fbfm", type=BandType.categorical)

    def test_type_accepts_string_value(self):
        """type field accepts string that maps to enum."""
        band = Band(key="fbfm", type="categorical", index=0)
        assert band.type == BandType.categorical

    def test_model_dump(self):
        """Model serializes correctly."""
        band = Band(
            key="fuel_load.1hr", type=BandType.continuous, unit="kg/m**2", index=1
        )
        data = band.model_dump()
        assert data == {
            "key": "fuel_load.1hr",
            "type": "continuous",
            "unit": "kg/m**2",
            "index": 1,
        }

    def test_dot_notation_keys_preserved(self):
        """Dot notation in keys is preserved."""
        band = Band(key="savr.live_herb", type=BandType.continuous, unit="1/m", index=5)
        assert band.key == "savr.live_herb"


class TestGeoreference:
    """Tests for Georeference model."""

    def test_minimal_valid_georeference(self):
        """Minimal georeference with required fields."""
        georef = Georeference(
            crs="EPSG:32610",
            transform=(30.0, 0.0, 500000.0, 0.0, -30.0, 4500000.0),
            shape=(100, 100),
        )
        assert georef.crs == "EPSG:32610"
        assert georef.transform == (30.0, 0.0, 500000.0, 0.0, -30.0, 4500000.0)
        assert georef.shape == (100, 100)

    def test_crs_is_required(self):
        """crs field is required."""
        with pytest.raises(ValidationError):
            Georeference(
                transform=(30.0, 0.0, 500000.0, 0.0, -30.0, 4500000.0),
                shape=(100, 100),
            )

    def test_transform_is_required(self):
        """transform field is required."""
        with pytest.raises(ValidationError):
            Georeference(crs="EPSG:32610", shape=(100, 100))

    def test_shape_is_required(self):
        """shape field is required."""
        with pytest.raises(ValidationError):
            Georeference(
                crs="EPSG:32610",
                transform=(30.0, 0.0, 500000.0, 0.0, -30.0, 4500000.0),
            )

    def test_transform_must_have_6_elements(self):
        """transform must have exactly 6 elements."""
        with pytest.raises(ValidationError):
            Georeference(
                crs="EPSG:32610",
                transform=(30.0, 0.0, 500000.0),  # Only 3 elements
                shape=(100, 100),
            )

    def test_shape_must_have_2_elements(self):
        """shape must have exactly 2 elements for 2D."""
        with pytest.raises(ValidationError):
            Georeference(
                crs="EPSG:32610",
                transform=(30.0, 0.0, 500000.0, 0.0, -30.0, 4500000.0),
                shape=(100,),  # Only 1 element
            )

    def test_model_dump(self):
        """Model serializes correctly."""
        georef = Georeference(
            crs="EPSG:32610",
            transform=(30.0, 0.0, 500000.0, 0.0, -30.0, 4500000.0),
            shape=(100, 100),
        )
        data = georef.model_dump()
        assert data == {
            "crs": "EPSG:32610",
            "transform": (30.0, 0.0, 500000.0, 0.0, -30.0, 4500000.0),
            "shape": (100, 100),
        }


class TestGeoreference3D:
    """Tests for Georeference3D model."""

    def test_minimal_valid_georeference_3d(self):
        """Minimal 3D georeference with required fields."""
        georef = Georeference3D(
            crs="EPSG:32610",
            transform=(1.0, 0.0, 500000.0, 0.0, -1.0, 4500000.0),
            shape=(50, 100, 100),
            z_resolution=1.0,
            z_origin=0.0,
        )
        assert georef.crs == "EPSG:32610"
        assert georef.shape == (50, 100, 100)
        assert georef.z_resolution == 1.0
        assert georef.z_origin == 0.0

    def test_shape_must_have_3_elements(self):
        """shape must have exactly 3 elements for 3D."""
        with pytest.raises(ValidationError):
            Georeference3D(
                crs="EPSG:32610",
                transform=(1.0, 0.0, 500000.0, 0.0, -1.0, 4500000.0),
                shape=(100, 100),  # Only 2 elements
                z_resolution=1.0,
                z_origin=0.0,
            )

    def test_z_resolution_is_required(self):
        """z_resolution field is required."""
        with pytest.raises(ValidationError):
            Georeference3D(
                crs="EPSG:32610",
                transform=(1.0, 0.0, 500000.0, 0.0, -1.0, 4500000.0),
                shape=(50, 100, 100),
                z_origin=0.0,
            )

    def test_z_origin_is_required(self):
        """z_origin field is required."""
        with pytest.raises(ValidationError):
            Georeference3D(
                crs="EPSG:32610",
                transform=(1.0, 0.0, 500000.0, 0.0, -1.0, 4500000.0),
                shape=(50, 100, 100),
                z_resolution=1.0,
            )

    def test_inherits_from_georeference(self):
        """Georeference3D inherits from Georeference."""
        assert issubclass(Georeference3D, Georeference)


class TestCreateGridRequestBase:
    """Tests for CreateGridRequestBase model.

    domain_id now comes from the URL path parameter, not the request body.
    """

    def test_minimal_valid_request(self):
        """Minimal request with no required fields (all have defaults)."""
        request = CreateGridRequestBase()
        assert request.name == ""
        assert request.description == ""
        assert request.tags == []
        assert request.modifications == []

    def test_name_defaults_to_empty_string(self):
        """name field defaults to empty string."""
        request = CreateGridRequestBase()
        assert request.name == ""

    def test_description_defaults_to_empty_string(self):
        """description field defaults to empty string."""
        request = CreateGridRequestBase()
        assert request.description == ""

    def test_tags_defaults_to_empty_list(self):
        """tags field defaults to empty list."""
        request = CreateGridRequestBase()
        assert request.tags == []

    def test_modifications_defaults_to_empty_list(self):
        """modifications field defaults to empty list."""
        request = CreateGridRequestBase()
        assert request.modifications == []

    def test_full_request_with_all_fields(self):
        """Full request with all optional fields."""
        request = CreateGridRequestBase(
            name="Test Grid",
            description="A test grid",
            tags=["test", "fuel"],
        )
        assert request.name == "Test Grid"
        assert request.description == "A test grid"
        assert request.tags == ["test", "fuel"]


class TestUpdateGridRequestBody:
    """Tests for UpdateGridRequestBody model."""

    def test_empty_update_is_valid(self):
        """Empty update body is valid (all fields optional)."""
        body = UpdateGridRequestBody()
        assert body.name is None
        assert body.description is None
        assert body.tags is None

    def test_update_name_only(self):
        """Can update just the name."""
        body = UpdateGridRequestBody(name="New Name")
        assert body.name == "New Name"
        assert body.description is None
        assert body.tags is None

    def test_update_description_only(self):
        """Can update just the description."""
        body = UpdateGridRequestBody(description="New Description")
        assert body.name is None
        assert body.description == "New Description"
        assert body.tags is None

    def test_update_tags_only(self):
        """Can update just the tags."""
        body = UpdateGridRequestBody(tags=["new", "tags"])
        assert body.name is None
        assert body.description is None
        assert body.tags == ["new", "tags"]

    def test_update_all_fields(self):
        """Can update all fields at once."""
        body = UpdateGridRequestBody(
            name="New Name",
            description="New Description",
            tags=["new", "tags"],
        )
        assert body.name == "New Name"
        assert body.description == "New Description"
        assert body.tags == ["new", "tags"]

    def test_model_dump_exclude_none(self):
        """model_dump with exclude_none returns only set fields."""
        body = UpdateGridRequestBody(name="New Name")
        data = body.model_dump(exclude_none=True)
        assert data == {"name": "New Name"}
        assert "description" not in data
        assert "tags" not in data


class TestGrid:
    """Tests for Grid model."""

    @pytest.fixture
    def minimal_grid_data(self):
        """Minimal valid grid data."""
        return {
            "id": "abc123",
            "domain_id": "domain_xyz",
            "status": JobStatus.pending,
            "created_on": datetime.now(),
            "modified_on": datetime.now(),
            "source": {
                "name": "landfire",
                "product": "fbfm40",
                "version": "2022",
                "description": "",
            },
            "bands": [{"key": "fbfm", "type": "categorical", "unit": None, "index": 0}],
        }

    def test_minimal_valid_grid(self, minimal_grid_data):
        """Grid with minimal required fields."""
        grid = Grid(**minimal_grid_data)
        assert grid.id == "abc123"
        assert grid.domain_id == "domain_xyz"
        assert grid.status == JobStatus.pending
        assert grid.name == ""
        assert grid.description == ""
        assert grid.tags == []
        assert grid.modifications == []
        assert grid.georeference is None

    def test_id_is_required(self, minimal_grid_data):
        """id field is required."""
        del minimal_grid_data["id"]
        with pytest.raises(ValidationError):
            Grid(**minimal_grid_data)

    def test_domain_id_is_required(self, minimal_grid_data):
        """domain_id field is required."""
        del minimal_grid_data["domain_id"]
        with pytest.raises(ValidationError):
            Grid(**minimal_grid_data)

    def test_status_is_required(self, minimal_grid_data):
        """status field is required."""
        del minimal_grid_data["status"]
        with pytest.raises(ValidationError):
            Grid(**minimal_grid_data)

    def test_source_is_required(self, minimal_grid_data):
        """source field is required."""
        del minimal_grid_data["source"]
        with pytest.raises(ValidationError):
            Grid(**minimal_grid_data)

    def test_bands_is_required(self, minimal_grid_data):
        """bands field is required."""
        del minimal_grid_data["bands"]
        with pytest.raises(ValidationError):
            Grid(**minimal_grid_data)

    def test_chunks_defaults_to_none(self, minimal_grid_data):
        """chunks defaults to None on Grid response."""
        grid = Grid(**minimal_grid_data)
        assert grid.chunks is None

    def test_chunks_can_be_set(self, minimal_grid_data):
        """chunks accepts a Chunks dict on Grid response."""
        minimal_grid_data["chunks"] = {
            "shape": (512, 512),
            "count": 4,
            "count_by_axis": {"y": 2, "x": 2},
        }
        grid = Grid(**minimal_grid_data)
        assert grid.chunks is not None
        assert grid.chunks.shape == (512, 512)
        assert grid.chunks.count == 4
        assert grid.chunks.count_by_axis == {"y": 2, "x": 2}

    def test_chunks_3d_round_trip(self, minimal_grid_data):
        """3D chunks layout round-trips through the Grid model."""
        minimal_grid_data["chunks"] = {
            "shape": (2, 512, 512),
            "count": 12,
            "count_by_axis": {"z": 3, "y": 2, "x": 2},
        }
        grid = Grid(**minimal_grid_data)
        assert grid.chunks.shape == (2, 512, 512)
        assert grid.chunks.count == 12
        assert grid.chunks.count_by_axis == {"z": 3, "y": 2, "x": 2}

    def test_chunks_count_and_count_by_axis_default_to_none(self, minimal_grid_data):
        """count and count_by_axis are optional (populated after processing)."""
        minimal_grid_data["chunks"] = {"shape": (512, 512)}
        grid = Grid(**minimal_grid_data)
        assert grid.chunks.shape == (512, 512)
        assert grid.chunks.count is None
        assert grid.chunks.count_by_axis is None

    def test_georeference_defaults_to_none(self, minimal_grid_data):
        """georeference defaults to None (populated by backend)."""
        grid = Grid(**minimal_grid_data)
        assert grid.georeference is None

    def test_georeference_can_be_set(self, minimal_grid_data):
        """georeference can be set when provided."""
        minimal_grid_data["georeference"] = {
            "crs": "EPSG:32610",
            "transform": (30.0, 0.0, 500000.0, 0.0, -30.0, 4500000.0),
            "shape": (100, 100),
        }
        grid = Grid(**minimal_grid_data)
        assert grid.georeference is not None
        assert grid.georeference.crs == "EPSG:32610"

    def test_status_accepts_string_value(self, minimal_grid_data):
        """status field accepts string that maps to JobStatus."""
        minimal_grid_data["status"] = "pending"
        grid = Grid(**minimal_grid_data)
        assert grid.status == JobStatus.pending

    def test_full_grid_with_all_fields(self, minimal_grid_data):
        """Grid with all optional fields."""
        minimal_grid_data["name"] = "Test Grid"
        minimal_grid_data["description"] = "A test grid"
        minimal_grid_data["tags"] = ["test", "fuel"]
        minimal_grid_data["georeference"] = {
            "crs": "EPSG:32610",
            "transform": (30.0, 0.0, 500000.0, 0.0, -30.0, 4500000.0),
            "shape": (100, 100),
        }

        grid = Grid(**minimal_grid_data)
        assert grid.name == "Test Grid"
        assert grid.description == "A test grid"
        assert grid.tags == ["test", "fuel"]
        assert grid.georeference is not None

    def test_bands_are_parsed_as_band_objects(self, minimal_grid_data):
        """bands are parsed as Band model instances."""
        grid = Grid(**minimal_grid_data)
        assert len(grid.bands) == 1
        assert isinstance(grid.bands[0], Band)
        assert grid.bands[0].key == "fbfm"

    def test_georeference_is_parsed_as_georeference(self, minimal_grid_data):
        """georeference is parsed as Georeference model when provided."""
        minimal_grid_data["georeference"] = {
            "crs": "EPSG:32610",
            "transform": (30.0, 0.0, 500000.0, 0.0, -30.0, 4500000.0),
            "shape": (100, 100),
        }
        grid = Grid(**minimal_grid_data)
        assert isinstance(grid.georeference, Georeference)
        assert grid.georeference.crs == "EPSG:32610"

    def test_georeference_3d_is_parsed(self, minimal_grid_data):
        """3D georeference is parsed as Georeference3D model."""
        minimal_grid_data["georeference"] = {
            "crs": "EPSG:32610",
            "transform": (1.0, 0.0, 500000.0, 0.0, -1.0, 4500000.0),
            "shape": (50, 100, 100),
            "z_resolution": 1.0,
            "z_origin": 0.0,
        }
        grid = Grid(**minimal_grid_data)
        assert isinstance(grid.georeference, Georeference3D)
        assert grid.georeference.z_resolution == 1.0


class TestListGridsResponse:
    """Tests for ListGridsResponse model."""

    @pytest.fixture
    def sample_grid_data(self):
        """Sample grid data for list response."""
        return {
            "id": "abc123",
            "domain_id": "domain_xyz",
            "name": "Test Grid",
            "description": "",
            "status": "pending",
            "created_on": datetime.now(),
            "modified_on": datetime.now(),
            "source": {
                "name": "landfire",
                "product": "fbfm40",
                "version": "2022",
                "description": "",
            },
            "bands": [{"key": "fbfm", "type": "categorical", "unit": None, "index": 0}],
            "georeference": None,
            "tags": [],
            "modifications": [],
        }

    def test_valid_list_response(self, sample_grid_data):
        """Valid list response with grids."""
        response = ListGridsResponse(
            grids=[sample_grid_data],
            current_page=0,
            page_size=100,
            total_items=1,
        )
        assert len(response.grids) == 1
        assert response.current_page == 0
        assert response.page_size == 100
        assert response.total_items == 1

    def test_empty_grids_list(self):
        """Empty grids list is valid."""
        response = ListGridsResponse(
            grids=[],
            current_page=0,
            page_size=100,
            total_items=0,
        )
        assert response.grids == []
        assert response.total_items == 0

    def test_grids_are_parsed_as_grid_objects(self, sample_grid_data):
        """grids are parsed as Grid model instances."""
        response = ListGridsResponse(
            grids=[sample_grid_data],
            current_page=0,
            page_size=100,
            total_items=1,
        )
        assert isinstance(response.grids[0], Grid)

    def test_grids_is_required(self):
        """grids field is required."""
        with pytest.raises(ValidationError):
            ListGridsResponse(current_page=0, page_size=100, total_items=0)

    def test_current_page_is_required(self, sample_grid_data):
        """current_page field is required."""
        with pytest.raises(ValidationError):
            ListGridsResponse(grids=[], page_size=100, total_items=0)

    def test_page_size_is_required(self, sample_grid_data):
        """page_size field is required."""
        with pytest.raises(ValidationError):
            ListGridsResponse(grids=[], current_page=0, total_items=0)

    def test_total_items_is_required(self, sample_grid_data):
        """total_items field is required."""
        with pytest.raises(ValidationError):
            ListGridsResponse(grids=[], current_page=0, page_size=100)
