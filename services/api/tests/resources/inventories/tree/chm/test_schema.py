"""
Unit tests for api/v2/resources/inventories/tree/chm/schema.py

Tests the CHM inventory extraction schema models and validation.
These are pure unit tests with no external dependencies.
"""

import pytest
from api.resources.inventories.tree.chm.schema import (
    ChmInventorySource,
    ChmSpikeFilter,
    CreateChmInventoryRequest,
    StemIsolationLmf,
    StemIsolationVwf,
)
from pydantic import ValidationError


class TestChmSpikeFilter:
    """Tests for the ChmSpikeFilter model."""

    def test_default_values(self):
        """Model initializes with sensible defaults."""
        f = ChmSpikeFilter()
        assert f.window == 3
        assert f.min_prominence == 10.0

    def test_custom_values(self):
        """Model accepts custom valid parameters."""
        f = ChmSpikeFilter(window=5, min_prominence=25.0)
        assert f.window == 5
        assert f.min_prominence == 25.0

    def test_even_window_rejected(self):
        """An even window raises ValidationError."""
        with pytest.raises(ValidationError, match="must be an odd integer"):
            ChmSpikeFilter(window=4)

    def test_window_below_three_rejected(self):
        """A window smaller than 3 pixels raises ValidationError."""
        with pytest.raises(ValidationError, match="at least 3 pixels"):
            ChmSpikeFilter(window=1)

    def test_non_positive_prominence_rejected(self):
        """min_prominence must be strictly positive."""
        with pytest.raises(ValidationError):
            ChmSpikeFilter(min_prominence=0.0)


class TestStemIsolationLmf:
    """Tests for StemIsolationLmf model."""

    def test_default_values(self):
        """Model initializes with correct default values."""
        algo = StemIsolationLmf()
        assert algo.name == "lmf"
        assert algo.min_height == 2.0
        assert algo.footprint_size == 3

    def test_name_is_always_lmf(self):
        """The name field cannot be set to anything other than 'lmf'."""
        with pytest.raises(ValidationError):
            StemIsolationLmf(name="watershed")

    def test_custom_values(self):
        """Model accepts custom valid parameters."""
        algo = StemIsolationLmf(min_height=5.5, footprint_size=7)
        assert algo.min_height == 5.5
        assert algo.footprint_size == 7

    def test_model_dump(self):
        """Model serializes correctly."""
        algo = StemIsolationLmf(min_height=3.0, footprint_size=5)
        data = algo.model_dump()
        assert data["name"] == "lmf"
        assert data["min_height"] == 3.0
        assert data["footprint_size"] == 5

    def test_even_footprint_size_rejected(self):
        """Even footprint_size raises ValidationError."""
        with pytest.raises(ValidationError, match="must be an odd integer"):
            StemIsolationLmf(footprint_size=4)

    def test_max_height_defaults_to_120(self):
        """max_height defaults to a physical ceiling above the tallest known tree."""
        assert StemIsolationLmf().max_height == 120.0

    def test_max_height_can_be_disabled_with_none(self):
        """max_height accepts null to disable the ceiling."""
        assert StemIsolationLmf(max_height=None).max_height is None

    def test_max_height_below_min_rejected(self):
        """max_height must exceed min_height."""
        with pytest.raises(ValidationError, match="must be greater than 'min_height'"):
            StemIsolationLmf(min_height=5.0, max_height=5.0)

    def test_spike_filter_defaults_to_none(self):
        """The spike filter is off (None) by default."""
        assert StemIsolationLmf().spike_filter is None

    def test_spike_filter_accepts_defaults_from_empty_dict(self):
        """An empty spike_filter object fills in the model defaults."""
        algo = StemIsolationLmf(spike_filter={})
        assert isinstance(algo.spike_filter, ChmSpikeFilter)
        assert algo.spike_filter.window == 3
        assert algo.spike_filter.min_prominence == 10.0

    def test_spike_filter_accepts_tuned_params(self):
        """A tuned spike_filter is parsed into a ChmSpikeFilter."""
        algo = StemIsolationLmf(spike_filter={"window": 5, "min_prominence": 20.0})
        assert algo.spike_filter.window == 5
        assert algo.spike_filter.min_prominence == 20.0


class TestStemIsolationVwf:
    """Tests for StemIsolationVwf model."""

    def test_default_values(self):
        """Model initializes with correct default values."""
        algo = StemIsolationVwf()
        assert algo.name == "vwf"
        assert algo.min_height == 2.0
        assert algo.spatial_resolution is None
        assert algo.crown_ratio == 0.10
        assert algo.crown_offset == 1.0

    def test_name_is_always_vwf(self):
        """The name field cannot be set to anything other than 'vwf'."""
        with pytest.raises(ValidationError):
            StemIsolationVwf(name="lmf")

    def test_custom_values(self):
        """Model accepts custom valid parameters."""
        algo = StemIsolationVwf(
            min_height=5.5, spatial_resolution=0.5, crown_ratio=0.15, crown_offset=2.0
        )
        assert algo.min_height == 5.5
        assert algo.spatial_resolution == 0.5
        assert algo.crown_ratio == 0.15
        assert algo.crown_offset == 2.0

    def test_max_height_defaults_to_120(self):
        """max_height defaults to a physical ceiling above the tallest known tree."""
        assert StemIsolationVwf().max_height == 120.0

    def test_max_height_can_be_disabled_with_none(self):
        """max_height accepts null to disable the ceiling."""
        assert StemIsolationVwf(max_height=None).max_height is None

    def test_max_height_below_min_rejected(self):
        """max_height must exceed min_height."""
        with pytest.raises(ValidationError, match="must be greater than 'min_height'"):
            StemIsolationVwf(min_height=10.0, max_height=8.0)

    def test_spike_filter_defaults_to_none(self):
        """The spike filter is off (None) by default."""
        assert StemIsolationVwf().spike_filter is None

    def test_spike_filter_accepts_tuned_params(self):
        """A tuned spike_filter is parsed into a ChmSpikeFilter."""
        algo = StemIsolationVwf(spike_filter={"window": 7, "min_prominence": 15.0})
        assert algo.spike_filter.window == 7
        assert algo.spike_filter.min_prominence == 15.0


class TestChmInventorySource:
    """Tests for ChmInventorySource model."""

    def test_valid_initialization_lmf(self):
        """Model initializes successfully with required fields using LMF."""
        source = ChmInventorySource(
            source_chm_grid_id="grid123",
            algorithm=StemIsolationLmf(),
        )
        assert source.name == "chm"
        assert source.source_chm_grid_id == "grid123"
        assert source.algorithm.name == "lmf"

    def test_valid_initialization_vwf(self):
        """Model initializes successfully with required fields using VWF."""
        source = ChmInventorySource(
            source_chm_grid_id="grid123",
            algorithm=StemIsolationVwf(),
        )
        assert source.algorithm.name == "vwf"

    def test_name_is_always_chm(self):
        """The name field cannot be overridden."""
        with pytest.raises(ValidationError):
            ChmInventorySource(
                name="pim",
                source_chm_grid_id="grid123",
                algorithm=StemIsolationLmf(),
            )

    def test_source_chm_grid_id_is_required(self):
        """The source_chm_grid_id field is required."""
        with pytest.raises(ValidationError):
            ChmInventorySource(algorithm=StemIsolationLmf())

    def test_algorithm_is_required(self):
        """The algorithm field is required."""
        with pytest.raises(ValidationError):
            ChmInventorySource(source_chm_grid_id="grid123")

    def test_source_chm_grid_checksum_defaults_to_none(self):
        """source_chm_grid_checksum defaults to None when not captured."""
        source = ChmInventorySource(
            source_chm_grid_id="grid123",
            algorithm=StemIsolationLmf(),
        )
        assert source.source_chm_grid_checksum is None

    def test_source_chm_grid_checksum_round_trips(self):
        """source_chm_grid_checksum is carried through serialization."""
        source = ChmInventorySource(
            source_chm_grid_id="grid123",
            source_chm_grid_checksum="sum123",
            algorithm=StemIsolationLmf(),
        )
        assert source.source_chm_grid_checksum == "sum123"
        assert source.model_dump()["source_chm_grid_checksum"] == "sum123"


class TestCreateChmInventoryRequest:
    """Tests for CreateChmInventoryRequest model."""

    def test_minimal_valid_request(self):
        """Minimal request with only the required source_chm_grid_id defaults to LMF."""
        request = CreateChmInventoryRequest(source_chm_grid_id="grid123")
        assert request.source_chm_grid_id == "grid123"
        assert isinstance(request.algorithm, StemIsolationLmf)

    def test_request_with_vwf_algorithm(self):
        """Request can be successfully created with the VWF algorithm."""
        request = CreateChmInventoryRequest(
            source_chm_grid_id="grid123", algorithm=StemIsolationVwf(min_height=3.0)
        )
        assert isinstance(request.algorithm, StemIsolationVwf)
        assert request.algorithm.min_height == 3.0
        assert request.algorithm.crown_ratio == 0.10  # Check default persisted

    def test_missing_source_grid_id_rejected(self):
        """Missing required source_chm_grid_id raises ValidationError."""
        with pytest.raises(ValidationError):
            CreateChmInventoryRequest(name="Failing request")

    def test_treatments_default_empty(self):
        """Treatments default to an empty list and are accepted."""
        request = CreateChmInventoryRequest(source_chm_grid_id="grid123")
        assert request.treatments == []

    def test_treatments_rejected(self):
        """A non-empty treatments list is rejected — CHM has no diameter."""
        with pytest.raises(ValidationError, match="diameter"):
            CreateChmInventoryRequest(
                source_chm_grid_id="grid123",
                treatments=[
                    {"metric": "diameter", "method": "from_below", "value": 30.0}
                ],
            )
