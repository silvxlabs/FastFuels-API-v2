"""
Unit tests for api/v2/resources/inventories/tree/pim/fusion/chm/schema.py

Tests the PIM-CHM fusion inventory schema models and validation.
These are pure unit tests with no external dependencies.
"""

import pytest
from api.resources.inventories.tree.pim.fusion.chm.schema import (
    CreatePimChmFusionInventoryRequest,
    PimChmFusionInventorySource,
    ReimputationMethod,
)
from pydantic import ValidationError


class TestReimputationMethod:
    """Tests for the ReimputationMethod model."""

    def test_default_values(self):
        """Defaults match the v1 production values."""
        method = ReimputationMethod()
        assert method.name == "reimputation"
        assert method.resolution == 7.5
        assert method.min_height == 2.0
        assert method.cover_threshold == 0.1

    def test_name_is_always_reimputation(self):
        """The name field cannot be set to anything else."""
        with pytest.raises(ValidationError):
            ReimputationMethod(name="surface_matching")

    def test_custom_values(self):
        """Model accepts custom valid parameters."""
        method = ReimputationMethod(
            resolution=10.0, min_height=1.0, cover_threshold=0.25
        )
        assert method.resolution == 10.0
        assert method.min_height == 1.0
        assert method.cover_threshold == 0.25

    def test_resolution_must_be_positive(self):
        """A non-positive resolution is rejected."""
        with pytest.raises(ValidationError):
            ReimputationMethod(resolution=0)

    def test_cover_threshold_bounds(self):
        """cover_threshold must be within [0, 1]."""
        with pytest.raises(ValidationError):
            ReimputationMethod(cover_threshold=1.5)
        with pytest.raises(ValidationError):
            ReimputationMethod(cover_threshold=-0.1)

    def test_min_height_non_negative(self):
        """min_height cannot be negative."""
        with pytest.raises(ValidationError):
            ReimputationMethod(min_height=-1.0)


class TestPimChmFusionInventorySource:
    """Tests for the PimChmFusionInventorySource model."""

    def test_valid_initialization(self):
        """Model initializes with required fields."""
        source = PimChmFusionInventorySource(
            source_pim_grid_id="pim123",
            source_chm_grid_id="chm123",
            method=ReimputationMethod(),
            point_process="inhomogeneous_poisson",
            seed=42,
        )
        assert source.name == "pim"
        assert source.fusion == ["chm"]
        assert source.source_pim_grid_id == "pim123"
        assert source.source_chm_grid_id == "chm123"
        assert source.method.name == "reimputation"

    def test_name_is_always_pim(self):
        """The name field cannot be overridden — the ?source=pim filter relies on it."""
        with pytest.raises(ValidationError):
            PimChmFusionInventorySource(
                name="chm",
                source_pim_grid_id="pim123",
                source_chm_grid_id="chm123",
                method=ReimputationMethod(),
                point_process="inhomogeneous_poisson",
                seed=42,
            )

    def test_fusion_defaults_to_chm(self):
        """fusion records the non-primary sources; defaults to ['chm'] here."""
        source = PimChmFusionInventorySource(
            source_pim_grid_id="pim123",
            source_chm_grid_id="chm123",
            method=ReimputationMethod(),
            point_process="inhomogeneous_poisson",
            seed=42,
        )
        assert source.model_dump()["fusion"] == ["chm"]

    def test_checksums_default_to_none(self):
        """Both source checksums default to None when not captured."""
        source = PimChmFusionInventorySource(
            source_pim_grid_id="pim123",
            source_chm_grid_id="chm123",
            method=ReimputationMethod(),
            point_process="inhomogeneous_poisson",
            seed=42,
        )
        assert source.source_pim_grid_checksum is None
        assert source.source_chm_grid_checksum is None

    def test_checksums_round_trip(self):
        """Both source checksums are carried through serialization."""
        source = PimChmFusionInventorySource(
            source_pim_grid_id="pim123",
            source_pim_grid_checksum="psum",
            source_chm_grid_id="chm123",
            source_chm_grid_checksum="csum",
            method=ReimputationMethod(),
            point_process="inhomogeneous_poisson",
            seed=42,
        )
        dumped = source.model_dump()
        assert dumped["source_pim_grid_checksum"] == "psum"
        assert dumped["source_chm_grid_checksum"] == "csum"


class TestCreatePimChmFusionInventoryRequest:
    """Tests for the CreatePimChmFusionInventoryRequest model."""

    def test_minimal_request_defaults_to_reimputation(self):
        """Minimal request with both grid ids defaults to the reimputation method."""
        request = CreatePimChmFusionInventoryRequest(
            source_pim_grid_id="pim123", source_chm_grid_id="chm123"
        )
        assert request.source_pim_grid_id == "pim123"
        assert request.source_chm_grid_id == "chm123"
        assert isinstance(request.method, ReimputationMethod)
        assert request.method.resolution == 7.5
        assert request.point_process == "inhomogeneous_poisson"

    def test_seed_is_generated_when_omitted(self):
        """A seed is generated when the client omits one."""
        request = CreatePimChmFusionInventoryRequest(
            source_pim_grid_id="pim123", source_chm_grid_id="chm123"
        )
        assert isinstance(request.seed, int)

    def test_missing_pim_grid_id_rejected(self):
        """Missing required source_pim_grid_id raises ValidationError."""
        with pytest.raises(ValidationError):
            CreatePimChmFusionInventoryRequest(source_chm_grid_id="chm123")

    def test_missing_chm_grid_id_rejected(self):
        """Missing required source_chm_grid_id raises ValidationError."""
        with pytest.raises(ValidationError):
            CreatePimChmFusionInventoryRequest(source_pim_grid_id="pim123")

    def test_unknown_method_name_rejected(self):
        """An unknown method name is rejected by the discriminated union."""
        with pytest.raises(ValidationError):
            CreatePimChmFusionInventoryRequest(
                source_pim_grid_id="pim123",
                source_chm_grid_id="chm123",
                method={"name": "not_a_method"},
            )

    def test_custom_method_knobs_parse(self):
        """Reimputation knobs supplied in the body are parsed and stored."""
        request = CreatePimChmFusionInventoryRequest(
            source_pim_grid_id="pim123",
            source_chm_grid_id="chm123",
            method={
                "name": "reimputation",
                "resolution": 10.0,
                "min_height": 1.5,
                "cover_threshold": 0.3,
            },
        )
        assert request.method.resolution == 10.0
        assert request.method.min_height == 1.5
        assert request.method.cover_threshold == 0.3

    def test_modifications_and_treatments_default_empty(self):
        """Modifications and treatments default to empty lists."""
        request = CreatePimChmFusionInventoryRequest(
            source_pim_grid_id="pim123", source_chm_grid_id="chm123"
        )
        assert request.modifications == []
        assert request.treatments == []
