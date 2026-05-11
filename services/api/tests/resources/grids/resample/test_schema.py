"""
Unit tests for api/v2/resources/grids/resample/schema.py

Tests the resample schema models against the alignment-based design
introduced in #205. ``ResamplingMethod`` is generated from rasterio's
canonical enum, so the tests focus on members the project depends on
rather than asserting an exact count.
"""

import pytest
from api.resources.grids.alignment import (
    GridAlignmentDomainTarget,
    GridAlignmentGridTarget,
    GridAlignmentNativeTarget,
)
from api.resources.grids.resample.examples import ALL_RESAMPLE_EXAMPLE_VALUES
from api.resources.grids.resample.schema import (
    CreateResampleRequest,
    ResampleSource,
    ResamplingMethod,
)
from pydantic import ValidationError


class TestResamplingMethod:
    """Tests for ResamplingMethod enum (generated from rasterio.enums.Resampling)."""

    def test_includes_common_methods(self):
        names = {m.value for m in ResamplingMethod}
        for name in (
            "nearest",
            "bilinear",
            "cubic",
            "cubic_spline",
            "lanczos",
            "average",
            "mode",
            "max",
            "min",
            "med",
            "sum",
            "rms",
        ):
            assert name in names

    def test_can_create_from_string(self):
        method = ResamplingMethod("bilinear")
        assert method == ResamplingMethod.bilinear

    def test_invalid_string_raises(self):
        with pytest.raises(ValueError):
            ResamplingMethod("invalid")


class TestResampleSource:
    """Tests for ResampleSource model."""

    def test_name_is_always_resample(self):
        source = ResampleSource(
            source_grid_id="abc123",
            alignment=GridAlignmentDomainTarget(resolution=2.0),
        )
        assert source.name == "resample"

    def test_name_cannot_be_overridden(self):
        with pytest.raises(ValidationError):
            ResampleSource(
                name="other",
                source_grid_id="abc123",
                alignment=GridAlignmentDomainTarget(resolution=2.0),
            )

    def test_source_grid_id_is_required(self):
        with pytest.raises(ValidationError):
            ResampleSource(
                alignment=GridAlignmentDomainTarget(resolution=2.0),
            )

    def test_alignment_is_required(self):
        with pytest.raises(ValidationError):
            ResampleSource(source_grid_id="abc123")

    def test_method_overrides_defaults_to_empty_dict(self):
        source = ResampleSource(
            source_grid_id="abc123",
            alignment=GridAlignmentDomainTarget(resolution=2.0),
        )
        assert source.method_overrides == {}

    def test_method_overrides_can_be_set(self):
        source = ResampleSource(
            source_grid_id="abc123",
            alignment=GridAlignmentDomainTarget(resolution=2.0),
            method_overrides={"fbfm": "nearest"},
        )
        assert source.method_overrides == {"fbfm": ResamplingMethod.nearest}

    def test_model_dump(self):
        source = ResampleSource(
            source_grid_id="abc123",
            alignment=GridAlignmentDomainTarget(
                resolution=2.0, method=ResamplingMethod.bilinear
            ),
            method_overrides={"fbfm": "nearest"},
        )
        data = source.model_dump()
        assert data["name"] == "resample"
        assert data["source_grid_id"] == "abc123"
        assert data["alignment"]["target"] == "domain"
        assert data["alignment"]["resolution"] == 2.0
        assert data["alignment"]["method"] == "bilinear"
        assert data["method_overrides"] == {"fbfm": "nearest"}


class TestCreateResampleRequest:
    """Tests for CreateResampleRequest model."""

    def test_minimal_valid_request_defaults_to_domain_target(self):
        request = CreateResampleRequest(source_grid_id="abc123")
        assert request.source_grid_id == "abc123"
        assert isinstance(request.alignment, GridAlignmentDomainTarget)
        assert request.alignment.resolution is None
        assert request.alignment.method is None
        assert request.method_overrides == {}
        assert request.name == ""
        assert request.description == ""
        assert request.tags == []
        assert request.modifications == []

    def test_source_grid_id_is_required(self):
        with pytest.raises(ValidationError):
            CreateResampleRequest()

    def test_alignment_resolution_must_be_at_least_1(self):
        # Resolution validation happens at the alignment-spec level.
        with pytest.raises(ValidationError):
            CreateResampleRequest(
                source_grid_id="abc123",
                alignment={"target": "domain", "resolution": 0.0},
            )
        with pytest.raises(ValidationError):
            CreateResampleRequest(
                source_grid_id="abc123",
                alignment={"target": "domain", "resolution": 0.5},
            )
        # Exactly 1m is valid
        request = CreateResampleRequest(
            source_grid_id="abc123",
            alignment={"target": "domain", "resolution": 1.0},
        )
        assert request.alignment.resolution == 1.0

    def test_alignment_native_target(self):
        request = CreateResampleRequest(
            source_grid_id="abc123",
            alignment={"target": "native", "resolution": 5.0},
        )
        assert isinstance(request.alignment, GridAlignmentNativeTarget)
        assert request.alignment.resolution == 5.0

    def test_alignment_grid_target_minimal(self):
        request = CreateResampleRequest(
            source_grid_id="abc123",
            alignment={"target": "grid", "grid_id": "xyz789"},
        )
        assert isinstance(request.alignment, GridAlignmentGridTarget)
        assert request.alignment.grid_id == "xyz789"
        assert request.alignment.resolution is None

    def test_alignment_grid_target_with_resolution(self):
        request = CreateResampleRequest(
            source_grid_id="abc123",
            alignment={
                "target": "grid",
                "grid_id": "xyz789",
                "resolution": 1.0,
            },
        )
        assert isinstance(request.alignment, GridAlignmentGridTarget)
        assert request.alignment.resolution == 1.0

    def test_method_overrides_defaults_to_empty_dict(self):
        request = CreateResampleRequest(
            source_grid_id="abc123",
            alignment={"target": "domain", "resolution": 2.0},
        )
        assert request.method_overrides == {}

    def test_full_request_with_all_fields(self):
        request = CreateResampleRequest(
            source_grid_id="abc123",
            alignment={
                "target": "domain",
                "resolution": 2.0,
                "method": "nearest",
            },
            method_overrides={"fbfm": "nearest"},
            name="Resampled grid",
            description="A resampled grid",
            tags=["resampled", "2m"],
        )
        assert request.source_grid_id == "abc123"
        assert request.alignment.resolution == 2.0
        assert request.alignment.method == ResamplingMethod.nearest
        assert request.method_overrides == {"fbfm": ResamplingMethod.nearest}
        assert request.name == "Resampled grid"
        assert request.description == "A resampled grid"
        assert request.tags == ["resampled", "2m"]


class TestExamplesValidateAgainstSchema:
    """Tests that documented examples are valid Pydantic inputs."""

    @pytest.mark.parametrize("example_name,example_value", ALL_RESAMPLE_EXAMPLE_VALUES)
    def test_example_validates_against_schema(self, example_name, example_value):
        request = CreateResampleRequest(**example_value)
        assert request.source_grid_id is not None
        assert request.alignment is not None
