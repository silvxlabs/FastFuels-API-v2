"""
Unit tests for api/v2/resources/grids/resample/schema.py

Tests the resample schema models: ResamplingMethod enum,
ResampleSource, and CreateResampleRequest.
These are pure unit tests with no external dependencies.
"""

import pytest
from api.resources.grids.resample.examples import ALL_RESAMPLE_EXAMPLE_VALUES
from api.resources.grids.resample.schema import (
    CreateResampleRequest,
    ResampleSource,
    ResamplingMethod,
)
from pydantic import ValidationError


class TestResamplingMethod:
    """Tests for ResamplingMethod enum."""

    def test_enum_values(self):
        """All resampling methods exist."""
        assert ResamplingMethod.nearest == "nearest"
        assert ResamplingMethod.bilinear == "bilinear"
        assert ResamplingMethod.cubic == "cubic"
        assert ResamplingMethod.cubic_spline == "cubic_spline"
        assert ResamplingMethod.lanczos == "lanczos"
        assert ResamplingMethod.average == "average"
        assert ResamplingMethod.mode == "mode"
        assert ResamplingMethod.max == "max"
        assert ResamplingMethod.min == "min"
        assert ResamplingMethod.median == "median"
        assert ResamplingMethod.first_quartile == "first_quartile"
        assert ResamplingMethod.third_quartile == "third_quartile"
        assert ResamplingMethod.sum == "sum"
        assert ResamplingMethod.root_mean_square == "root_mean_square"

    def test_enum_count(self):
        """Exactly 14 resampling methods."""
        assert len(ResamplingMethod) == 14

    def test_can_create_from_string(self):
        """Can create enum from string value."""
        method = ResamplingMethod("bilinear")
        assert method == ResamplingMethod.bilinear

    def test_invalid_string_raises(self):
        """Invalid string raises ValueError."""
        with pytest.raises(ValueError):
            ResamplingMethod("invalid")


class TestResampleSource:
    """Tests for ResampleSource model."""

    def test_name_is_always_resample(self):
        """The name field is always 'resample'."""
        source = ResampleSource(
            source_grid_id="abc123",
            target_resolution=2.0,
            method="bilinear",
        )
        assert source.name == "resample"

    def test_name_cannot_be_overridden(self):
        """The name field cannot be set to anything other than 'resample'."""
        with pytest.raises(ValidationError):
            ResampleSource(
                name="other",
                source_grid_id="abc123",
                target_resolution=2.0,
                method="bilinear",
            )

    def test_source_grid_id_is_required(self):
        """The source_grid_id field is required."""
        with pytest.raises(ValidationError):
            ResampleSource(
                target_resolution=2.0,
                method="bilinear",
            )

    def test_target_resolution_is_required(self):
        """The target_resolution field is required."""
        with pytest.raises(ValidationError):
            ResampleSource(
                source_grid_id="abc123",
                method="bilinear",
            )

    def test_method_is_required(self):
        """The method field is required."""
        with pytest.raises(ValidationError):
            ResampleSource(
                source_grid_id="abc123",
                target_resolution=2.0,
            )

    def test_method_overrides_defaults_to_empty_dict(self):
        """method_overrides defaults to empty dict."""
        source = ResampleSource(
            source_grid_id="abc123",
            target_resolution=2.0,
            method="bilinear",
        )
        assert source.method_overrides == {}

    def test_method_overrides_can_be_set(self):
        """method_overrides stores per-band overrides."""
        source = ResampleSource(
            source_grid_id="abc123",
            target_resolution=2.0,
            method="bilinear",
            method_overrides={"fbfm": "nearest"},
        )
        assert source.method_overrides == {"fbfm": ResamplingMethod.nearest}

    def test_model_dump(self):
        """Model serializes correctly."""
        source = ResampleSource(
            source_grid_id="abc123",
            target_resolution=2.0,
            method="bilinear",
            method_overrides={"fbfm": "nearest"},
        )
        data = source.model_dump()
        assert data["name"] == "resample"
        assert data["source_grid_id"] == "abc123"
        assert data["target_resolution"] == 2.0
        assert data["method"] == "bilinear"
        assert data["method_overrides"] == {"fbfm": "nearest"}


class TestCreateResampleRequest:
    """Tests for CreateResampleRequest model."""

    def test_minimal_valid_request(self):
        """Minimal request with only required fields."""
        request = CreateResampleRequest(
            source_grid_id="abc123",
            resolution=2.0,
        )
        assert request.source_grid_id == "abc123"
        assert request.resolution == 2.0
        assert request.method == ResamplingMethod.bilinear
        assert request.method_overrides == {}
        assert request.name == ""
        assert request.description == ""
        assert request.tags == []
        assert request.modifications == []

    def test_source_grid_id_is_required(self):
        """source_grid_id field is required."""
        with pytest.raises(ValidationError):
            CreateResampleRequest(resolution=2.0)

    def test_resolution_is_required(self):
        """resolution field is required."""
        with pytest.raises(ValidationError):
            CreateResampleRequest(source_grid_id="abc123")

    def test_resolution_must_be_at_least_1(self):
        """resolution must be >= 1 meter."""
        with pytest.raises(ValidationError):
            CreateResampleRequest(source_grid_id="abc123", resolution=0)
        with pytest.raises(ValidationError):
            CreateResampleRequest(source_grid_id="abc123", resolution=0.5)
        with pytest.raises(ValidationError):
            CreateResampleRequest(source_grid_id="abc123", resolution=-1)
        # Exactly 1m is valid
        request = CreateResampleRequest(source_grid_id="abc123", resolution=1.0)
        assert request.resolution == 1.0

    def test_method_defaults_to_bilinear(self):
        """method defaults to bilinear."""
        request = CreateResampleRequest(
            source_grid_id="abc123",
            resolution=2.0,
        )
        assert request.method == ResamplingMethod.bilinear

    def test_method_can_be_overridden(self):
        """method can be set to a different resampling method."""
        request = CreateResampleRequest(
            source_grid_id="abc123",
            resolution=2.0,
            method="nearest",
        )
        assert request.method == ResamplingMethod.nearest

    def test_method_overrides_defaults_to_empty_dict(self):
        """method_overrides defaults to empty dict."""
        request = CreateResampleRequest(
            source_grid_id="abc123",
            resolution=2.0,
        )
        assert request.method_overrides == {}

    def test_name_defaults_to_empty_string(self):
        """name defaults to empty string."""
        request = CreateResampleRequest(
            source_grid_id="abc123",
            resolution=2.0,
        )
        assert request.name == ""

    def test_tags_defaults_to_empty_list(self):
        """tags defaults to empty list."""
        request = CreateResampleRequest(
            source_grid_id="abc123",
            resolution=2.0,
        )
        assert request.tags == []

    def test_modifications_defaults_to_empty_list(self):
        """modifications defaults to empty list."""
        request = CreateResampleRequest(
            source_grid_id="abc123",
            resolution=2.0,
        )
        assert request.modifications == []

    def test_full_request_with_all_fields(self):
        """Full request with all optional fields set."""
        request = CreateResampleRequest(
            source_grid_id="abc123",
            resolution=2.0,
            method="nearest",
            method_overrides={"fbfm": "nearest"},
            name="Resampled grid",
            description="A resampled grid",
            tags=["resampled", "2m"],
        )
        assert request.source_grid_id == "abc123"
        assert request.resolution == 2.0
        assert request.method == ResamplingMethod.nearest
        assert request.method_overrides == {"fbfm": ResamplingMethod.nearest}
        assert request.name == "Resampled grid"
        assert request.description == "A resampled grid"
        assert request.tags == ["resampled", "2m"]


class TestExamplesValidateAgainstSchema:
    """Tests that documented examples are valid Pydantic inputs."""

    @pytest.mark.parametrize("example_name,example_value", ALL_RESAMPLE_EXAMPLE_VALUES)
    def test_example_validates_against_schema(self, example_name, example_value):
        """Each example from ALL_RESAMPLE_EXAMPLE_VALUES creates a valid request."""
        request = CreateResampleRequest(**example_value)
        assert request.source_grid_id is not None
        assert request.resolution > 0
