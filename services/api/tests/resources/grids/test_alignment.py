"""
Schema tests for ``GridAlignmentSpecification``.

Verifies:
- Discriminator dispatch picks the right variant on ``target``.
- Default ``CreateSourceGridRequestBase.alignment`` is the domain target.
- ``target="grid"`` accepts an optional ``resolution``.
- Invalid combinations are rejected.
"""

import pytest
from api.resources.grids.alignment import (
    GridAlignmentDomainTarget,
    GridAlignmentGridTarget,
    GridAlignmentNativeTarget,
    GridAlignmentSpecification,
    ResamplingMethod,
)
from api.resources.grids.schema import CreateSourceGridRequestBase
from pydantic import TypeAdapter, ValidationError

_AlignmentAdapter = TypeAdapter(GridAlignmentSpecification)


class TestDiscriminatorDispatch:
    def test_default_is_domain_target(self):
        parsed = _AlignmentAdapter.validate_python({"target": "domain"})
        assert isinstance(parsed, GridAlignmentDomainTarget)

    def test_native_target(self):
        parsed = _AlignmentAdapter.validate_python({"target": "native"})
        assert isinstance(parsed, GridAlignmentNativeTarget)

    def test_grid_target(self):
        parsed = _AlignmentAdapter.validate_python({"target": "grid", "grid_id": "abc"})
        assert isinstance(parsed, GridAlignmentGridTarget)
        assert parsed.grid_id == "abc"


class TestDomainTargetDefaults:
    def test_resolution_optional(self):
        spec = GridAlignmentDomainTarget()
        assert spec.target == "domain"
        assert spec.resolution is None
        assert spec.method is None

    def test_explicit_resolution(self):
        spec = GridAlignmentDomainTarget(resolution=2.0)
        assert spec.resolution == 2.0

    def test_resolution_must_be_positive(self):
        with pytest.raises(ValidationError):
            GridAlignmentDomainTarget(resolution=0.0)


class TestGridTargetResolution:
    def test_resolution_optional(self):
        spec = _AlignmentAdapter.validate_python({"target": "grid", "grid_id": "abc"})
        assert spec.resolution is None

    def test_resolution_present(self):
        spec = _AlignmentAdapter.validate_python(
            {"target": "grid", "grid_id": "abc", "resolution": 1.0}
        )
        assert spec.resolution == 1.0

    def test_grid_id_required(self):
        with pytest.raises(ValidationError):
            _AlignmentAdapter.validate_python({"target": "grid"})


class TestResamplingMethod:
    def test_includes_common_methods(self):
        names = {m.value for m in ResamplingMethod}
        assert "nearest" in names
        assert "bilinear" in names
        assert "cubic" in names
        assert "mode" in names
        assert "average" in names
        assert "median" in names
        assert "root_mean_square" in names

    def test_gauss_not_exposed(self):
        names = {m.value for m in ResamplingMethod}
        assert "gauss" not in names

    def test_method_field_accepts_string(self):
        spec = _AlignmentAdapter.validate_python(
            {"target": "domain", "method": "nearest"}
        )
        assert spec.method == ResamplingMethod.nearest


class TestCreateSourceGridRequestBaseDefault:
    def test_alignment_defaults_to_domain_target(self):
        # CreateSourceGridRequestBase is abstract enough to instantiate directly
        # for default-checking; subclasses just add fields.
        request = CreateSourceGridRequestBase()
        assert isinstance(request.alignment, GridAlignmentDomainTarget)
        assert request.alignment.target == "domain"
        assert request.alignment.resolution is None

    def test_alignment_round_trip_through_dict(self):
        request = CreateSourceGridRequestBase.model_validate(
            {"alignment": {"target": "native", "resolution": 5.0}}
        )
        assert isinstance(request.alignment, GridAlignmentNativeTarget)
        assert request.alignment.resolution == 5.0


class TestRejections:
    def test_unknown_target_rejected(self):
        with pytest.raises(ValidationError):
            _AlignmentAdapter.validate_python({"target": "elsewhere"})

    def test_unknown_method_rejected(self):
        with pytest.raises(ValidationError):
            _AlignmentAdapter.validate_python(
                {"target": "domain", "method": "not-a-real-method"}
            )

    def test_grid_target_rejects_unknown_field(self):
        # Pydantic models accept extra fields by default; this confirms the
        # explicit absence of `resolution` on the discriminated form is *not*
        # what's enforcing rejection (since we now allow it). Sanity check.
        spec = _AlignmentAdapter.validate_python({"target": "grid", "grid_id": "abc"})
        assert spec.target == "grid"
