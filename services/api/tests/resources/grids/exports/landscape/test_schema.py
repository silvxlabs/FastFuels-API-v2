"""
Unit tests for the landscape export request schema and examples.

Pure unit tests — no Firestore, no HTTP. Behavioral tests for the router
live in test_router.py.
"""

import pytest
from api.resources.grids.exports.landscape.examples import (
    CREATE_LANDSCAPE_EXPORT_OPENAPI_EXAMPLES,
)
from api.resources.grids.exports.landscape.schema import (
    LandscapeExportAlignmentDomainTarget,
    LandscapeExportAlignmentGridTarget,
    LandscapeExportRequest,
    LandscapeExportSource,
    LandscapeFieldSource,
)
from pydantic import ValidationError

_ROLE_FIELDS = [
    "elevation",
    "slope",
    "aspect",
    "fuel_model",
    "canopy_cover",
    "canopy_height",
    "canopy_base_height",
    "canopy_bulk_density",
]


def _minimal_request_kwargs() -> dict:
    """Return a minimal valid LandscapeExportRequest as kwargs."""
    return {
        "fire_behavior_fuel_model": "fbfm40",
        "elevation": LandscapeFieldSource(grid_id="topo", band="elevation"),
        "slope": LandscapeFieldSource(grid_id="topo", band="slope"),
        "aspect": LandscapeFieldSource(grid_id="topo", band="aspect"),
        "fuel_model": LandscapeFieldSource(grid_id="fbfm", band="fbfm"),
        "canopy_cover": LandscapeFieldSource(grid_id="canopy", band="cc"),
        "canopy_height": LandscapeFieldSource(grid_id="canopy", band="chm"),
        "canopy_base_height": LandscapeFieldSource(grid_id="canopy", band="cbh"),
        "canopy_bulk_density": LandscapeFieldSource(grid_id="canopy", band="cbd"),
    }


class TestLandscapeFieldSource:
    def test_basic(self):
        source = LandscapeFieldSource(grid_id="g1", band="b1")
        assert source.grid_id == "g1"
        assert source.band == "b1"

    def test_grid_id_required(self):
        with pytest.raises(ValidationError):
            LandscapeFieldSource(band="b1")

    def test_band_required(self):
        with pytest.raises(ValidationError):
            LandscapeFieldSource(grid_id="g1")


class TestAlignmentDomainTarget:
    def test_defaults_to_landfire_native(self):
        spec = LandscapeExportAlignmentDomainTarget()
        assert spec.target == "domain"
        assert spec.resolution == 30.0

    def test_custom_resolution(self):
        spec = LandscapeExportAlignmentDomainTarget(resolution=10.0)
        assert spec.resolution == 10.0

    def test_positive_required(self):
        with pytest.raises(ValidationError):
            LandscapeExportAlignmentDomainTarget(resolution=0)
        with pytest.raises(ValidationError):
            LandscapeExportAlignmentDomainTarget(resolution=-30.0)


class TestAlignmentGridTarget:
    def test_basic(self):
        spec = LandscapeExportAlignmentGridTarget(target="grid", grid_id="master_xyz")
        assert spec.target == "grid"
        assert spec.grid_id == "master_xyz"

    def test_grid_id_required(self):
        with pytest.raises(ValidationError):
            LandscapeExportAlignmentGridTarget(target="grid")


class TestAlignmentOnRequest:
    """Alignment is discriminated on `target` at the request level."""

    def test_alignment_defaults_when_omitted(self):
        request = LandscapeExportRequest(**_minimal_request_kwargs())
        assert request.alignment.target == "domain"
        assert request.alignment.resolution == 30.0

    def test_grid_alignment(self):
        request = LandscapeExportRequest(
            **_minimal_request_kwargs(),
            alignment={"target": "grid", "grid_id": "master_xyz"},
        )
        assert request.alignment.target == "grid"
        assert request.alignment.grid_id == "master_xyz"

    def test_partial_alignment_defaults_target_to_domain(self):
        # `target` may be omitted when alignment is supplied — it defaults to
        # "domain" so `{"resolution": 10}` is accepted.
        request = LandscapeExportRequest(
            **_minimal_request_kwargs(),
            alignment={"resolution": 10.0},
        )
        assert request.alignment.target == "domain"
        assert request.alignment.resolution == 10.0

    def test_unknown_target_rejected(self):
        with pytest.raises(ValidationError):
            LandscapeExportRequest(
                **_minimal_request_kwargs(),
                alignment={"target": "native", "resolution": 30.0},
            )


class TestLandscapeExportRequest:
    def test_minimal_valid(self):
        request = LandscapeExportRequest(**_minimal_request_kwargs())
        assert request.fire_behavior_fuel_model == "fbfm40"
        assert request.canopy_height.band == "chm"
        assert request.canopy_base_height.band == "cbh"
        assert request.canopy_bulk_density.band == "cbd"
        assert request.expiration_days == 7
        assert request.name == ""
        assert request.description == ""
        assert request.tags == []

    @pytest.mark.parametrize("missing", _ROLE_FIELDS)
    def test_required_role_missing(self, missing):
        kwargs = _minimal_request_kwargs()
        kwargs.pop(missing)
        with pytest.raises(ValidationError):
            LandscapeExportRequest(**kwargs)

    def test_fire_behavior_fuel_model_required(self):
        kwargs = _minimal_request_kwargs()
        kwargs.pop("fire_behavior_fuel_model")
        with pytest.raises(ValidationError):
            LandscapeExportRequest(**kwargs)

    def test_fbfm13_accepted(self):
        kwargs = _minimal_request_kwargs()
        kwargs["fire_behavior_fuel_model"] = "fbfm13"
        request = LandscapeExportRequest(**kwargs)
        assert request.fire_behavior_fuel_model == "fbfm13"

    def test_unknown_fuel_model_classification_rejected(self):
        kwargs = _minimal_request_kwargs()
        kwargs["fire_behavior_fuel_model"] = "fccs"
        with pytest.raises(ValidationError):
            LandscapeExportRequest(**kwargs)

    def test_expiration_days_clamped(self):
        with pytest.raises(ValidationError):
            LandscapeExportRequest(**_minimal_request_kwargs(), expiration_days=0)
        with pytest.raises(ValidationError):
            LandscapeExportRequest(**_minimal_request_kwargs(), expiration_days=8)

    def test_metadata_passthrough(self):
        request = LandscapeExportRequest(
            **_minimal_request_kwargs(),
            expiration_days=3,
            name="my landscape",
            description="for FlamMap",
            tags=["a", "b"],
        )
        assert request.expiration_days == 3
        assert request.name == "my landscape"
        assert request.description == "for FlamMap"
        assert request.tags == ["a", "b"]


def _georeference() -> dict:
    return {
        "crs": "EPSG:5070",
        "transform": [30.0, 0.0, -1379265.0, 0.0, -30.0, 2781015.0],
        "shape": [39, 50],
    }


class TestLandscapeExportSource:
    def test_minimal(self):
        source = LandscapeExportSource(
            domain_id="d1",
            **_minimal_request_kwargs(),
            alignment={"target": "domain", "resolution": 30.0},
            georeference=_georeference(),
        )
        assert source.name == "landscape"
        assert source.domain_id == "d1"
        assert source.fire_behavior_fuel_model == "fbfm40"
        assert source.canopy_height.band == "chm"
        assert source.georeference.crs == "EPSG:5070"
        assert source.georeference.shape == (39, 50)

    def test_name_is_pinned(self):
        # `name` is a Literal["landscape"]; assigning anything else fails.
        with pytest.raises(ValidationError):
            LandscapeExportSource(
                name="geotiff",  # type: ignore[arg-type]
                domain_id="d1",
                **_minimal_request_kwargs(),
                alignment={"target": "domain", "resolution": 30.0},
                georeference=_georeference(),
            )

    def test_georeference_is_required(self):
        with pytest.raises(ValidationError):
            LandscapeExportSource(
                domain_id="d1",
                **_minimal_request_kwargs(),
                alignment={"target": "domain", "resolution": 30.0},
            )

    def test_georeference_is_typed_not_a_blob(self):
        """A free-form dict is rejected — the lattice has a schema."""
        with pytest.raises(ValidationError):
            LandscapeExportSource(
                domain_id="d1",
                **_minimal_request_kwargs(),
                alignment={"target": "domain", "resolution": 30.0},
                georeference={"landscape_grid": {}},
            )


class TestExampleValidation:
    """Every documented example must pass schema validation."""

    @pytest.mark.parametrize(
        "name,example",
        [
            (key, ex["value"])
            for key, ex in CREATE_LANDSCAPE_EXPORT_OPENAPI_EXAMPLES.items()
        ],
    )
    def test_example_validates(self, name, example):
        LandscapeExportRequest(**example)
