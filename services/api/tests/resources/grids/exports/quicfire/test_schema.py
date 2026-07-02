"""
Unit tests for the QUIC-Fire export request schema and examples.

Pure unit tests — no Firestore, no HTTP. Behavioral tests for the router
live in test_router.py.
"""

import pytest
from api.resources.grids.exports.quicfire.examples import (
    CREATE_QUICFIRE_EXPORT_OPENAPI_EXAMPLES,
)
from api.resources.grids.exports.quicfire.schema import (
    FieldSource,
    QUICFireExportAlignmentDomainTarget,
    QUICFireExportAlignmentGridTarget,
    QuicfireExportRequest,
    QuicfireExportSource,
)
from pydantic import ValidationError


def _domain_alignment() -> dict:
    return {"target": "domain", "dx": 2.0, "dy": 2.0, "dz": 1.0}


def _minimal_request_kwargs() -> dict:
    """Return a minimal valid QuicfireExportRequest as kwargs."""
    return {
        "alignment": _domain_alignment(),
        "canopy_bulk_density": FieldSource(
            grid_id="tree", band="bulk_density.foliage.live"
        ),
        "canopy_moisture": FieldSource(grid_id="tree", band="fuel_moisture.live"),
        "surface_fuel_load": FieldSource(grid_id="lookup", band="fuel_load.1hr"),
        "surface_fuel_depth": FieldSource(grid_id="lookup", band="fuel_depth"),
        "surface_moisture": FieldSource(grid_id="uniform", band="fuel_moisture.1hr"),
    }


class TestFieldSource:
    def test_basic(self):
        source = FieldSource(grid_id="g1", band="b1")
        assert source.grid_id == "g1"
        assert source.band == "b1"

    def test_grid_id_required(self):
        with pytest.raises(ValidationError):
            FieldSource(band="b1")

    def test_band_required(self):
        with pytest.raises(ValidationError):
            FieldSource(grid_id="g1")


class TestAlignmentDomainTarget:
    def test_basic(self):
        spec = QUICFireExportAlignmentDomainTarget(dx=2.0, dy=2.0, dz=1.0)
        assert spec.target == "domain"
        assert spec.dx == 2.0
        assert spec.dy == 2.0
        assert spec.dz == 1.0

    def test_dx_dy_must_be_equal(self):
        with pytest.raises(ValidationError) as exc:
            QUICFireExportAlignmentDomainTarget(dx=2.0, dy=3.0, dz=1.0)
        assert "must be equal" in str(exc.value)

    @pytest.mark.parametrize("field", ["dx", "dy", "dz"])
    def test_positive_required(self, field):
        kwargs = {"dx": 2.0, "dy": 2.0, "dz": 1.0}
        kwargs[field] = 0
        with pytest.raises(ValidationError):
            QUICFireExportAlignmentDomainTarget(**kwargs)
        kwargs[field] = -1.0
        with pytest.raises(ValidationError):
            QUICFireExportAlignmentDomainTarget(**kwargs)

    def test_defaults_match_qf_recommended(self):
        spec = QUICFireExportAlignmentDomainTarget()
        assert spec.target == "domain"
        assert spec.dx == 2.0
        assert spec.dy == 2.0
        assert spec.dz == 1.0


class TestAlignmentGridTarget:
    def test_basic(self):
        spec = QUICFireExportAlignmentGridTarget(target="grid", grid_id="master_xyz")
        assert spec.target == "grid"
        assert spec.grid_id == "master_xyz"

    def test_grid_id_required(self):
        with pytest.raises(ValidationError):
            QUICFireExportAlignmentGridTarget(target="grid")


class TestAlignmentOnRequest:
    """Alignment is discriminated on `target` at the request level."""

    def test_domain_alignment(self):
        request = QuicfireExportRequest(**_minimal_request_kwargs())
        assert request.alignment.target == "domain"
        assert request.alignment.dx == 2.0
        assert request.alignment.dy == 2.0
        assert request.alignment.dz == 1.0

    def test_grid_alignment(self):
        kwargs = _minimal_request_kwargs()
        kwargs["alignment"] = {"target": "grid", "grid_id": "master_xyz"}
        request = QuicfireExportRequest(**kwargs)
        assert request.alignment.target == "grid"
        assert request.alignment.grid_id == "master_xyz"

    def test_alignment_defaults_when_omitted(self):
        kwargs = _minimal_request_kwargs()
        kwargs.pop("alignment")
        request = QuicfireExportRequest(**kwargs)
        assert request.alignment.target == "domain"
        assert request.alignment.dx == 2.0
        assert request.alignment.dy == 2.0
        assert request.alignment.dz == 1.0

    def test_partial_alignment_defaults_target_to_domain(self):
        # `target` may be omitted when alignment is supplied — it defaults to
        # "domain" so `{"dx": 1, "dy": 1}` is accepted.
        kwargs = _minimal_request_kwargs()
        kwargs["alignment"] = {"dx": 1.0, "dy": 1.0}
        request = QuicfireExportRequest(**kwargs)
        assert request.alignment.target == "domain"
        assert request.alignment.dx == 1.0
        assert request.alignment.dy == 1.0
        assert request.alignment.dz == 1.0  # vertical default preserved

    def test_unknown_target_rejected(self):
        kwargs = _minimal_request_kwargs()
        kwargs["alignment"] = {"target": "native", "dx": 2.0, "dy": 2.0, "dz": 1.0}
        with pytest.raises(ValidationError):
            QuicfireExportRequest(**kwargs)


class TestQuicfireExportRequest:
    def test_minimal_valid(self):
        request = QuicfireExportRequest(**_minimal_request_kwargs())
        assert request.canopy_savr is None
        assert request.surface_savr is None
        assert request.topography is None
        assert request.expiration_days == 7
        assert request.name == ""
        assert request.description == ""
        assert request.tags == []

    def test_with_topography(self):
        request = QuicfireExportRequest(
            **_minimal_request_kwargs(),
            topography=FieldSource(grid_id="topo", band="elevation"),
        )
        assert request.topography is not None
        assert request.topography.grid_id == "topo"
        assert request.topography.band == "elevation"

    def test_with_both_savr_roles(self):
        request = QuicfireExportRequest(
            **_minimal_request_kwargs(),
            canopy_savr=FieldSource(grid_id="tree", band="savr.foliage"),
            surface_savr=FieldSource(grid_id="lookup", band="savr.1hr"),
        )
        assert request.canopy_savr is not None
        assert request.surface_savr is not None

    @pytest.mark.parametrize(
        "missing",
        [
            "canopy_bulk_density",
            "canopy_moisture",
            "surface_fuel_load",
            "surface_fuel_depth",
            "surface_moisture",
        ],
    )
    def test_required_role_missing(self, missing):
        kwargs = _minimal_request_kwargs()
        kwargs.pop(missing)
        with pytest.raises(ValidationError):
            QuicfireExportRequest(**kwargs)

    def test_savr_pairing_canopy_only_rejected(self):
        with pytest.raises(ValidationError) as exc:
            QuicfireExportRequest(
                **_minimal_request_kwargs(),
                canopy_savr=FieldSource(grid_id="tree", band="savr.foliage"),
            )
        assert "canopy_savr and surface_savr" in str(exc.value)

    def test_savr_pairing_surface_only_rejected(self):
        with pytest.raises(ValidationError) as exc:
            QuicfireExportRequest(
                **_minimal_request_kwargs(),
                surface_savr=FieldSource(grid_id="lookup", band="savr.1hr"),
            )
        assert "canopy_savr and surface_savr" in str(exc.value)

    def test_savr_pairing_neither_ok(self):
        QuicfireExportRequest(**_minimal_request_kwargs())

    def test_expiration_days_clamped(self):
        with pytest.raises(ValidationError):
            QuicfireExportRequest(**_minimal_request_kwargs(), expiration_days=0)
        with pytest.raises(ValidationError):
            QuicfireExportRequest(**_minimal_request_kwargs(), expiration_days=8)

    def test_metadata_passthrough(self):
        request = QuicfireExportRequest(
            **_minimal_request_kwargs(),
            expiration_days=3,
            name="my export",
            description="for QF run",
            tags=["a", "b"],
        )
        assert request.expiration_days == 3
        assert request.name == "my export"
        assert request.description == "for QF run"
        assert request.tags == ["a", "b"]

    def test_merge_field_defaults(self):
        request = QuicfireExportRequest(**_minimal_request_kwargs())
        assert request.rhof_merge == "sum"
        assert request.moist_merge == "max"  # v1-parity default
        assert request.savr_merge == "weighted_avg"

    def test_moist_merge_weighted_avg_accepted(self):
        request = QuicfireExportRequest(
            **_minimal_request_kwargs(), moist_merge="weighted_avg"
        )
        assert request.moist_merge == "weighted_avg"

    def test_moist_merge_max_accepted(self):
        request = QuicfireExportRequest(**_minimal_request_kwargs(), moist_merge="max")
        assert request.moist_merge == "max"

    @pytest.mark.parametrize(
        "field,bad_value",
        [
            ("rhof_merge", "overwrite"),
            ("rhof_merge", "max"),
            ("moist_merge", "overwrite"),
            ("moist_merge", "additive"),
            ("savr_merge", "overwrite"),
            ("savr_merge", "max"),
        ],
    )
    def test_merge_field_rejects_unsupported_values(self, field, bad_value):
        kwargs = _minimal_request_kwargs()
        kwargs[field] = bad_value
        with pytest.raises(ValidationError):
            QuicfireExportRequest(**kwargs)


class TestQuicfireExportSource:
    def test_minimal(self):
        source = QuicfireExportSource(
            domain_id="d1",
            **_minimal_request_kwargs(),
            resolved={"domain": {}, "fire_grid": {}, "roles": {}},
        )
        assert source.name == "quicfire"
        assert source.domain_id == "d1"
        assert source.alignment.target == "domain"
        assert source.canopy_savr is None
        assert source.surface_savr is None
        assert source.topography is None
        assert source.rhof_merge == "sum"
        assert source.moist_merge == "max"
        assert source.savr_merge == "weighted_avg"

    def test_name_is_pinned(self):
        # `name` is a Literal["quicfire"]; assigning anything else fails.
        with pytest.raises(ValidationError):
            QuicfireExportSource(
                name="zarr",  # type: ignore[arg-type]
                domain_id="d1",
                **_minimal_request_kwargs(),
                resolved={"domain": {}, "fire_grid": {}, "roles": {}},
            )

    def test_resolved_is_required(self):
        with pytest.raises(ValidationError):
            QuicfireExportSource(domain_id="d1", **_minimal_request_kwargs())


class TestExampleValidation:
    """Every documented example must pass schema validation."""

    @pytest.mark.parametrize(
        "name,example",
        [
            (key, ex["value"])
            for key, ex in CREATE_QUICFIRE_EXPORT_OPENAPI_EXAMPLES.items()
        ],
    )
    def test_example_validates(self, name, example):
        QuicfireExportRequest(**example)
