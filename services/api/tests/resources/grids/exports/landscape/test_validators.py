"""
Unit tests for the landscape export validator helpers.

Every helper in `validators.py` takes plain dicts (or a Pydantic request)
and raises `HTTPException(422)` on failure — no Firestore, no async — so
each can be exercised with hand-built fixtures here. The Firestore loading
layer (`_load_all_grids`) and the live-router behavior of the orchestrator
live in test_router.py.
"""

import pytest
from api.resources.grids.exports.landscape.schema import (
    LandscapeExportRequest,
    LandscapeFieldSource,
)
from api.resources.grids.exports.landscape.validators import (
    _MAX_CELLS,
    _ROLE_CONTRACT,
    _build_landscape_grid,
    _check_cell_count_cap,
    _check_fuel_model_declaration,
    _check_role_alignment,
    _check_role_contract,
    _iter_roles,
    _make_source,
)
from fastapi import HTTPException

# Canonical landscape lattice used by every helper test below.
_CRS = "EPSG:32611"
_RES = 30.0
_NX = 100
_NY = 50
_ORIGIN_X = 720210.0
_ORIGIN_Y = 5190660.0
_LANDSCAPE_TRANSFORM = [_RES, 0.0, _ORIGIN_X, 0.0, -_RES, _ORIGIN_Y]
_LANDSCAPE_GRID = {
    "nx": _NX,
    "ny": _NY,
    "dx": _RES,
    "dy": _RES,
    "transform": _LANDSCAPE_TRANSFORM,
    "crs": _CRS,
}


def _grid_doc(
    *,
    shape: tuple[int, ...] = (_NY, _NX),
    transform: list[float] = _LANDSCAPE_TRANSFORM,
    crs: str | None = _CRS,
    bands: list[dict] | None = None,
    source: dict | None = None,
) -> dict:
    """Hand-build a Firestore-shaped grid doc for the helpers to consume."""
    return {
        "georeference": {
            "shape": list(shape),
            "transform": list(transform),
            "crs": crs,
        },
        "bands": bands or [],
        "source": source or {},
    }


def _topo_doc(**overrides) -> dict:
    defaults = {
        "bands": [
            {"key": "elevation", "type": "continuous", "unit": "m", "index": 0},
            {"key": "slope", "type": "continuous", "unit": "deg", "index": 1},
            {"key": "aspect", "type": "continuous", "unit": "deg", "index": 2},
        ],
    }
    return _grid_doc(**(defaults | overrides))


def _fbfm_doc(**overrides) -> dict:
    defaults = {
        "bands": [{"key": "fbfm", "type": "categorical", "unit": None, "index": 0}],
        "source": {"name": "landfire", "product": "fbfm40"},
    }
    return _grid_doc(**(defaults | overrides))


def _canopy_doc(**overrides) -> dict:
    defaults = {
        "bands": [
            {"key": "cc", "type": "continuous", "unit": "%", "index": 0},
            {"key": "chm", "type": "continuous", "unit": "m", "index": 1},
            {"key": "cbh", "type": "continuous", "unit": "m", "index": 2},
            {"key": "cbd", "type": "continuous", "unit": "kg/m**3", "index": 3},
        ],
    }
    return _grid_doc(**(defaults | overrides))


def _minimal_request(alignment_override: dict | None = None) -> LandscapeExportRequest:
    kwargs = {
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
    if alignment_override is not None:
        kwargs["alignment"] = alignment_override
    return LandscapeExportRequest(**kwargs)


class TestIterRoles:
    def test_landfire_band_order(self):
        names = [n for n, _ in _iter_roles(_minimal_request())]
        assert names == [
            "elevation",
            "slope",
            "aspect",
            "fuel_model",
            "canopy_cover",
            "canopy_height",
            "canopy_base_height",
            "canopy_bulk_density",
        ]


class TestCheckRoleContract:
    def test_valid(self):
        # Should not raise.
        _check_role_contract(
            _topo_doc(),
            LandscapeFieldSource(grid_id="topo", band="elevation"),
            "elevation",
        )

    def test_categorical_role_skips_unit_check(self):
        # fuel_model has no unit in the contract; a unit-less band passes.
        _check_role_contract(
            _fbfm_doc(),
            LandscapeFieldSource(grid_id="fbfm", band="fbfm"),
            "fuel_model",
        )

    def test_missing_band(self):
        with pytest.raises(HTTPException) as exc:
            _check_role_contract(
                _topo_doc(),
                LandscapeFieldSource(grid_id="topo", band="nope"),
                "elevation",
            )
        assert exc.value.status_code == 422

    def test_wrong_unit(self):
        doc = _topo_doc(
            bands=[{"key": "elevation", "type": "continuous", "unit": "ft", "index": 0}]
        )
        with pytest.raises(HTTPException) as exc:
            _check_role_contract(
                doc,
                LandscapeFieldSource(grid_id="topo", band="elevation"),
                "elevation",
            )
        assert exc.value.status_code == 422

    def test_3d_grid_rejected(self):
        doc = _topo_doc(shape=(10, _NY, _NX))
        with pytest.raises(HTTPException) as exc:
            _check_role_contract(
                doc,
                LandscapeFieldSource(grid_id="topo", band="elevation"),
                "elevation",
            )
        assert exc.value.status_code == 422


class TestCheckFuelModelDeclaration:
    def test_builtin_fbfm40_with_fbfm40_declared(self):
        # Should not raise.
        _check_fuel_model_declaration(_fbfm_doc(), "fbfm", "fbfm40")

    def test_builtin_fbfm40_with_fbfm13_declared_rejected(self):
        with pytest.raises(HTTPException) as exc:
            _check_fuel_model_declaration(_fbfm_doc(), "fbfm", "fbfm13")
        assert exc.value.status_code == 422
        assert "FBFM40" in exc.value.detail
        assert "fbfm13" in exc.value.detail

    def test_custom_grid_declaration_trusted(self):
        # Uploaded/custom grids have non-fbfm40 provenance; either
        # declaration is accepted.
        uploaded = _fbfm_doc(source={"name": "upload"})
        _check_fuel_model_declaration(uploaded, "fbfm", "fbfm13")
        _check_fuel_model_declaration(uploaded, "fbfm", "fbfm40")

    def test_missing_source_tolerated(self):
        doc = _fbfm_doc(source={})
        doc["source"] = None
        _check_fuel_model_declaration(doc, "fbfm", "fbfm13")


class TestBuildLandscapeGrid:
    def test_domain_target_default_30m(self):
        domain = {
            "bbox": [
                _ORIGIN_X,
                _ORIGIN_Y - _NY * _RES,
                _ORIGIN_X + _NX * _RES,
                _ORIGIN_Y,
            ],
            "crs": {"type": "name", "properties": {"name": _CRS}},
        }
        lattice = _build_landscape_grid(_minimal_request(), domain, None)
        assert lattice["nx"] == _NX
        assert lattice["ny"] == _NY
        assert lattice["dx"] == _RES
        assert lattice["transform"][2] == pytest.approx(_ORIGIN_X)
        assert lattice["transform"][5] == pytest.approx(_ORIGIN_Y)
        assert lattice["crs"] == _CRS

    def test_unpadded_bbox_pads_via_ceil(self):
        # Bbox extends 1 m past a clean 30 m corner — width should ceil up.
        domain = {
            "bbox": [
                _ORIGIN_X,
                _ORIGIN_Y - _NY * _RES,
                _ORIGIN_X + _NX * _RES + 1.0,
                _ORIGIN_Y,
            ],
            "crs": {"type": "name", "properties": {"name": _CRS}},
        }
        lattice = _build_landscape_grid(_minimal_request(), domain, None)
        assert lattice["nx"] == _NX + 1
        assert lattice["ny"] == _NY

    def test_custom_resolution(self):
        domain = {
            "bbox": [_ORIGIN_X, _ORIGIN_Y - 300, _ORIGIN_X + 300, _ORIGIN_Y],
            "crs": {"type": "name", "properties": {"name": _CRS}},
        }
        req = _minimal_request(
            alignment_override={"target": "domain", "resolution": 10.0}
        )
        lattice = _build_landscape_grid(req, domain, None)
        assert lattice["dx"] == 10.0
        assert lattice["nx"] == 30
        assert lattice["ny"] == 30

    def test_grid_target_uses_referenced_lattice(self):
        ref_transform = [10.0, 0.0, 700000.0, 0.0, -10.0, 5000000.0]
        ref_doc = _grid_doc(shape=(33, 77), transform=ref_transform, crs="EPSG:32612")
        domain = {"bbox": [0.0, 0.0, 0.0, 0.0], "crs": {"properties": {"name": _CRS}}}
        req = _minimal_request(
            alignment_override={"target": "grid", "grid_id": "master"}
        )
        lattice = _build_landscape_grid(req, domain, ref_doc)
        assert lattice["nx"] == 77
        assert lattice["ny"] == 33
        assert lattice["dx"] == 10.0
        assert lattice["crs"] == "EPSG:32612"
        assert lattice["transform"][2] == pytest.approx(700000.0)
        assert lattice["transform"][5] == pytest.approx(5000000.0)

    def test_grid_target_missing_georeference_rejected(self):
        bad_doc = {"georeference": None}
        domain = {"bbox": [0, 0, 0, 0], "crs": {"properties": {"name": _CRS}}}
        req = _minimal_request(
            alignment_override={"target": "grid", "grid_id": "master"}
        )
        with pytest.raises(HTTPException) as exc:
            _build_landscape_grid(req, domain, bad_doc)
        assert exc.value.status_code == 422


class TestCheckRoleAlignment:
    @pytest.fixture
    def lattice(self):
        return dict(_LANDSCAPE_GRID)

    def test_exact_match_passes(self, lattice):
        _check_role_alignment(
            _topo_doc(),
            LandscapeFieldSource(grid_id="topo", band="elevation"),
            "elevation",
            lattice,
        )

    def test_oversized_role_passes(self, lattice):
        # Role grid extends one cell in every direction past the landscape.
        oversized_transform = [
            _RES,
            0.0,
            _ORIGIN_X - _RES,
            0.0,
            -_RES,
            _ORIGIN_Y + _RES,
        ]
        doc = _topo_doc(shape=(_NY + 2, _NX + 2), transform=oversized_transform)
        _check_role_alignment(
            doc,
            LandscapeFieldSource(grid_id="topo", band="elevation"),
            "elevation",
            lattice,
        )

    def test_crs_mismatch_rejected(self, lattice):
        doc = _topo_doc(crs="EPSG:4326")
        with pytest.raises(HTTPException) as exc:
            _check_role_alignment(
                doc,
                LandscapeFieldSource(grid_id="topo", band="elevation"),
                "elevation",
                lattice,
            )
        assert exc.value.status_code == 422
        assert "CRS" in exc.value.detail

    def test_equivalent_crs_spellings_pass(self, lattice):
        # Grid georeference is stored as an EPSG string; the landscape grid
        # inherits its CRS from the domain, which may be in OGC URN form.
        lattice["crs"] = "urn:ogc:def:crs:EPSG::32611"
        doc = _topo_doc(crs="EPSG:32611")
        _check_role_alignment(
            doc,
            LandscapeFieldSource(grid_id="topo", band="elevation"),
            "elevation",
            lattice,
        )

    def test_cell_size_mismatch_rejected(self, lattice):
        doc = _topo_doc(transform=[2.0, 0.0, _ORIGIN_X, 0.0, -2.0, _ORIGIN_Y])
        with pytest.raises(HTTPException) as exc:
            _check_role_alignment(
                doc,
                LandscapeFieldSource(grid_id="topo", band="elevation"),
                "elevation",
                lattice,
            )
        assert exc.value.status_code == 422
        detail = exc.value.detail
        # Message names both resolutions and offers a copy-pasteable fix.
        assert "resolution mismatch" in detail
        assert "2.0 m" in detail  # the grid's resolution
        assert "30.0 m" in detail  # the landscape resolution
        assert '"alignment": {"resolution": 2.0}' in detail

    def test_off_lattice_origin_rejected(self, lattice):
        # Shifted by 15 m — half a cell. Not on the 30 m lattice.
        doc = _topo_doc(transform=[_RES, 0.0, _ORIGIN_X + 15.0, 0.0, -_RES, _ORIGIN_Y])
        with pytest.raises(HTTPException) as exc:
            _check_role_alignment(
                doc,
                LandscapeFieldSource(grid_id="topo", band="elevation"),
                "elevation",
                lattice,
            )
        assert exc.value.status_code == 422
        assert "lattice" in exc.value.detail

    def test_undersized_coverage_rejected(self, lattice):
        # Lattice-aligned but only covers half the landscape.
        doc = _topo_doc(shape=(_NY // 2, _NX // 2))
        with pytest.raises(HTTPException) as exc:
            _check_role_alignment(
                doc,
                LandscapeFieldSource(grid_id="topo", band="elevation"),
                "elevation",
                lattice,
            )
        assert exc.value.status_code == 422
        assert "cover" in exc.value.detail


class TestCheckCellCountCap:
    def test_under_cap_passes(self):
        _check_cell_count_cap({"nx": 1000, "ny": 1000})

    def test_over_cap_rejected(self):
        with pytest.raises(HTTPException) as exc:
            _check_cell_count_cap({"nx": _MAX_CELLS + 1, "ny": 1})
        assert exc.value.status_code == 422
        assert "exceeds cap" in exc.value.detail


class TestMakeSource:
    def test_round_trip(self):
        source = _make_source(_minimal_request(), "test-domain", dict(_LANDSCAPE_GRID))
        assert source.name == "landscape"
        assert source.domain_id == "test-domain"
        assert source.fire_behavior_fuel_model == "fbfm40"
        assert source.alignment.target == "domain"
        assert source.canopy_height.band == "chm"
        assert source.canopy_base_height.band == "cbh"
        assert source.canopy_bulk_density.band == "cbd"

    def test_georeference_is_the_resolved_lattice(self):
        """The working lattice dict collapses to a plain Georeference —
        `nx`/`ny`/`dx`/`dy` are all derivable from shape and transform."""
        source = _make_source(_minimal_request(), "test-domain", dict(_LANDSCAPE_GRID))
        geo = source.georeference
        assert geo.crs == _CRS
        assert geo.shape == (_NY, _NX)
        assert tuple(geo.transform) == tuple(_LANDSCAPE_TRANSFORM)
        assert geo.transform[0] == _RES  # dx
        assert -geo.transform[4] == _RES  # dy


class TestRoleContractTable:
    """Static contract — make sure no role drifts silently."""

    def test_all_roles_covered(self):
        expected = {
            "elevation",
            "slope",
            "aspect",
            "fuel_model",
            "canopy_cover",
            "canopy_height",
            "canopy_base_height",
            "canopy_bulk_density",
        }
        assert set(_ROLE_CONTRACT) == expected

    def test_units_are_canonical(self):
        assert _ROLE_CONTRACT["elevation"] == "m"
        assert _ROLE_CONTRACT["slope"] == "deg"
        assert _ROLE_CONTRACT["aspect"] == "deg"
        assert _ROLE_CONTRACT["fuel_model"] is None
        assert _ROLE_CONTRACT["canopy_cover"] == "%"
        assert _ROLE_CONTRACT["canopy_height"] == "m"
        assert _ROLE_CONTRACT["canopy_base_height"] == "m"
        assert _ROLE_CONTRACT["canopy_bulk_density"] == "kg/m**3"
