"""
Unit tests for the QUIC-Fire validator helpers.

Every helper in `validators.py` takes plain dicts (or a Pydantic request)
and raises `HTTPException(422)` on failure — no Firestore, no async — so
each can be exercised with hand-built fixtures here. The Firestore loading
layer (`_load_all_grids`) and the live-router behavior of the orchestrator
live in `test_router.py`.
"""

import pytest
from api.resources.grids.exports.quicfire.schema import (
    FieldSource,
    QuicfireExportRequest,
)
from api.resources.grids.exports.quicfire.validators import (
    _MAX_CELLS,
    _ROLE_CONTRACT,
    _build_fire_grid,
    _check_3d_role_vertical,
    _check_cell_count_cap,
    _check_role_alignment,
    _check_role_contract,
    _domain_crs_string,
    _iter_roles,
    _make_source,
)
from fastapi import HTTPException

# Canonical fire-grid lattice used by every helper test below.
_CRS = "EPSG:32611"
_DX = 2.0
_DZ = 1.0
_NX = 100
_NY = 50
_NZ = 10
_ORIGIN_X = 720226.0
_ORIGIN_Y = 5190646.0
_FIRE_TRANSFORM = [_DX, 0.0, _ORIGIN_X, 0.0, -_DX, _ORIGIN_Y]
_FIRE_GRID = {
    "nx": _NX,
    "ny": _NY,
    "nz": _NZ,
    "dx": _DX,
    "dy": _DX,
    "dz": _DZ,
    "transform": _FIRE_TRANSFORM,
    "z_origin": 0.0,
    "crs": _CRS,
}


def _grid_doc(
    *,
    shape: tuple[int, ...],
    transform: list[float] = _FIRE_TRANSFORM,
    crs: str | None = _CRS,
    z_resolution: float | None = None,
    z_origin: float | None = None,
    bands: list[dict] | None = None,
) -> dict:
    """Hand-build a Firestore-shaped grid doc for the helpers to consume."""
    geo: dict = {"shape": list(shape), "transform": list(transform), "crs": crs}
    if z_resolution is not None:
        geo["z_resolution"] = z_resolution
    if z_origin is not None:
        geo["z_origin"] = z_origin
    return {"georeference": geo, "bands": bands or []}


def _canopy_doc(**overrides) -> dict:
    defaults = {
        "shape": (_NZ, _NY, _NX),
        "transform": _FIRE_TRANSFORM,
        "z_resolution": _DZ,
        "z_origin": 0.0,
        "bands": [
            {
                "key": "bulk_density.foliage.live",
                "type": "continuous",
                "unit": "kg/m**3",
                "index": 0,
            },
            {
                "key": "fuel_moisture.live",
                "type": "continuous",
                "unit": "%",
                "index": 1,
            },
            {"key": "savr.foliage", "type": "continuous", "unit": "1/m", "index": 2},
        ],
    }
    return _grid_doc(**(defaults | overrides))


def _surface_doc(**overrides) -> dict:
    defaults = {
        "shape": (_NY, _NX),
        "transform": _FIRE_TRANSFORM,
        "bands": [
            {
                "key": "fuel_load.1hr",
                "type": "continuous",
                "unit": "kg/m**2",
                "index": 0,
            },
            {"key": "fuel_depth", "type": "continuous", "unit": "m", "index": 1},
            {"key": "savr.1hr", "type": "continuous", "unit": "1/m", "index": 2},
        ],
    }
    return _grid_doc(**(defaults | overrides))


def _minimal_request(alignment_override: dict | None = None) -> QuicfireExportRequest:
    kwargs = {
        "canopy_bulk_density": FieldSource(
            grid_id="canopy", band="bulk_density.foliage.live"
        ),
        "canopy_moisture": FieldSource(grid_id="canopy", band="fuel_moisture.live"),
        "surface_fuel_load": FieldSource(grid_id="surface", band="fuel_load.1hr"),
        "surface_fuel_depth": FieldSource(grid_id="surface", band="fuel_depth"),
        "surface_moisture": FieldSource(grid_id="surface", band="fuel_moisture.1hr"),
    }
    if alignment_override is not None:
        kwargs["alignment"] = alignment_override
    return QuicfireExportRequest(**kwargs)


class TestIterRoles:
    def test_required_only(self):
        roles = _iter_roles(_minimal_request())
        names = [n for n, _ in roles]
        assert names == [
            "canopy_bulk_density",
            "canopy_moisture",
            "surface_fuel_load",
            "surface_fuel_depth",
            "surface_moisture",
        ]

    def test_with_topography(self):
        req = QuicfireExportRequest(
            **{
                "canopy_bulk_density": FieldSource(
                    grid_id="c", band="bulk_density.foliage.live"
                ),
                "canopy_moisture": FieldSource(grid_id="c", band="fuel_moisture.live"),
                "surface_fuel_load": FieldSource(grid_id="s", band="fuel_load.1hr"),
                "surface_fuel_depth": FieldSource(grid_id="s", band="fuel_depth"),
                "surface_moisture": FieldSource(grid_id="m", band="fuel_moisture.1hr"),
                "topography": FieldSource(grid_id="t", band="elevation"),
            }
        )
        names = [n for n, _ in _iter_roles(req)]
        assert "topography" in names
        assert len(names) == 6


class TestDomainCrsString:
    def test_from_geojson_dict(self):
        assert (
            _domain_crs_string({"crs": {"type": "name", "properties": {"name": _CRS}}})
            == _CRS
        )

    def test_from_string(self):
        assert _domain_crs_string({"crs": _CRS}) == _CRS

    def test_missing(self):
        assert _domain_crs_string({}) is None


class TestCheckRoleContract:
    def test_valid(self):
        # Should not raise.
        _check_role_contract(
            _canopy_doc(),
            FieldSource(grid_id="canopy", band="bulk_density.foliage.live"),
            "canopy_bulk_density",
        )

    def test_missing_band(self):
        doc = _canopy_doc(
            bands=[
                {"key": "other", "type": "continuous", "unit": "kg/m**3", "index": 0}
            ]
        )
        with pytest.raises(HTTPException) as exc:
            _check_role_contract(
                doc,
                FieldSource(grid_id="canopy", band="bulk_density.foliage.live"),
                "canopy_bulk_density",
            )
        assert exc.value.status_code == 422

    def test_wrong_unit(self):
        doc = _canopy_doc(
            bands=[
                {
                    "key": "bulk_density.foliage.live",
                    "type": "continuous",
                    "unit": "kg/m**2",
                    "index": 0,
                }
            ]
        )
        with pytest.raises(HTTPException) as exc:
            _check_role_contract(
                doc,
                FieldSource(grid_id="canopy", band="bulk_density.foliage.live"),
                "canopy_bulk_density",
            )
        assert exc.value.status_code == 422

    def test_wrong_dimensionality(self):
        # 2D doc with the canopy band — canopy_bulk_density expects 3D.
        doc = _grid_doc(
            shape=(_NY, _NX),
            bands=[
                {
                    "key": "bulk_density.foliage.live",
                    "type": "continuous",
                    "unit": "kg/m**3",
                    "index": 0,
                }
            ],
        )
        with pytest.raises(HTTPException) as exc:
            _check_role_contract(
                doc,
                FieldSource(grid_id="canopy", band="bulk_density.foliage.live"),
                "canopy_bulk_density",
            )
        assert exc.value.status_code == 422


class TestBuildFireGridDomainTarget:
    def test_padded_domain_clean_lattice(self):
        domain = {
            "bbox": [
                _ORIGIN_X,
                _ORIGIN_Y - _NY * _DX,
                _ORIGIN_X + _NX * _DX,
                _ORIGIN_Y,
            ],
            "crs": {"type": "name", "properties": {"name": _CRS}},
        }
        req = _minimal_request()
        fg = _build_fire_grid(req, domain, _canopy_doc()["georeference"], None)
        assert fg["nx"] == _NX
        assert fg["ny"] == _NY
        assert fg["nz"] == _NZ
        assert fg["dx"] == _DX
        assert fg["dy"] == _DX
        assert fg["dz"] == _DZ
        assert fg["transform"][2] == pytest.approx(_ORIGIN_X)
        assert fg["transform"][5] == pytest.approx(_ORIGIN_Y)
        assert fg["crs"] == _CRS

    def test_unpadded_bbox_pads_via_ceil(self):
        # Bbox extends 1.94 m past a clean 2 m corner — width should ceil up.
        domain = {
            "bbox": [
                _ORIGIN_X,
                _ORIGIN_Y - _NY * _DX,
                _ORIGIN_X + _NX * _DX + 1.94,
                _ORIGIN_Y,
            ],
            "crs": {"type": "name", "properties": {"name": _CRS}},
        }
        fg = _build_fire_grid(
            _minimal_request(), domain, _canopy_doc()["georeference"], None
        )
        assert fg["nx"] == _NX + 1  # ceil of 1.94 m extra at dx=2 m
        assert fg["ny"] == _NY

    def test_custom_dx(self):
        domain = {
            "bbox": [_ORIGIN_X, _ORIGIN_Y - 80, _ORIGIN_X + 80, _ORIGIN_Y],
            "crs": {"type": "name", "properties": {"name": _CRS}},
        }
        req = _minimal_request(
            alignment_override={"target": "domain", "dx": 4, "dy": 4, "dz": 0.5}
        )
        fg = _build_fire_grid(req, domain, _canopy_doc()["georeference"], None)
        assert fg["dx"] == 4.0
        assert fg["dy"] == 4.0
        assert fg["nx"] == 20  # 80 / 4
        assert fg["ny"] == 20


class TestBuildFireGridGridTarget:
    def test_uses_referenced_grid_lattice(self):
        # Alignment grid sits offset from anything the Domain bbox would
        # produce; the fire grid should match it verbatim.
        ref_origin_x = 700000.0
        ref_origin_y = 5000000.0
        ref_transform = [_DX, 0.0, ref_origin_x, 0.0, -_DX, ref_origin_y]
        ref_doc = _grid_doc(shape=(33, 77), transform=ref_transform, crs="EPSG:32612")
        domain = {"bbox": [0.0, 0.0, 0.0, 0.0], "crs": {"properties": {"name": _CRS}}}
        req = _minimal_request(
            alignment_override={"target": "grid", "grid_id": "master"}
        )

        fg = _build_fire_grid(req, domain, _canopy_doc()["georeference"], ref_doc)
        assert fg["nx"] == 77
        assert fg["ny"] == 33
        assert fg["dx"] == _DX
        assert fg["crs"] == "EPSG:32612"
        assert fg["transform"][2] == pytest.approx(ref_origin_x)
        assert fg["transform"][5] == pytest.approx(ref_origin_y)
        # Vertical still from canopy.
        assert fg["nz"] == _NZ
        assert fg["dz"] == _DZ

    def test_missing_georeference_rejected(self):
        bad_doc = {"georeference": None}
        domain = {"bbox": [0, 0, 0, 0], "crs": {"properties": {"name": _CRS}}}
        req = _minimal_request(
            alignment_override={"target": "grid", "grid_id": "master"}
        )
        with pytest.raises(HTTPException) as exc:
            _build_fire_grid(req, domain, _canopy_doc()["georeference"], bad_doc)
        assert exc.value.status_code == 422


class TestCheckRoleAlignment:
    @pytest.fixture
    def fire_grid(self):
        return dict(_FIRE_GRID)

    def test_exact_match_passes(self, fire_grid):
        _check_role_alignment(
            _surface_doc(),
            FieldSource(grid_id="surface", band="fuel_load.1hr"),
            "surface_fuel_load",
            fire_grid,
        )

    def test_oversized_role_passes(self, fire_grid):
        # Role grid extends one cell west, north, east, south of fire grid.
        oversized_transform = [_DX, 0.0, _ORIGIN_X - _DX, 0.0, -_DX, _ORIGIN_Y + _DX]
        doc = _surface_doc(shape=(_NY + 2, _NX + 2), transform=oversized_transform)
        _check_role_alignment(
            doc,
            FieldSource(grid_id="surface", band="fuel_load.1hr"),
            "surface_fuel_load",
            fire_grid,
        )

    def test_crs_mismatch_rejected(self, fire_grid):
        doc = _surface_doc(crs="EPSG:4326")
        with pytest.raises(HTTPException) as exc:
            _check_role_alignment(
                doc,
                FieldSource(grid_id="surface", band="fuel_load.1hr"),
                "surface_fuel_load",
                fire_grid,
            )
        assert exc.value.status_code == 422
        assert "CRS" in exc.value.detail

    def test_cell_size_mismatch_rejected(self, fire_grid):
        doc = _surface_doc(transform=[30.0, 0.0, _ORIGIN_X, 0.0, -30.0, _ORIGIN_Y])
        with pytest.raises(HTTPException) as exc:
            _check_role_alignment(
                doc,
                FieldSource(grid_id="surface", band="fuel_load.1hr"),
                "surface_fuel_load",
                fire_grid,
            )
        assert exc.value.status_code == 422
        detail = exc.value.detail
        # Message names both resolutions and offers a copy-pasteable fix.
        assert "resolution mismatch" in detail
        assert "30.0 m" in detail  # the grid's resolution
        assert "2.0 m" in detail  # the fire-grid resolution
        assert '"alignment": {"dx": 30.0, "dy": 30.0}' in detail

    def test_off_lattice_origin_rejected(self, fire_grid):
        # Shifted by 0.5 m — half a cell. Not on the 2 m lattice.
        doc = _surface_doc(transform=[_DX, 0.0, _ORIGIN_X + 0.5, 0.0, -_DX, _ORIGIN_Y])
        with pytest.raises(HTTPException) as exc:
            _check_role_alignment(
                doc,
                FieldSource(grid_id="surface", band="fuel_load.1hr"),
                "surface_fuel_load",
                fire_grid,
            )
        assert exc.value.status_code == 422
        assert "lattice" in exc.value.detail

    def test_undersized_coverage_rejected(self, fire_grid):
        # Lattice-aligned but only covers half the fire grid.
        doc = _surface_doc(shape=(_NY // 2, _NX // 2))
        with pytest.raises(HTTPException) as exc:
            _check_role_alignment(
                doc,
                FieldSource(grid_id="surface", band="fuel_load.1hr"),
                "surface_fuel_load",
                fire_grid,
            )
        assert exc.value.status_code == 422
        assert "cover" in exc.value.detail


class TestCheck3dRoleVertical:
    @pytest.fixture
    def fire_grid(self):
        return dict(_FIRE_GRID)

    def test_2d_role_is_noop(self, fire_grid):
        _check_3d_role_vertical(
            _surface_doc(),
            FieldSource(grid_id="surface", band="fuel_load.1hr"),
            "surface_fuel_load",
            fire_grid,
        )

    def test_matching_z_passes(self, fire_grid):
        _check_3d_role_vertical(
            _canopy_doc(),
            FieldSource(grid_id="canopy", band="bulk_density.foliage.live"),
            "canopy_bulk_density",
            fire_grid,
        )

    def test_nz_mismatch_rejected(self, fire_grid):
        doc = _canopy_doc(shape=(_NZ + 5, _NY, _NX))
        with pytest.raises(HTTPException) as exc:
            _check_3d_role_vertical(
                doc,
                FieldSource(grid_id="canopy", band="bulk_density.foliage.live"),
                "canopy_bulk_density",
                fire_grid,
            )
        assert exc.value.status_code == 422

    def test_dz_mismatch_rejected(self, fire_grid):
        doc = _canopy_doc(z_resolution=0.5)
        with pytest.raises(HTTPException) as exc:
            _check_3d_role_vertical(
                doc,
                FieldSource(grid_id="canopy", band="bulk_density.foliage.live"),
                "canopy_bulk_density",
                fire_grid,
            )
        assert exc.value.status_code == 422

    def test_z_origin_mismatch_rejected(self, fire_grid):
        doc = _canopy_doc(z_origin=2.0)
        with pytest.raises(HTTPException) as exc:
            _check_3d_role_vertical(
                doc,
                FieldSource(grid_id="canopy", band="bulk_density.foliage.live"),
                "canopy_bulk_density",
                fire_grid,
            )
        assert exc.value.status_code == 422


class TestCheckCellCountCap:
    def test_under_cap_passes(self):
        _check_cell_count_cap({"nx": 100, "ny": 100, "nz": 10})

    def test_over_cap_rejected(self):
        with pytest.raises(HTTPException) as exc:
            _check_cell_count_cap({"nx": _MAX_CELLS + 1, "ny": 1, "nz": 1})
        assert exc.value.status_code == 422
        assert "exceeds cap" in exc.value.detail


class TestMakeSource:
    def test_round_trip(self):
        req = _minimal_request()
        source = _make_source(req, "test-domain", dict(_FIRE_GRID))
        assert source.name == "quicfire"
        assert source.domain_id == "test-domain"
        assert source.alignment.target == "domain"
        assert source.resolved["fire_grid"]["nx"] == _NX
        assert source.canopy_bulk_density.grid_id == "canopy"
        assert source.topography is None

    def test_with_optional_roles(self):
        req = QuicfireExportRequest(
            canopy_bulk_density=FieldSource(
                grid_id="c", band="bulk_density.foliage.live"
            ),
            canopy_moisture=FieldSource(grid_id="c", band="fuel_moisture.live"),
            canopy_savr=FieldSource(grid_id="c", band="savr.foliage"),
            surface_fuel_load=FieldSource(grid_id="s", band="fuel_load.1hr"),
            surface_fuel_depth=FieldSource(grid_id="s", band="fuel_depth"),
            surface_moisture=FieldSource(grid_id="m", band="fuel_moisture.1hr"),
            surface_savr=FieldSource(grid_id="s", band="savr.1hr"),
            topography=FieldSource(grid_id="t", band="elevation"),
        )
        source = _make_source(req, "test-domain", dict(_FIRE_GRID))
        assert source.topography.grid_id == "t"
        assert source.canopy_savr.band == "savr.foliage"
        assert source.surface_savr.band == "savr.1hr"


class TestRoleContractTable:
    """Static contract — make sure no role drifts silently."""

    def test_all_roles_covered(self):
        expected = {
            "canopy_bulk_density",
            "canopy_moisture",
            "canopy_savr",
            "surface_fuel_load",
            "surface_fuel_depth",
            "surface_moisture",
            "surface_savr",
            "topography",
        }
        assert set(_ROLE_CONTRACT) == expected

    def test_3d_roles_marked_as_3d(self):
        assert _ROLE_CONTRACT["canopy_bulk_density"][0] == 3
        assert _ROLE_CONTRACT["canopy_moisture"][0] == 3
        assert _ROLE_CONTRACT["canopy_savr"][0] == 3

    def test_2d_roles_marked_as_2d(self):
        for role in (
            "surface_fuel_load",
            "surface_fuel_depth",
            "surface_moisture",
            "surface_savr",
            "topography",
        ):
            assert _ROLE_CONTRACT[role][0] == 2
