"""
Tests for lib.alignment.

Pure-math tests for grid-alignment helpers used by both the API (validation)
and Griddle (handler runtime).
"""

import geopandas as gpd
import pytest
from affine import Affine
from shapely.geometry import box

from lib.alignment import (
    lattice_from_bounds,
    resolve_alignment_destination,
    target_grid_bounds,
)


def _make_domain(
    minx=500_000.0,
    miny=4_000_000.0,
    maxx=500_500.0,
    maxy=4_000_400.0,
    crs="EPSG:32610",
):
    return gpd.GeoDataFrame({"geometry": [box(minx, miny, maxx, maxy)]}, crs=crs)


class TestLatticeFromBounds:
    def test_exact_division(self):
        transform, (h, w) = lattice_from_bounds((0, 0, 100, 50), 10.0)
        assert (h, w) == (5, 10)
        assert transform.a == pytest.approx(10.0)
        assert transform.e == pytest.approx(-10.0)
        assert transform.c == pytest.approx(0.0)
        # f is upper-left y; for ceil-bounds at miny=0 + h*res = 50, that's f.
        assert transform.f == pytest.approx(50.0)

    def test_ceil_rounding_grows_extent(self):
        # 100m wide, 7m cells -> ceil(100/7) = 15 cells -> covers 105m.
        transform, (h, w) = lattice_from_bounds((0, 0, 100, 100), 7.0)
        assert (h, w) == (15, 15)
        assert transform.a == pytest.approx(7.0)
        # f = miny + h * resolution = 0 + 15 * 7 = 105
        assert transform.f == pytest.approx(105.0)

    def test_anchor_at_lower_left(self):
        bounds = (500_000.0, 4_000_000.0, 500_500.0, 4_000_400.0)
        transform, _ = lattice_from_bounds(bounds, 2.0)
        # transform.c == minx, transform.f == miny + h * res
        assert transform.c == pytest.approx(500_000.0)

    def test_minimum_one_cell(self):
        # Sub-resolution bounds get clamped to 1 cell minimum.
        transform, (h, w) = lattice_from_bounds((0, 0, 0.1, 0.1), 1.0)
        assert (h, w) == (1, 1)


class TestTargetGridBounds:
    def test_round_trip_with_lattice_from_bounds(self):
        bounds = (1000.0, 2000.0, 1100.0, 2050.0)
        transform, shape = lattice_from_bounds(bounds, 5.0)
        georef = {
            "transform": tuple(transform)[:6],
            "shape": shape,
        }
        # Recovered bounds should equal what we put in (extents may grow
        # via ceil, but for clean divisions they match).
        assert target_grid_bounds(georef) == pytest.approx(
            (1000.0, 2000.0, 1100.0, 2050.0)
        )

    def test_with_north_up_transform(self):
        # 10m cells, 3 cols, 2 rows, anchored at (0, 0) lower-left -> upper-left (0, 20).
        georef = {
            "transform": (10.0, 0.0, 0.0, 0.0, -10.0, 20.0),
            "shape": (2, 3),
        }
        assert target_grid_bounds(georef) == pytest.approx((0.0, 0.0, 30.0, 20.0))

    def test_3d_shape_uses_xy_only(self):
        # Static voxel grids persist a (z, y, x) shape; only the trailing
        # two dims describe the raster footprint.
        georef = {
            "transform": (10.0, 0.0, 0.0, 0.0, -10.0, 20.0),
            "shape": (50, 2, 3),
        }
        assert target_grid_bounds(georef) == pytest.approx((0.0, 0.0, 30.0, 20.0))


class TestResolveAlignmentDomain:
    def test_default_resolution_uses_source_native(self):
        domain = _make_domain()
        dest = resolve_alignment_destination(
            {"target": "domain"},
            domain,
            None,
            source_native_resolution=30.0,
        )
        assert dest["destination_crs"] == domain.crs
        # 500m / 30 ceil = 17 cells x 14 cells (400/30 ceil = 14)
        assert dest["destination_shape"] == (14, 17)

    def test_explicit_resolution(self):
        domain = _make_domain()
        dest = resolve_alignment_destination(
            {"target": "domain", "resolution": 2.0},
            domain,
            None,
            30.0,
        )
        # 500/2 = 250 cells x 400/2 = 200 cells
        assert dest["destination_shape"] == (200, 250)
        assert dest["destination_transform"].a == pytest.approx(2.0)


class TestResolveAlignmentNative:
    def test_no_resolution_returns_empty_kwargs(self):
        domain = _make_domain()
        dest = resolve_alignment_destination(
            {"target": "native"},
            domain,
            None,
            30.0,
        )
        assert dest == {}

    def test_with_resolution_returns_crs_only(self):
        domain = _make_domain()
        dest = resolve_alignment_destination(
            {"target": "native", "resolution": 5.0},
            domain,
            None,
            30.0,
        )
        assert "destination_crs" in dest
        assert "destination_transform" not in dest
        assert "destination_shape" not in dest


class TestResolveAlignmentGrid:
    def _target_doc(self):
        return {
            "georeference": {
                "crs": "EPSG:32610",
                "transform": (10.0, 0.0, 0.0, 0.0, -10.0, 20.0),
                "shape": (2, 3),
            }
        }

    def test_exact_match_when_resolution_none(self):
        domain = _make_domain()
        dest = resolve_alignment_destination(
            {"target": "grid", "grid_id": "x"},
            domain,
            self._target_doc(),
            30.0,
        )
        assert dest["destination_crs"] == "EPSG:32610"
        assert dest["destination_transform"] == Affine(10.0, 0.0, 0.0, 0.0, -10.0, 20.0)
        assert dest["destination_shape"] == (2, 3)

    def test_same_anchor_new_resolution(self):
        domain = _make_domain()
        dest = resolve_alignment_destination(
            {"target": "grid", "grid_id": "x", "resolution": 1.0},
            domain,
            self._target_doc(),
            30.0,
        )
        assert dest["destination_crs"] == "EPSG:32610"
        # Same target bounds (0, 0, 30, 20) at 1m -> 30 x 20
        assert dest["destination_shape"] == (20, 30)
        # Origin preserved
        assert dest["destination_transform"].c == pytest.approx(0.0)
        assert dest["destination_transform"].f == pytest.approx(20.0)

    def test_missing_target_grid_doc_raises(self):
        with pytest.raises(ValueError, match="target='grid'"):
            resolve_alignment_destination(
                {"target": "grid", "grid_id": "x"},
                _make_domain(),
                None,
                30.0,
            )

    def test_3d_target_grid_no_resolution_uses_xy_shape(self):
        # A persisted static voxel grid stores ``shape`` as (z, y, x). The
        # alignment math is purely raster, so it must read the trailing two
        # dims for the destination footprint.
        target_doc = {
            "georeference": {
                "crs": "EPSG:32610",
                "transform": (10.0, 0.0, 0.0, 0.0, -10.0, 20.0),
                "shape": (50, 2, 3),
            }
        }
        dest = resolve_alignment_destination(
            {"target": "grid", "grid_id": "x"},
            _make_domain(),
            target_doc,
            30.0,
        )
        # destination_shape is the 2D raster footprint, never (z, y, x).
        assert dest["destination_shape"] == (2, 3)
        assert dest["destination_transform"] == Affine(10.0, 0.0, 0.0, 0.0, -10.0, 20.0)

    def test_3d_target_grid_with_resolution_uses_xy_shape(self):
        target_doc = {
            "georeference": {
                "crs": "EPSG:32610",
                # 30m cells, 10x10 footprint, 50 z-layers.
                "transform": (30.0, 0.0, 0.0, 0.0, -30.0, 300.0),
                "shape": (50, 10, 10),
            }
        }
        dest = resolve_alignment_destination(
            {"target": "grid", "grid_id": "x", "resolution": 1.0},
            _make_domain(),
            target_doc,
            30.0,
        )
        # 300m bounds at 1m -> 300x300 footprint.
        assert dest["destination_shape"] == (300, 300)


class TestResolveAlignmentInvalid:
    def test_unknown_target(self):
        with pytest.raises(ValueError, match="unknown alignment target"):
            resolve_alignment_destination(
                {"target": "elsewhere"},
                _make_domain(),
                None,
                30.0,
            )


class TestResolveAlignmentExtentBuffer:
    """``extent_buffer_cells`` must extend the destination lattice by N
    output cells on every side for ``target='domain'`` and ``target='grid'``.
    For ``target='native'`` the buffer is applied at the clip step inside
    ``extract_window`` and the helper passes through unchanged.
    """

    def test_domain_target_expands_lattice_by_n_cells(self):
        domain = _make_domain()
        dest = resolve_alignment_destination(
            {"target": "domain", "resolution": 2.0},
            domain,
            None,
            30.0,
            extent_buffer_cells=4,
        )
        # 500 / 2 = 250 cells + 8 buffer = 258; 400 / 2 = 200 + 8 = 208.
        assert dest["destination_shape"] == (208, 258)
        # Origin shifts by 4 cells * 2 m = 8 m left/down.
        assert dest["destination_transform"].c == pytest.approx(500_000.0 - 8.0)

    def test_domain_target_expands_at_native_resolution_when_unspecified(self):
        domain = _make_domain()
        dest = resolve_alignment_destination(
            {"target": "domain"},
            domain,
            None,
            source_native_resolution=30.0,
            extent_buffer_cells=2,
        )
        # 500/30 ceil = 17 + 4 buffer = 21 along x; 400/30 ceil = 14 + 4 = 18 along y.
        assert dest["destination_shape"] == (18, 21)
        # Origin shifts by 2 cells * 30 m = 60 m.
        assert dest["destination_transform"].c == pytest.approx(500_000.0 - 60.0)

    def test_grid_target_no_resolution_expands_in_target_cells(self):
        target_doc = {
            "georeference": {
                "crs": "EPSG:32610",
                # 30m cells, anchored lower-left at (0, 0), shape (10, 10) -> 300x300.
                "transform": (30.0, 0.0, 0.0, 0.0, -30.0, 300.0),
                "shape": (10, 10),
            }
        }
        dest = resolve_alignment_destination(
            {"target": "grid", "grid_id": "x"},
            _make_domain(),
            target_doc,
            30.0,
            extent_buffer_cells=3,
        )
        # 10 + 6 = 16 cells per axis. Origin shifts 3 cells * 30 m = 90 m left/down.
        assert dest["destination_shape"] == (16, 16)
        assert dest["destination_transform"].c == pytest.approx(-90.0)
        # Cells stay 30 m and origin shift is exactly an integer multiple, so
        # buffered cells nest with the unbuffered target lattice.
        assert dest["destination_transform"].a == pytest.approx(30.0)

    def test_grid_target_with_resolution_expands_in_new_cells(self):
        target_doc = {
            "georeference": {
                "crs": "EPSG:32610",
                "transform": (30.0, 0.0, 0.0, 0.0, -30.0, 300.0),
                "shape": (10, 10),
            }
        }
        dest = resolve_alignment_destination(
            {"target": "grid", "grid_id": "x", "resolution": 1.0},
            _make_domain(),
            target_doc,
            30.0,
            extent_buffer_cells=5,
        )
        # 300 m at 1 m = 300 cells; +10 buffer = 310 per axis.
        assert dest["destination_shape"] == (310, 310)
        # Origin shifts 5 cells * 1 m = 5 m left/down.
        assert dest["destination_transform"].c == pytest.approx(-5.0)
        assert dest["destination_transform"].a == pytest.approx(1.0)

    def test_native_target_passes_through_buffer_unchanged(self):
        # Native target relies on extract_window's clip path for buffering;
        # the alignment destination is unchanged by extent_buffer_cells.
        domain = _make_domain()
        dest_no_buffer = resolve_alignment_destination(
            {"target": "native", "resolution": 5.0}, domain, None, 30.0
        )
        dest_with_buffer = resolve_alignment_destination(
            {"target": "native", "resolution": 5.0},
            domain,
            None,
            30.0,
            extent_buffer_cells=4,
        )
        assert dest_no_buffer == dest_with_buffer

    def test_zero_buffer_matches_no_buffer(self):
        domain = _make_domain()
        baseline = resolve_alignment_destination(
            {"target": "domain", "resolution": 2.0}, domain, None, 30.0
        )
        with_zero = resolve_alignment_destination(
            {"target": "domain", "resolution": 2.0},
            domain,
            None,
            30.0,
            extent_buffer_cells=0,
        )
        assert baseline["destination_shape"] == with_zero["destination_shape"]
        assert baseline["destination_transform"] == with_zero["destination_transform"]
