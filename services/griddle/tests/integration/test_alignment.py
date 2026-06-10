"""
Integration tests for the grid alignment feature (issue #205).

These exercise the full griddle pipeline against real LANDFIRE COGs and
verify that the alignment specification controls the output lattice
end-to-end. The Blue Mountain domain (EPSG:32611, ~1 sq km) is used
throughout.

Each test compares the resulting zarr's georeference (CRS, transform,
shape) to what the alignment helpers should have produced. They guard
against regressions in:

- ``RasterConnection.extract_window`` destination-kwargs routing.
- ``lib.alignment.resolve_alignment_destination`` dispatch on the three
  alignment targets.
- The single-reprojection invariant — output lattices should match the
  destination computed up front, not whatever rasterio's default
  reprojection produces.
"""

import json

import geopandas as gpd
import pytest

from lib.alignment import lattice_from_bounds
from lib.testing import SHARED_TEST_DOMAINS_DIR

BLUE_MTN_CRS = "EPSG:32611"


def _blue_mtn_bounds() -> tuple[float, float, float, float]:
    """Return Blue Mountain domain bounds in EPSG:32611 as floats."""
    with open(SHARED_TEST_DOMAINS_DIR / "blue_mtn.json") as f:
        data = json.load(f)
    features = []
    for f in data["features"]:
        coords = f["geometry"]["coordinates"]
        if isinstance(coords, str):
            f["geometry"]["coordinates"] = json.loads(coords)
        features.append(f)
    gdf = gpd.GeoDataFrame.from_features(features, crs=BLUE_MTN_CRS)
    minx, miny, maxx, maxy = gdf.total_bounds
    return float(minx), float(miny), float(maxx), float(maxy)


def _origin(ds, var: str) -> tuple[float, float]:
    """Return (transform.c, transform.f) — upper-left x, upper-left y."""
    t = ds[var].rio.transform()
    return float(t.c), float(t.f)


def _cell_size(ds, var: str) -> float:
    return abs(float(ds[var].rio.transform().a))


def test_domain_target_default_resolution_lands_at_domain_origin(griddle_runner):
    """target='domain' with no explicit resolution falls back to LANDFIRE's
    native ~30m and anchors the output at the domain's lower-left corner."""
    result = griddle_runner(
        "blue_mtn.json",
        "landfire_fbfm40.json",
        source_overrides={"alignment": {"target": "domain"}},
    )
    ds = result.ds

    minx, miny, maxx, maxy = _blue_mtn_bounds()
    expected_transform, expected_shape = lattice_from_bounds(
        (minx, miny, maxx, maxy), _cell_size(ds, "fbfm")
    )

    origin_x, origin_y = _origin(ds, "fbfm")
    assert origin_x == pytest.approx(minx, abs=0.01)
    assert origin_y == pytest.approx(float(expected_transform.f), abs=0.01)
    assert ds["fbfm"].shape == expected_shape
    assert str(ds.rio.crs).endswith("32611")


def test_domain_target_explicit_resolution(griddle_runner):
    """target='domain' with explicit resolution=10 yields a 10m lattice
    anchored at the domain's lower-left."""
    result = griddle_runner(
        "blue_mtn.json",
        "landfire_fbfm40.json",
        source_overrides={"alignment": {"target": "domain", "resolution": 10.0}},
    )
    ds = result.ds

    minx, miny, maxx, maxy = _blue_mtn_bounds()
    expected_transform, expected_shape = lattice_from_bounds(
        (minx, miny, maxx, maxy), 10.0
    )

    assert _cell_size(ds, "fbfm") == pytest.approx(10.0)
    origin_x, origin_y = _origin(ds, "fbfm")
    assert origin_x == pytest.approx(minx, abs=0.01)
    assert origin_y == pytest.approx(float(expected_transform.f), abs=0.01)
    assert ds["fbfm"].shape == expected_shape


def test_native_target_does_not_anchor_at_domain_origin(griddle_runner):
    """target='native' preserves the source pixel anchor — the output's
    upper-left x should NOT be equal to the domain's minx (it's snapped to
    the LANDFIRE source pixel grid instead)."""
    result = griddle_runner(
        "blue_mtn.json",
        "landfire_fbfm40.json",
        source_overrides={"alignment": {"target": "native"}},
    )
    ds = result.ds

    minx, _, _, _ = _blue_mtn_bounds()
    origin_x, _ = _origin(ds, "fbfm")
    # Source pixel grid is offset from the domain origin by some fraction
    # of a pixel — they should differ by at least the floating-point
    # tolerance used in our composition validator.
    assert abs(origin_x - minx) > 1e-3


def test_cross_source_compose_at_2m(griddle_runner):
    """Two domain-anchored grids at the same resolution share an exact
    transform — the composition invariant required for QUIC-Fire (#74)
    and LCP (#187) exports."""
    fbfm = griddle_runner(
        "blue_mtn.json",
        "landfire_fbfm40.json",
        source_overrides={"alignment": {"target": "domain", "resolution": 2.0}},
    )
    topo = griddle_runner(
        "blue_mtn.json",
        "landfire_topography.json",
        source_overrides={"alignment": {"target": "domain", "resolution": 2.0}},
    )

    fbfm_transform = tuple(fbfm.ds["fbfm"].rio.transform())[:6]
    topo_transform = tuple(topo.ds["elevation"].rio.transform())[:6]
    assert fbfm_transform == topo_transform
    assert fbfm.ds["fbfm"].shape == topo.ds["elevation"].shape


def test_grid_target_exact_match(griddle_runner):
    """target='grid' with no resolution should produce a lattice
    byte-equal to the target grid's georeference."""
    target = griddle_runner(
        "blue_mtn.json",
        "landfire_fbfm40.json",
        source_overrides={"alignment": {"target": "domain", "resolution": 10.0}},
    )
    target_transform = tuple(target.ds["fbfm"].rio.transform())[:6]
    target_shape = target.ds["fbfm"].shape

    # Resample the FBFM40 grid with target='grid' pointing at itself
    # (any target grid works; we use the FBFM40 we just built).
    aligned = griddle_runner(
        "blue_mtn.json",
        "resample_bilinear.json",
        source_overrides={
            "source_grid_id": target.grid_id,
            "alignment": {"target": "grid", "grid_id": target.grid_id},
            "method_overrides": {"fbfm": "nearest"},
        },
    )

    aligned_transform = tuple(aligned.ds["fbfm"].rio.transform())[:6]
    assert aligned_transform == target_transform
    assert aligned.ds["fbfm"].shape == target_shape


def test_grid_target_new_resolution_preserves_origin(griddle_runner):
    """target='grid' with explicit resolution should preserve the target's
    CRS and origin (transform[2], transform[5]) but recompute shape at the
    new cell size."""
    target = griddle_runner(
        "blue_mtn.json",
        "landfire_fbfm40.json",
        source_overrides={"alignment": {"target": "domain", "resolution": 30.0}},
    )
    target_origin = (
        float(target.ds["fbfm"].rio.transform().c),
        float(target.ds["fbfm"].rio.transform().f),
    )

    aligned = griddle_runner(
        "blue_mtn.json",
        "resample_bilinear.json",
        source_overrides={
            "source_grid_id": target.grid_id,
            "alignment": {
                "target": "grid",
                "grid_id": target.grid_id,
                "resolution": 10.0,
            },
            "method_overrides": {"fbfm": "nearest"},
        },
    )

    aligned_transform = aligned.ds["fbfm"].rio.transform()
    assert (float(aligned_transform.c), float(aligned_transform.f)) == pytest.approx(
        target_origin, abs=0.01
    )
    assert abs(float(aligned_transform.a)) == pytest.approx(10.0)
    # 30m target cell at 10m output = 3× along each axis.
    assert aligned.ds["fbfm"].shape[0] == target.ds["fbfm"].shape[0] * 3
    assert aligned.ds["fbfm"].shape[1] == target.ds["fbfm"].shape[1] * 3


def test_resample_domain_target(griddle_runner):
    """Resample with target='domain' should land on the domain-origin lattice
    independent of the source grid's anchor."""
    source = griddle_runner(
        "blue_mtn.json",
        "landfire_fbfm40.json",
        source_overrides={"alignment": {"target": "native"}},
    )

    resampled = griddle_runner(
        "blue_mtn.json",
        "resample_bilinear.json",
        source_overrides={
            "source_grid_id": source.grid_id,
            "alignment": {"target": "domain", "resolution": 30.0},
            "method_overrides": {"fbfm": "nearest"},
        },
    )

    minx, miny, maxx, maxy = _blue_mtn_bounds()
    expected_transform, expected_shape = lattice_from_bounds(
        (minx, miny, maxx, maxy), 30.0
    )
    transform = resampled.ds["fbfm"].rio.transform()
    assert float(transform.c) == pytest.approx(minx, abs=0.01)
    assert float(transform.f) == pytest.approx(float(expected_transform.f), abs=0.01)
    assert resampled.ds["fbfm"].shape == expected_shape
