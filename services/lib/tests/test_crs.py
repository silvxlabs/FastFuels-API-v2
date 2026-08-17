"""Unit tests for lib.crs."""

import numpy as np
import pytest
from pyproj import Transformer

from lib.crs import DEFAULT_TOLERANCE, crs_equal, reproject
from lib.laz import CANONICAL_SCALE


def test_identical_strings():
    assert crs_equal("EPSG:5070", "EPSG:5070")


def test_epsg_vs_ogc_urn():
    # The exact spellings from the QUIC-Fire export bug report.
    assert crs_equal("EPSG:5070", "urn:ogc:def:crs:EPSG::5070")
    assert crs_equal("urn:ogc:def:crs:EPSG::5070", "EPSG:5070")


def test_different_crs():
    assert not crs_equal("EPSG:5070", "EPSG:4326")


def test_both_none():
    assert crs_equal(None, None)


def test_one_none():
    assert not crs_equal("EPSG:5070", None)
    assert not crs_equal(None, "EPSG:5070")


# 3DEP publishes in EPSG:3857 and domains are commonly UTM, so this is the pair
# the point cloud chain actually runs.
WEB_MERCATOR_TO_UTM = Transformer.from_crs("EPSG:3857", "EPSG:32612", always_xy=True)
# Montana, where the Blackfoot fixtures are.
CENTRE = (-1.25e7, 5.9e6)


def box_of_points(span, count=20_000, seed=0):
    """Points scattered over a `span`-metre box, in the source CRS."""
    rng = np.random.default_rng(seed)
    return (
        rng.uniform(CENTRE[0] - span / 2, CENTRE[0] + span / 2, count),
        rng.uniform(CENTRE[1] - span / 2, CENTRE[1] + span / 2, count),
    )


@pytest.mark.parametrize("span", [1.0, 100.0, 500.0, 3_200.0, 16_000.0])
def test_matches_exact_within_tolerance(span):
    """Across the range of node extents, the fit holds to what it promises."""
    x, y = box_of_points(span)
    ax, ay = reproject(WEB_MERCATOR_TO_UTM, x, y)
    ex, ey = WEB_MERCATOR_TO_UTM.transform(x, y)
    assert np.hypot(ax - ex, ay - ey).max() <= DEFAULT_TOLERANCE


def test_stored_coordinates_are_unchanged():
    """The test that matters: what lands in the file is scaled int32, not metres.

    A node's worth of points quantised the way `lib.laz` stores them. A value
    within the fit's error of a half-count boundary can round either way, so a
    handful of differences is expected; more would mean real precision is going.
    """
    x, y = box_of_points(3_200.0, count=50_000)
    ax, ay = reproject(WEB_MERCATOR_TO_UTM, x, y)
    ex, ey = WEB_MERCATOR_TO_UTM.transform(x, y)
    differing = (np.round(ax / CANONICAL_SCALE) != np.round(ex / CANONICAL_SCALE)) | (
        np.round(ay / CANONICAL_SCALE) != np.round(ey / CANONICAL_SCALE)
    )
    assert differing.mean() < 1e-3


def test_falls_back_when_the_box_is_too_wide():
    """A span the quadratic cannot hold must return exact values, not close ones."""
    x, y = box_of_points(400_000.0)
    ax, ay = reproject(WEB_MERCATOR_TO_UTM, x, y)
    ex, ey = WEB_MERCATOR_TO_UTM.transform(x, y)
    # Bit-for-bit, because falling back returns the exact call's own output.
    assert np.array_equal(ax, ex) and np.array_equal(ay, ey)


def test_tolerance_is_honoured_when_tightened():
    """Asking for more than a quadratic can give falls back rather than silently missing."""
    x, y = box_of_points(16_000.0)
    ax, ay = reproject(WEB_MERCATOR_TO_UTM, x, y, tolerance=1e-9)
    ex, ey = WEB_MERCATOR_TO_UTM.transform(x, y)
    assert np.array_equal(ax, ex) and np.array_equal(ay, ey)


@pytest.mark.parametrize(
    "x, y",
    [
        (np.array([]), np.array([])),
        (np.array([-1.25e7]), np.array([5.9e6])),
        # Every point in one place: there is no box to fit over.
        (np.full(100, -1.25e7), np.full(100, 5.9e6)),
    ],
    ids=["empty", "single point", "no extent"],
)
def test_degenerate_inputs_use_the_exact_transform(x, y):
    ax, ay = reproject(WEB_MERCATOR_TO_UTM, x, y)
    ex, ey = WEB_MERCATOR_TO_UTM.transform(x, y)
    assert np.array_equal(ax, ex) and np.array_equal(ay, ey)


@pytest.mark.parametrize(
    "target, centre",
    [
        ("EPSG:32612", (-1.25e7, 5.9e6)),
        ("EPSG:32610", (-1.36e7, 5.7e6)),
        ("EPSG:26912", (-1.25e7, 5.9e6)),
        ("EPSG:5070", (-9.0e6, 4.0e6)),
    ],
    ids=["UTM 12N", "UTM 10N", "NAD83 UTM 12N", "CONUS Albers"],
)
@pytest.mark.parametrize("span", [500.0, 5_000.0, 16_000.0, 200_000.0])
def test_promise_holds_across_crs_and_span(target, centre, span):
    """Either the fit was used and is inside tolerance, or exact values came back.

    A domain can be in any projected CRS, and how well a quadratic matches the
    map varies with it — Albers gives up around 10 km where UTM holds past 13.
    The guarantee may not: whatever the pair, a caller gets tolerance or exact.
    """
    transformer = Transformer.from_crs("EPSG:3857", target, always_xy=True)
    rng = np.random.default_rng(1)
    # Corners and edges included, where a least-squares residual peaks.
    x = (
        np.concatenate(
            [
                rng.uniform(-span / 2, span / 2, 20_000),
                np.repeat([-span / 2, span / 2], 2_000),
            ]
        )
        + centre[0]
    )
    y = (
        np.concatenate(
            [
                rng.uniform(-span / 2, span / 2, 20_000),
                rng.uniform(-span / 2, span / 2, 4_000),
            ]
        )
        + centre[1]
    )
    ax, ay = reproject(transformer, x, y)
    ex, ey = transformer.transform(x, y)
    fell_back = np.array_equal(ax, ex) and np.array_equal(ay, ey)
    assert fell_back or np.hypot(ax - ex, ay - ey).max() <= DEFAULT_TOLERANCE


@pytest.mark.parametrize("target", ["EPSG:32612", "EPSG:5070"])
def test_fit_is_used_at_the_size_a_real_node_covers(target):
    """The speedup is worth nothing if the guard rejects everything it sees.

    Cached 3DEP nodes span 101 m to 3202 m, so the fit has to hold there.
    """
    transformer = Transformer.from_crs("EPSG:3857", target, always_xy=True)
    x, y = box_of_points(3_200.0)
    ax, _ = reproject(transformer, x, y)
    ex, _ = transformer.transform(x, y)
    assert not np.array_equal(ax, ex), "fell back at a size it should handle"


def test_collinear_points_still_reproject():
    """A node clipped to a sliver has extent on one axis only."""
    x = np.linspace(CENTRE[0] - 200, CENTRE[0] + 200, 5_000)
    y = np.full(5_000, CENTRE[1])
    ax, ay = reproject(WEB_MERCATOR_TO_UTM, x, y)
    ex, ey = WEB_MERCATOR_TO_UTM.transform(x, y)
    assert np.hypot(ax - ex, ay - ey).max() <= DEFAULT_TOLERANCE
