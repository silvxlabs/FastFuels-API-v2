"""Manual parity gate: static FlatGeobuf reader vs. live Overpass.

SKIPPED by default. Overpass blocks cloud-provider IPs (the very reason for this
migration), so this cannot run in CI — run it locally on a non-cloud network:

    RUN_OSM_PARITY=1 uv run --extra parity \\
        pytest tests/integration/handlers/test_osm_parity.py -v

It confirms the static per-state OSM snapshot reproduces what Overpass would have
returned for the same ROI: identical osm_id sets and matching geometries
(Hausdorff distance ~0) for the features both sources agree on. Covers a
single-state ROI, a border-straddling ROI (cross-border dedup), and a
reservoir/basin ROI (water bodies folded into the FGB at build time).

``osmnx`` is imported lazily inside each test so collecting this file never
requires it (it lives in the ``parity`` extra, not the default test env), and
``RUN_OSM_PARITY`` must be set for any test here to execute.
"""

import os

import pytest
from etcher.handlers.road import ROAD_DATA
from etcher.osm_source import read_osm_features
from shapely.geometry import box

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_OSM_PARITY") != "1",
    reason="set RUN_OSM_PARITY=1 to run the live-Overpass parity gate "
    "(local, non-cloud IP only)",
)

# (minx, miny, maxx, maxy) in EPSG:4326. Starting points near the Montana test
# domain — swap in ROIs you trust for the comparison you care about.
SINGLE_STATE_BBOX = (-114.05, 46.84, -113.95, 46.92)  # Missoula, MT (interior)
BORDER_BBOX = (-114.40, 46.55, -114.30, 46.65)  # MT / ID state line
RESERVOIR_BBOX = (-111.60, 43.85, -111.50, 43.95)  # reservoir, ID

WATER_TAGS = {
    "natural": ["water"],
    "waterway": True,
    "landuse": ["reservoir", "basin"],
}

HAUSDORFF_TOL_DEG = 1e-6


def _overpass_features(bbox, tags, keep_geom_types):
    """Live Overpass features for ``tags`` in ``bbox``, flattened with an
    integer ``osm_id`` column and restricted to ``keep_geom_types``.

    Filtering to the geometry types the FlatGeobuf layer stores keeps the
    comparison apples-to-apples (Overpass also returns node/point variants the
    snapshot omits).
    """
    import osmnx as ox

    gdf = ox.features_from_polygon(box(*bbox), tags=tags)
    gdf = gdf[gdf.geom_type.isin(keep_geom_types)]
    # osmnx 2.x indexes by (element, id); the integer OSM id is the "id" level.
    flat = gdf.reset_index()
    id_col = next(c for c in ("id", "osmid", "osm_id") if c in flat.columns)
    flat = flat.rename(columns={id_col: "osm_id"})
    flat["osm_id"] = flat["osm_id"].astype("int64")
    return flat


def _assert_parity(fgb_map, overpass_map, label):
    """Assert identical osm_id sets and matching geometries for shared ids."""
    fgb_ids = set(fgb_map)
    op_ids = set(overpass_map)
    only_fgb = fgb_ids - op_ids
    only_op = op_ids - fgb_ids
    assert not only_fgb and not only_op, (
        f"{label}: osm_id mismatch — only in FGB: {sorted(only_fgb)[:10]}, "
        f"only in Overpass: {sorted(only_op)[:10]}"
    )
    for osm_id in fgb_ids:
        dist = fgb_map[osm_id].hausdorff_distance(overpass_map[osm_id])
        assert dist < HAUSDORFF_TOL_DEG, (
            f"{label}: geometry drift for osm_id {osm_id}: Hausdorff {dist}"
        )


@pytest.mark.parametrize(
    "bbox", [SINGLE_STATE_BBOX, BORDER_BBOX], ids=["single_state", "border"]
)
def test_road_parity(bbox):
    """Buffered-input road features match live Overpass for the same ROI."""
    fgb = read_osm_features(bbox, "road")
    fgb = fgb[fgb["highway"].isin(ROAD_DATA.keys())]
    fgb_map = dict(zip(fgb["osm_id"].astype("int64"), fgb.geometry))

    # Overpass returns every highway type; restrict to the ones we buffer.
    op = _overpass_features(bbox, {"highway": True}, ["LineString"])
    op = op[op["highway"].isin(ROAD_DATA.keys())]
    overpass_map = dict(zip(op["osm_id"], op.geometry))

    _assert_parity(fgb_map, overpass_map, f"road {bbox}")


@pytest.mark.parametrize(
    "bbox", [RESERVOIR_BBOX, SINGLE_STATE_BBOX], ids=["reservoir", "single_state"]
)
def test_water_parity(bbox):
    """Water features (incl. reservoir/basin bodies) match live Overpass."""
    fgb = read_osm_features(bbox, "water")
    fgb_map = dict(zip(fgb["osm_id"].astype("int64"), fgb.geometry))

    op = _overpass_features(bbox, WATER_TAGS, ["LineString", "Polygon", "MultiPolygon"])
    overpass_map = dict(zip(op["osm_id"], op.geometry))

    _assert_parity(fgb_map, overpass_map, f"water {bbox}")
