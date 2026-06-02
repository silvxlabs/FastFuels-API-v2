"""Offline unit tests for the per-state OSM FlatGeobuf reader.

These never touch GCS: ``gpd.read_file`` and the GCS existence check are mocked,
so the tests exercise routing, cross-border dedup, and error handling purely in
memory. The synthetic states index has two adjacent states sharing the x=10
border:

    state_a -> box(0, 0, 10, 10)
    state_b -> box(10, 0, 20, 10)
"""

from unittest.mock import patch

import geopandas as gpd
import pytest
from etcher import osm_source
from etcher.errors import ProcessingError
from shapely.geometry import LineString, box


@pytest.fixture(autouse=True)
def _reset_states_cache():
    """The process-wide states cache must not leak across tests."""
    osm_source._states.cache_clear()
    yield
    osm_source._states.cache_clear()


def _states_gdf():
    return gpd.GeoDataFrame(
        {"slug": ["state_a", "state_b"], "name": ["State A", "State B"]},
        geometry=[box(0, 0, 10, 10), box(10, 0, 20, 10)],
        crs="EPSG:4326",
    )


def _layer_gdf(osm_ids, *, with_osm_id=True):
    """A per-state road layer with the given osm_ids."""
    data = {
        "highway": ["residential"] * len(osm_ids),
        "name": [f"r{i}" for i in osm_ids],
    }
    if with_osm_id:
        data = {"osm_id": list(osm_ids), **data}
    return gpd.GeoDataFrame(
        data,
        geometry=[LineString([(1, 1), (2, 2)]) for _ in osm_ids],
        crs="EPSG:4326",
    )


def _make_read_file(per_state, raises=None):
    """Build a ``gpd.read_file`` side effect.

    Returns the synthetic index for the states-index path, otherwise the
    configured per-state layer (or raises the configured exception for a slug).
    """
    raises = raises or {}

    def _read(path, bbox=None, **kwargs):
        if path == osm_source.STATES_INDEX:
            return _states_gdf()
        slug = path.rsplit("/", 1)[1].removesuffix(".fgb")
        if slug in raises:
            raise raises[slug]
        return per_state[slug]

    return _read


class TestIntersectingStateSlugs:
    def test_single_state(self):
        with patch.object(osm_source.gpd, "read_file", side_effect=_make_read_file({})):
            assert osm_source._intersecting_state_slugs((1, 1, 2, 2)) == ["state_a"]

    def test_straddling_border(self):
        with patch.object(osm_source.gpd, "read_file", side_effect=_make_read_file({})):
            slugs = osm_source._intersecting_state_slugs((9, 1, 11, 2))
        assert set(slugs) == {"state_a", "state_b"}

    def test_states_index_read_once_and_cached(self):
        read = _make_read_file({})
        with patch.object(osm_source.gpd, "read_file", side_effect=read) as mock_read:
            osm_source._intersecting_state_slugs((1, 1, 2, 2))
            osm_source._intersecting_state_slugs((1, 1, 2, 2))
        # The index is read on the first call only; the second is served from cache.
        assert mock_read.call_count == 1


class TestReadOsmFeatures:
    def test_single_state(self):
        per_state = {"state_a": _layer_gdf([1, 2])}
        with patch.object(
            osm_source.gpd, "read_file", side_effect=_make_read_file(per_state)
        ):
            gdf = osm_source.read_osm_features((1, 1, 2, 2), "road")
        assert sorted(gdf["osm_id"].tolist()) == [1, 2]
        assert gdf.crs == "EPSG:4326"

    def test_cross_border_dedup(self):
        # A way crossing the border appears in both states' extracts (osm_id 2).
        per_state = {
            "state_a": _layer_gdf([1, 2]),
            "state_b": _layer_gdf([2, 3]),
        }
        with patch.object(
            osm_source.gpd, "read_file", side_effect=_make_read_file(per_state)
        ):
            gdf = osm_source.read_osm_features((9, 1, 11, 2), "road")
        assert sorted(gdf["osm_id"].tolist()) == [1, 2, 3]

    def test_empty_roi(self):
        with patch.object(osm_source.gpd, "read_file", side_effect=_make_read_file({})):
            gdf = osm_source.read_osm_features((100, 100, 101, 101), "road")
        assert len(gdf) == 0
        assert gdf.crs == "EPSG:4326"

    def test_absent_layer_skipped(self):
        # state_a's layer is genuinely missing -> skip it; state_b still contributes.
        per_state = {"state_b": _layer_gdf([3, 4])}
        read = _make_read_file(per_state, raises={"state_a": FileNotFoundError()})
        with (
            patch.object(osm_source.gpd, "read_file", side_effect=read),
            patch.object(osm_source, "_layer_exists", return_value=False),
        ):
            gdf = osm_source.read_osm_features((9, 1, 11, 2), "road")
        assert sorted(gdf["osm_id"].tolist()) == [3, 4]

    def test_read_error_raises(self):
        # The layer exists but the read failed -> propagate, do not swallow.
        per_state = {"state_b": _layer_gdf([3, 4])}
        read = _make_read_file(per_state, raises={"state_a": RuntimeError("boom")})
        with (
            patch.object(osm_source.gpd, "read_file", side_effect=read),
            patch.object(osm_source, "_layer_exists", return_value=True),
            pytest.raises(ProcessingError),
        ):
            osm_source.read_osm_features((9, 1, 11, 2), "road")

    def test_invalid_feature_type(self):
        with pytest.raises(ValueError):
            osm_source.read_osm_features((1, 1, 2, 2), "trees")

    def test_no_osm_id_column_no_dedup(self):
        per_state = {"state_a": _layer_gdf([1, 1], with_osm_id=False)}
        with patch.object(
            osm_source.gpd, "read_file", side_effect=_make_read_file(per_state)
        ):
            gdf = osm_source.read_osm_features((1, 1, 2, 2), "road")
        # Without an osm_id column there is no dedup; both rows survive.
        assert len(gdf) == 2
