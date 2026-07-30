"""
Unit tests for lakitu.ept octree traversal.

All offline: hierarchy pages and ept.json documents are supplied by a fake
session, so nothing here touches the network.
"""

from unittest.mock import MagicMock

import pytest
from lakitu.ept import (
    MAX_DEPTH,
    MAX_HIERARCHY_PAGES,
    EptMetadata,
    _overlaps_2d,
    fetch_metadata,
    node_bounds,
    walk_hierarchy,
)
from pyproj import CRS

from lib.errors import ProcessingError

# A root cube 1024 units on a side, so node bounds divide evenly at every depth.
ROOT = (0.0, 0.0, 0.0, 1024.0, 1024.0, 1024.0)

EPT_DOCUMENT = {
    "bounds": list(ROOT),
    "boundsConforming": [0.0, 0.0, 0.0, 1024.0, 1024.0, 512.0],
    "dataType": "laszip",
    "hierarchyType": "json",
    "points": 1000,
    "span": 256,
    "srs": {"authority": "EPSG", "horizontal": "3857"},
    "schema": [{"name": "X"}, {"name": "Y"}, {"name": "Z"}, {"name": "Intensity"}],
}


def fake_session(responses: dict) -> MagicMock:
    """Build a session returning canned JSON keyed by URL suffix."""
    session = MagicMock()

    def get(url, **kwargs):
        response = MagicMock()
        for suffix, payload in responses.items():
            if url.endswith(suffix):
                response.status_code = 200
                response.json.return_value = payload
                response.raise_for_status.return_value = None
                return response
        response.status_code = 404
        response.raise_for_status.side_effect = Exception(f"404 {url}")
        return response

    session.get.side_effect = get
    return session


def metadata(bounds=ROOT) -> EptMetadata:
    return EptMetadata(
        url="https://example.test/ACQ/ept.json",
        base_url="https://example.test/ACQ",
        bounds=bounds,
        bounds_conforming=bounds,
        crs=CRS.from_user_input("EPSG:3857"),
        vertical_crs=None,
        point_count=1000,
        dimension_names=("X", "Y", "Z"),
    )


class TestNodeBounds:
    """Tests for octree key to bounding box conversion."""

    def test_root_covers_the_whole_cube(self):
        assert node_bounds(ROOT, 0, 0, 0, 0) == ROOT

    def test_depth_one_splits_in_eight(self):
        assert node_bounds(ROOT, 1, 0, 0, 0) == (0, 0, 0, 512, 512, 512)
        assert node_bounds(ROOT, 1, 1, 1, 1) == (512, 512, 512, 1024, 1024, 1024)

    def test_deep_key(self):
        # At depth 8 each node is 1024 / 256 = 4 units across.
        assert node_bounds(ROOT, 8, 3, 0, 0) == (12, 0, 0, 16, 4, 4)

    def test_children_tile_their_parent_exactly(self):
        """No gaps and no overlap, or points would be lost or duplicated."""
        parent = node_bounds(ROOT, 2, 1, 1, 1)
        total = 0.0
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    child = node_bounds(ROOT, 3, 2 + dx, 2 + dy, 2 + dz)
                    total += (child[3] - child[0]) * (child[4] - child[1])
                    assert child[0] >= parent[0] and child[3] <= parent[3]
        parent_area = (parent[3] - parent[0]) * (parent[4] - parent[1])
        # Four columns of two stacked children each cover the parent footprint.
        assert total == pytest.approx(parent_area * 2)


class TestOverlap:
    """Tests for the node/query intersection test."""

    def test_overlapping_boxes_match(self):
        assert _overlaps_2d((0, 0, 0, 10, 10, 10), (5, 5, 15, 15))

    def test_disjoint_boxes_do_not_match(self):
        assert not _overlaps_2d((0, 0, 0, 10, 10, 10), (20, 20, 30, 30))

    def test_touching_boxes_do_not_match(self):
        """Adjacent nodes share a face; including them fetches dead weight."""
        assert not _overlaps_2d((0, 0, 0, 10, 10, 10), (10, 0, 20, 10))

    def test_elevation_is_ignored(self):
        """A node far above or below the query must still be selected.

        The octree splits in z as well, so a node at depth 8 covers one thin
        elevation slice. A domain has no elevation extent, and every slice is
        wanted — testing z here would select nothing at all.
        """
        high_node = (0, 0, 900, 10, 10, 1000)
        low_node = (0, 0, 0, 10, 10, 100)
        query = (0, 0, 10, 10)
        assert _overlaps_2d(high_node, query)
        assert _overlaps_2d(low_node, query)


class TestFetchMetadata:
    """Tests for reading and validating ept.json."""

    def test_reads_a_valid_document(self):
        session = fake_session({"ept.json": EPT_DOCUMENT})
        meta = fetch_metadata(session, "https://example.test/ACQ/ept.json")
        assert meta.bounds == ROOT
        assert meta.crs.to_epsg() == 3857
        assert meta.base_url == "https://example.test/ACQ"
        assert meta.dimension_names == ("X", "Y", "Z", "Intensity")

    def test_rejects_unsupported_data_type(self):
        """Binary and zstandard EPT exist and laspy cannot read either."""
        session = fake_session({"ept.json": {**EPT_DOCUMENT, "dataType": "binary"}})
        with pytest.raises(ProcessingError) as exc:
            fetch_metadata(session, "https://example.test/ACQ/ept.json")
        assert exc.value.code == "EPT_METADATA_ERROR"

    def test_rejects_unsupported_hierarchy_type(self):
        session = fake_session({"ept.json": {**EPT_DOCUMENT, "hierarchyType": "gzip"}})
        with pytest.raises(ProcessingError) as exc:
            fetch_metadata(session, "https://example.test/ACQ/ept.json")
        assert exc.value.code == "EPT_METADATA_ERROR"

    def test_rejects_missing_crs(self):
        """Without a CRS the points cannot be placed, so do not guess one."""
        session = fake_session({"ept.json": {**EPT_DOCUMENT, "srs": {}}})
        with pytest.raises(ProcessingError) as exc:
            fetch_metadata(session, "https://example.test/ACQ/ept.json")
        assert exc.value.code == "EPT_METADATA_ERROR"

    def test_falls_back_to_wkt_crs(self):
        wkt = CRS.from_user_input("EPSG:26912").to_wkt()
        session = fake_session({"ept.json": {**EPT_DOCUMENT, "srs": {"wkt": wkt}}})
        meta = fetch_metadata(session, "https://example.test/ACQ/ept.json")
        assert meta.crs.to_epsg() == 26912

    def test_reads_a_declared_vertical_crs(self):
        """Recorded when the survey declares it — it is a label, not a transform."""
        srs = {"authority": "EPSG", "horizontal": "3857", "vertical": "5703"}
        session = fake_session({"ept.json": {**EPT_DOCUMENT, "srs": srs}})
        meta = fetch_metadata(session, "https://example.test/ACQ/ept.json")
        assert meta.vertical_crs == "EPSG:5703"

    def test_undeclared_vertical_crs_is_none(self):
        """Most acquisitions declare nothing; inventing one would be a guess."""
        session = fake_session({"ept.json": EPT_DOCUMENT})
        meta = fetch_metadata(session, "https://example.test/ACQ/ept.json")
        assert meta.vertical_crs is None

    def test_unreachable_index_raises(self):
        session = fake_session({})
        with pytest.raises(ProcessingError) as exc:
            fetch_metadata(session, "https://example.test/ACQ/ept.json")
        assert exc.value.code == "EPT_METADATA_ERROR"


class TestWalkHierarchy:
    """Tests for selecting nodes from the index."""

    def test_selects_overlapping_nodes_at_every_depth(self):
        """EPT is additive: a parent's points are not repeated by its children.

        Every overlapping node at every depth must be selected, or the result
        is a thinned cloud rather than a full-density one.
        """
        hierarchy = {
            "0-0-0-0": 100,
            "1-0-0-0": 50,
            "1-1-1-1": 50,
            "2-0-0-0": 25,
        }
        session = fake_session({"0-0-0-0.json": hierarchy})
        nodes = walk_hierarchy(session, metadata(), (0, 0, 200, 200))

        keys = {node.key for node in nodes}
        assert "0-0-0-0" in keys
        assert "1-0-0-0" in keys
        assert "2-0-0-0" in keys
        # The far corner node does not overlap the query.
        assert "1-1-1-1" not in keys

    def test_skips_empty_nodes(self):
        session = fake_session({"0-0-0-0.json": {"0-0-0-0": 0, "1-0-0-0": 10}})
        nodes = walk_hierarchy(session, metadata(), (0, 0, 100, 100))
        assert {node.key for node in nodes} == {"1-0-0-0"}

    def test_follows_a_deeper_index_page(self):
        """A -1 count points at another page holding the real counts."""
        session = fake_session(
            {
                "0-0-0-0.json": {"0-0-0-0": 100, "1-0-0-0": -1},
                "1-0-0-0.json": {"1-0-0-0": 40, "2-0-0-0": 20},
            }
        )
        nodes = walk_hierarchy(session, metadata(), (0, 0, 100, 100))
        by_key = {node.key: node.count for node in nodes}
        # The page's own entry must overwrite the -1 placeholder.
        assert by_key["1-0-0-0"] == 40
        assert by_key["2-0-0-0"] == 20

    def test_unresolved_page_raises(self):
        """A page that does not supply its own count would loop forever."""
        session = fake_session(
            {
                "0-0-0-0.json": {"0-0-0-0": 100, "1-0-0-0": -1},
                "1-0-0-0.json": {"2-0-0-0": 20},
            }
        )
        with pytest.raises(ProcessingError) as exc:
            walk_hierarchy(session, metadata(), (0, 0, 100, 100))
        assert exc.value.code == "EPT_METADATA_ERROR"

    def test_self_referencing_page_raises(self):
        """A cycle in the index must fail rather than spin."""
        session = fake_session(
            {
                "0-0-0-0.json": {"0-0-0-0": 100, "1-0-0-0": -1},
                "1-0-0-0.json": {"1-0-0-0": -1},
            }
        )
        with pytest.raises(ProcessingError) as exc:
            walk_hierarchy(session, metadata(), (0, 0, 100, 100))
        assert exc.value.code == "EPT_METADATA_ERROR"

    def test_unbounded_descent_is_stopped(self):
        """A tree that never bottoms out must fail, not descend forever.

        Every node here claims a deeper page, so the walk would recurse without
        end. Either the depth or the page ceiling catches it; what matters is
        that it terminates with a structured error.
        """

        def get(url, **kwargs):
            response = MagicMock()
            response.status_code = 200
            response.raise_for_status.return_value = None
            key = url.rsplit("/", 1)[-1].removesuffix(".json")
            depth = int(key.split("-")[0])
            response.json.return_value = {
                key: -1 if depth else 100,
                f"{depth + 1}-0-0-0": -1,
            }
            return response

        session = MagicMock()
        session.get.side_effect = get
        with pytest.raises(ProcessingError) as exc:
            walk_hierarchy(session, metadata(), (0, 0, 100, 100))
        assert exc.value.code == "EPT_METADATA_ERROR"
        assert session.get.call_count <= MAX_HIERARCHY_PAGES

    def test_deepest_allowed_level_is_not_treated_as_malformed(self):
        """Children are pushed one level past whatever exists.

        Judging depth before checking the index would fail a perfectly good
        tree whose deepest populated level happens to be the limit.
        """
        # The walk only descends through nodes the index claims, so the whole
        # ancestor chain has to exist for the deepest one to be reached.
        chain = {f"{d}-0-0-0": 5 for d in range(MAX_DEPTH + 1)}
        session = fake_session({"0-0-0-0.json": chain})
        nodes = walk_hierarchy(session, metadata(), (0, 0, 1024, 1024))
        assert f"{MAX_DEPTH}-0-0-0" in {node.key for node in nodes}

    def test_a_node_past_the_depth_limit_raises(self):
        """The guard still fires for a tree that genuinely goes too deep."""
        chain = {f"{d}-0-0-0": 5 for d in range(MAX_DEPTH + 2)}
        session = fake_session({"0-0-0-0.json": chain})
        with pytest.raises(ProcessingError) as exc:
            walk_hierarchy(session, metadata(), (0, 0, 1024, 1024))
        assert exc.value.code == "EPT_METADATA_ERROR"

    def test_nodes_are_ordered_shallowest_first(self):
        session = fake_session(
            {"0-0-0-0.json": {"0-0-0-0": 5, "1-0-0-0": 5, "2-0-0-0": 5}}
        )
        nodes = walk_hierarchy(session, metadata(), (0, 0, 100, 100))
        assert [node.depth for node in nodes] == sorted(node.depth for node in nodes)

    def test_node_data_url(self):
        session = fake_session({"0-0-0-0.json": {"0-0-0-0": 5}})
        node = walk_hierarchy(session, metadata(), (0, 0, 100, 100))[0]
        assert node.data_url == "https://example.test/ACQ/ept-data/0-0-0-0.laz"
