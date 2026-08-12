"""Unit tests for the per-node point-processing chain."""

import pytest
from lakitu import chain
from lakitu.ept import EptNode
from pyproj import CRS

from lib.errors import ProcessingError


def test_corrupt_node_is_reported_as_fetch_failure(monkeypatch):
    """Archive corruption remains an EPT error after decoding moves to a worker."""
    node = EptNode(
        key="1-0-0-0",
        depth=1,
        count=10,
        bounds=(0.0, 0.0, 0.0, 1.0, 1.0, 1.0),
        base_url="https://example.test/acquisition",
    )
    crs_wkt = CRS.from_epsg(32612).to_wkt()

    def fetch_corrupt_node(*args, **kwargs):
        assert kwargs["raw"] is True
        yield node, b"not a LAZ file"

    monkeypatch.setattr(chain, "fetch_nodes", fetch_corrupt_node)

    with pytest.raises(ProcessingError) as exc:
        list(
            chain.stream_records(
                session=None,
                plan=[(0, node)],
                sources=[(crs_wkt, None, (0.0, 0.0, 1.0, 1.0))],
                dst_crs_wkt=crs_wkt,
                header_bounds=(0.0, 0.0, 1.0, 1.0),
                point_format_id=6,
                workers=1,
                download_workers=1,
                batch=1,
            )
        )

    assert exc.value.code == "EPT_FETCH_FAILED"
    assert node.data_url in exc.value.traceback
