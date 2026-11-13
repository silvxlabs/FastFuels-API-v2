"""Live read check against Overture's public S3 bucket.

SKIPPED by default; reads real (public, anonymous) data over the network:

    RUN_OVERTURE_LIVE=1 uv run --extra test \\
        pytest tests/integration/handlers/test_overture_live.py -v -s

Counts change between releases, so they are asserted as ranges.
"""

import os
import time
from datetime import date

import pytest
from etcher.overture_source import (
    building_files,
    latest_release,
    parse_release,
    read_overture_buildings,
)

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_OVERTURE_LIVE") != "1",
    reason="set RUN_OVERTURE_LIVE=1 to read live Overture data from S3",
)

BIGFORK_BBOX = (-114.085846, 48.054491, -114.064148, 48.070509)


def test_latest_release_is_plausible():
    release = latest_release()
    release_date, _ = parse_release(release)
    assert (date.today() - release_date).days < 120


def test_read_bigfork_buildings():
    release = latest_release()
    start = time.monotonic()
    gdf = read_overture_buildings(BIGFORK_BBOX, release)
    elapsed = time.monotonic() - start
    print(f"release={release} buildings={len(gdf)} seconds={elapsed:.1f}")

    assert 450 <= len(gdf) <= 650
    assert gdf.crs == "EPSG:4326"
    assert set(gdf.geom_type) <= {"Polygon", "MultiPolygon"}
    assert gdf["height"].notna().mean() > 0.5
    assert all(len(s) > 0 and "dataset" in s[0] for s in gdf["sources"])


def test_bigfork_reads_few_files():
    # The STAC index narrows a release's ~500 building files to the one or two
    # whose bbox covers the ROI; scanning all of them exhausts etcher's memory.
    paths = building_files(BIGFORK_BBOX, latest_release())
    assert 1 <= len(paths) <= 2
    assert all("/theme=buildings/type=building/" in p for p in paths)
