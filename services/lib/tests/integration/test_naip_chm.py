"""Integration test for lib.naip_chm against the real index in TABLES_BUCKET.

Requires GCP credentials with read access to the tables bucket.
"""

from pyproj import Transformer
from shapely.geometry import box

from lib.naip_chm import naip_chm_coverage, select_naip_chm_tiles


def _mt_id_box():
    """50 km box (+-25 km in EPSG:32611) centered on (-114.5, 46.6), the MT/ID line."""
    x, y = Transformer.from_crs(4326, 32611, always_xy=True).transform(-114.5, 46.6)
    return box(x - 25_000, y - 25_000, x + 25_000, y + 25_000)


def test_mt_id_coverage():
    coverage = naip_chm_coverage(_mt_id_box(), "EPSG:32611")
    years = {c.year: c for c in coverage.years}

    assert [c.year for c in coverage.years] == sorted(years, reverse=True)
    assert min(years) == 2007
    assert years[2023].coverage == "full"
    assert years[2021].coverage == "full"
    assert years[2025].coverage == "partial"
    assert 62.0 < years[2025].percent < 63.5
    assert years[2024].coverage == "partial"
    assert 0.0 < years[2024].percent < 0.5
    assert coverage.default_year == 2023


def test_mt_id_default_selection():
    tiles, year = select_naip_chm_tiles(_mt_id_box(), "EPSG:32611", None)
    assert year == 2023
    assert tiles
    assert all(t.year == 2023 for t in tiles)
