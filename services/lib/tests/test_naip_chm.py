"""Unit tests for lib.naip_chm against a small synthetic index on local disk."""

import json
from datetime import UTC, datetime

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import shapely
from fsspec.implementations.local import LocalFileSystem
from pyproj import Transformer
from shapely.geometry import box

from lib.naip_chm import (
    NaipChmYearCoverage,
    naip_chm_coverage,
    select_naip_chm_tiles,
)

INDEX_YEARS = [2010, 2021, 2022, 2023, 2024, 2025]
DOMAIN = box(-114.0, 46.0, -113.5, 46.5)
FS = LocalFileSystem()


def _quads(year: int, xmin: float, ymin: float, size: float = 0.25, n: int = 2):
    """n x n tiles of ``size`` degrees starting at (xmin, ymin)."""
    return [
        (
            year,
            box(
                xmin + i * size,
                ymin + j * size,
                xmin + (i + 1) * size,
                ymin + (j + 1) * size,
            ),
        )
        for i in range(n)
        for j in range(n)
    ]


def _write_index(path, row_groups: list[list[tuple[int, shapely.Polygon]]]) -> str:
    """Write an index with one parquet row group per entry of ``row_groups``."""
    schema = pa.schema(
        [
            ("chm_url", pa.string()),
            ("year", pa.int16()),
            ("bbox_xmin", pa.float64()),
            ("bbox_ymin", pa.float64()),
            ("bbox_xmax", pa.float64()),
            ("bbox_ymax", pa.float64()),
            ("footprint", pa.binary()),
            ("datetime", pa.timestamp("us", tz="UTC")),
            ("source_gsd", pa.float64()),
            ("pipeline", pa.string()),
            ("model", pa.string()),
        ],
        metadata={b"naip_chm:years": json.dumps(INDEX_YEARS).encode()},
    )
    with pq.ParquetWriter(path, schema) as writer:
        for rg, tiles in enumerate(row_groups):
            rows = []
            for k, (year, footprint) in enumerate(tiles):
                xmin, ymin, xmax, ymax = footprint.bounds
                rows.append(
                    {
                        "chm_url": f"https://example.test/{year}/rg{rg}_{k}.tif",
                        "year": year,
                        "bbox_xmin": xmin,
                        "bbox_ymin": ymin,
                        "bbox_xmax": xmax,
                        "bbox_ymax": ymax,
                        "footprint": shapely.to_wkb(footprint),
                        "datetime": datetime(year, 7, 1, tzinfo=UTC),
                        "source_gsd": 0.6,
                        "pipeline": "v2-pc",
                        "model": "test",
                    }
                )
            writer.write_table(pa.Table.from_pylist(rows, schema=schema))
    return str(path)


@pytest.fixture
def index_path(tmp_path):
    """RG0 covers the domain; RG1 and RG2 are far away."""
    near = [
        *_quads(2023, -114.0, 46.0),  # full
        *_quads(2025, -114.0, 46.0, n=1),  # west half, south
        (2025, box(-114.0, 46.25, -113.75, 46.5)),  # west half, north
        (2024, box(-113.5, 46.0, -113.25, 46.5)),  # touches the east edge only
    ]
    far_east = _quads(2022, -100.0, 40.0)
    far_south = _quads(2021, -114.0, 30.0)
    return _write_index(tmp_path / "index.parquet", [near, far_east, far_south])


@pytest.fixture
def fallback_index_path(tmp_path):
    """No single year covers the domain."""
    tiles = [
        (2025, box(-114.0, 46.0, -113.75, 46.5)),  # west half
        (2023, box(-114.0, 46.0, -113.75, 46.5)),  # west half, shadowed by 2025
        (2023, box(-113.75, 46.0, -113.5, 46.25)),  # east, south
        (2021, box(-114.0, 46.25, -113.5, 46.5)),  # north half, adds only east-north
        (2010, box(-114.0, 46.0, -113.5, 46.25)),  # south half, adds nothing
    ]
    return _write_index(tmp_path / "fallback.parquet", [tiles])


def _by_year(coverage) -> dict[int, NaipChmYearCoverage]:
    return {c.year: c for c in coverage.years}


def test_coverage_classification(index_path):
    coverage = naip_chm_coverage(DOMAIN, 4326, index_path=index_path, filesystem=FS)
    years = _by_year(coverage)

    assert [c.year for c in coverage.years] == sorted(INDEX_YEARS, reverse=True)
    assert years[2023] == NaipChmYearCoverage(2023, 100.0, "full")
    assert years[2025].coverage == "partial"
    # Equal-area: splitting by longitude halves the area exactly.
    assert years[2025].percent == pytest.approx(50.0, abs=1e-6)
    assert years[2024] == NaipChmYearCoverage(2024, 0.0, "none")
    assert coverage.default_year == 2023


def test_years_without_nearby_tiles_are_none(index_path):
    years = _by_year(
        naip_chm_coverage(DOMAIN, 4326, index_path=index_path, filesystem=FS)
    )
    # 2022/2021 only have tiles far away; 2010 has no tiles at all.
    for year in (2022, 2021, 2010):
        assert years[year] == NaipChmYearCoverage(year, 0.0, "none")


def test_row_group_pruning(index_path, monkeypatch):
    requested = []
    original = pq.ParquetFile.read_row_groups

    def spy(self, row_groups, *args, **kwargs):
        requested.append(list(row_groups))
        return original(self, row_groups, *args, **kwargs)

    monkeypatch.setattr(pq.ParquetFile, "read_row_groups", spy)

    naip_chm_coverage(DOMAIN, 4326, index_path=index_path, filesystem=FS)
    naip_chm_coverage(
        box(-99.9, 40.1, -99.8, 40.2), 4326, index_path=index_path, filesystem=FS
    )
    naip_chm_coverage(
        box(-80.0, 30.0, -79.0, 31.0), 4326, index_path=index_path, filesystem=FS
    )
    assert requested == [[0], [1], []]


def test_projected_crs_domain(index_path):
    utm = Transformer.from_crs(4326, 32612, always_xy=True)
    domain_utm = shapely.transform(
        shapely.segmentize(DOMAIN, 0.001),
        lambda x, y: utm.transform(x, y),
        interleaved=False,
    )
    years = _by_year(
        naip_chm_coverage(
            domain_utm, "EPSG:32612", index_path=index_path, filesystem=FS
        )
    )
    assert years[2023].coverage == "full"
    assert years[2025].coverage == "partial"
    assert years[2025].percent == pytest.approx(50.0, abs=0.01)
    assert years[2024].coverage == "none"


def test_partially_outside_domain_is_partial(index_path):
    # Extends 0.1 deg north of every tile.
    domain = box(-114.0, 46.0, -113.5, 46.6)
    years = _by_year(
        naip_chm_coverage(domain, 4326, index_path=index_path, filesystem=FS)
    )
    assert years[2023].coverage == "partial"
    assert 80.0 < years[2023].percent < 85.0


def test_select_default_year(index_path):
    tiles, year = select_naip_chm_tiles(
        DOMAIN, 4326, None, index_path=index_path, filesystem=FS
    )
    assert year == 2023
    assert len(tiles) == 4
    assert {t.year for t in tiles} == {2023}
    tile = tiles[0]
    assert tile.url.startswith("https://example.test/2023/")
    assert tile.datetime == datetime(2023, 7, 1, tzinfo=UTC)
    assert tile.source_gsd == 0.6
    assert (tile.pipeline, tile.model) == ("v2-pc", "test")
    assert tile.footprint.intersects(DOMAIN)


def test_select_explicit_year(index_path):
    tiles, year = select_naip_chm_tiles(
        DOMAIN, 4326, 2025, index_path=index_path, filesystem=FS
    )
    assert year == 2025
    assert len(tiles) == 2
    assert all(t.year == 2025 for t in tiles)


@pytest.mark.parametrize("year", [2024, 2022, 2010, 2030])
def test_select_explicit_year_without_coverage_is_empty(index_path, year):
    tiles, resolved = select_naip_chm_tiles(
        DOMAIN, 4326, year, index_path=index_path, filesystem=FS
    )
    assert tiles == []
    assert resolved == year


def test_fallback_greedy_newest_first(fallback_index_path):
    coverage = naip_chm_coverage(
        DOMAIN, 4326, index_path=fallback_index_path, filesystem=FS
    )
    assert coverage.default_year is None

    tiles, year = select_naip_chm_tiles(
        DOMAIN, 4326, None, index_path=fallback_index_path, filesystem=FS
    )
    assert year is None
    assert [t.year for t in tiles] == [2025, 2023, 2021]
    assert tiles[1].footprint.equals(box(-113.75, 46.0, -113.5, 46.25))
    assert shapely.union_all([t.footprint for t in tiles]).covers(DOMAIN)


def test_fallback_partial_union(tmp_path):
    # Nothing covers the north-east quarter: fallback keeps what exists.
    path = _write_index(
        tmp_path / "holes.parquet",
        [
            [
                (2025, box(-114.0, 46.0, -113.75, 46.5)),
                (2022, box(-113.75, 46.0, -113.5, 46.25)),
                (2021, box(-114.0, 46.0, -113.75, 46.25)),
            ]
        ],
    )
    tiles, year = select_naip_chm_tiles(
        DOMAIN, 4326, None, index_path=path, filesystem=FS
    )
    assert year is None
    assert [t.year for t in tiles] == [2025, 2022]


def test_domain_without_area_raises(index_path):
    with pytest.raises(ValueError):
        naip_chm_coverage(
            shapely.LineString([(-114.0, 46.0), (-113.5, 46.5)]),
            4326,
            index_path=index_path,
            filesystem=FS,
        )
