"""NAIP-CHM v1 tile index: lookup, per-year coverage, and tile selection.

The index (built by ``scripts/build_naip_chm_index.py``) is a location-sorted
parquet with one row per raster. A lookup reads only the row groups whose bbox
column statistics overlap the domain, so it touches a few MB of a ~130 MB file.

GDAL-free on purpose: the API imports this module (coverage endpoint and
create-time check), so it uses only pyarrow, gcsfs, shapely and pyproj.

Coverage is measured in an equal-area CRS (EPSG:6933). A year is ``full`` when
the uncovered share of the domain is at most ``FULL_COVERAGE_TOLERANCE``
(relative), ``none`` when no footprint overlaps the domain with positive area,
and ``partial`` otherwise.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import shapely
from pyproj import CRS, Transformer
from shapely.geometry.base import BaseGeometry

from lib.config import TABLES_BUCKET
from lib.gcs import get_gcsfs_client

NAIP_CHM_INDEX_PATH = f"gs://{TABLES_BUCKET}/naip_chm_v1_index.parquet"

# Relative share of the domain area that may stay uncovered and still count as
# full coverage. Also the minimum new area a fallback tile must add.
FULL_COVERAGE_TOLERANCE = 1e-6

_EQUAL_AREA_CRS = CRS.from_epsg(6933)
_WGS84 = CRS.from_epsg(4326)
_YEARS_KEY = b"naip_chm:years"
_BBOX_COLUMNS = ("bbox_xmin", "bbox_ymin", "bbox_xmax", "bbox_ymax")
_COVERAGE_COLUMNS = ["year", *_BBOX_COLUMNS, "footprint"]
_TILE_COLUMNS = [
    "chm_url",
    "year",
    *_BBOX_COLUMNS,
    "footprint",
    "datetime",
    "source_gsd",
    "pipeline",
    "model",
]
# Domain edges are densified before reprojection so curved edges keep their area.
_DENSIFY_SEGMENTS = 64


@dataclass(frozen=True)
class NaipChmTile:
    url: str
    year: int
    footprint: BaseGeometry  # EPSG:4326
    datetime: datetime
    source_gsd: float
    pipeline: str
    model: str


@dataclass(frozen=True)
class NaipChmYearCoverage:
    year: int
    percent: float
    coverage: Literal["full", "partial", "none"]


@dataclass(frozen=True)
class NaipChmCoverage:
    years: list[NaipChmYearCoverage]  # every index year, newest first
    default_year: int | None  # latest full year


@dataclass(frozen=True)
class _Candidates:
    index_years: list[int]
    table: pa.Table  # rows with positive-area overlap
    footprints_ea: np.ndarray  # footprints in EPSG:6933, aligned with table
    domain_ea: BaseGeometry
    domain_area: float


def naip_chm_coverage(
    geometry: BaseGeometry,
    crs,
    *,
    index_path: str = NAIP_CHM_INDEX_PATH,
    filesystem=None,
) -> NaipChmCoverage:
    """Per-year NAIP-CHM coverage of a domain geometry given in ``crs``."""
    candidates = _load_candidates(
        geometry, crs, _COVERAGE_COLUMNS, index_path, filesystem
    )
    return _coverage(candidates)


def select_naip_chm_tiles(
    geometry: BaseGeometry,
    crs,
    year: int | None,
    *,
    index_path: str = NAIP_CHM_INDEX_PATH,
    filesystem=None,
) -> tuple[list[NaipChmTile], int | None]:
    """Tiles to mosaic for a domain, newest year first, and the resolved year.

    An explicit ``year`` returns that year's overlapping tiles (possibly none).
    With ``year=None`` the latest full year is used; if no year is full, a
    greedy newest-to-oldest fallback keeps each tile that covers domain area not
    already covered by newer kept tiles, and the resolved year is ``None``.
    """
    candidates = _load_candidates(geometry, crs, _TILE_COLUMNS, index_path, filesystem)
    years = candidates.table.column("year").to_numpy()

    if year is None:
        year = _coverage(candidates).default_year
        if year is None:
            return _fallback_tiles(candidates), None

    rows = np.flatnonzero(years == year)
    return _tiles(candidates.table, rows), year


def _load_candidates(
    geometry: BaseGeometry, crs, columns: list[str], index_path: str, filesystem
) -> _Candidates:
    crs = CRS.from_user_input(crs)
    xmin, ymin, xmax, ymax = geometry.bounds
    dense = shapely.segmentize(
        geometry, max(xmax - xmin, ymax - ymin) / _DENSIFY_SEGMENTS
    )
    domain_ll = _reproject(dense, crs, _WGS84)
    domain_ea = _reproject(dense, crs, _EQUAL_AREA_CRS)
    domain_area = domain_ea.area
    if domain_area <= 0:
        raise ValueError("Domain geometry has no area")

    index_years, table = _read_index(
        index_path, filesystem or get_gcsfs_client(), domain_ll.bounds, columns
    )

    footprints_ll = shapely.from_wkb(table.column("footprint").to_numpy(False))
    footprints_ea = _reproject_array(footprints_ll, _WGS84, _EQUAL_AREA_CRS)
    overlap = shapely.area(shapely.intersection(footprints_ea, domain_ea)) > 0
    keep = pa.array(overlap, type=pa.bool_())
    return _Candidates(
        index_years=index_years,
        table=table.filter(keep),
        footprints_ea=footprints_ea[overlap],
        domain_ea=domain_ea,
        domain_area=domain_area,
    )


def _read_index(
    index_path: str, filesystem, bounds: tuple[float, float, float, float], columns
) -> tuple[list[int], pa.Table]:
    """Read the rows whose bbox overlaps ``bounds`` (EPSG:4326) plus the index years."""
    # No readahead cache: pre_buffer already coalesces the column-chunk ranges.
    with filesystem.open(index_path, "rb", cache_type="none") as f:
        pf = pq.ParquetFile(f, pre_buffer=True)
        index_years = json.loads(pf.metadata.metadata[_YEARS_KEY])
        row_groups = _matching_row_groups(pf.metadata, bounds)
        table = pf.read_row_groups(row_groups, columns=columns)

    xmin, ymin, xmax, ymax = bounds
    mask = (
        (table.column("bbox_xmin").to_numpy() <= xmax)
        & (table.column("bbox_xmax").to_numpy() >= xmin)
        & (table.column("bbox_ymin").to_numpy() <= ymax)
        & (table.column("bbox_ymax").to_numpy() >= ymin)
    )
    return index_years, table.filter(pa.array(mask, type=pa.bool_()))


def _matching_row_groups(
    metadata: pq.FileMetaData, bounds: tuple[float, float, float, float]
) -> list[int]:
    """Row groups whose bbox column statistics may overlap ``bounds``."""
    xmin, ymin, xmax, ymax = bounds
    names = metadata.schema.names
    idx = {name: names.index(name) for name in _BBOX_COLUMNS}
    matches = []
    for i in range(metadata.num_row_groups):
        rg = metadata.row_group(i)
        stats = {name: rg.column(j).statistics for name, j in idx.items()}
        if not all(s is not None and s.has_min_max for s in stats.values()):
            matches.append(i)
            continue
        if (
            stats["bbox_xmin"].min <= xmax
            and stats["bbox_xmax"].max >= xmin
            and stats["bbox_ymin"].min <= ymax
            and stats["bbox_ymax"].max >= ymin
        ):
            matches.append(i)
    return matches


def _coverage(candidates: _Candidates) -> NaipChmCoverage:
    years = candidates.table.column("year").to_numpy()
    results = []
    for year in sorted(candidates.index_years, reverse=True):
        footprints = candidates.footprints_ea[years == year]
        if len(footprints) == 0:
            results.append(NaipChmYearCoverage(year, 0.0, "none"))
            continue
        covered = shapely.union_all(footprints).intersection(candidates.domain_ea).area
        uncovered_share = 1.0 - covered / candidates.domain_area
        if uncovered_share <= FULL_COVERAGE_TOLERANCE:
            results.append(NaipChmYearCoverage(year, 100.0, "full"))
        else:
            percent = 100.0 * covered / candidates.domain_area
            results.append(NaipChmYearCoverage(year, percent, "partial"))
    default_year = next((r.year for r in results if r.coverage == "full"), None)
    return NaipChmCoverage(years=results, default_year=default_year)


def _fallback_tiles(candidates: _Candidates) -> list[NaipChmTile]:
    table = candidates.table
    years = table.column("year").to_numpy()
    urls = table.column("chm_url").to_pylist()
    order = sorted(range(len(urls)), key=lambda i: (-int(years[i]), urls[i]))
    min_gain = FULL_COVERAGE_TOLERANCE * candidates.domain_area

    remaining = candidates.domain_ea
    kept = []
    for i in order:
        footprint = candidates.footprints_ea[i]
        if footprint.intersection(remaining).area > min_gain:
            kept.append(i)
            remaining = remaining.difference(footprint)
            if remaining.area <= min_gain:
                break
    return _tiles(table, np.asarray(kept, dtype=np.int64))


def _tiles(table: pa.Table, rows: np.ndarray) -> list[NaipChmTile]:
    """Build tiles for ``rows``, ordered newest year first, then by URL."""
    records = table.take(pa.array(rows, type=pa.int64())).to_pylist()
    tiles = [
        NaipChmTile(
            url=r["chm_url"],
            year=int(r["year"]),
            footprint=shapely.from_wkb(r["footprint"]),
            datetime=r["datetime"],
            source_gsd=r["source_gsd"],
            pipeline=r["pipeline"],
            model=r["model"],
        )
        for r in records
    ]
    return sorted(tiles, key=lambda t: (-t.year, t.url))


def _reproject(geometry: BaseGeometry, src: CRS, dst: CRS) -> BaseGeometry:
    if src == dst:
        return geometry
    return _reproject_array(np.asarray([geometry]), src, dst)[0]


def _reproject_array(geometries: np.ndarray, src: CRS, dst: CRS) -> np.ndarray:
    transformer = Transformer.from_crs(src, dst, always_xy=True)

    def transform(coords: np.ndarray) -> np.ndarray:
        x, y = transformer.transform(coords[:, 0], coords[:, 1])
        return np.column_stack([x, y])

    return shapely.transform(geometries, transform)
