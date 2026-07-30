"""
USGS 3DEP Entwine Point Tile (EPT) discovery.

Shared logic for finding the 3DEP lidar acquisitions that cover a domain and
deciding which of them to read. Used by both the API (the point cloud coverage
pre-flight check) and lakitu (the point cloud fetch worker).

This is the **point cloud** side of 3DEP and is deliberately separate from
``lib.threedep``, which discovers 3DEP *elevation raster* tiles. They are
different USGS products served from different endpoints in different formats;
the only thing they share is the program name.

USGS publishes its public lidar as EPT — a streamable octree of LAZ files — at
one directory per acquisition under ``usgs-lidar-public``. The acquisitions and
their footprints are catalogued in a single GeoJSON file, so working out what is
available for a domain is a geometry problem, not an S3 listing problem.

All functions are synchronous. The API wraps blocking I/O calls with
asyncio.to_thread() to avoid blocking the event loop.
"""

import logging
import os
from dataclasses import dataclass
from datetime import UTC, datetime

import geopandas as gpd
import requests
from shapely import make_valid, set_precision
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry

from lib.config import RASTERS_BUCKET

logger = logging.getLogger(__name__)

# Live catalog of every public 3DEP EPT acquisition, maintained by Hobu. It is a
# ~9 MB GeoJSON of acquisition footprints with a point count per acquisition.
CATALOG_URL = (
    "https://raw.githubusercontent.com/hobuinc/usgs-lidar/master/"
    "boundaries/resources.geojson"
)

# Our own mirror of the same data as GeoParquet. The live catalog is a file in a
# GitHub repository with no availability guarantee, and it backs a user-facing
# endpoint, so we read the mirror first and fall back to the source.
CATALOG_MIRROR = f"gs://{RASTERS_BUCKET}/catalogs/usgs_3dep_ept.parquet"

EPT_BASE_URL = "https://s3-us-west-2.amazonaws.com/usgs-lidar-public"

# Equal-area projection used only to size whole acquisition footprints, which
# can span a state and would be badly distorted in a domain's local CRS.
EQUAL_AREA_CRS = "EPSG:8857"

# An acquisition covering at least this much of the domain is used on its own.
# Every additional acquisition introduces a seam — a different flight, date, and
# point density — which shows up as a discontinuity in derived products, so a
# single near-complete acquisition beats a mosaic that is nominally complete.
SINGLE_DATASET_THRESHOLD = 0.99

# Acquisitions contributing less than this fraction of the domain are dropped.
# Node selection is by bounding box, so a hairline sliver along one edge has a
# full-width bbox and would pull in an entire column of octree nodes to keep a
# handful of points.
MIN_CONTRIBUTION_FRACTION = 1e-4

# Selection stops once this little of the domain is left uncovered.
COVERAGE_EPSILON = 1e-4

# Ceiling on the points a single fetch may read. The output LAZ is built in
# memory (LAZ writers need a seekable target, and worker-local disk is RAM), so
# this is what bounds the worker's peak memory.
MAX_POINTS = int(os.getenv("LAKITU_MAX_POINTS", 200_000_000))

# Grid size that geometry is snapped to before differencing, in metres. Repeated
# difference operations otherwise leave slivers along shared edges.
_PRECISION_GRID = 0.01

# How long a loaded catalog is reused, in seconds. USGS publishes new surveys
# periodically, and instances here are long-lived, so caching for the life of
# the process would let two instances answer the same coverage question
# differently for as long as they both stayed warm.
CATALOG_TTL = float(os.getenv("EPT_CATALOG_TTL_SECONDS", 3600))

_catalog: gpd.GeoDataFrame | None = None
_catalog_fetched_on: datetime | None = None


class EptCatalogError(Exception):
    """The 3DEP EPT catalog could not be read from the mirror or the source."""


class DatasetNotFoundError(ValueError):
    """A requested acquisition name does not exist in the 3DEP EPT catalog."""


class DatasetOutsideDomainError(ValueError):
    """A requested acquisition exists but does not overlap the domain."""


@dataclass(frozen=True)
class EptDataset:
    """One 3DEP acquisition selected to cover part of a domain."""

    name: str
    url: str
    count: int
    contribution_fraction: float
    estimated_density: float
    estimated_points: int


@dataclass(frozen=True)
class EptSelection:
    """The acquisitions chosen to cover a domain, and what each one covers.

    ``datasets`` is ordered and ``contributions`` is index-aligned with it. The
    contributions are disjoint: each is the part of the domain that acquisition
    is responsible for, with the parts earlier acquisitions already cover
    removed. Reading each acquisition only within its own contribution is what
    keeps overlapping acquisitions from contributing the same ground twice.
    """

    datasets: list[EptDataset]
    contributions: list[BaseGeometry]
    coverage_fraction: float
    estimated_point_count: int
    catalog_fetched_on: datetime | None = None


def ept_url(name: str) -> str:
    """Return the ``ept.json`` URL for a 3DEP acquisition name."""
    return f"{EPT_BASE_URL}/{name}/ept.json"


def load_catalog(refresh: bool = False) -> gpd.GeoDataFrame:
    """Load the 3DEP EPT acquisition catalog, caching it in the process.

    Reads the GeoParquet mirror first and falls back to the live GeoJSON. The
    result is cached for ``CATALOG_TTL`` seconds, since it is several megabytes
    that change only when USGS publishes new acquisitions.

    Never call this at import time. Worker processes fork after import, and the
    parse is slow enough to matter on an API cold start.

    Args:
        refresh: Re-read the catalog even if the cached copy is still fresh.

    Returns:
        GeoDataFrame in EPSG:4326 with columns ``name``, ``url``, ``count``.

    Raises:
        EptCatalogError: If neither the mirror nor the source can be read.
    """
    global _catalog, _catalog_fetched_on

    fresh = (
        _catalog_fetched_on is not None
        and (datetime.now(UTC) - _catalog_fetched_on).total_seconds() < CATALOG_TTL
    )
    if _catalog is not None and fresh and not refresh:
        return _catalog

    try:
        catalog = gpd.read_parquet(CATALOG_MIRROR)
        logger.info(f"Loaded 3DEP EPT catalog from mirror: {len(catalog)} datasets")
    except Exception as mirror_error:
        logger.warning(
            f"3DEP EPT catalog mirror unavailable ({mirror_error}); "
            "falling back to the live catalog"
        )
        try:
            response = requests.get(CATALOG_URL, timeout=60)
            response.raise_for_status()
            catalog = gpd.read_file(response.text)
        except Exception as source_error:
            raise EptCatalogError(
                f"Could not read the 3DEP EPT catalog from {CATALOG_MIRROR} "
                f"({mirror_error}) or {CATALOG_URL} ({source_error})."
            ) from source_error
        logger.info(f"Loaded 3DEP EPT catalog from source: {len(catalog)} datasets")

    _catalog = catalog
    _catalog_fetched_on = datetime.now(UTC)
    return _catalog


def search_3dep_ept(roi: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Find the 3DEP EPT acquisitions that overlap a region of interest.

    Args:
        roi: Region of interest. Any CRS; results are returned in the ROI's CRS.

    Returns:
        GeoDataFrame of overlapping acquisitions in the ROI's CRS, with columns
        ``name``, ``url``, ``count``, and ``density`` (points per square metre,
        computed over the whole acquisition footprint). Empty if nothing
        overlaps.

    Raises:
        EptCatalogError: If the catalog cannot be read.
    """
    catalog = load_catalog()

    # Query in the catalog's own CRS rather than assuming one. The published
    # catalog is EPSG:4326 and the mirror preserves that, but nothing in the
    # format guarantees it.
    search_area = roi.to_crs(catalog.crs).union_all()

    hits = catalog.iloc[list(catalog.sindex.query(search_area, predicate="intersects"))]
    if hits.empty:
        return hits.to_crs(roi.crs)

    # Acquisition footprints can span a whole state, so their areas are measured
    # in an equal-area projection rather than the domain's local CRS.
    footprint_areas = hits.to_crs(EQUAL_AREA_CRS).area
    hits = hits.to_crs(roi.crs).copy()
    hits["density"] = (hits["count"] / footprint_areas.values).astype(float)
    return hits


def select_datasets(
    roi: gpd.GeoDataFrame,
    candidates: gpd.GeoDataFrame,
    pinned: list[str] | None = None,
) -> EptSelection:
    """Choose which acquisitions to read, and what part of the domain from each.

    With no pin, prefers a single acquisition that covers essentially all of the
    domain — fewer acquisitions means fewer seams. Failing that, it fills the
    domain greedily by marginal coverage: repeatedly take whichever remaining
    acquisition covers the most of what is still uncovered, breaking ties by
    density and then by name so the result is deterministic.

    With a pin, the listed acquisitions are used in the order given and nothing
    else is considered. Order is meaningful: where two pinned acquisitions
    overlap, the earlier one wins.

    Args:
        roi: Region of interest. Its CRS is used for all geometry operations.
        candidates: Output of ``search_3dep_ept`` for the same ROI.
        pinned: Acquisition names to use, in priority order. None to choose
            automatically.

    Returns:
        The selected acquisitions and their disjoint contributions. Empty
        ``datasets`` with ``coverage_fraction`` 0.0 when nothing covers the ROI.

    Raises:
        DatasetNotFoundError: A pinned name is not in the catalog.
        DatasetOutsideDomainError: A pinned acquisition does not overlap the ROI.
    """
    roi_geom = _clean(roi.union_all())
    roi_area = roi_geom.area

    if pinned:
        ordered = _order_by_pin(candidates, pinned)
    elif candidates.empty:
        ordered = []
    else:
        ordered = _order_by_coverage(candidates, roi_geom, roi_area)

    datasets: list[EptDataset] = []
    contributions: list[BaseGeometry] = []
    remaining = roi_geom

    for name, url, count, density in ordered:
        if remaining.is_empty or remaining.area / roi_area < COVERAGE_EPSILON:
            break

        footprint = _clean(_footprint(candidates, name))
        contribution = _clean(remaining.intersection(footprint))
        fraction = contribution.area / roi_area
        if contribution.is_empty or fraction < MIN_CONTRIBUTION_FRACTION:
            continue

        estimated_points = int(density * contribution.area)
        datasets.append(
            EptDataset(
                name=name,
                url=url,
                count=int(count),
                contribution_fraction=fraction,
                estimated_density=float(density),
                estimated_points=estimated_points,
            )
        )
        contributions.append(contribution)
        remaining = _clean(remaining.difference(footprint))

    coverage_fraction = sum(d.contribution_fraction for d in datasets)
    return EptSelection(
        datasets=datasets,
        contributions=contributions,
        # Contributions are disjoint by construction, so summing them is a true
        # union. Summing raw per-acquisition overlaps would not be: acquisitions
        # overlap each other freely and routinely sum past 100%.
        coverage_fraction=min(coverage_fraction, 1.0),
        estimated_point_count=sum(d.estimated_points for d in datasets),
        catalog_fetched_on=_catalog_fetched_on,
    )


def _order_by_pin(
    candidates: gpd.GeoDataFrame, pinned: list[str]
) -> list[tuple[str, str, int, float]]:
    """Order candidates by an explicit user pin, validating every name."""
    known = set(load_catalog()["name"])
    overlapping = set() if candidates.empty else set(candidates["name"])

    ordered = []
    for name in pinned:
        if name not in known:
            raise DatasetNotFoundError(
                f"'{name}' is not a USGS 3DEP lidar acquisition."
            )
        if name not in overlapping:
            raise DatasetOutsideDomainError(
                f"3DEP acquisition '{name}' does not overlap this domain."
            )
        row = candidates.loc[candidates["name"] == name].iloc[0]
        ordered.append((name, row["url"], row["count"], row["density"]))
    return ordered


def _order_by_coverage(
    candidates: gpd.GeoDataFrame, roi_geom: BaseGeometry, roi_area: float
) -> list[tuple[str, str, int, float]]:
    """Order candidates automatically: single near-complete, else marginal greedy."""
    rows = {
        row["name"]: (row["name"], row["url"], row["count"], float(row["density"]))
        for _, row in candidates.iterrows()
    }
    footprints = {row["name"]: _clean(row.geometry) for _, row in candidates.iterrows()}
    density = {name: rows[name][3] for name in rows}

    complete = [
        name
        for name, geom in footprints.items()
        if roi_geom.intersection(geom).area / roi_area >= SINGLE_DATASET_THRESHOLD
    ]
    if complete:
        # Densest wins among acquisitions that each cover the domain on their own.
        return [rows[max(complete, key=lambda n: (density[n], n))]]

    ordered = []
    remaining = roi_geom
    pool = list(rows)

    while pool and not remaining.is_empty:
        marginal = {n: remaining.intersection(footprints[n]).area for n in pool}
        best = max(pool, key=lambda n: (marginal[n], density[n], n))
        if marginal[best] / roi_area < MIN_CONTRIBUTION_FRACTION:
            break

        ordered.append(rows[best])
        remaining = _clean(remaining.difference(footprints[best]))
        pool.remove(best)

    return ordered


def _footprint(candidates: gpd.GeoDataFrame, name: str) -> BaseGeometry:
    """Return the footprint geometry of a named acquisition."""
    return candidates.loc[candidates["name"] == name].geometry.iloc[0]


def _clean(geom: BaseGeometry) -> BaseGeometry:
    """Snap a geometry to a fixed grid and keep only its polygonal parts.

    Differencing footprints repeatedly produces slivers, and can yield a
    GeometryCollection carrying stray lines and points that later point-in-
    polygon tests cannot consume. Snapping first makes shared edges exactly
    shared so adjacent contributions neither overlap nor leave a gap.
    """
    geom = make_valid(set_precision(geom, _PRECISION_GRID))
    if isinstance(geom, GeometryCollection):
        parts = [g for g in geom.geoms if isinstance(g, (Polygon, MultiPolygon))]
        geom = MultiPolygon(
            [
                p
                for g in parts
                for p in (g.geoms if isinstance(g, MultiPolygon) else [g])
            ]
        )
    return geom
