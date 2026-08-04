"""
USGS 3DEP point cloud handler.

Reads public airborne lidar from 3DEP for a domain, clips it to the domain,
reprojects it to the domain's CRS, and writes it as a partitioned Parquet
dataset.

This is the point cloud side of 3DEP. The elevation raster side lives in
griddle and shares nothing with it but the program name.
"""

import logging
from collections.abc import Callable

import geopandas as gpd
import numpy as np
import shapely
from pyproj import CRS
from shapely.geometry.base import BaseGeometry

from lakitu.chain import stream_records
from lakitu.ept import fetch_metadata, make_session, walk_hierarchy
from lakitu.parquet_writer import write_parquet
from lakitu.storage import cloud_location
from lib.entwine import (
    MAX_POINTS,
    DatasetNotFoundError,
    DatasetOutsideDomainError,
    EptCatalogError,
    EptSelection,
    search_3dep_ept,
    select_datasets,
)
from lib.errors import ProcessingError
from lib.laz import build_output_header, point_format_id_for_dimensions

logger = logging.getLogger(__name__)

# Maximum length of a domain edge, in metres, before it is broken into segments
# for reprojection. A straight line in the domain's CRS is a curve in the
# acquisition's, and reprojecting only the corners would clip the bulge.
_SEGMENT_LENGTH = 100.0


def handle_3dep(
    point_cloud: dict,
    source: dict,
    domain_gdf: gpd.GeoDataFrame,
    progress: Callable[[str, int | None], None],
    point_cloud_id: str,
) -> dict:
    """Fetch, clip, and reproject 3DEP lidar for a domain.

    Args:
        point_cloud: Point cloud document from Firestore.
        source: The document's source block, carrying any pinned acquisitions.
        domain_gdf: Domain geometry, in the domain's CRS.
        progress: Progress callback (message, percent).
        point_cloud_id: Resource id, which decides where the dataset is written.

    Returns:
        Dict with ``georeference``, ``summary``, ``size_bytes``, and
        ``source_extra``. Points are written to GCS here rather than returned:
        the dataset is many files and never exists whole in memory.

    Raises:
        ProcessingError: If no lidar covers the domain, the fetch would exceed
            the point budget, or the acquisition cannot be read.
    """
    domain_crs = domain_gdf.crs
    domain_geom = domain_gdf.union_all()

    progress("Finding available 3DEP lidar", 5)
    selection = _resolve(domain_gdf, source)

    session = make_session()
    metadata = []
    all_nodes: list[tuple[int, list]] = []
    total_points = 0

    progress("Reading 3DEP index", 10)
    for index, dataset in enumerate(selection.datasets):
        meta = fetch_metadata(session, dataset.url)
        projected = _project_contribution(
            selection.contributions[index], domain_crs, meta.crs
        )
        nodes = walk_hierarchy(session, meta, projected.bounds)
        metadata.append(meta)
        all_nodes.append((index, nodes))
        total_points += _estimate_kept_points(nodes, projected)

    # Checking before any node is downloaded costs one index walk and avoids
    # pulling gigabytes that would only be discarded.
    if total_points > MAX_POINTS:
        raise ProcessingError(
            code="POINT_BUDGET_EXCEEDED",
            message=(
                f"This domain covers roughly {total_points:,} 3DEP lidar "
                f"points, more than the {MAX_POINTS:,} point limit for a "
                "single point cloud."
            ),
            suggestion="Use a smaller domain.",
        )

    logger.info(f"Reading {total_points:,} points from {len(metadata)} acquisitions")

    summary, bounds, size_bytes = _write_points(
        session,
        selection,
        metadata,
        all_nodes,
        domain_geom,
        domain_crs,
        progress,
        point_cloud_id,
    )

    if summary["point_count"] == 0:
        raise ProcessingError(
            code="EMPTY_POINT_CLOUD",
            message="No 3DEP lidar points fall inside this domain.",
            suggestion=("Check coverage for the domain before creating a point cloud."),
        )

    return {
        "size_bytes": size_bytes,
        "georeference": {
            "crs": _crs_name(domain_crs),
            "bounds": bounds,
        },
        "summary": summary,
        "source_extra": {
            "datasets": [d.name for d in selection.datasets],
            "coverage_fraction": selection.coverage_fraction,
            "catalog_fetched_on": (
                selection.catalog_fetched_on.isoformat()
                if selection.catalog_fetched_on
                else None
            ),
        },
    }


def _resolve(domain_gdf: gpd.GeoDataFrame, source: dict) -> EptSelection:
    """Re-resolve which acquisitions to read.

    The API resolved this when the request came in, but the catalog is live and
    the job may have queued for a while, so the pin — not the earlier answer —
    is what carries forward.

    Raises:
        ProcessingError: EPT_CATALOG_UNAVAILABLE if the catalog cannot be read,
            COVERAGE_ERROR if nothing usable covers the domain.
    """
    pinned = source.get("requested_datasets")
    try:
        selection = select_datasets(
            domain_gdf, search_3dep_ept(domain_gdf), pinned=pinned
        )
    except EptCatalogError as e:
        # Transient and retryable. Without this the bare Exception handler in
        # main.py turns an upstream outage into UNEXPECTED_FAILURE, which tells
        # the user to contact support about someone else's downtime.
        raise ProcessingError(
            code="EPT_CATALOG_UNAVAILABLE",
            message=(
                "The USGS 3DEP lidar catalog could not be reached. Please try "
                "again shortly."
            ),
            suggestion="Retry the request in a few minutes.",
            traceback=str(e),
        ) from e
    except (DatasetNotFoundError, DatasetOutsideDomainError) as e:
        # The API validates the pin at create time, so reaching here means the
        # catalog changed under a queued job.
        raise ProcessingError(
            code="COVERAGE_ERROR",
            message=str(e),
            suggestion=(
                "Check the acquisitions available for this domain with GET "
                "/domains/{domain_id}/pointclouds/3dep/coverage."
            ),
        ) from e

    if not selection.datasets:
        raise ProcessingError(
            code="COVERAGE_ERROR",
            message="No USGS 3DEP lidar is available for this domain.",
            suggestion=(
                "Check coverage with GET "
                "/domains/{domain_id}/pointclouds/3dep/coverage."
            ),
        )
    return selection


def _project_contribution(
    contribution: BaseGeometry, domain_crs: CRS, target_crs: CRS
) -> BaseGeometry:
    """Reproject a contribution into an acquisition's CRS.

    Densified first: its edges are straight in the domain's CRS but curved in
    the acquisition's, and transforming only the corners would produce a shape
    that cuts the curve off.
    """
    densified = shapely.segmentize(contribution, max_segment_length=_SEGMENT_LENGTH)
    return gpd.GeoSeries([densified], crs=domain_crs).to_crs(target_crs).iloc[0]


def _estimate_kept_points(nodes: list, projected: BaseGeometry) -> int:
    """Estimate how many of a box's points fall inside the contribution itself.

    Nodes are selected by bounding box, but only points inside the acquisition's
    own contribution are kept. Where two acquisitions split a domain along an
    irregular seam, both boxes approximate the whole domain, so summing raw node
    counts charges the whole domain to each acquisition and roughly doubles the
    total — enough to reject a fetch that was comfortably within budget.

    Scaling by the contribution's share of its own bounding box corrects that,
    assuming points are spread evenly. Rough, but it feeds a guard rail rather
    than a reported figure, and both areas are measured in the acquisition's CRS
    so the ratio is dimensionless.
    """
    node_points = sum(node.count for node in nodes)
    if not node_points:
        return 0

    min_x, min_y, max_x, max_y = projected.bounds
    box_area = (max_x - min_x) * (max_y - min_y)
    if box_area <= 0:
        return node_points
    return int(node_points * min(1.0, projected.area / box_area))


def _write_points(
    session,
    selection: EptSelection,
    metadata: list,
    all_nodes: list[tuple[int, list]],
    domain_geom: BaseGeometry,
    domain_crs: CRS,
    progress: Callable[[str, int | None], None],
    point_cloud_id: str,
) -> tuple[dict, list[float], int]:
    """Stream every selected node through the chain and into Parquet.

    Returns:
        ``(summary, bounds, size_bytes)``, where summary carries
        ``point_count``, ``point_classes`` and ``density``, and bounds is
        ``[min_x, min_y, min_z, max_x, max_y, max_z]``.
    """
    # Point formats can differ between acquisitions, and a merged output has to
    # pick one, so it is decided once up front from what the sources declare.
    point_format_id = point_format_id_for_dimensions(
        name for meta in metadata for name in meta.dimension_names
    )
    header_bounds = domain_geom.bounds
    header = build_output_header(
        domain_crs, header_bounds, point_format_id=point_format_id
    )

    # Contributions are only needed to arbitrate between acquisitions. With one
    # acquisition the domain shape alone decides, and testing the polygon is far
    # more expensive than the rectangle it sits in.
    multi_source = len(selection.datasets) > 1
    axis_aligned = domain_geom.equals(domain_geom.envelope)

    plan = []
    for index, nodes in all_nodes:
        clip = selection.contributions[index] if multi_source else domain_geom
        polygon = None if (axis_aligned and not multi_source) else clip
        plan.append((metadata[index], nodes, polygon, clip.bounds))

    # The writer partitions on the domain's own horizontal extent. Elevations
    # come from the sources: horizontal reprojection leaves them untouched, so
    # the source z-range transfers directly, and build_output_header reads only
    # the two horizontal minima so it cannot supply them.
    z_low = min(meta.bounds_conforming[2] for meta in metadata)
    z_high = max(meta.bounds_conforming[5] for meta in metadata)
    info = {
        "mins": np.array([header_bounds[0], header_bounds[1], z_low]),
        "maxs": np.array([header_bounds[2], header_bounds[3], z_high]),
        "scales": np.asarray(header.scales),
        "offsets": np.asarray(header.offsets),
    }

    stats = _PointStats(info["scales"], info["offsets"])
    total_nodes = sum(len(nodes) for _, nodes in all_nodes)
    reporter = _NodeProgress(progress, total_nodes, stats)

    records = stream_records(
        session,
        plan,
        domain_crs.to_wkt(),
        header_bounds,
        point_format_id,
        on_node=reporter,
    )

    progress("Writing point cloud", 15)
    bucket, prefix = cloud_location(point_cloud_id)
    result = write_parquet(stats.observe(records), info, bucket, prefix)
    return stats.summary(), stats.bounds(), result["output_bytes"]


class _PointStats:
    """Folds written points into the statistics the resource reports.

    Accumulated on the way past rather than read back afterwards, so what is
    reported always describes what was stored.
    """

    def __init__(self, scales, offsets):
        self._scales = np.asarray(scales)
        self._offsets = np.asarray(offsets)
        self.count = 0
        # Classification is a uint8, so a flag per value beats accumulating a
        # set: no sort, no Python-level set union, per record.
        self._seen_class = np.zeros(256, dtype=bool)
        self._mins = np.full(3, np.iinfo(np.int32).max, dtype=np.int64)
        self._maxs = np.full(3, np.iinfo(np.int32).min, dtype=np.int64)

    def observe(self, records):
        """Fold each record's extremes in, then pass it straight through.

        Reduces over the stored millimetre integers and scales only the six
        surviving scalars at the end. Scaling every point here would repeat, on
        the busiest thread in the process, work the writer already does to route
        the point.
        """
        for record in records:
            for axis, name in enumerate(("X", "Y", "Z")):
                column = record[name]
                self._mins[axis] = min(self._mins[axis], int(column.min()))
                self._maxs[axis] = max(self._maxs[axis], int(column.max()))
            self._seen_class[record["classification"]] = True
            self.count += record.size
            yield record

    def bounds(self) -> list[float]:
        if self.count == 0:
            return [*self._offsets.tolist(), *self._offsets.tolist()]
        mins = self._mins * self._scales + self._offsets
        maxs = self._maxs * self._scales + self._offsets
        return [*mins.tolist(), *maxs.tolist()]

    def summary(self) -> dict:
        bounds = self.bounds()
        area = (bounds[3] - bounds[0]) * (bounds[4] - bounds[1]) if self.count else 0.0
        return {
            "point_count": self.count,
            "point_classes": [int(c) for c in np.flatnonzero(self._seen_class)],
            "density": float(self.count / area) if area > 0 else 0.0,
        }


class _NodeProgress:
    """Reports read progress over the 15-85% band, at most every 5%."""

    def __init__(self, progress, total_nodes, stats):
        self._progress = progress
        self._total = max(total_nodes, 1)
        self._stats = stats
        self._done = 0
        self._last = 0

    def __call__(self) -> None:
        self._done += 1
        percent = 15 + int(70 * self._done / self._total)
        if percent >= self._last + 5:
            self._last = percent
            self._progress(
                f"Reading 3DEP lidar ({self._stats.count:,} points)", percent
            )


def _crs_name(crs: CRS) -> str:
    """Return an authority code for a CRS, falling back to WKT."""
    code = crs.to_epsg()
    return f"EPSG:{code}" if code else crs.to_wkt()
