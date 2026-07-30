"""
USGS 3DEP point cloud handler.

Reads public airborne lidar from 3DEP for a domain, clips it to the domain,
reprojects it to the domain's CRS, and returns it as a single LAZ.

This is the point cloud side of 3DEP. The elevation raster side lives in
griddle and shares nothing with it but the program name.
"""

import logging
from collections.abc import Callable

import geopandas as gpd
import numpy as np
import shapely
from pyproj import CRS, Transformer
from shapely.geometry.base import BaseGeometry

from lakitu.ept import fetch_metadata, fetch_nodes, make_session, walk_hierarchy
from lib.entwine import MAX_POINTS, EptSelection, search_3dep_ept, select_datasets
from lib.errors import ProcessingError
from lib.laz import (
    LazAccumulator,
    build_output_header,
    normalize_record,
    point_format_id_for_dimensions,
)

logger = logging.getLogger(__name__)

# 3DEP publishes orthometric heights on NAVD88. Elevations pass through
# untouched, so this is recorded rather than applied.
VERTICAL_DATUM = "NAVD88"
VERTICAL_CRS = "EPSG:5703"

# Maximum length of a domain edge, in metres, before it is broken into segments
# for reprojection. A straight line in the domain's CRS is a curve in the
# acquisition's, and reprojecting only the corners would clip the bulge.
_SEGMENT_LENGTH = 100.0


def handle_3dep(
    point_cloud: dict,
    source: dict,
    domain_gdf: gpd.GeoDataFrame,
    progress: Callable[[str, int | None], None],
) -> dict:
    """Fetch, clip, and reproject 3DEP lidar for a domain.

    Args:
        point_cloud: Point cloud document from Firestore.
        source: The document's source block, carrying any pinned acquisitions.
        domain_gdf: Domain geometry, in the domain's CRS.
        progress: Progress callback (message, percent).

    Returns:
        Dict with ``buffer``, ``georeference``, ``summary``, and ``source_extra``.

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
        query = _query_box(selection.contributions[index], domain_crs, meta.crs)
        nodes = walk_hierarchy(session, meta, query)
        metadata.append(meta)
        all_nodes.append((index, nodes))
        total_points += sum(node.count for node in nodes)

    # The index gives an exact upper bound on what the fetch would read, and
    # masking only ever removes points. Checking here costs one index walk and
    # avoids downloading gigabytes we would have to throw away.
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

    accumulator, written = _read_points(
        session, selection, metadata, all_nodes, domain_geom, domain_crs, progress
    )

    if written == 0:
        raise ProcessingError(
            code="EMPTY_POINT_CLOUD",
            message="No 3DEP lidar points fall inside this domain.",
            suggestion=("Check coverage for the domain before creating a point cloud."),
        )

    progress("Writing point cloud", 85)
    buffer, summary, bounds = accumulator.finish()

    return {
        "buffer": buffer,
        "georeference": {
            "crs": _crs_name(domain_crs),
            "vertical_crs": VERTICAL_CRS,
            "bounds": bounds,
        },
        "summary": summary,
        "source_extra": {
            "datasets": [d.name for d in selection.datasets],
            "coverage_fraction": selection.coverage_fraction,
            "vertical_datum": VERTICAL_DATUM,
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
    """
    pinned = source.get("requested_datasets")
    selection = select_datasets(domain_gdf, search_3dep_ept(domain_gdf), pinned=pinned)
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


def _query_box(
    contribution: BaseGeometry, domain_crs: CRS, target_crs: CRS
) -> tuple[float, float, float, float]:
    """Bounding box of a contribution in the acquisition's CRS.

    The contribution is densified before reprojection: its edges are straight
    in the domain's CRS but curved in the acquisition's, and transforming only
    the corners would produce a box that cuts the curve off.
    """
    densified = shapely.segmentize(contribution, max_segment_length=_SEGMENT_LENGTH)
    reprojected = (
        gpd.GeoSeries([densified], crs=domain_crs).to_crs(target_crs).total_bounds
    )
    return tuple(reprojected)


def _read_points(
    session,
    selection: EptSelection,
    metadata: list,
    all_nodes: list[tuple[int, list]],
    domain_geom: BaseGeometry,
    domain_crs: CRS,
    progress: Callable[[str, int | None], None],
) -> tuple[LazAccumulator, int]:
    """Read every selected node, clip it, and accumulate it into one LAZ."""
    # Point formats can differ between acquisitions, and laspy refuses to write
    # records whose format differs from the file's, so the output format is
    # decided once up front from what the sources declare they carry.
    point_format_id = point_format_id_for_dimensions(
        name for meta in metadata for name in meta.dimension_names
    )
    header = build_output_header(
        domain_crs, domain_geom.bounds, point_format_id=point_format_id
    )
    accumulator = LazAccumulator(header)

    # Contributions are only needed to arbitrate between acquisitions. With one
    # acquisition the domain shape alone decides, and testing the polygon is
    # far more expensive than the rectangle it sits in.
    multi_source = len(selection.datasets) > 1
    axis_aligned = domain_geom.equals(domain_geom.envelope)

    total_nodes = sum(len(nodes) for _, nodes in all_nodes)
    done = 0
    last_reported = 0

    for index, nodes in all_nodes:
        if not nodes:
            continue
        transformer = Transformer.from_crs(
            metadata[index].crs, domain_crs, always_xy=True
        )
        clip = selection.contributions[index] if multi_source else domain_geom
        prepared = None if (axis_aligned and not multi_source) else clip
        min_x, min_y, max_x, max_y = clip.bounds

        for node, points in fetch_nodes(session, nodes):
            x, y = transformer.transform(np.asarray(points.x), np.asarray(points.y))
            z = np.asarray(points.z)

            # Cheap rectangle test first; the polygon test only runs for the
            # points that survive it, and only when the shape needs it.
            keep = (x >= min_x) & (x <= max_x) & (y >= min_y) & (y <= max_y)
            if prepared is not None and keep.any():
                # Boundary-exclusive: on a shared edge between two
                # acquisitions, dropping a point is safer than duplicating it.
                keep[keep] = shapely.contains_xy(prepared, x[keep], y[keep])

            if keep.any():
                accumulator.append(
                    normalize_record(
                        points.points[keep], header, x[keep], y[keep], z[keep]
                    )
                )

            done += 1
            percent = 10 + int(70 * done / total_nodes)
            if percent >= last_reported + 5:
                last_reported = percent
                progress(
                    f"Reading 3DEP lidar ({accumulator.point_count:,} points)",
                    percent,
                )

    return accumulator, accumulator.point_count


def _crs_name(crs: CRS) -> str:
    """Return an authority code for a CRS, falling back to WKT."""
    code = crs.to_epsg()
    return f"EPSG:{code}" if code else crs.to_wkt()
