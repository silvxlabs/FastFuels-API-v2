"""
Handler dispatch for Lakitu.

Routes point cloud requests to the appropriate handler based on source type.
"""

from collections.abc import Callable

import geopandas as gpd

from lakitu.handlers import threedep
from lib.errors import ProcessingError


def dispatch_handler(
    point_cloud: dict,
    domain_gdf: gpd.GeoDataFrame,
    progress_callback: Callable[[str, int | None], None],
) -> dict:
    """Route to the appropriate handler based on point cloud source type.

    Args:
        point_cloud: Point cloud document from Firestore
        domain_gdf: Domain geometry as GeoDataFrame
        progress_callback: Function to report progress (message, percent)

    Returns:
        Dict with 'buffer', 'georeference', 'summary', and 'source_extra' keys

    Raises:
        ProcessingError: If source type is unknown or processing fails
    """
    source = point_cloud["source"]
    source_name = source["name"]

    match source_name:
        case "3dep":
            return threedep.handle_3dep(
                point_cloud, source, domain_gdf, progress_callback
            )
        case _:
            raise ProcessingError(
                code="UNKNOWN_SOURCE",
                message=f"Unknown source type: {source_name}",
                suggestion="Check that the point cloud source type is supported.",
            )
