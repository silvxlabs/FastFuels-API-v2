"""
Handler dispatch for Features.

Routes feature generation requests to the appropriate handler based on
feature type and source product.
"""

import geopandas as gpd

from feature.errors import ProcessingError
from feature.handlers import road, water


def dispatch_handler(
    feature: dict, domain_gdf: gpd.GeoDataFrame, progress_callback
) -> dict:
    """Route to appropriate handler based on feature type and source product.

    Returns dict with 'georeference' key.
    """
    feature_type = feature.get("type")
    source = feature.get("source", {})
    product = source.get("product")

    match feature_type:
        case "road":
            match product:
                case "osm":
                    return road.handle_osm(
                        feature, source, domain_gdf, progress_callback
                    )
                case _:
                    raise ProcessingError(
                        code="UNKNOWN_PRODUCT",
                        message=f"Unknown product '{product}' for road features.",
                        suggestion="Currently only 'osm' is supported for roads.",
                    )

        case "water":
            match product:
                case "osm":
                    return water.handle_osm(
                        feature, source, domain_gdf, progress_callback
                    )
                case _:
                    raise ProcessingError(
                        code="UNKNOWN_PRODUCT",
                        message=f"Unknown product '{product}' for water features.",
                        suggestion="Currently only 'osm' is supported for water.",
                    )

        case _:
            raise ProcessingError(
                code="UNKNOWN_FEATURE_TYPE",
                message=f"Unknown feature type: {feature_type}",
                suggestion="Supported feature types are 'road' and 'water'.",
            )
