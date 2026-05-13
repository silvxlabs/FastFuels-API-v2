"""
Handler dispatch for Standgen.

Routes inventory requests to the appropriate handler based on source type.
"""

import geopandas as gpd

from lib.errors import ProcessingError
from standgen.handlers import chm, modifications, pim


def dispatch_handler(
    inventory: dict, domain_gdf: gpd.GeoDataFrame, progress_callback
) -> dict:
    """Route to appropriate handler based on inventory source type.

    Returns dict with 'georeference' key.
    """
    source = inventory["source"]
    source_name = source["name"]

    match source_name:
        case "pim":
            return pim.handle_pim(inventory, source, domain_gdf, progress_callback)
        case "chm":
            return chm.handle_chm(inventory, source, domain_gdf, progress_callback)
        case "modifications":
            return modifications.handle_modifications(
                inventory, source, domain_gdf, progress_callback
            )
        case _:
            raise ProcessingError(
                code="UNKNOWN_SOURCE",
                message=f"Unknown source type: {source_name}",
                suggestion="Check that the inventory source type is supported.",
            )
