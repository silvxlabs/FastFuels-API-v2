"""
Pacific Northwest flag for the uncompacted crown ratio model.

The core model has separate coefficient rows for California, Oregon and
Washington. A job gets the flag from its domain centroid only; a domain that
straddles the region boundary uses whichever side its centroid falls on.
"""

from functools import cache
from pathlib import Path

import geopandas as gpd
from shapely.geometry.base import BaseGeometry

_REGION_PATH = Path(__file__).parent / "data" / "pacific_northwest_4326.geojson"


@cache
def _pacific_northwest() -> BaseGeometry:
    return gpd.read_file(_REGION_PATH).geometry.union_all()


def in_pacific_northwest(domain_gdf: gpd.GeoDataFrame) -> bool:
    """Return True if the domain centroid lies in California, Oregon or Washington."""
    centroid = domain_gdf.geometry.union_all().centroid
    point = gpd.GeoSeries([centroid], crs=domain_gdf.crs).to_crs(4326).iloc[0]
    return bool(_pacific_northwest().covers(point))
