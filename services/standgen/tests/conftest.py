"""
Shared pytest fixtures for standgen tests.
"""

import geopandas as gpd
import pytest
from shapely.geometry import box


@pytest.fixture
def domain_gdf():
    return gpd.GeoDataFrame(
        geometry=[box(-105.0, 40.0, -104.99, 40.01)],
        crs="EPSG:4326",
    )
