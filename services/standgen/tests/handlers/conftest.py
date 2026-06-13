"""Shared fixtures for standgen handler tests.

These mock a domain and a tree dataframe in the shape handlers consume. Defined
here rather than in a single test module so every handler test (chm, gdam, ...)
reuses one definition instead of copying it.
"""

import dask.dataframe as dd
import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box


@pytest.fixture
def mock_domain_gdf():
    """A domain polygon in a metric CRS (EPSG:32610 is a UTM zone, meters)."""
    return gpd.GeoDataFrame(geometry=[box(0, 0, 100, 100)], crs="EPSG:32610")


@pytest.fixture
def mock_trees_ddf():
    """A minimal tree Dask DataFrame (x, y, height) as handlers produce/consume."""
    pdf = pd.DataFrame({"x": [10.0, 50.0], "y": [20.0, 60.0], "height": [15.0, 25.0]})
    return dd.from_pandas(pdf, npartitions=1)
