"""Shared fixtures and constants for standgen handler tests.

These mock a domain and a tree dataframe in the shape handlers consume. Defined
here rather than in a single test module so every handler test (chm, gdam, ...)
reuses one definition instead of copying it.
"""

import dask.dataframe as dd
import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

BASE_INVENTORY_COLUMNS = [
    {"key": "x", "type": "continuous", "unit": "m"},
    {"key": "y", "type": "continuous", "unit": "m"},
    {"key": "fia_species_code", "type": "categorical"},
    {"key": "fia_status_code", "type": "categorical"},
    {"key": "dbh", "type": "continuous", "unit": "cm"},
    {"key": "height", "type": "continuous", "unit": "m"},
    {"key": "crown_ratio", "type": "continuous"},
]

CHM_INVENTORY_COLUMNS = [
    {"key": "x", "type": "continuous", "unit": "m"},
    {"key": "y", "type": "continuous", "unit": "m"},
    {"key": "height", "type": "continuous", "unit": "m"},
]


@pytest.fixture
def mock_domain_gdf():
    """A domain polygon in a metric CRS (EPSG:32610 is a UTM zone, meters)."""
    return gpd.GeoDataFrame(geometry=[box(0, 0, 100, 100)], crs="EPSG:32610")


@pytest.fixture
def mock_trees_ddf():
    """A minimal tree Dask DataFrame (x, y, height) as handlers produce/consume."""
    pdf = pd.DataFrame({"x": [10.0, 50.0], "y": [20.0, 60.0], "height": [15.0, 25.0]})
    return dd.from_pandas(pdf, npartitions=1)


"""Shared fixtures and constants for standgen handler tests."""
