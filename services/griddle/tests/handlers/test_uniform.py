"""
Tests for uniform grid handler.
"""

import json
from dataclasses import dataclass
from unittest.mock import MagicMock

import geopandas as gpd
import numpy as np
import pytest
import rioxarray  # noqa: F401
import xarray as xr
from griddle.handlers.uniform import create_uniform_grid
from pyproj import CRS
from shapely.geometry import box

from lib.testing import SHARED_TEST_DOMAINS_DIR

DOMAINS_DIR = SHARED_TEST_DOMAINS_DIR


@dataclass
class DomainFixture:
    """Test domain configuration."""

    name: str
    description: str
    file_name: str


DOMAIN_FIXTURES = [
    DomainFixture(
        name="Blue Mountain",
        description="~1 sq km in Blue Mountain Recreation Area, Montana",
        file_name="blue_mtn.json",
    ),
]


@pytest.fixture(params=DOMAIN_FIXTURES, ids=[d.name for d in DOMAIN_FIXTURES])
def test_domain(request) -> DomainFixture:
    """Return test domain configuration."""
    return request.param


@pytest.fixture
def roi(test_domain) -> gpd.GeoDataFrame:
    """Load domain as a GeoDataFrame."""
    with open(DOMAINS_DIR / test_domain.file_name) as f:
        domain = json.load(f)

    crs = domain["crs"]["properties"]["name"]
    return gpd.GeoDataFrame.from_features(domain["features"], crs=crs)


@pytest.fixture
def projected_domain():
    """A small domain in UTM (EPSG:32611) with a known shape (20x10) at 10m
    resolution."""
    geom = box(300000, 4100000, 300100, 4100200)
    return gpd.GeoDataFrame(geometry=[geom], crs="EPSG:32611")


class TestCreateUniformGrid:
    """Tests for create_uniform_grid function."""

    def test_returns_dataset(self, roi):
        """create_uniform_grid returns an xr.Dataset."""
        bands = [{"quantity": "fuel_moisture.1hr", "value": 6.0}]
        progress = MagicMock()

        result = create_uniform_grid(roi, bands, 10.0, progress)

        assert isinstance(result, xr.Dataset)

    def test_variable_names_match_band_keys(self, roi):
        """Dataset variable names match the quantity keys."""
        bands = [
            {"quantity": "fuel_moisture.1hr", "value": 6.0},
            {"quantity": "fuel_load.1hr", "value": 0.15},
        ]
        progress = MagicMock()

        result = create_uniform_grid(roi, bands, 10.0, progress)

        assert list(result.data_vars) == ["fuel_moisture.1hr", "fuel_load.1hr"]

    def test_values_are_constant(self, roi):
        """Each variable is filled with the constant value."""
        bands = [
            {"quantity": "fuel_moisture.1hr", "value": 6.0},
            {"quantity": "fuel_load.1hr", "value": 0.15},
        ]
        progress = MagicMock()

        result = create_uniform_grid(roi, bands, 10.0, progress)

        np.testing.assert_array_equal(
            result["fuel_moisture.1hr"].values,
            np.full(result["fuel_moisture.1hr"].shape, 6.0),
        )
        np.testing.assert_array_almost_equal(
            result["fuel_load.1hr"].values,
            np.full(result["fuel_load.1hr"].shape, 0.15),
        )

    def test_int_value_dtype(self, roi):
        """Integer values produce int32 arrays."""
        bands = [{"quantity": "fuel_moisture.1hr", "value": 6}]
        progress = MagicMock()

        result = create_uniform_grid(roi, bands, 10.0, progress)

        assert result["fuel_moisture.1hr"].dtype == np.int32

    def test_float_value_dtype(self, roi):
        """Float values produce float64 arrays."""
        bands = [{"quantity": "fuel_load.1hr", "value": 0.15}]
        progress = MagicMock()

        result = create_uniform_grid(roi, bands, 10.0, progress)

        assert result["fuel_load.1hr"].dtype == np.float64

    def test_crs_set_correctly(self, roi):
        """CRS is written to the dataset via rioxarray."""
        bands = [{"quantity": "fuel_moisture.1hr", "value": 6.0}]
        progress = MagicMock()

        result = create_uniform_grid(roi, bands, 10.0, progress)

        assert result.rio.crs == roi.crs

    def test_transform_set(self, roi):
        """Affine transform is written to the dataset."""
        bands = [{"quantity": "fuel_moisture.1hr", "value": 6.0}]
        progress = MagicMock()

        result = create_uniform_grid(roi, bands, 10.0, progress)

        transform = result["fuel_moisture.1hr"].rio.transform()
        assert transform is not None
        # Resolution should be 10m
        assert abs(transform.a) == pytest.approx(10.0)
        assert abs(transform.e) == pytest.approx(10.0)

    def test_grid_shape_matches_resolution(self, projected_domain):
        """Grid dimensions match domain extent / resolution."""
        # Domain is 100m x 200m, resolution 10m -> 10 x 20
        bands = [{"quantity": "fuel_moisture.1hr", "value": 6.0}]
        progress = MagicMock()

        result = create_uniform_grid(projected_domain, bands, 10.0, progress)

        assert result.rio.width == 10
        assert result.rio.height == 20

    def test_crs_matches_domain(self, roi):
        """Output CRS matches the domain's projected CRS."""
        bands = [{"quantity": "fuel_moisture.1hr", "value": 6.0}]
        progress = MagicMock()

        result = create_uniform_grid(roi, bands, 10.0, progress)

        assert result.rio.crs == CRS.from_epsg(32611)

    def test_calls_progress_callback(self, roi):
        """create_uniform_grid reports progress."""
        bands = [{"quantity": "fuel_moisture.1hr", "value": 6.0}]
        progress = MagicMock()

        create_uniform_grid(roi, bands, 10.0, progress)

        assert progress.call_count >= 2

    def test_multiple_bands_same_shape(self, roi):
        """All bands in a multi-band grid have the same shape."""
        bands = [
            {"quantity": "fuel_moisture.1hr", "value": 6.0},
            {"quantity": "fuel_moisture.10hr", "value": 8.0},
            {"quantity": "fuel_depth", "value": 0.3},
        ]
        progress = MagicMock()

        result = create_uniform_grid(roi, bands, 10.0, progress)

        shapes = [result[var].shape for var in result.data_vars]
        assert all(s == shapes[0] for s in shapes)
