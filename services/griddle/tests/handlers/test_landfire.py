"""
Tests for LANDFIRE handler.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock

import geopandas as gpd
import pytest
import rioxarray  # noqa: F401
import xarray as xr
from griddle.handlers.landfire import fetch_fbfm40, fetch_topography

TEST_DATA_DIR = Path(__file__).parent.parent / "data"
DOMAINS_DIR = TEST_DATA_DIR / "domains"


@dataclass
class DomainFixture:
    """Test domain configuration."""

    name: str
    description: str
    file_name: str
    expected_shape: tuple  # (y, x)


DOMAIN_FIXTURES = [
    DomainFixture(
        name="Blue Mountain",
        description="~1 sq km in Blue Mountain Recreation Area, Montana",
        file_name="blue_mtn.json",
        expected_shape=(47, 61),
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


class TestFetchFbfm40:
    """Integration tests for fetch_fbfm40."""

    def test_returns_dataset(self, roi):
        """fetch_fbfm40 returns a Dataset."""
        result = fetch_fbfm40(roi=roi)
        assert isinstance(result, xr.Dataset)

    def test_has_fbfm_variable(self, roi):
        """Dataset contains a 'fbfm' variable."""
        result = fetch_fbfm40(roi=roi)
        assert "fbfm" in result.data_vars

    def test_fbfm_shape(self, test_domain, roi):
        """The fbfm variable has the expected spatial shape."""
        result = fetch_fbfm40(roi=roi)
        assert result["fbfm"].shape == test_domain.expected_shape

    def test_fbfm_dtype(self, roi):
        """The fbfm variable is int16 (categorical codes)."""
        result = fetch_fbfm40(roi=roi)
        assert result["fbfm"].dtype == "int16"

    def test_crs_preserved(self, roi):
        """CRS is preserved via rioxarray."""
        result = fetch_fbfm40(roi=roi)
        assert result.rio.crs == roi.crs

    def test_fbfm_values_in_range(self, roi):
        """FBFM40 codes should be <= 204."""
        result = fetch_fbfm40(roi=roi)
        assert result["fbfm"].values.max() <= 204


class TestFetchTopography:
    """Integration tests for fetch_topography."""

    def test_returns_dataset(self, roi):
        """fetch_topography always returns a Dataset."""
        progress = MagicMock()
        result = fetch_topography(
            roi=roi,
            version="2020",
            bands=["elevation"],
            progress=progress,
        )
        assert isinstance(result, xr.Dataset)

    def test_fetch_all_bands(self, test_domain, roi):
        """fetch_topography with all bands returns three variables."""
        progress = MagicMock()
        result = fetch_topography(
            roi=roi,
            version="2020",
            bands=["elevation", "slope", "aspect"],
            progress=progress,
        )

        assert list(result.data_vars) == ["elevation", "slope", "aspect"]
        for var in result.data_vars:
            assert result[var].shape == test_domain.expected_shape
        assert result.rio.crs == roi.crs

    def test_fetch_single_band(self, test_domain, roi):
        """fetch_topography with one band returns Dataset with one variable."""
        progress = MagicMock()
        result = fetch_topography(
            roi=roi,
            version="2020",
            bands=["elevation"],
            progress=progress,
        )

        assert list(result.data_vars) == ["elevation"]
        assert result["elevation"].shape == test_domain.expected_shape
        assert result.rio.crs == roi.crs

    def test_fetch_two_bands(self, test_domain, roi):
        """fetch_topography with two bands returns correct subset."""
        progress = MagicMock()
        result = fetch_topography(
            roi=roi,
            version="2020",
            bands=["slope", "aspect"],
            progress=progress,
        )

        assert list(result.data_vars) == ["slope", "aspect"]

    def test_variable_names_match_request(self, roi):
        """Variable names match the requested band names and order."""
        progress = MagicMock()
        result = fetch_topography(
            roi=roi,
            version="2020",
            bands=["aspect", "elevation"],
            progress=progress,
        )

        assert list(result.data_vars) == ["aspect", "elevation"]

    def test_elevation_data_is_numeric(self, roi):
        """Elevation values are numeric (float or int)."""
        progress = MagicMock()
        result = fetch_topography(
            roi=roi,
            version="2020",
            bands=["elevation"],
            progress=progress,
        )

        assert result["elevation"].dtype.kind in ("f", "i")

    def test_calls_progress_callback(self, roi):
        """fetch_topography reports progress for each band."""
        progress = MagicMock()
        fetch_topography(
            roi=roi,
            version="2020",
            bands=["elevation", "slope"],
            progress=progress,
        )

        assert progress.call_count == 2
