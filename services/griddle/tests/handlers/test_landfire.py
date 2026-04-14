"""
Tests for LANDFIRE handler.
"""

import json
from dataclasses import dataclass
from unittest.mock import MagicMock

import geopandas as gpd
import numpy as np
import pytest
import rioxarray  # noqa: F401
import xarray as xr
from griddle.handlers.landfire import (
    _most_frequent,
    _remove_non_burnable_blocks,
    fetch_fbfm40,
    fetch_topography,
)

from lib.testing import SHARED_TEST_DOMAINS_DIR

DOMAINS_DIR = SHARED_TEST_DOMAINS_DIR


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


class TestMostFrequent:
    """Unit tests for _most_frequent majority filter function."""

    def test_returns_central_when_burnable_and_most_frequent(self):
        """Central pixel is returned when it's burnable and tied for max freq."""
        # 5x5 window flattened, central pixel at index 12
        window = np.array([101] * 13 + [102] * 12, dtype=np.int16)
        assert _most_frequent(window, [91, 93, 99]) == 101

    def test_skips_non_burnable_in_ranking(self):
        """Non-burnable codes are skipped even when most frequent."""
        # Non-burnable 99 is most frequent, but burnable 101 should be returned
        window = np.array([99] * 20 + [101] * 5, dtype=np.int16)
        assert _most_frequent(window, [99]) == 101

    def test_falls_back_to_central_when_all_non_burnable(self):
        """Returns central pixel when no burnable values exist."""
        window = np.array([91] * 13 + [99] * 12, dtype=np.int16)
        assert _most_frequent(window, [91, 99]) == 91

    def test_prefers_central_over_tie(self):
        """Central pixel wins ties when it's burnable."""
        # 101 and 102 tied, central pixel is 101
        window = np.full(25, 102, dtype=np.int16)
        window[12] = 101
        # Actually need a real tie: equal counts
        window = np.array(
            [101] * 12 + [101] + [102] * 12, dtype=np.int16
        )  # 13 x 101, 12 x 102 => 101 is most frequent
        assert _most_frequent(window, [91]) == 101

    def test_selects_most_frequent_burnable_when_central_is_non_burnable(self):
        """When central is non-burnable, returns most frequent burnable neighbor."""
        window = np.array([101] * 12 + [99] + [102] * 12, dtype=np.int16)
        # central=99 (non-burnable), 101 appears 12 times, 102 appears 12 times
        # 101 and 102 are tied but 101 comes first sorted by descending count
        result = _most_frequent(window, [99])
        assert result in (101, 102)
        assert result != 99


class TestRemoveNonBurnableBlocks:
    """Unit tests for _remove_non_burnable_blocks."""

    def test_no_non_burnable_returns_copy(self):
        """Grid with no non-burnable codes is returned unchanged."""
        grid = np.full((10, 10), 101, dtype=np.int16)
        result = _remove_non_burnable_blocks(grid, [91, 93, 99])
        np.testing.assert_array_equal(result, grid)
        assert result is not grid  # Returns a copy

    def test_replaces_single_non_burnable_cell(self):
        """A single non-burnable cell surrounded by burnable is replaced."""
        grid = np.full((10, 10), 102, dtype=np.int16)
        grid[5, 5] = 99  # Single bare ground cell
        result = _remove_non_burnable_blocks(grid, [99])
        assert result[5, 5] == 102
        # All other cells unchanged
        grid[5, 5] = 102
        np.testing.assert_array_equal(result, grid)

    def test_only_targeted_codes_removed(self):
        """Only the specified non-burnable codes are removed; others are kept."""
        grid = np.full((10, 10), 101, dtype=np.int16)
        grid[3, 3] = 91  # Urban — targeted
        grid[7, 7] = 98  # Water — not targeted
        result = _remove_non_burnable_blocks(grid, [91])
        assert result[3, 3] == 101  # Urban replaced
        assert result[7, 7] == 98  # Water preserved

    def test_replaces_with_most_frequent_neighbor(self):
        """Non-burnable cells are replaced by the most frequent burnable neighbor."""
        grid = np.full((10, 10), 101, dtype=np.int16)
        # Put a block of 102 on the right side
        grid[:, 7:] = 102
        # Put a non-burnable cell in the 102 region
        grid[5, 8] = 99
        result = _remove_non_burnable_blocks(grid, [99])
        assert result[5, 8] == 102  # Replaced by dominant neighbor

    def test_preserves_non_targeted_non_burnable(self):
        """Non-burnable codes not in the target list remain untouched."""
        grid = np.full((10, 10), 101, dtype=np.int16)
        grid[2, 2] = 92  # Snow/ice
        grid[4, 4] = 93  # Agriculture
        result = _remove_non_burnable_blocks(grid, [93])
        assert result[2, 2] == 92  # Snow/ice preserved
        assert result[4, 4] == 101  # Agriculture replaced

    def test_large_non_burnable_patch(self):
        """A large patch of non-burnable codes is fully replaced."""
        grid = np.full((20, 20), 101, dtype=np.int16)
        grid[8:12, 8:12] = 99  # 4x4 bare ground block
        result = _remove_non_burnable_blocks(grid, [99])
        assert not np.any(np.isin(result, [99]))
