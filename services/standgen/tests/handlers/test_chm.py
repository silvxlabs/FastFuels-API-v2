"""Tests for standgen CHM handler and LMF algorithm."""

from unittest.mock import MagicMock, patch

import dask.dataframe as dd
import geopandas as gpd
import numpy as np
import pytest
import rioxarray  # noqa: F401 - registers .rio accessor
import xarray as xr
from shapely.geometry import Point, box
from standgen.errors import ProcessingError
from standgen.handlers.chm import find_treetops_lmf, handle_chm

# --- Fixtures for LMF Algorithm ---


@pytest.fixture
def sample_chm_da():
    """Create a small xarray DataArray mimicking a CHM grid.

    Contains two clear peaks (heights 15 and 20) and a flat background (height 1).
    """
    data = np.array(
        [
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 15.0, 1.0, 1.0, 1.0],  # Peak 1 at index (1, 1)
            [1.0, 1.0, 5.0, 1.0, 1.0],  # Sub-peak, might be filtered by footprint
            [1.0, 1.0, 1.0, 20.0, 1.0],  # Peak 2 at index (3, 3)
            [1.0, 1.0, 1.0, 1.0, 1.0],
        ],
        dtype=float,
    )

    # 10m resolution grid
    x = np.array([0.0, 10.0, 20.0, 30.0, 40.0])
    y = np.array([40.0, 30.0, 20.0, 10.0, 0.0])

    da = xr.DataArray(data, dims=("y", "x"), coords={"y": y, "x": x})
    da = da.rio.write_crs("EPSG:32610")
    return da


# --- Fixtures for Handler ---


@pytest.fixture
def mock_inventory():
    return {
        "id": "test-inv-123",
        "domain_id": "test-domain",
        "source": {
            "name": "chm",
            "source_chm_grid_id": "test-grid-id",
            "algorithm": {
                "name": "lmf",
                "min_height": 2.0,
                "footprint_size": 3,
            },
        },
        "modifications": [],
    }


@pytest.fixture
def mock_domain_gdf():
    return gpd.GeoDataFrame(geometry=[box(0, 0, 100, 100)], crs="EPSG:32610")


# --- LMF Algorithm Tests ---


class TestFindTreetopsLmf:
    def test_basic_tree_extraction(self, sample_chm_da):
        """Extracts the correct peaks above the threshold."""
        gdf = find_treetops_lmf(sample_chm_da, min_height=10.0, footprint_size=3)

        # Should find exactly 2 trees (heights 15 and 20)
        assert len(gdf) == 2
        assert "height" in gdf.columns
        assert "geometry" in gdf.columns
        assert gdf.crs == "EPSG:32610"

        # Verify heights are correct
        heights = set(gdf["height"].values)
        assert heights == {15.0, 20.0}

        # Verify coordinates (Peak 1 at x=10, y=30; Peak 2 at x=30, y=10)
        coords = set((geom.x, geom.y) for geom in gdf.geometry)
        assert coords == {(10.0, 30.0), (30.0, 10.0)}

    def test_min_height_filter(self, sample_chm_da):
        """Filters out peaks below the minimum height threshold."""
        gdf = find_treetops_lmf(sample_chm_da, min_height=18.0, footprint_size=3)

        # Should only find the 20m tree
        assert len(gdf) == 1
        assert gdf["height"].iloc[0] == 20.0

    def test_empty_result_returns_valid_gdf(self, sample_chm_da):
        """Returns an empty GeoDataFrame with correct schema if no trees found."""
        gdf = find_treetops_lmf(sample_chm_da, min_height=50.0, footprint_size=3)

        assert len(gdf) == 0
        assert "height" in gdf.columns
        assert "geometry" in gdf.columns
        assert gdf.crs == "EPSG:32610"

    def test_even_footprint_raises_value_error(self, sample_chm_da):
        """Algorithm strictly requires odd footprint sizes."""
        with pytest.raises(ValueError, match="odd integer"):
            find_treetops_lmf(sample_chm_da, min_height=2.0, footprint_size=4)

    def test_flat_top_tree_returns_center(self):
        """A plateau of equal heights should return a single center point."""
        # 5x5 plateau of height 20 in the middle
        data = np.zeros((10, 10))
        data[3:6, 3:6] = 20.0

        x = np.arange(10) * 1.0
        y = np.arange(10)[::-1] * 1.0
        da = xr.DataArray(data, dims=("y", "x"), coords={"y": y, "x": x})
        da = da.rio.write_crs("EPSG:32610")

        gdf = find_treetops_lmf(da, min_height=5.0, footprint_size=3)

        # Should only identify 1 tree for the entire plateau
        assert len(gdf) == 1
        assert gdf["height"].iloc[0] == 20.0


# --- CHM Handler Tests ---


class TestHandleChm:
    @patch("standgen.handlers.chm.get_document")
    @patch("standgen.handlers.chm.load_grid")
    @patch("standgen.handlers.chm.save_parquet")
    @patch("standgen.handlers.chm.find_treetops_lmf")
    def test_successful_handler_execution(
        self, mock_find, mock_save, mock_load, mock_get, mock_inventory, mock_domain_gdf
    ):
        """Handler correctly routes data and returns georeference."""
        # Setup mocks
        mock_snapshot = MagicMock()
        mock_snapshot.to_dict.return_value = {"id": "grid-123"}
        mock_get.return_value = (None, mock_snapshot)

        mock_ds = xr.Dataset({"chm": xr.DataArray([1, 2, 3])})
        mock_load.return_value = mock_ds

        mock_trees_gdf = gpd.GeoDataFrame(
            {"height": [15.0, 20.0], "geometry": [Point(0, 0), Point(10, 10)]},
            crs="EPSG:32610",
        )
        mock_find.return_value = mock_trees_gdf

        progress = MagicMock()

        # Execute
        result = handle_chm(
            mock_inventory, mock_inventory["source"], mock_domain_gdf, progress
        )

        # Verify georeference was returned
        assert "georeference" in result
        assert result["georeference"]["crs"] == "EPSG:32610"

        # Verify save_parquet was called with a Dask DataFrame
        mock_save.assert_called_once()
        args, _ = mock_save.call_args
        assert args[0] == "test-inv-123"
        assert isinstance(args[1], dd.DataFrame)

        # Verify base columns are present in the output dask dataframe
        saved_ddf = args[1]
        assert "height" in saved_ddf.columns
        assert "dbh" in saved_ddf.columns
        assert "fia_species_code" in saved_ddf.columns

    @patch("standgen.handlers.chm.get_document")
    def test_missing_grid_raises_processing_error(
        self, mock_get, mock_inventory, mock_domain_gdf
    ):
        from lib.firestore import DocumentNotFoundError

        mock_get.side_effect = DocumentNotFoundError("Grid not found")

        with pytest.raises(ProcessingError) as exc_info:
            handle_chm(
                mock_inventory, mock_inventory["source"], mock_domain_gdf, MagicMock()
            )
        assert exc_info.value.code == "SOURCE_GRID_NOT_FOUND"

    @patch("standgen.handlers.chm.get_document")
    @patch("standgen.handlers.chm.load_grid")
    def test_missing_chm_band_raises_processing_error(
        self, mock_load, mock_get, mock_inventory, mock_domain_gdf
    ):
        mock_get.return_value = (None, MagicMock())
        # Return dataset missing the 'chm' band
        mock_load.return_value = xr.Dataset({"wrong_band": xr.DataArray([1])})

        with pytest.raises(ProcessingError) as exc_info:
            handle_chm(
                mock_inventory, mock_inventory["source"], mock_domain_gdf, MagicMock()
            )
        assert exc_info.value.code == "MISSING_BAND"

    @patch("standgen.handlers.chm.get_document")
    @patch("standgen.handlers.chm.load_grid")
    def test_unsupported_algorithm_raises_processing_error(
        self, mock_load, mock_get, mock_inventory, mock_domain_gdf
    ):
        mock_get.return_value = (None, MagicMock())
        mock_load.return_value = xr.Dataset({"chm": xr.DataArray([1])})

        # Change algorithm name
        mock_inventory["source"]["algorithm"]["name"] = "watershed"

        with pytest.raises(ProcessingError) as exc_info:
            handle_chm(
                mock_inventory, mock_inventory["source"], mock_domain_gdf, MagicMock()
            )
        assert exc_info.value.code == "UNSUPPORTED_ALGORITHM"

    @patch("standgen.handlers.chm.get_document")
    @patch("standgen.handlers.chm.load_grid")
    @patch("standgen.handlers.chm.find_treetops_lmf")
    def test_algorithm_value_error_mapped_to_processing_error(
        self, mock_find, mock_load, mock_get, mock_inventory, mock_domain_gdf
    ):
        mock_get.return_value = (None, MagicMock())
        mock_load.return_value = xr.Dataset({"chm": xr.DataArray([1])})

        # Force the algorithm to throw a ValueError
        mock_find.side_effect = ValueError("footprint_size must be an odd integer")

        with pytest.raises(ProcessingError) as exc_info:
            handle_chm(
                mock_inventory, mock_inventory["source"], mock_domain_gdf, MagicMock()
            )
        assert exc_info.value.code == "INVALID_ALGORITHM_PARAMS"
