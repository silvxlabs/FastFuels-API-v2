"""Tests for standgen CHM handler and FastFuels stem isolation algorithms."""

from unittest.mock import MagicMock, patch

import dask.dataframe as dd
import geopandas as gpd
import pandas as pd
import pytest
import rioxarray  # noqa: F401 - registers .rio accessor
import xarray as xr
from shapely.geometry import box
from standgen.errors import ProcessingError
from standgen.handlers.chm import handle_chm

# --- Fixtures for Handler ---


@pytest.fixture
def mock_inventory_lmf():
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
def mock_inventory_vwf():
    return {
        "id": "test-inv-vwf",
        "domain_id": "test-domain",
        "source": {
            "name": "chm",
            "source_chm_grid_id": "test-grid-id",
            "algorithm": {
                "name": "vwf",
                "min_height": 5.0,
                "crown_ratio": 0.15,
                "crown_offset": 1.0,
            },
        },
        "modifications": [],
    }


@pytest.fixture
def mock_domain_gdf():
    # EPSG:32610 is a UTM zone (meters)
    return gpd.GeoDataFrame(geometry=[box(0, 0, 100, 100)], crs="EPSG:32610")


@pytest.fixture
def mock_trees_ddf():
    """Returns a Dask DataFrame matching the output of fastfuels_core functions."""
    pdf = pd.DataFrame({"x": [10.0, 50.0], "y": [20.0, 60.0], "height": [15.0, 25.0]})
    return dd.from_pandas(pdf, npartitions=1)


# --- CHM Handler Tests ---


class TestHandleChm:
    def _setup_mock_grid(self, mock_get, mock_load, crs="EPSG:32610", resolution=1.0):
        """Helper to set up standard Firestore and Grid dataset mocks."""
        mock_snapshot = MagicMock()
        mock_snapshot.to_dict.return_value = {"id": "grid-123"}
        mock_get.return_value = (None, mock_snapshot)

        # Create a mock DataArray with the necessary rioxarray properties
        da = xr.DataArray([[1, 2], [3, 4]])
        da = da.rio.write_crs(crs)
        # rioxarray resolution returns a tuple (x_res, y_res)
        da.rio.write_transform(
            # Basic affine transform to set a 1.0m resolution
            __import__("affine").Affine(resolution, 0.0, 0.0, 0.0, -resolution, 0.0),
            inplace=True,
        )

        mock_ds = xr.Dataset({"chm": da})
        mock_load.return_value = mock_ds
        return da

    @patch("standgen.handlers.chm.get_document")
    @patch("standgen.handlers.chm.load_grid")
    @patch("standgen.handlers.chm.save_parquet")
    @patch("standgen.handlers.chm.fixed_window_filter")
    def test_successful_lmf_execution(
        self,
        mock_fixed_filter,
        mock_save,
        mock_load,
        mock_get,
        mock_inventory_lmf,
        mock_domain_gdf,
        mock_trees_ddf,
    ):
        """Handler correctly routes LMF, translates parameters, and outputs Parquet."""
        self._setup_mock_grid(mock_get, mock_load, resolution=1.0)
        mock_fixed_filter.return_value = mock_trees_ddf
        progress = MagicMock()

        # Execute
        result = handle_chm(
            mock_inventory_lmf, mock_inventory_lmf["source"], mock_domain_gdf, progress
        )

        # Verify algorithm was called with correct pixel-to-meter translation
        # footprint_size (3) * spatial_res (1.0) = 3.0 meters
        mock_fixed_filter.assert_called_once()
        _, kwargs = mock_fixed_filter.call_args
        assert kwargs["window_size_meters"] == 3.0
        assert kwargs["min_height"] == 2.0

        # Verify saving logic
        mock_save.assert_called_once()
        args, _ = mock_save.call_args
        saved_ddf = args[1]

        assert isinstance(saved_ddf, dd.DataFrame)
        assert sorted(saved_ddf.columns.tolist()) == ["height", "x", "y"]
        assert result["georeference"]["crs"] == "EPSG:32610"

    @patch("standgen.handlers.chm.get_document")
    @patch("standgen.handlers.chm.load_grid")
    @patch("standgen.handlers.chm.save_parquet")
    @patch("standgen.handlers.chm.variable_window_filter")
    def test_successful_vwf_execution(
        self,
        mock_var_filter,
        mock_save,
        mock_load,
        mock_get,
        mock_inventory_vwf,
        mock_domain_gdf,
        mock_trees_ddf,
    ):
        """Handler correctly routes VWF and passes exact parameters."""
        self._setup_mock_grid(mock_get, mock_load, resolution=0.5)
        mock_var_filter.return_value = mock_trees_ddf
        progress = MagicMock()

        # Execute
        handle_chm(
            mock_inventory_vwf, mock_inventory_vwf["source"], mock_domain_gdf, progress
        )

        # Verify algorithm was called with correct VWF params and dynamic resolution
        mock_var_filter.assert_called_once()
        _, kwargs = mock_var_filter.call_args
        assert kwargs["spatial_resolution"] == 0.5
        assert kwargs["crown_ratio"] == 0.15
        assert kwargs["crown_offset"] == 1.0

    @patch("standgen.handlers.chm.get_document")
    def test_missing_grid_raises_processing_error(
        self, mock_get, mock_inventory_lmf, mock_domain_gdf
    ):
        from lib.firestore import DocumentNotFoundError

        mock_get.side_effect = DocumentNotFoundError("Grid not found")

        with pytest.raises(ProcessingError) as exc_info:
            handle_chm(
                mock_inventory_lmf,
                mock_inventory_lmf["source"],
                mock_domain_gdf,
                MagicMock(),
            )
        assert exc_info.value.code == "SOURCE_GRID_NOT_FOUND"

    @patch("standgen.handlers.chm.get_document")
    @patch("standgen.handlers.chm.load_grid")
    def test_missing_chm_band_raises_processing_error(
        self, mock_load, mock_get, mock_inventory_lmf, mock_domain_gdf
    ):
        mock_get.return_value = (None, MagicMock())
        # Return dataset missing the 'chm' band
        mock_load.return_value = xr.Dataset({"wrong_band": xr.DataArray([1])})

        with pytest.raises(ProcessingError) as exc_info:
            handle_chm(
                mock_inventory_lmf,
                mock_inventory_lmf["source"],
                mock_domain_gdf,
                MagicMock(),
            )
        assert exc_info.value.code == "MISSING_BAND"

    @patch("standgen.handlers.chm.get_document")
    @patch("standgen.handlers.chm.load_grid")
    def test_unsupported_algorithm_raises_processing_error(
        self, mock_load, mock_get, mock_inventory_lmf, mock_domain_gdf
    ):
        self._setup_mock_grid(mock_get, mock_load)
        mock_inventory_lmf["source"]["algorithm"]["name"] = "watershed"

        with pytest.raises(ProcessingError) as exc_info:
            handle_chm(
                mock_inventory_lmf,
                mock_inventory_lmf["source"],
                mock_domain_gdf,
                MagicMock(),
            )
        assert exc_info.value.code == "UNSUPPORTED_ALGORITHM"

    @patch("standgen.handlers.chm.get_document")
    @patch("standgen.handlers.chm.load_grid")
    @patch("standgen.handlers.chm.fixed_window_filter")
    def test_algorithm_value_error_mapped_to_processing_error(
        self,
        mock_fixed_filter,
        mock_load,
        mock_get,
        mock_inventory_lmf,
        mock_domain_gdf,
    ):
        self._setup_mock_grid(mock_get, mock_load)
        # Force the underlying FastFuels logic to throw a validation error
        mock_fixed_filter.side_effect = ValueError("min_height cannot be negative")

        with pytest.raises(ProcessingError) as exc_info:
            handle_chm(
                mock_inventory_lmf,
                mock_inventory_lmf["source"],
                mock_domain_gdf,
                MagicMock(),
            )
        assert exc_info.value.code == "INVALID_ALGORITHM_PARAMS"
        assert "negative" in exc_info.value.message
