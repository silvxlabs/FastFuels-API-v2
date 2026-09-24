"""Tests for standgen CHM handler and FastFuels stem isolation algorithms."""

from unittest.mock import MagicMock, patch

import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest
import rioxarray  # noqa: F401 - registers .rio accessor
import xarray as xr
from affine import Affine
from standgen.handlers.chm import handle_chm

from lib.errors import ProcessingError

from .conftest import CHM_INVENTORY_COLUMNS

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
        "columns": CHM_INVENTORY_COLUMNS,
        "type": "tree",
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
        "columns": CHM_INVENTORY_COLUMNS,
        "type": "tree",
    }


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

    @patch("standgen.handlers.chm.count_inventory_rows")
    @patch("standgen.handlers.chm.get_document")
    @patch("standgen.handlers.chm.load_grid")
    @patch("standgen.handlers.chm.save_parquet_with_summary")
    @patch("standgen.handlers.chm.fixed_window_filter")
    def test_successful_lmf_execution(
        self,
        mock_fixed_filter,
        mock_save,
        mock_load,
        mock_get,
        mock_count,
        mock_inventory_lmf,
        mock_domain_gdf,
        mock_trees_ddf,
    ):
        """Handler correctly routes LMF, translates parameters, and outputs Parquet."""
        self._setup_mock_grid(mock_get, mock_load, resolution=1.0)
        mock_fixed_filter.return_value = mock_trees_ddf
        mock_count.return_value = 2
        mock_save.return_value = ("gs://test-bucket/test-inv-123", {}, None)
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
        saved_columns = args[2]

        assert isinstance(saved_ddf, dd.DataFrame)
        assert saved_ddf.columns.tolist() == ["tree_id", "x", "y", "height"]
        assert result["georeference"]["crs"] == "EPSG:32610"
        assert "columns" in result
        assert saved_columns == mock_inventory_lmf["columns"]

    @patch("standgen.handlers.chm.count_inventory_rows")
    @patch("standgen.handlers.chm.get_document")
    @patch("standgen.handlers.chm.load_grid")
    @patch("standgen.handlers.chm.save_parquet_with_summary")
    @patch("standgen.handlers.chm.variable_window_filter")
    def test_successful_vwf_execution(
        self,
        mock_var_filter,
        mock_save,
        mock_load,
        mock_get,
        mock_count,
        mock_inventory_vwf,
        mock_domain_gdf,
        mock_trees_ddf,
    ):
        """Handler correctly routes VWF and passes exact parameters."""
        self._setup_mock_grid(mock_get, mock_load, resolution=0.5)
        mock_var_filter.return_value = mock_trees_ddf
        mock_save.return_value = ("gs://test-bucket/test-inv-vwf", {}, None)
        mock_count.return_value = 2
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

    def _mock_grid_with_values(self, mock_get, mock_load, values):
        """Mock a CHM grid holding `values` (a 2D array) at 1 m resolution."""
        chm = xr.DataArray(np.asarray(values, dtype=float)).rio.write_crs("EPSG:32610")
        chm.rio.write_transform(Affine(1.0, 0.0, 0.0, 0.0, -1.0, 0.0), inplace=True)
        mock_snapshot = MagicMock()
        mock_snapshot.to_dict.return_value = {"id": "grid-123"}
        mock_get.return_value = (None, mock_snapshot)
        mock_load.return_value = xr.Dataset({"chm": chm})
        return chm

    @patch("standgen.handlers.chm.count_inventory_rows")
    @patch("standgen.handlers.chm.get_document")
    @patch("standgen.handlers.chm.load_grid")
    @patch("standgen.handlers.chm.save_parquet_with_summary")
    @patch("standgen.handlers.chm.variable_window_filter")
    def test_max_height_clips_tall_artifacts_before_detection(
        self,
        mock_var_filter,
        mock_save,
        mock_load,
        mock_get,
        mock_count,
        mock_inventory_vwf,
        mock_domain_gdf,
        mock_trees_ddf,
    ):
        """CHM returns above max_height are zeroed before the filter runs; valid
        pixels and the grid's georeferencing are left untouched."""
        chm = self._mock_grid_with_values(
            mock_get, mock_load, [[10.0, 20.0], [30.0, 400.0]]
        )
        mock_var_filter.return_value = mock_trees_ddf
        mock_save.return_value = ("gs://test-bucket/test-inv-vwf", {}, None)
        mock_count.return_value = 2

        source = mock_inventory_vwf["source"]
        source["algorithm"]["max_height"] = 120.0

        handle_chm(mock_inventory_vwf, source, mock_domain_gdf, MagicMock())

        _, kwargs = mock_var_filter.call_args
        clipped = kwargs["chm_da"]
        assert float(clipped.max()) <= 120.0
        # 400 m artifact -> 0; the real 30 m return is preserved verbatim.
        assert float(clipped.values[1, 1]) == 0.0
        assert float(clipped.values[1, 0]) == 30.0
        # Georeferencing must survive the clip (downstream reproject reads it).
        assert clipped.rio.crs == chm.rio.crs

    @patch("standgen.handlers.chm.count_inventory_rows")
    @patch("standgen.handlers.chm.get_document")
    @patch("standgen.handlers.chm.load_grid")
    @patch("standgen.handlers.chm.save_parquet_with_summary")
    @patch("standgen.handlers.chm.variable_window_filter")
    def test_nodata_nan_filled_to_zero_before_detection(
        self,
        mock_var_filter,
        mock_save,
        mock_load,
        mock_get,
        mock_count,
        mock_inventory_vwf,
        mock_domain_gdf,
        mock_trees_ddf,
    ):
        """Invalid CHM pixels (NaN nodata) are zero-filled before the filter runs so
        they can't suppress real treetops at nodata boundaries (#430); valid pixels
        and the grid's georeferencing are left untouched."""
        chm = self._mock_grid_with_values(
            mock_get, mock_load, [[10.0, np.nan], [np.nan, 30.0]]
        )
        mock_var_filter.return_value = mock_trees_ddf
        mock_save.return_value = ("gs://test-bucket/test-inv-vwf", {}, None)
        mock_count.return_value = 2

        source = mock_inventory_vwf["source"]
        source["algorithm"]["max_height"] = None

        handle_chm(mock_inventory_vwf, source, mock_domain_gdf, MagicMock())

        _, kwargs = mock_var_filter.call_args
        filled = kwargs["chm_da"]
        # No NaN reaches the core; nodata pixels become 0.
        assert not bool(np.isnan(filled.values).any())
        assert float(filled.values[0, 1]) == 0.0
        assert float(filled.values[1, 0]) == 0.0
        # Real returns are preserved verbatim.
        assert float(filled.values[0, 0]) == 10.0
        assert float(filled.values[1, 1]) == 30.0
        # Georeferencing must survive the fill (downstream reproject reads it).
        assert filled.rio.crs == chm.rio.crs
        assert filled.rio.transform() == chm.rio.transform()

    @patch("standgen.handlers.chm.count_inventory_rows")
    @patch("standgen.handlers.chm.get_document")
    @patch("standgen.handlers.chm.load_grid")
    @patch("standgen.handlers.chm.save_parquet_with_summary")
    @patch("standgen.handlers.chm.variable_window_filter")
    def test_nodata_and_over_max_both_zeroed(
        self,
        mock_var_filter,
        mock_save,
        mock_load,
        mock_get,
        mock_count,
        mock_inventory_vwf,
        mock_domain_gdf,
        mock_trees_ddf,
    ):
        """The nodata fill and the max_height cap compose: NaN nodata and over-max
        returns both land at 0, while valid in-range pixels pass through."""
        chm = self._mock_grid_with_values(
            mock_get, mock_load, [[10.0, np.nan], [400.0, 30.0]]
        )
        mock_var_filter.return_value = mock_trees_ddf
        mock_save.return_value = ("gs://test-bucket/test-inv-vwf", {}, None)
        mock_count.return_value = 2

        source = mock_inventory_vwf["source"]
        source["algorithm"]["max_height"] = 120.0

        handle_chm(mock_inventory_vwf, source, mock_domain_gdf, MagicMock())

        _, kwargs = mock_var_filter.call_args
        result = kwargs["chm_da"]
        assert not bool(np.isnan(result.values).any())
        assert float(result.values[0, 1]) == 0.0  # NaN nodata -> 0
        assert float(result.values[1, 0]) == 0.0  # 400 m over-max -> 0
        assert float(result.values[0, 0]) == 10.0  # valid, preserved
        assert float(result.values[1, 1]) == 30.0  # valid, preserved
        assert result.rio.crs == chm.rio.crs

    @patch("standgen.handlers.chm.count_inventory_rows")
    @patch("standgen.handlers.chm.get_document")
    @patch("standgen.handlers.chm.load_grid")
    @patch("standgen.handlers.chm.save_parquet_with_summary")
    @patch("standgen.handlers.chm.variable_window_filter")
    def test_max_height_none_leaves_chm_unclipped(
        self,
        mock_var_filter,
        mock_save,
        mock_load,
        mock_get,
        mock_count,
        mock_inventory_vwf,
        mock_domain_gdf,
        mock_trees_ddf,
    ):
        """max_height=None disables the ceiling; tall returns pass through."""
        self._mock_grid_with_values(mock_get, mock_load, [[10.0, 20.0], [30.0, 400.0]])
        mock_var_filter.return_value = mock_trees_ddf
        mock_save.return_value = ("gs://test-bucket/test-inv-vwf", {}, None)
        mock_count.return_value = 2

        source = mock_inventory_vwf["source"]
        source["algorithm"]["max_height"] = None

        handle_chm(mock_inventory_vwf, source, mock_domain_gdf, MagicMock())

        _, kwargs = mock_var_filter.call_args
        assert float(kwargs["chm_da"].max()) == 400.0


def _cone_chm(size=21, peak=20.0, slope=1.0, chunks=None):
    """A single cone on a 1 m CHM, apex at the center cell."""
    rows, cols = np.mgrid[0:size, 0:size]
    c = size // 2
    values = peak - slope * np.hypot(rows - c, cols - c)
    chm = xr.DataArray(
        values,
        dims=("y", "x"),
        coords={"y": size - 0.5 - np.arange(size), "x": 0.5 + np.arange(size)},
    )
    chm = chm.rio.write_crs("EPSG:32610")
    chm = chm.rio.write_transform(Affine(1.0, 0.0, 0.0, 0.0, -1.0, float(size)))
    if chunks is not None:
        chm = chm.chunk(chunks)
    return chm


class TestHandleChmCrownSegmentation:
    SEGMENTATION = {
        "method": "dalponte2016",
        "radius_estimator": "area_equivalent",
        "min_relative_height": 0.45,
        "min_relative_crown_height": 0.55,
        "max_crown_radius": 3.0,
    }

    def _inventory(self, crown_segmentation):
        source = {
            "name": "chm",
            "source_chm_grid_id": "test-grid-id",
            "algorithm": {
                "name": "lmf",
                "min_height": 2.0,
                "max_height": 120.0,
                "footprint_size": 3,
            },
            "crown_segmentation": crown_segmentation,
        }
        columns = list(CHM_INVENTORY_COLUMNS)
        if crown_segmentation:
            columns.append({"key": "crown_radius", "type": "continuous", "unit": "m"})
        return {
            "id": "test-inv-seg",
            "domain_id": "test-domain",
            "source": source,
            "modifications": [],
            "columns": columns,
            "type": "tree",
        }

    def _run(self, chm, inventory, domain_gdf):
        """Run the handler on `chm`; return the frame it would write."""
        saved = {}

        def fake_save(inventory_id, ddf, columns, *args, **kwargs):
            saved["df"] = ddf.compute()
            return "gs://test", {}, None

        snapshot = MagicMock()
        snapshot.to_dict.return_value = {"id": "grid"}
        with (
            patch("standgen.handlers.chm.get_document", return_value=(None, snapshot)),
            patch(
                "standgen.handlers.chm.load_grid",
                return_value=xr.Dataset({"chm": chm}),
            ),
            patch("standgen.handlers.chm.save_parquet_with_summary", fake_save),
            patch("standgen.handlers.chm.count_inventory_rows", return_value=None),
        ):
            handle_chm(inventory, inventory["source"], domain_gdf, MagicMock())
        return saved["df"]

    def test_without_segmentation_has_no_crown_radius(self, mock_domain_gdf):
        with patch("standgen.handlers.chm.dalponte2016") as mock_segment:
            df = self._run(_cone_chm(), self._inventory(None), mock_domain_gdf)
        mock_segment.assert_not_called()
        assert df.columns.tolist() == ["tree_id", "x", "y", "height"]

    @pytest.mark.parametrize("chunks", [None, 8])
    def test_isolated_cone_clipped_at_max_crown_radius(self, mock_domain_gdf, chunks):
        """Every cell within 3 m of the apex qualifies, so the crown is the 29
        lattice cells of a radius-3 disc."""
        df = self._run(
            _cone_chm(chunks=chunks),
            self._inventory(self.SEGMENTATION),
            mock_domain_gdf,
        )
        assert df.columns.tolist() == ["tree_id", "x", "y", "height", "crown_radius"]
        assert len(df) == 1
        assert df["crown_radius"].iloc[0] == pytest.approx(np.sqrt(29 / np.pi))

    def test_every_tree_gets_at_least_one_cell(self, mock_domain_gdf):
        """A treetop whose neighbours all fail the height tests keeps its own cell."""
        chm = _cone_chm(slope=15.0)  # neighbours drop below 0.45 x apex at once
        df = self._run(chm, self._inventory(self.SEGMENTATION), mock_domain_gdf)
        assert len(df) == 1
        assert df["crown_radius"].iloc[0] == pytest.approx(np.sqrt(1 / np.pi))

    def test_crown_radius_survives_reprojection(self, mock_domain_gdf):
        domain_gdf = mock_domain_gdf.to_crs("EPSG:32611")
        df = self._run(_cone_chm(), self._inventory(self.SEGMENTATION), domain_gdf)
        assert df.columns.tolist() == ["tree_id", "x", "y", "height", "crown_radius"]
        assert df["crown_radius"].iloc[0] == pytest.approx(np.sqrt(29 / np.pi))

    def test_detection_graph_computed_once(self, mock_domain_gdf):
        calls = []

        def partition(i):
            calls.append(i)
            return pd.DataFrame({"x": [10.5], "y": [10.5], "height": [20.0]})

        meta = pd.DataFrame({"x": [], "y": [], "height": []})
        ddf = dd.from_map(partition, [0], meta=meta)
        with patch("standgen.handlers.chm.fixed_window_filter", return_value=ddf):
            df = self._run(
                _cone_chm(), self._inventory(self.SEGMENTATION), mock_domain_gdf
            )
        assert calls == [0]
        assert len(df) == 1

    def test_segmentation_value_error_is_processing_error(self, mock_domain_gdf):
        with (
            patch(
                "standgen.handlers.chm.dalponte2016",
                side_effect=ValueError("two treetops fall in the same CHM cell"),
            ),
            pytest.raises(ProcessingError) as exc_info,
        ):
            self._run(_cone_chm(), self._inventory(self.SEGMENTATION), mock_domain_gdf)
        assert exc_info.value.code == "INVALID_CROWN_SEGMENTATION_PARAMS"
