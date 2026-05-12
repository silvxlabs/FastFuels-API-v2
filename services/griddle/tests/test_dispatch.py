"""
Tests for dispatch module.
"""

from unittest.mock import MagicMock, patch

import geopandas as gpd
import pytest
from griddle.dispatch import (
    dispatch_handler,
    handle_3dep,
    handle_canopy,
    handle_landfire,
    handle_lookup,
    handle_pim,
    handle_resample,
    handle_uniform,
)
from griddle.errors import ProcessingError


class TestHandleLandfire:
    """Tests for handle_landfire function."""

    def test_unknown_product_raises(self):
        """handle_landfire raises ProcessingError for unknown product."""
        mock_gdf = MagicMock(spec=gpd.GeoDataFrame)
        progress = MagicMock()

        source = {"product": "unknown_product"}

        with pytest.raises(ProcessingError) as exc_info:
            handle_landfire(mock_gdf, source, progress)

        assert exc_info.value.code == "UNKNOWN_PRODUCT"
        assert "unknown_product" in exc_info.value.message

    @patch("griddle.dispatch.landfire.fetch_fbfm40")
    def test_routes_fbfm40_to_handler(self, mock_fetch):
        """handle_landfire routes fbfm40 product to fetch_fbfm40."""
        mock_gdf = MagicMock(spec=gpd.GeoDataFrame)
        mock_result = MagicMock()
        mock_fetch.return_value = mock_result
        progress = MagicMock()

        source = {"product": "fbfm40", "version": "2022"}

        result = handle_landfire(mock_gdf, source, progress)

        mock_fetch.assert_called_once_with(
            mock_gdf,
            "2022",
            remove_non_burnable=None,
            extent_buffer_cells=0,
            alignment={"target": "domain"},
            target_grid_doc=None,
        )
        assert result == mock_result

    @patch("griddle.dispatch.landfire.fetch_fbfm40")
    def test_fbfm40_default_version(self, mock_fetch):
        """handle_landfire uses 2024 as default version for fbfm40."""
        mock_gdf = MagicMock(spec=gpd.GeoDataFrame)
        progress = MagicMock()

        source = {"product": "fbfm40"}  # No version specified

        handle_landfire(mock_gdf, source, progress)

        # Check that fetch_fbfm40 was called with default version
        _, call_kwargs = mock_fetch.call_args
        assert call_kwargs == {} or mock_fetch.call_args[0][1] == "2024"

    @patch("griddle.dispatch.landfire.fetch_fbfm40")
    def test_fbfm40_calls_progress_callback(self, mock_fetch):
        """handle_landfire reports progress."""
        mock_gdf = MagicMock(spec=gpd.GeoDataFrame)
        progress = MagicMock()

        source = {"product": "fbfm40", "version": "2022"}

        handle_landfire(mock_gdf, source, progress)

        progress.assert_called_once()
        call_args = progress.call_args[0]
        assert "LANDFIRE" in call_args[0]
        assert "fbfm40" in call_args[0]

    @patch("griddle.dispatch.landfire.fetch_fccs")
    def test_routes_fccs_to_handler(self, mock_fetch):
        """handle_landfire routes fccs product to fetch_fccs."""
        mock_gdf = MagicMock(spec=gpd.GeoDataFrame)
        mock_result = MagicMock()
        mock_fetch.return_value = mock_result
        progress = MagicMock()

        source = {"product": "fccs", "version": "2023"}

        result = handle_landfire(mock_gdf, source, progress)

        mock_fetch.assert_called_once_with(
            mock_gdf,
            "2023",
            extent_buffer_cells=0,
            alignment={"target": "domain"},
            target_grid_doc=None,
        )
        assert result == mock_result

    @patch("griddle.dispatch.landfire.fetch_fccs")
    def test_fccs_default_version(self, mock_fetch):
        """handle_landfire uses 2023 as default version for fccs."""
        mock_gdf = MagicMock(spec=gpd.GeoDataFrame)
        mock_fetch.return_value = MagicMock()
        progress = MagicMock()

        source = {"product": "fccs"}  # No version specified

        handle_landfire(mock_gdf, source, progress)

        call_args = mock_fetch.call_args[0]
        assert call_args[1] == "2023"

    @patch("griddle.dispatch.landfire.fetch_fccs")
    def test_fccs_calls_progress(self, mock_fetch):
        """handle_landfire reports progress for fccs."""
        mock_gdf = MagicMock(spec=gpd.GeoDataFrame)
        mock_fetch.return_value = MagicMock()
        progress = MagicMock()

        source = {"product": "fccs", "version": "2023"}

        handle_landfire(mock_gdf, source, progress)

        progress.assert_called_once()
        call_args = progress.call_args[0]
        assert "LANDFIRE" in call_args[0]
        assert "fccs" in call_args[0]

    @patch("griddle.dispatch.landfire.fetch_topography")
    def test_routes_topography_to_handler(self, mock_fetch):
        """handle_landfire routes topography product to fetch_topography."""
        mock_gdf = MagicMock(spec=gpd.GeoDataFrame)
        mock_result = MagicMock()
        mock_fetch.return_value = mock_result
        progress = MagicMock()

        source = {
            "product": "topography",
            "version": "2020",
            "bands": ["elevation", "slope", "aspect"],
        }

        result = handle_landfire(mock_gdf, source, progress)

        mock_fetch.assert_called_once_with(
            mock_gdf,
            "2020",
            ["elevation", "slope", "aspect"],
            progress,
            extent_buffer_cells=0,
            alignment={"target": "domain"},
            target_grid_doc=None,
        )
        assert result == mock_result

    @patch("griddle.dispatch.landfire.fetch_topography")
    def test_topography_passes_bands(self, mock_fetch):
        """handle_landfire passes band list to topography handler."""
        mock_gdf = MagicMock(spec=gpd.GeoDataFrame)
        progress = MagicMock()

        source = {
            "product": "topography",
            "version": "2020",
            "bands": ["elevation"],
        }

        handle_landfire(mock_gdf, source, progress)

        call_args = mock_fetch.call_args[0]
        assert call_args[2] == ["elevation"]

    @patch("griddle.dispatch.landfire.fetch_topography")
    def test_topography_calls_progress(self, mock_fetch):
        """handle_landfire reports progress for topography."""
        mock_gdf = MagicMock(spec=gpd.GeoDataFrame)
        progress = MagicMock()

        source = {
            "product": "topography",
            "version": "2020",
            "bands": ["elevation"],
        }

        handle_landfire(mock_gdf, source, progress)

        progress.assert_called_once()
        call_args = progress.call_args[0]
        assert "LANDFIRE" in call_args[0]
        assert "topography" in call_args[0]


class TestDispatchHandler:
    """Tests for dispatch_handler function."""

    @patch("griddle.dispatch.handle_landfire")
    def test_routes_landfire_source(self, mock_handle_landfire):
        """dispatch_handler routes landfire source to handle_landfire."""
        mock_result = MagicMock()
        mock_handle_landfire.return_value = mock_result
        mock_gdf = MagicMock(spec=gpd.GeoDataFrame)
        progress = MagicMock()

        grid = {
            "source": {"name": "landfire", "product": "fbfm40"},
            "domain_id": "test-domain-id",
        }

        result = dispatch_handler(grid, mock_gdf, progress)

        mock_handle_landfire.assert_called_once_with(mock_gdf, grid["source"], progress)
        assert result == mock_result

    def test_unknown_source_raises(self):
        """dispatch_handler raises ProcessingError for unknown source."""
        mock_gdf = MagicMock(spec=gpd.GeoDataFrame)
        progress = MagicMock()

        grid = {
            "source": {"name": "unknown_source"},
            "domain_id": "test-domain-id",
        }

        with pytest.raises(ProcessingError) as exc_info:
            dispatch_handler(grid, mock_gdf, progress)

        assert exc_info.value.code == "UNKNOWN_SOURCE"
        assert "unknown_source" in exc_info.value.message

    @patch("griddle.dispatch.handle_lookup")
    def test_routes_lookup_source(self, mock_handle_lookup):
        """dispatch_handler routes lookup source to handle_lookup."""
        mock_result = MagicMock()
        mock_handle_lookup.return_value = mock_result
        mock_gdf = MagicMock(spec=gpd.GeoDataFrame)
        progress = MagicMock()

        grid = {
            "source": {"name": "lookup", "table": "fbfm40", "source_grid_id": "src-id"},
            "bands": [{"key": "fuel_load.1hr"}],
        }

        result = dispatch_handler(grid, mock_gdf, progress)

        mock_handle_lookup.assert_called_once_with(grid, grid["source"], progress)
        assert result == mock_result


class TestDispatchHandlerPim:
    """Tests for dispatch_handler routing to pim."""

    @patch("griddle.dispatch.handle_pim")
    def test_routes_pim_source(self, mock_handle_pim):
        """dispatch_handler routes pim source to handle_pim."""
        mock_result = MagicMock()
        mock_handle_pim.return_value = mock_result
        mock_gdf = MagicMock(spec=gpd.GeoDataFrame)
        progress = MagicMock()

        grid = {
            "source": {
                "name": "pim",
                "product": "treemap",
                "version": "2022",
                "bands": ["tm_id"],
            },
            "domain_id": "test-domain-id",
        }

        result = dispatch_handler(grid, mock_gdf, progress)

        mock_handle_pim.assert_called_once_with(mock_gdf, grid["source"], progress)
        assert result == mock_result


class TestHandlePim:
    """Tests for handle_pim function."""

    @patch("griddle.dispatch.pim.fetch_treemap")
    def test_routes_treemap_to_handler(self, mock_fetch):
        """handle_pim routes treemap product to fetch_treemap."""
        mock_gdf = MagicMock(spec=gpd.GeoDataFrame)
        mock_result = MagicMock()
        mock_fetch.return_value = mock_result
        progress = MagicMock()

        source = {
            "product": "treemap",
            "version": "2022",
            "bands": ["tm_id", "plt_cn"],
        }

        result = handle_pim(mock_gdf, source, progress)

        mock_fetch.assert_called_once_with(
            mock_gdf,
            "2022",
            ["tm_id", "plt_cn"],
            progress,
            extent_buffer_cells=0,
            alignment={"target": "domain"},
            target_grid_doc=None,
        )
        assert result == mock_result

    @patch("griddle.dispatch.pim.fetch_treemap")
    def test_default_version(self, mock_fetch):
        """handle_pim uses 2022 as default version."""
        mock_gdf = MagicMock(spec=gpd.GeoDataFrame)
        progress = MagicMock()

        source = {"product": "treemap", "bands": ["tm_id"]}

        handle_pim(mock_gdf, source, progress)

        call_args = mock_fetch.call_args[0]
        assert call_args[1] == "2022"

    @patch("griddle.dispatch.pim.fetch_treemap")
    def test_passes_bands_list(self, mock_fetch):
        """handle_pim passes bands list to treemap handler."""
        mock_gdf = MagicMock(spec=gpd.GeoDataFrame)
        progress = MagicMock()

        source = {
            "product": "treemap",
            "version": "2022",
            "bands": ["tm_id"],
        }

        handle_pim(mock_gdf, source, progress)

        call_args = mock_fetch.call_args[0]
        assert call_args[2] == ["tm_id"]

    @patch("griddle.dispatch.pim.fetch_treemap")
    def test_default_bands_when_missing(self, mock_fetch):
        """handle_pim defaults to tm_id band when bands key is missing."""
        mock_gdf = MagicMock(spec=gpd.GeoDataFrame)
        progress = MagicMock()

        source = {"product": "treemap", "version": "2022"}

        handle_pim(mock_gdf, source, progress)

        call_args = mock_fetch.call_args[0]
        assert call_args[2] == ["tm_id"]

    def test_unknown_product_raises(self):
        """handle_pim raises ProcessingError for unknown product."""
        mock_gdf = MagicMock(spec=gpd.GeoDataFrame)
        progress = MagicMock()

        source = {"product": "unknown_product"}

        with pytest.raises(ProcessingError) as exc_info:
            handle_pim(mock_gdf, source, progress)

        assert exc_info.value.code == "UNKNOWN_PRODUCT"
        assert "unknown_product" in exc_info.value.message

    @patch("griddle.dispatch.pim.fetch_treemap")
    def test_calls_progress_callback(self, mock_fetch):
        """handle_pim reports progress."""
        mock_gdf = MagicMock(spec=gpd.GeoDataFrame)
        progress = MagicMock()

        source = {"product": "treemap", "version": "2022", "bands": ["tm_id"]}

        handle_pim(mock_gdf, source, progress)

        progress.assert_called()
        call_args = progress.call_args_list[0][0]
        assert "PIM" in call_args[0]
        assert "treemap" in call_args[0]


class TestHandleLookup:
    """Tests for handle_lookup function."""

    @patch("griddle.dispatch.lookup.fbfm40_lookup")
    def test_routes_fbfm40_table(self, mock_fbfm40_lookup):
        """handle_lookup routes fbfm40 table to fbfm40_lookup."""
        mock_result = MagicMock()
        mock_fbfm40_lookup.return_value = mock_result
        progress = MagicMock()

        grid = {
            "bands": [{"key": "fuel_load.1hr"}, {"key": "fuel_depth"}],
        }
        source = {
            "table": "fbfm40",
            "source_grid_id": "test-source-grid-id",
        }

        result = handle_lookup(grid, source, progress)

        mock_fbfm40_lookup.assert_called_once_with(
            source_grid_id="test-source-grid-id",
            bands=grid["bands"],
            progress=progress,
        )
        assert result == mock_result

    def test_unknown_table_raises(self):
        """handle_lookup raises ProcessingError for unknown table."""
        progress = MagicMock()

        grid = {"bands": []}
        source = {"table": "unknown_table", "source_grid_id": "id"}

        with pytest.raises(ProcessingError) as exc_info:
            handle_lookup(grid, source, progress)

        assert exc_info.value.code == "UNKNOWN_TABLE"
        assert "unknown_table" in exc_info.value.message

    @patch("griddle.dispatch.lookup.fbfm40_lookup")
    def test_calls_progress_callback(self, mock_fbfm40_lookup):
        """handle_lookup reports progress."""
        progress = MagicMock()

        grid = {"bands": [{"key": "fuel_load.1hr"}]}
        source = {"table": "fbfm40", "source_grid_id": "id"}

        handle_lookup(grid, source, progress)

        progress.assert_called()
        call_args = progress.call_args_list[0][0]
        assert "fbfm40" in call_args[0]


def _stub_source_grid_doc():
    """Snapshot returned by the patched ``get_document`` lookup that
    ``handle_resample`` performs on the source grid."""
    snapshot = MagicMock()
    snapshot.to_dict.return_value = {
        "bands": [{"key": "elevation", "type": "continuous"}],
    }
    return snapshot


class TestDispatchHandlerResample:
    """Tests for dispatch_handler routing to resample."""

    @patch("griddle.dispatch.get_document")
    @patch("griddle.dispatch.handle_resample")
    def test_routes_resample_source(self, mock_handle_resample, mock_get_doc):
        """dispatch_handler routes resample source to handle_resample."""
        mock_result = MagicMock()
        mock_handle_resample.return_value = mock_result
        mock_gdf = MagicMock(spec=gpd.GeoDataFrame)
        progress = MagicMock()
        mock_get_doc.return_value = (MagicMock(), _stub_source_grid_doc())

        grid = {
            "source": {
                "name": "resample",
                "source_grid_id": "src-id",
                "alignment": {"target": "domain", "resolution": 10.0},
                "method_overrides": {},
            },
        }

        result = dispatch_handler(grid, mock_gdf, progress)

        mock_handle_resample.assert_called_once_with(mock_gdf, grid["source"], progress)
        assert result == mock_result


class TestHandleResample:
    """Tests for handle_resample function."""

    @patch("griddle.dispatch.get_document")
    @patch("griddle.dispatch.resample.resample_grid")
    def test_routes_to_resample_handler(self, mock_resample_grid, mock_get_doc):
        """handle_resample calls resample.resample_grid with correct params."""
        mock_result = MagicMock()
        mock_resample_grid.return_value = mock_result
        mock_gdf = MagicMock(spec=gpd.GeoDataFrame)
        progress = MagicMock()
        mock_get_doc.return_value = (MagicMock(), _stub_source_grid_doc())

        source = {
            "source_grid_id": "test-source-grid-id",
            "alignment": {"target": "domain", "resolution": 10.0, "method": "bilinear"},
            "method_overrides": {"fbfm": "nearest"},
        }

        result = handle_resample(mock_gdf, source, progress)

        mock_resample_grid.assert_called_once_with(
            source_grid_id="test-source-grid-id",
            alignment={
                "target": "domain",
                "resolution": 10.0,
                "method": "bilinear",
            },
            method_overrides={"fbfm": "nearest"},
            domain_gdf=mock_gdf,
            target_grid_doc=None,
            band_types={"elevation": "continuous"},
            progress=progress,
        )
        assert result == mock_result

    @patch("griddle.dispatch.get_document")
    @patch("griddle.dispatch.resample.resample_grid")
    def test_calls_progress_callback(self, mock_resample_grid, mock_get_doc):
        """handle_resample reports progress."""
        progress = MagicMock()
        mock_gdf = MagicMock(spec=gpd.GeoDataFrame)
        mock_get_doc.return_value = (MagicMock(), _stub_source_grid_doc())

        source = {
            "source_grid_id": "id",
            "alignment": {"target": "domain", "resolution": 10.0},
        }

        handle_resample(mock_gdf, source, progress)

        progress.assert_called()

    @patch("griddle.dispatch.get_document")
    @patch("griddle.dispatch.resample.resample_grid")
    def test_passes_empty_overrides_when_missing(
        self, mock_resample_grid, mock_get_doc
    ):
        """When source has no method_overrides key, passes empty dict."""
        progress = MagicMock()
        mock_gdf = MagicMock(spec=gpd.GeoDataFrame)
        mock_get_doc.return_value = (MagicMock(), _stub_source_grid_doc())

        source = {
            "source_grid_id": "id",
            "alignment": {"target": "domain", "resolution": 10.0},
            # No method_overrides key
        }

        handle_resample(mock_gdf, source, progress)

        call_kwargs = mock_resample_grid.call_args[1]
        assert call_kwargs["method_overrides"] == {}


class TestDispatchHandlerUniform:
    """Tests for dispatch_handler routing to uniform."""

    @patch("griddle.dispatch.handle_uniform")
    def test_routes_uniform_source(self, mock_handle_uniform):
        """dispatch_handler routes uniform source to handle_uniform."""
        mock_result = MagicMock()
        mock_handle_uniform.return_value = mock_result
        mock_gdf = MagicMock(spec=gpd.GeoDataFrame)
        progress = MagicMock()

        grid = {
            "source": {
                "name": "uniform",
                "bands": [{"quantity": "fuel_moisture.1hr", "value": 6.0}],
                "resolution": 2.0,
            },
            "domain_id": "test-domain-id",
        }

        result = dispatch_handler(grid, mock_gdf, progress)

        mock_handle_uniform.assert_called_once_with(mock_gdf, grid["source"], progress)
        assert result == mock_result


class TestHandleUniform:
    """Tests for handle_uniform function."""

    @patch("griddle.dispatch.uniform.create_uniform_grid")
    def test_routes_to_handler(self, mock_create):
        """handle_uniform calls create_uniform_grid with correct params."""
        mock_gdf = MagicMock(spec=gpd.GeoDataFrame)
        mock_result = MagicMock()
        mock_create.return_value = mock_result
        progress = MagicMock()

        source = {
            "bands": [{"quantity": "fuel_moisture.1hr", "value": 6.0}],
            "resolution": 2.0,
        }

        result = handle_uniform(mock_gdf, source, progress)

        mock_create.assert_called_once_with(
            domain_gdf=mock_gdf,
            bands=source["bands"],
            resolution=2.0,
            progress=progress,
        )
        assert result == mock_result

    @patch("griddle.dispatch.uniform.create_uniform_grid")
    def test_calls_progress_callback(self, mock_create):
        """handle_uniform reports progress."""
        mock_gdf = MagicMock(spec=gpd.GeoDataFrame)
        progress = MagicMock()

        source = {
            "bands": [{"quantity": "fuel_moisture.1hr", "value": 6.0}],
            "resolution": 2.0,
        }

        handle_uniform(mock_gdf, source, progress)

        progress.assert_called()
        call_args = progress.call_args_list[0][0]
        assert "uniform" in call_args[0].lower()


class TestDispatchHandlerCanopy:
    """Tests for dispatch_handler routing to canopy."""

    @patch("griddle.dispatch.handle_canopy")
    def test_routes_canopy_source(self, mock_handle_canopy):
        """dispatch_handler routes canopy source to handle_canopy."""
        mock_result = MagicMock()
        mock_handle_canopy.return_value = mock_result
        mock_gdf = MagicMock(spec=gpd.GeoDataFrame)
        progress = MagicMock()

        grid = {
            "source": {
                "name": "canopy",
                "product": "meta",
                "version": "2",
            },
            "domain_id": "test-domain-id",
        }

        result = dispatch_handler(grid, mock_gdf, progress)

        mock_handle_canopy.assert_called_once_with(mock_gdf, grid["source"], progress)
        assert result == mock_result


class TestHandleCanopy:
    """Tests for handle_canopy function."""

    @patch("griddle.dispatch.chm.fetch_meta_chm")
    def test_routes_meta_to_handler(self, mock_fetch):
        """handle_canopy routes meta product to fetch_meta_chm."""
        mock_gdf = MagicMock(spec=gpd.GeoDataFrame)
        mock_dataset = MagicMock()
        mock_metadata = {
            "tiles": ["url"],
            "tile_source": None,
            "tile_count": 1,
            "native_crs": "EPSG:32611",
            "acquisition_dates": None,
        }
        mock_fetch.return_value = (mock_dataset, mock_metadata)
        progress = MagicMock()

        source = {
            "product": "meta",
            "version": "2",
        }

        result = handle_canopy(mock_gdf, source, progress)

        mock_fetch.assert_called_once_with(
            mock_gdf,
            "2",
            progress,
            extent_buffer_cells=0,
            alignment={"target": "domain"},
            target_grid_doc=None,
        )
        assert result == mock_dataset
        assert source["tile_metadata"] == mock_metadata

    @patch("griddle.dispatch.chm.fetch_meta_chm")
    def test_default_version(self, mock_fetch):
        """handle_canopy uses 2 as default version."""
        mock_gdf = MagicMock(spec=gpd.GeoDataFrame)
        mock_fetch.return_value = (MagicMock(), {})
        progress = MagicMock()

        source = {"product": "meta"}  # No version specified

        handle_canopy(mock_gdf, source, progress)

        call_args = mock_fetch.call_args[0]
        assert call_args[1] == "2"

    def test_unknown_product_raises(self):
        """handle_canopy raises ProcessingError for unknown product."""
        mock_gdf = MagicMock(spec=gpd.GeoDataFrame)
        progress = MagicMock()

        source = {"product": "unknown_product"}

        with pytest.raises(ProcessingError) as exc_info:
            handle_canopy(mock_gdf, source, progress)

        assert exc_info.value.code == "UNKNOWN_PRODUCT"
        assert "unknown_product" in exc_info.value.message

    @patch("griddle.dispatch.chm.fetch_naip_chm")
    def test_routes_naip_to_handler(self, mock_fetch):
        """handle_canopy routes naip product to fetch_naip_chm."""
        mock_gdf = MagicMock(spec=gpd.GeoDataFrame)
        mock_dataset = MagicMock()
        mock_metadata = {
            "tiles": ["url"],
            "tile_source": None,
            "tile_count": 1,
            "native_crs": "EPSG:32611",
            "acquisition_dates": None,
        }
        mock_fetch.return_value = (mock_dataset, mock_metadata)
        progress = MagicMock()

        source = {
            "product": "naip",
            "version": "2020",
        }

        result = handle_canopy(mock_gdf, source, progress)

        mock_fetch.assert_called_once_with(
            mock_gdf,
            progress,
            extent_buffer_cells=0,
            alignment={"target": "domain"},
            target_grid_doc=None,
        )
        assert result == mock_dataset
        assert source["tile_metadata"] == mock_metadata

    @patch("griddle.dispatch.chm.fetch_meta_chm")
    def test_calls_progress_callback(self, mock_fetch):
        """handle_canopy reports progress."""
        mock_gdf = MagicMock(spec=gpd.GeoDataFrame)
        mock_fetch.return_value = (MagicMock(), {})
        progress = MagicMock()

        source = {"product": "meta", "version": "2"}

        handle_canopy(mock_gdf, source, progress)

        progress.assert_called()
        call_args = progress.call_args_list[0][0]
        assert "canopy" in call_args[0]
        assert "meta" in call_args[0]

    @patch("griddle.dispatch.chm.fetch_meta_chm")
    @pytest.mark.parametrize(
        "version,expected_license,expected_license_url",
        [
            ("1", "CC-BY-4.0", "https://creativecommons.org/licenses/by/4.0/"),
            (
                "2",
                "DINOv3",
                "https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md",
            ),
        ],
    )
    def test_meta_populates_attribution(
        self, mock_fetch, version, expected_license, expected_license_url
    ):
        """handle_canopy populates attribution for each meta version."""
        mock_gdf = MagicMock(spec=gpd.GeoDataFrame)
        mock_fetch.return_value = (MagicMock(), {})
        progress = MagicMock()

        source = {"product": "meta", "version": version}

        handle_canopy(mock_gdf, source, progress)

        assert "attribution" in source
        attr = source["attribution"]
        assert attr["license_name"] == expected_license
        assert attr["license_url"] == expected_license_url
        assert "registry.opendata.aws" in attr["access_url"]
        assert attr["accessed_on"]  # ISO date string
        assert (
            "arXiv:2603.06382" in attr["citation"]
            if version == "2"
            else "High Resolution Canopy Height Maps" in attr["citation"]
        )

    @patch("griddle.dispatch.chm.fetch_naip_chm")
    def test_naip_does_not_populate_attribution(self, mock_fetch):
        """handle_canopy does not populate attribution for naip product."""
        mock_gdf = MagicMock(spec=gpd.GeoDataFrame)
        mock_fetch.return_value = (MagicMock(), {})
        progress = MagicMock()

        source = {"product": "naip", "version": "2020"}

        handle_canopy(mock_gdf, source, progress)

        assert "attribution" not in source


class TestDispatchHandler3dep:
    """Tests for dispatch_handler routing to 3dep."""

    @patch("griddle.dispatch.handle_3dep")
    def test_routes_3dep_source(self, mock_handle_3dep):
        """dispatch_handler routes 3dep source to handle_3dep."""
        mock_result = MagicMock()
        mock_handle_3dep.return_value = mock_result
        mock_gdf = MagicMock(spec=gpd.GeoDataFrame)
        progress = MagicMock()

        grid = {
            "source": {
                "name": "3dep",
                "product": "topography",
                "source_resolution": 10,
                "bands": ["elevation"],
            },
            "domain_id": "test-domain-id",
        }

        result = dispatch_handler(grid, mock_gdf, progress)

        mock_handle_3dep.assert_called_once_with(mock_gdf, grid["source"], progress)
        assert result == mock_result


class TestHandle3dep:
    """Tests for handle_3dep function."""

    @patch("griddle.dispatch.threedep.fetch_topography")
    def test_routes_topography_to_handler(self, mock_fetch):
        """handle_3dep routes topography product to fetch_topography."""
        mock_gdf = MagicMock(spec=gpd.GeoDataFrame)
        mock_dataset = MagicMock()
        mock_metadata = {"tiles": ["url"], "tile_count": 1}
        mock_fetch.return_value = (mock_dataset, mock_metadata)
        progress = MagicMock()

        grid = {
            "source": {
                "name": "3dep",
                "product": "topography",
                "source_resolution": 10,
                "bands": ["elevation", "slope", "aspect"],
            },
        }

        result = handle_3dep(mock_gdf, grid["source"], progress)

        mock_fetch.assert_called_once_with(
            mock_gdf,
            10,
            ["elevation", "slope", "aspect"],
            progress,
            extent_buffer_cells=0,
            alignment={"target": "domain"},
            target_grid_doc=None,
        )
        assert result == mock_dataset

    @patch("griddle.dispatch.threedep.fetch_topography")
    def test_nests_tile_metadata_in_source(self, mock_fetch):
        """handle_3dep stores tile metadata under source['tile_metadata']."""
        mock_gdf = MagicMock(spec=gpd.GeoDataFrame)
        mock_dataset = MagicMock()
        mock_metadata = {
            "tiles": ["https://example.com/tile.tif"],
            "tile_source": None,
            "tile_count": 1,
            "native_crs": "EPSG:4326",
            "acquisition_dates": None,
        }
        mock_fetch.return_value = (mock_dataset, mock_metadata)
        progress = MagicMock()

        source = {
            "name": "3dep",
            "product": "topography",
            "source_resolution": 10,
            "bands": ["elevation"],
        }

        handle_3dep(mock_gdf, source, progress)

        assert source["tile_metadata"] == mock_metadata
        assert source["tile_metadata"]["tile_count"] == 1
        assert source["tile_metadata"]["tiles"] == ["https://example.com/tile.tif"]
        assert source["tile_metadata"]["native_crs"] == "EPSG:4326"

    @patch("griddle.dispatch.threedep.fetch_topography")
    def test_default_source_resolution(self, mock_fetch):
        """handle_3dep uses 10 as default source_resolution."""
        mock_gdf = MagicMock(spec=gpd.GeoDataFrame)
        mock_fetch.return_value = (MagicMock(), {})
        progress = MagicMock()

        source = {"product": "topography", "bands": ["elevation"]}

        handle_3dep(mock_gdf, source, progress)

        call_args = mock_fetch.call_args[0]
        assert call_args[1] == 10

    def test_unknown_product_raises(self):
        """handle_3dep raises ProcessingError for unknown product."""
        mock_gdf = MagicMock(spec=gpd.GeoDataFrame)
        progress = MagicMock()

        source = {"product": "unknown_product", "source_resolution": 10}

        with pytest.raises(ProcessingError) as exc_info:
            handle_3dep(mock_gdf, source, progress)

        assert exc_info.value.code == "UNKNOWN_PRODUCT"
        assert "unknown_product" in exc_info.value.message

    @patch("griddle.dispatch.threedep.fetch_topography")
    def test_calls_progress_callback(self, mock_fetch):
        """handle_3dep reports progress."""
        mock_gdf = MagicMock(spec=gpd.GeoDataFrame)
        mock_fetch.return_value = (MagicMock(), {})
        progress = MagicMock()

        source = {
            "product": "topography",
            "source_resolution": 10,
            "bands": ["elevation"],
        }

        handle_3dep(mock_gdf, source, progress)

        progress.assert_called()
        call_args = progress.call_args_list[0][0]
        assert "3DEP" in call_args[0]
        assert "topography" in call_args[0]
