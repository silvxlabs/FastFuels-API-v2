"""
Tests for dispatch module.
"""

import json
from unittest.mock import MagicMock, patch

import geopandas as gpd
import pytest
from griddle.dispatch import (
    dispatch_handler,
    handle_chm,
    handle_landfire,
    handle_lookup,
    handle_pim,
    handle_resample,
    handle_uniform,
    load_domain_gdf,
)
from griddle.errors import ProcessingError


class TestLoadDomainGdf:
    """Tests for load_domain_gdf function."""

    @patch("griddle.dispatch.get_document")
    def test_returns_geodataframe(self, mock_get_document):
        """load_domain_gdf returns a GeoDataFrame."""
        mock_snapshot = MagicMock()
        mock_snapshot.to_dict.return_value = {
            "crs": {"properties": {"name": "EPSG:4326"}, "type": "name"},
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [
                            [
                                [-120.0, 38.0],
                                [-119.0, 38.0],
                                [-119.0, 39.0],
                                [-120.0, 39.0],
                                [-120.0, 38.0],
                            ]
                        ],
                    },
                    "properties": {},
                }
            ],
        }
        mock_get_document.return_value = (None, mock_snapshot)

        result = load_domain_gdf("test-domain-id")

        assert isinstance(result, gpd.GeoDataFrame)
        assert result.crs.to_epsg() == 4326
        assert len(result) == 1

    @patch("griddle.dispatch.get_document")
    def test_multiple_features(self, mock_get_document):
        """load_domain_gdf handles multiple features."""
        mock_snapshot = MagicMock()
        mock_snapshot.to_dict.return_value = {
            "crs": {"properties": {"name": "EPSG:32610"}, "type": "name"},
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]],
                    },
                    "properties": {},
                },
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [
                            [[20, 20], [30, 20], [30, 30], [20, 30], [20, 20]]
                        ],
                    },
                    "properties": {},
                },
            ],
        }
        mock_get_document.return_value = (None, mock_snapshot)

        result = load_domain_gdf("test-domain-id")

        assert len(result) == 2
        bounds = result.total_bounds
        assert bounds[0] == 0  # minx
        assert bounds[1] == 0  # miny
        assert bounds[2] == 30  # maxx
        assert bounds[3] == 30  # maxy

    @patch("griddle.dispatch.get_document")
    def test_empty_features_raises(self, mock_get_document):
        """Raise error when domain has no features."""
        mock_snapshot = MagicMock()
        mock_snapshot.to_dict.return_value = {"crs": "EPSG:4326", "features": []}
        mock_get_document.return_value = (None, mock_snapshot)

        with pytest.raises(ProcessingError) as exc_info:
            load_domain_gdf("test-domain-id")

        assert exc_info.value.code == "EMPTY_DOMAIN"

    @patch("griddle.dispatch.get_document")
    def test_missing_features_raises(self, mock_get_document):
        """Raise error when domain has no features key."""
        mock_snapshot = MagicMock()
        mock_snapshot.to_dict.return_value = {"crs": "EPSG:4326"}
        mock_get_document.return_value = (None, mock_snapshot)

        with pytest.raises(ProcessingError) as exc_info:
            load_domain_gdf("test-domain-id")

        assert exc_info.value.code == "EMPTY_DOMAIN"

    @patch("griddle.dispatch.get_document")
    def test_default_crs(self, mock_get_document):
        """Use EPSG:4326 as default CRS when not specified."""
        mock_snapshot = MagicMock()
        mock_snapshot.to_dict.return_value = {
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [
                            [
                                [-120.0, 38.0],
                                [-119.0, 38.0],
                                [-119.0, 39.0],
                                [-120.0, 38.0],
                            ]
                        ],
                    },
                    "properties": {},
                }
            ],
        }
        mock_get_document.return_value = (None, mock_snapshot)

        result = load_domain_gdf("test-domain-id")

        assert result.crs.to_epsg() == 4326

    @patch("griddle.dispatch.get_document")
    def test_crs_as_geojson_object(self, mock_get_document):
        """Handle CRS stored as GeoJSON CRS object (production Firestore format)."""
        mock_snapshot = MagicMock()
        mock_snapshot.to_dict.return_value = {
            "crs": {"properties": {"name": "EPSG:32611"}, "type": "name"},
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [
                            [
                                [300000, 4100000],
                                [310000, 4100000],
                                [310000, 4110000],
                                [300000, 4110000],
                                [300000, 4100000],
                            ]
                        ],
                    },
                    "properties": {},
                }
            ],
        }
        mock_get_document.return_value = (None, mock_snapshot)

        result = load_domain_gdf("test-domain-id")

        assert isinstance(result, gpd.GeoDataFrame)
        assert result.crs.to_epsg() == 32611
        assert len(result) == 1

    @patch("griddle.dispatch.get_document")
    def test_coordinates_as_json_strings(self, mock_get_document):
        """Handle coordinates stored as JSON strings (Firestore serialization)."""
        mock_snapshot = MagicMock()
        mock_snapshot.to_dict.return_value = {
            "crs": {"properties": {"name": "EPSG:4326"}, "type": "name"},
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": json.dumps(
                            [
                                [
                                    [-120.0, 38.0],
                                    [-119.0, 38.0],
                                    [-119.0, 39.0],
                                    [-120.0, 39.0],
                                    [-120.0, 38.0],
                                ]
                            ]
                        ),
                    },
                    "properties": {},
                }
            ],
        }
        mock_get_document.return_value = (None, mock_snapshot)

        result = load_domain_gdf("test-domain-id")

        assert isinstance(result, gpd.GeoDataFrame)
        assert result.crs.to_epsg() == 4326
        assert len(result) == 1
        bounds = result.total_bounds
        assert bounds[0] == pytest.approx(-120.0)
        assert bounds[2] == pytest.approx(-119.0)

    @patch("griddle.dispatch.get_document")
    def test_realistic_firestore_domain(self, mock_get_document):
        """End-to-end test with realistic Firestore domain data.

        Combines both serialization quirks: CRS as GeoJSON object and
        coordinates as JSON strings.
        """
        mock_snapshot = MagicMock()
        mock_snapshot.to_dict.return_value = {
            "crs": {"properties": {"name": "EPSG:32611"}, "type": "name"},
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": json.dumps(
                            [
                                [
                                    [300000.0, 4100000.0],
                                    [310000.0, 4100000.0],
                                    [310000.0, 4110000.0],
                                    [300000.0, 4110000.0],
                                    [300000.0, 4100000.0],
                                ]
                            ]
                        ),
                    },
                    "properties": {},
                }
            ],
        }
        mock_get_document.return_value = (None, mock_snapshot)

        result = load_domain_gdf("test-domain-id")

        assert isinstance(result, gpd.GeoDataFrame)
        assert result.crs.to_epsg() == 32611
        assert len(result) == 1
        bounds = result.total_bounds
        assert bounds[0] == pytest.approx(300000.0)
        assert bounds[2] == pytest.approx(310000.0)

    @patch("griddle.dispatch.get_document")
    def test_domain_not_found_raises(self, mock_get_document):
        """Raise ProcessingError when domain not found."""
        from lib.firestore import DocumentNotFoundError

        mock_get_document.side_effect = DocumentNotFoundError("domains", "missing-id")

        with pytest.raises(ProcessingError) as exc_info:
            load_domain_gdf("missing-id")

        assert exc_info.value.code == "DOMAIN_NOT_FOUND"

    @patch("griddle.dispatch.get_document")
    def test_invalid_geometry_raises(self, mock_get_document):
        """Raise ProcessingError when geometry is malformed."""
        mock_snapshot = MagicMock()
        mock_snapshot.to_dict.return_value = {
            "crs": {"properties": {"name": "EPSG:4326"}, "type": "name"},
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": "not-valid-json{{{",
                    },
                    "properties": {},
                }
            ],
        }
        mock_get_document.return_value = (None, mock_snapshot)

        with pytest.raises(ProcessingError) as exc_info:
            load_domain_gdf("test-domain-id")

        assert exc_info.value.code == "INVALID_GEOMETRY"


class TestHandleLandfire:
    """Tests for handle_landfire function."""

    @patch("griddle.dispatch.landfire.fetch_fbfm40")
    @patch("griddle.dispatch.load_domain_gdf")
    def test_routes_fbfm40_to_handler(self, mock_load_domain, mock_fetch):
        """handle_landfire routes fbfm40 product to fetch_fbfm40."""
        mock_gdf = MagicMock(spec=gpd.GeoDataFrame)
        mock_load_domain.return_value = mock_gdf
        mock_result = MagicMock()
        mock_fetch.return_value = mock_result
        progress = MagicMock()

        grid = {"domain_id": "test-domain-id"}
        source = {"product": "fbfm40", "version": "2022"}

        result = handle_landfire(grid, source, progress)

        mock_load_domain.assert_called_once_with("test-domain-id")
        mock_fetch.assert_called_once_with(mock_gdf, "2022")
        assert result == mock_result

    @patch("griddle.dispatch.landfire.fetch_fbfm40")
    @patch("griddle.dispatch.load_domain_gdf")
    def test_default_version(self, mock_load_domain, mock_fetch):
        """handle_landfire uses 2022 as default version."""
        mock_load_domain.return_value = MagicMock(spec=gpd.GeoDataFrame)
        progress = MagicMock()

        grid = {"domain_id": "test-domain-id"}
        source = {"product": "fbfm40"}  # No version specified

        handle_landfire(grid, source, progress)

        # Check that fetch_fbfm40 was called with default version
        _, call_kwargs = mock_fetch.call_args
        assert call_kwargs == {} or mock_fetch.call_args[0][1] == "2022"

    @patch("griddle.dispatch.load_domain_gdf")
    def test_unknown_product_raises(self, mock_load_domain):
        """handle_landfire raises ProcessingError for unknown product."""
        mock_load_domain.return_value = MagicMock(spec=gpd.GeoDataFrame)
        progress = MagicMock()

        grid = {"domain_id": "test-domain-id"}
        source = {"product": "unknown_product"}

        with pytest.raises(ProcessingError) as exc_info:
            handle_landfire(grid, source, progress)

        assert exc_info.value.code == "UNKNOWN_PRODUCT"
        assert "unknown_product" in exc_info.value.message

    @patch("griddle.dispatch.landfire.fetch_fbfm40")
    @patch("griddle.dispatch.load_domain_gdf")
    def test_calls_progress_callback(self, mock_load_domain, mock_fetch):
        """handle_landfire reports progress."""
        mock_load_domain.return_value = MagicMock(spec=gpd.GeoDataFrame)
        progress = MagicMock()

        grid = {"domain_id": "test-domain-id"}
        source = {"product": "fbfm40", "version": "2022"}

        handle_landfire(grid, source, progress)

        progress.assert_called_once()
        call_args = progress.call_args[0]
        assert "LANDFIRE" in call_args[0]
        assert "fbfm40" in call_args[0]

    @patch("griddle.dispatch.landfire.fetch_topography")
    @patch("griddle.dispatch.load_domain_gdf")
    def test_routes_topography_to_handler(self, mock_load_domain, mock_fetch):
        """handle_landfire routes topography product to fetch_topography."""
        mock_gdf = MagicMock(spec=gpd.GeoDataFrame)
        mock_load_domain.return_value = mock_gdf
        mock_result = MagicMock()
        mock_fetch.return_value = mock_result
        progress = MagicMock()

        grid = {"domain_id": "test-domain-id"}
        source = {
            "product": "topography",
            "version": "2020",
            "bands": ["elevation", "slope", "aspect"],
        }

        result = handle_landfire(grid, source, progress)

        mock_load_domain.assert_called_once_with("test-domain-id")
        mock_fetch.assert_called_once_with(
            mock_gdf, "2020", ["elevation", "slope", "aspect"], progress
        )
        assert result == mock_result

    @patch("griddle.dispatch.landfire.fetch_topography")
    @patch("griddle.dispatch.load_domain_gdf")
    def test_topography_passes_bands(self, mock_load_domain, mock_fetch):
        """handle_landfire passes band list to topography handler."""
        mock_load_domain.return_value = MagicMock(spec=gpd.GeoDataFrame)
        progress = MagicMock()

        grid = {"domain_id": "test-domain-id"}
        source = {
            "product": "topography",
            "version": "2020",
            "bands": ["elevation"],
        }

        handle_landfire(grid, source, progress)

        call_args = mock_fetch.call_args[0]
        assert call_args[2] == ["elevation"]

    @patch("griddle.dispatch.landfire.fetch_topography")
    @patch("griddle.dispatch.load_domain_gdf")
    def test_topography_calls_progress(self, mock_load_domain, mock_fetch):
        """handle_landfire reports progress for topography."""
        mock_load_domain.return_value = MagicMock(spec=gpd.GeoDataFrame)
        progress = MagicMock()

        grid = {"domain_id": "test-domain-id"}
        source = {
            "product": "topography",
            "version": "2020",
            "bands": ["elevation"],
        }

        handle_landfire(grid, source, progress)

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
        progress = MagicMock()

        grid = {
            "source": {"name": "landfire", "product": "fbfm40"},
            "domain_id": "test-domain-id",
        }

        result = dispatch_handler(grid, progress)

        mock_handle_landfire.assert_called_once_with(grid, grid["source"], progress)
        assert result == mock_result

    def test_unknown_source_raises(self):
        """dispatch_handler raises ProcessingError for unknown source."""
        progress = MagicMock()

        grid = {
            "source": {"name": "unknown_source"},
            "domain_id": "test-domain-id",
        }

        with pytest.raises(ProcessingError) as exc_info:
            dispatch_handler(grid, progress)

        assert exc_info.value.code == "UNKNOWN_SOURCE"
        assert "unknown_source" in exc_info.value.message

    @patch("griddle.dispatch.handle_lookup")
    def test_routes_lookup_source(self, mock_handle_lookup):
        """dispatch_handler routes lookup source to handle_lookup."""
        mock_result = MagicMock()
        mock_handle_lookup.return_value = mock_result
        progress = MagicMock()

        grid = {
            "source": {"name": "lookup", "table": "fbfm40", "source_grid_id": "src-id"},
            "bands": [{"key": "fuel_load.1hr"}],
        }

        result = dispatch_handler(grid, progress)

        mock_handle_lookup.assert_called_once_with(grid, grid["source"], progress)
        assert result == mock_result


class TestDispatchHandlerPim:
    """Tests for dispatch_handler routing to pim."""

    @patch("griddle.dispatch.handle_pim")
    def test_routes_pim_source(self, mock_handle_pim):
        """dispatch_handler routes pim source to handle_pim."""
        mock_result = MagicMock()
        mock_handle_pim.return_value = mock_result
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

        result = dispatch_handler(grid, progress)

        mock_handle_pim.assert_called_once_with(grid, grid["source"], progress)
        assert result == mock_result


class TestHandlePim:
    """Tests for handle_pim function."""

    @patch("griddle.dispatch.pim.fetch_treemap")
    @patch("griddle.dispatch.load_domain_gdf")
    def test_routes_treemap_to_handler(self, mock_load_domain, mock_fetch):
        """handle_pim routes treemap product to fetch_treemap."""
        mock_gdf = MagicMock(spec=gpd.GeoDataFrame)
        mock_load_domain.return_value = mock_gdf
        mock_result = MagicMock()
        mock_fetch.return_value = mock_result
        progress = MagicMock()

        grid = {"domain_id": "test-domain-id"}
        source = {
            "product": "treemap",
            "version": "2022",
            "bands": ["tm_id", "plt_cn"],
        }

        result = handle_pim(grid, source, progress)

        mock_load_domain.assert_called_once_with("test-domain-id")
        mock_fetch.assert_called_once_with(
            mock_gdf, "2022", ["tm_id", "plt_cn"], progress
        )
        assert result == mock_result

    @patch("griddle.dispatch.pim.fetch_treemap")
    @patch("griddle.dispatch.load_domain_gdf")
    def test_default_version(self, mock_load_domain, mock_fetch):
        """handle_pim uses 2022 as default version."""
        mock_load_domain.return_value = MagicMock(spec=gpd.GeoDataFrame)
        progress = MagicMock()

        grid = {"domain_id": "test-domain-id"}
        source = {"product": "treemap", "bands": ["tm_id"]}

        handle_pim(grid, source, progress)

        call_args = mock_fetch.call_args[0]
        assert call_args[1] == "2022"

    @patch("griddle.dispatch.pim.fetch_treemap")
    @patch("griddle.dispatch.load_domain_gdf")
    def test_passes_bands_list(self, mock_load_domain, mock_fetch):
        """handle_pim passes bands list to treemap handler."""
        mock_load_domain.return_value = MagicMock(spec=gpd.GeoDataFrame)
        progress = MagicMock()

        grid = {"domain_id": "test-domain-id"}
        source = {
            "product": "treemap",
            "version": "2022",
            "bands": ["tm_id"],
        }

        handle_pim(grid, source, progress)

        call_args = mock_fetch.call_args[0]
        assert call_args[2] == ["tm_id"]

    @patch("griddle.dispatch.pim.fetch_treemap")
    @patch("griddle.dispatch.load_domain_gdf")
    def test_default_bands_when_missing(self, mock_load_domain, mock_fetch):
        """handle_pim defaults to tm_id band when bands key is missing."""
        mock_load_domain.return_value = MagicMock(spec=gpd.GeoDataFrame)
        progress = MagicMock()

        grid = {"domain_id": "test-domain-id"}
        source = {"product": "treemap", "version": "2022"}

        handle_pim(grid, source, progress)

        call_args = mock_fetch.call_args[0]
        assert call_args[2] == ["tm_id"]

    @patch("griddle.dispatch.load_domain_gdf")
    def test_unknown_product_raises(self, mock_load_domain):
        """handle_pim raises ProcessingError for unknown product."""
        mock_load_domain.return_value = MagicMock(spec=gpd.GeoDataFrame)
        progress = MagicMock()

        grid = {"domain_id": "test-domain-id"}
        source = {"product": "unknown_product"}

        with pytest.raises(ProcessingError) as exc_info:
            handle_pim(grid, source, progress)

        assert exc_info.value.code == "UNKNOWN_PRODUCT"
        assert "unknown_product" in exc_info.value.message

    @patch("griddle.dispatch.pim.fetch_treemap")
    @patch("griddle.dispatch.load_domain_gdf")
    def test_calls_progress_callback(self, mock_load_domain, mock_fetch):
        """handle_pim reports progress."""
        mock_load_domain.return_value = MagicMock(spec=gpd.GeoDataFrame)
        progress = MagicMock()

        grid = {"domain_id": "test-domain-id"}
        source = {"product": "treemap", "version": "2022", "bands": ["tm_id"]}

        handle_pim(grid, source, progress)

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


class TestDispatchHandlerResample:
    """Tests for dispatch_handler routing to resample."""

    @patch("griddle.dispatch.handle_resample")
    def test_routes_resample_source(self, mock_handle_resample):
        """dispatch_handler routes resample source to handle_resample."""
        mock_result = MagicMock()
        mock_handle_resample.return_value = mock_result
        progress = MagicMock()

        grid = {
            "source": {
                "name": "resample",
                "source_grid_id": "src-id",
                "target_resolution": 10.0,
                "method": "bilinear",
            },
        }

        result = dispatch_handler(grid, progress)

        mock_handle_resample.assert_called_once_with(grid, grid["source"], progress)
        assert result == mock_result


class TestHandleResample:
    """Tests for handle_resample function."""

    @patch("griddle.dispatch.resample.resample_grid")
    def test_routes_to_resample_handler(self, mock_resample_grid):
        """handle_resample calls resample.resample_grid with correct params."""
        mock_result = MagicMock()
        mock_resample_grid.return_value = mock_result
        progress = MagicMock()

        grid = {}
        source = {
            "source_grid_id": "test-source-grid-id",
            "target_resolution": 10.0,
            "method": "bilinear",
            "method_overrides": {"fbfm": "nearest"},
        }

        result = handle_resample(grid, source, progress)

        mock_resample_grid.assert_called_once_with(
            source_grid_id="test-source-grid-id",
            target_resolution=10.0,
            method="bilinear",
            method_overrides={"fbfm": "nearest"},
            progress=progress,
        )
        assert result == mock_result

    @patch("griddle.dispatch.resample.resample_grid")
    def test_calls_progress_callback(self, mock_resample_grid):
        """handle_resample reports progress."""
        progress = MagicMock()

        grid = {}
        source = {
            "source_grid_id": "id",
            "target_resolution": 10.0,
            "method": "bilinear",
        }

        handle_resample(grid, source, progress)

        progress.assert_called()

    @patch("griddle.dispatch.resample.resample_grid")
    def test_passes_empty_overrides_when_missing(self, mock_resample_grid):
        """When source has no method_overrides key, passes empty dict."""
        progress = MagicMock()

        grid = {}
        source = {
            "source_grid_id": "id",
            "target_resolution": 10.0,
            "method": "bilinear",
            # No method_overrides key
        }

        handle_resample(grid, source, progress)

        call_kwargs = mock_resample_grid.call_args[1]
        assert call_kwargs["method_overrides"] == {}


class TestDispatchHandlerUniform:
    """Tests for dispatch_handler routing to uniform."""

    @patch("griddle.dispatch.handle_uniform")
    def test_routes_uniform_source(self, mock_handle_uniform):
        """dispatch_handler routes uniform source to handle_uniform."""
        mock_result = MagicMock()
        mock_handle_uniform.return_value = mock_result
        progress = MagicMock()

        grid = {
            "source": {
                "name": "uniform",
                "bands": [{"quantity": "fuel_moisture.1hr", "value": 6.0}],
                "resolution": 2.0,
            },
            "domain_id": "test-domain-id",
        }

        result = dispatch_handler(grid, progress)

        mock_handle_uniform.assert_called_once_with(grid, grid["source"], progress)
        assert result == mock_result


class TestHandleUniform:
    """Tests for handle_uniform function."""

    @patch("griddle.dispatch.uniform.create_uniform_grid")
    @patch("griddle.dispatch.load_domain_gdf")
    def test_routes_to_handler(self, mock_load_domain, mock_create):
        """handle_uniform calls create_uniform_grid with correct params."""
        mock_gdf = MagicMock(spec=gpd.GeoDataFrame)
        mock_load_domain.return_value = mock_gdf
        mock_result = MagicMock()
        mock_create.return_value = mock_result
        progress = MagicMock()

        grid = {"domain_id": "test-domain-id"}
        source = {
            "bands": [{"quantity": "fuel_moisture.1hr", "value": 6.0}],
            "resolution": 2.0,
        }

        result = handle_uniform(grid, source, progress)

        mock_load_domain.assert_called_once_with("test-domain-id")
        mock_create.assert_called_once_with(
            domain_gdf=mock_gdf,
            bands=source["bands"],
            resolution=2.0,
            progress=progress,
        )
        assert result == mock_result

    @patch("griddle.dispatch.uniform.create_uniform_grid")
    @patch("griddle.dispatch.load_domain_gdf")
    def test_calls_progress_callback(self, mock_load_domain, mock_create):
        """handle_uniform reports progress."""
        mock_load_domain.return_value = MagicMock(spec=gpd.GeoDataFrame)
        progress = MagicMock()

        grid = {"domain_id": "test-domain-id"}
        source = {
            "bands": [{"quantity": "fuel_moisture.1hr", "value": 6.0}],
            "resolution": 2.0,
        }

        handle_uniform(grid, source, progress)

        progress.assert_called()
        call_args = progress.call_args_list[0][0]
        assert "uniform" in call_args[0].lower()


class TestDispatchHandlerChm:
    """Tests for dispatch_handler routing to chm."""

    @patch("griddle.dispatch.handle_chm")
    def test_routes_chm_source(self, mock_handle_chm):
        """dispatch_handler routes chm source to handle_chm."""
        mock_result = MagicMock()
        mock_handle_chm.return_value = mock_result
        progress = MagicMock()

        grid = {
            "source": {
                "name": "chm",
                "product": "meta",
                "version": "2024",
            },
            "domain_id": "test-domain-id",
        }

        result = dispatch_handler(grid, progress)

        mock_handle_chm.assert_called_once_with(grid, grid["source"], progress)
        assert result == mock_result


class TestHandleChm:
    """Tests for handle_chm function."""

    @patch("griddle.dispatch.chm.fetch_meta_chm")
    @patch("griddle.dispatch.load_domain_gdf")
    def test_routes_meta_to_handler(self, mock_load_domain, mock_fetch):
        """handle_chm routes meta product to fetch_meta_chm."""
        mock_gdf = MagicMock(spec=gpd.GeoDataFrame)
        mock_load_domain.return_value = mock_gdf
        mock_result = MagicMock()
        mock_fetch.return_value = mock_result
        progress = MagicMock()

        grid = {"domain_id": "test-domain-id"}
        source = {
            "product": "meta",
            "version": "2024",
        }

        result = handle_chm(grid, source, progress)

        mock_load_domain.assert_called_once_with("test-domain-id")
        mock_fetch.assert_called_once_with(mock_gdf, "2024", progress)
        assert result == mock_result

    @patch("griddle.dispatch.chm.fetch_meta_chm")
    @patch("griddle.dispatch.load_domain_gdf")
    def test_default_version(self, mock_load_domain, mock_fetch):
        """handle_chm uses 2024 as default version."""
        mock_load_domain.return_value = MagicMock(spec=gpd.GeoDataFrame)
        progress = MagicMock()

        grid = {"domain_id": "test-domain-id"}
        source = {"product": "meta"}  # No version specified

        handle_chm(grid, source, progress)

        call_args = mock_fetch.call_args[0]
        assert call_args[1] == "2024"

    @patch("griddle.dispatch.load_domain_gdf")
    def test_unknown_product_raises(self, mock_load_domain):
        """handle_chm raises ProcessingError for unknown product."""
        mock_load_domain.return_value = MagicMock(spec=gpd.GeoDataFrame)
        progress = MagicMock()

        grid = {"domain_id": "test-domain-id"}
        source = {"product": "unknown_product"}

        with pytest.raises(ProcessingError) as exc_info:
            handle_chm(grid, source, progress)

        assert exc_info.value.code == "UNKNOWN_PRODUCT"
        assert "unknown_product" in exc_info.value.message

    @patch("griddle.dispatch.chm.fetch_meta_chm")
    @patch("griddle.dispatch.load_domain_gdf")
    def test_calls_progress_callback(self, mock_load_domain, mock_fetch):
        """handle_chm reports progress."""
        mock_load_domain.return_value = MagicMock(spec=gpd.GeoDataFrame)
        progress = MagicMock()

        grid = {"domain_id": "test-domain-id"}
        source = {"product": "meta", "version": "2024"}

        handle_chm(grid, source, progress)

        progress.assert_called()
        call_args = progress.call_args_list[0][0]
        assert "CHM" in call_args[0]
        assert "meta" in call_args[0]
