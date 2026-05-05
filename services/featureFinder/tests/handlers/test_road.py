"""Tests for road feature handler."""

from unittest.mock import MagicMock, patch

import geopandas as gpd
import pytest
from featureFinder.handlers.road import (
    buffer_roads,
    compute_georeference,
    handle_osm,
)
from shapely.geometry import LineString, Polygon


@pytest.fixture
def sample_domain_gdf():
    """Create a sample domain GeoDataFrame in a projected CRS (UTM)."""
    # A simple 1km x 1km square
    poly = Polygon(
        [(500000, 4500000), (501000, 4500000), (501000, 4501000), (500000, 4501000)]
    )
    return gpd.GeoDataFrame({"geometry": [poly]}, crs="EPSG:32610")


@pytest.fixture
def sample_osm_roads():
    """Create a sample OSM response with lines in WGS84."""
    return gpd.GeoDataFrame(
        {
            "highway": ["motorway", "path", "invalid_type"],
            "name": ["Highway 1", "Local Trail", "Fake Road"],
            "geometry": [
                LineString([(-120.0, 40.0), (-120.0, 40.01)]),
                LineString([(-120.01, 40.0), (-120.01, 40.01)]),
                LineString([(-120.02, 40.0), (-120.02, 40.01)]),
            ],
        },
        crs="EPSG:4326",
    )


class TestBufferRoads:
    def test_missing_highway_column(self):
        """Should return an empty GDF if the 'highway' column is missing."""
        gdf = gpd.GeoDataFrame(
            {"geometry": [LineString([(0, 0), (1, 1)])]}, crs="EPSG:4326"
        )
        result = buffer_roads(gdf)
        assert result.empty
        assert result.crs == "EPSG:4326"

    def test_drops_invalid_highways(self, sample_osm_roads):
        """Should drop roads with tags not in ROAD_DATA."""
        result = buffer_roads(sample_osm_roads)
        # Only 'motorway' and 'path' should remain, 'invalid_type' dropped
        assert len(result) == 2
        assert set(result["highway"].values) == {"motorway", "path"}

    def test_buffers_lines_to_polygons(self, sample_osm_roads):
        """Should convert LineStrings to Polygons representing road width."""
        result = buffer_roads(sample_osm_roads)

        # Ensure they are now polygons
        assert all(geom.geom_type == "Polygon" for geom in result.geometry)

        # Ensure it returns in EPSG:4326
        assert result.crs == "EPSG:4326"

    def test_empty_input(self):
        """Should handle an empty input gracefully."""
        gdf = gpd.GeoDataFrame({"highway": [], "geometry": []}, crs="EPSG:4326")
        result = buffer_roads(gdf)
        assert result.empty


class TestComputeGeoreference:
    def test_compute_georeference(self, sample_domain_gdf):
        """Should correctly extract CRS and bounds."""
        geo = compute_georeference(sample_domain_gdf)

        assert geo["crs"] == "EPSG:32610"
        assert len(geo["bounds"]) == 4
        assert geo["bounds"] == [500000.0, 4500000.0, 501000.0, 4501000.0]


class TestHandleOsm:
    @pytest.fixture
    def mock_feature(self):
        return {"id": "feat-123", "domain_id": "dom-456", "type": "road"}

    @patch("featureFinder.handlers.road.save_geojson")
    @patch("featureFinder.handlers.road.ox.features_from_polygon")
    def test_happy_path(
        self, mock_osmnx, mock_save, mock_feature, sample_domain_gdf, sample_osm_roads
    ):
        """Should fetch, buffer, format, and save the roads."""
        mock_osmnx.return_value = sample_osm_roads
        progress = MagicMock()

        result = handle_osm(mock_feature, {}, sample_domain_gdf, progress)

        # Ensure OSMnx was called
        mock_osmnx.assert_called_once()

        # Ensure save_geojson was called
        mock_save.assert_called_once()
        args, _ = mock_save.call_args

        assert args[0] == "dom-456"
        assert args[1] == "feat-123"
        saved_gdf = args[2]

        # Saved GDF should have specific columns
        assert list(saved_gdf.columns) == ["geometry", "type", "name"]

        # Saved GDF should project back to the domain's native CRS
        assert saved_gdf.crs == sample_domain_gdf.crs

        # Ensure georeference is returned
        assert "georeference" in result
        assert result["georeference"]["crs"] == "EPSG:32610"

    @patch("featureFinder.handlers.road.save_geojson")
    @patch("featureFinder.handlers.road.ox.features_from_polygon")
    def test_empty_osmnx_response(
        self, mock_osmnx, mock_save, mock_feature, sample_domain_gdf
    ):
        """Should handle an empty response from OSMnx gracefully."""
        # Return an empty GDF
        mock_osmnx.return_value = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        progress = MagicMock()

        handle_osm(mock_feature, {}, sample_domain_gdf, progress)

        # save_geojson should be called with an empty GDF in the native CRS
        mock_save.assert_called_once()
        saved_gdf = mock_save.call_args[0][2]
        assert saved_gdf.empty
        assert saved_gdf.crs == sample_domain_gdf.crs

    @patch("featureFinder.handlers.road.save_geojson")
    @patch("featureFinder.handlers.road.ox.features_from_polygon")
    def test_osmnx_raises_exception(
        self, mock_osmnx, mock_save, mock_feature, sample_domain_gdf
    ):
        """Should handle connection errors or internal OSMnx exceptions gracefully."""
        # Simulate network error or No Data found exception from OSMnx
        mock_osmnx.side_effect = Exception("OSM API is down")
        progress = MagicMock()

        handle_osm(mock_feature, {}, sample_domain_gdf, progress)

        # Should swallow the exception and save an empty GeoJSON
        mock_save.assert_called_once()
        saved_gdf = mock_save.call_args[0][2]
        assert saved_gdf.empty
        assert saved_gdf.crs == sample_domain_gdf.crs
