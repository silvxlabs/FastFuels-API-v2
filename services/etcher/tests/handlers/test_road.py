"""Tests for road feature handler."""

from unittest.mock import MagicMock, patch

import geopandas as gpd
import pytest
from etcher.handlers.road import (
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
    """Create a sample OSM FlatGeobuf read result with lines in WGS84."""
    return gpd.GeoDataFrame(
        {
            "osm_id": [1, 2, 3],
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

    @patch("etcher.handlers.road.save_features")
    @patch("etcher.handlers.road.read_osm_features")
    def test_happy_path(
        self, mock_read, mock_save, mock_feature, sample_domain_gdf, sample_osm_roads
    ):
        """Should read, buffer, format, and save the roads."""
        mock_read.return_value = sample_osm_roads
        progress = MagicMock()

        result = handle_osm(mock_feature, {}, sample_domain_gdf, progress)

        # Ensure the OSM source was read for roads.
        mock_read.assert_called_once()
        assert mock_read.call_args.args[1] == "road"

        # Ensure save_features was called
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

    @patch("etcher.handlers.road.save_features")
    @patch("etcher.handlers.road.read_osm_features")
    def test_empty_reader_response(
        self, mock_read, mock_save, mock_feature, sample_domain_gdf
    ):
        """An empty read (no roads / absent layer) saves an empty Parquet."""
        # Return an empty GDF
        mock_read.return_value = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        progress = MagicMock()

        handle_osm(mock_feature, {}, sample_domain_gdf, progress)

        # save_features should be called with an empty GDF in the native CRS
        mock_save.assert_called_once()
        saved_gdf = mock_save.call_args[0][2]
        assert saved_gdf.empty
        assert saved_gdf.crs == sample_domain_gdf.crs

    @patch("etcher.handlers.road.save_features")
    @patch("etcher.handlers.road.read_osm_features")
    def test_reader_error_propagates(
        self, mock_read, mock_save, mock_feature, sample_domain_gdf
    ):
        """A real read error must propagate, not be saved as an empty result."""
        # A genuine GCS/read failure (not an absent layer) raises from the reader.
        mock_read.side_effect = RuntimeError("vsigs read failed")
        progress = MagicMock()

        with pytest.raises(RuntimeError):
            handle_osm(mock_feature, {}, sample_domain_gdf, progress)

        # Nothing should be written when the read fails.
        mock_save.assert_not_called()

    @patch("etcher.handlers.road.save_features")
    @patch("etcher.handlers.road.read_osm_features")
    def test_extent_buffer_widens_read_bbox_and_georeference(
        self, mock_read, mock_save, mock_feature, sample_domain_gdf, sample_osm_roads
    ):
        """A non-zero extent_buffer_m enlarges the read bbox and the georeference."""
        mock_read.return_value = sample_osm_roads
        progress = MagicMock()

        # First call: no buffer.
        result_no_buf = handle_osm(mock_feature, {}, sample_domain_gdf, progress)
        bbox_no_buf = mock_read.call_args.args[0]
        georef_no_buf = result_no_buf["georeference"]["bounds"]

        # Second call: 50 m buffer.
        result_buf = handle_osm(
            mock_feature,
            {"extent_buffer_m": 50},
            sample_domain_gdf,
            progress,
        )
        bbox_buf = mock_read.call_args.args[0]
        georef_buf = result_buf["georeference"]["bounds"]

        # The read bbox (minx, miny, maxx, maxy) should now be wider on every edge.
        assert bbox_buf[0] < bbox_no_buf[0]
        assert bbox_buf[1] < bbox_no_buf[1]
        assert bbox_buf[2] > bbox_no_buf[2]
        assert bbox_buf[3] > bbox_no_buf[3]

        # Georeference bounds should be strictly larger on every edge.
        assert georef_buf[0] < georef_no_buf[0]
        assert georef_buf[1] < georef_no_buf[1]
        assert georef_buf[2] > georef_no_buf[2]
        assert georef_buf[3] > georef_no_buf[3]
