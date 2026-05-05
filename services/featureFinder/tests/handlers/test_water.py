"""Tests for water feature handler."""

from unittest.mock import MagicMock, patch

import geopandas as gpd
import pandas as pd
import pytest
from featureFinder.handlers.water import (
    buffer_water_features,
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
def sample_osm_water():
    """Create a sample OSM response with mixed geometries in WGS84."""
    # A river (LineString), a lake (Polygon), and an unmapped waterway (LineString)
    return gpd.GeoDataFrame(
        {
            "waterway": ["river", pd.NA, "unknown_stream"],
            "natural": [pd.NA, "water", pd.NA],
            "name": ["Colorado River", "Lake Mead", "Mystery Creek"],
            "geometry": [
                LineString([(-120.0, 40.0), (-120.0, 40.01)]),
                Polygon(
                    [(-120.1, 40.1), (-120.1, 40.2), (-120.2, 40.2), (-120.2, 40.1)]
                ),
                LineString([(-120.3, 40.3), (-120.3, 40.31)]),
            ],
        },
        crs="EPSG:4326",
    )


class TestBufferWaterFeatures:
    def test_empty_input(self):
        """Should return an empty GDF if the input is empty."""
        gdf = gpd.GeoDataFrame({"geometry": []}, crs="EPSG:4326")
        result = buffer_water_features(gdf)
        assert result.empty

    def test_missing_waterway_column(self):
        """Should return the GDF as-is if 'waterway' column is missing."""
        poly = Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])
        gdf = gpd.GeoDataFrame({"geometry": [poly]}, crs="EPSG:4326")

        result = buffer_water_features(gdf)

        assert len(result) == 1
        assert result.geom_type.iloc[0] == "Polygon"
        assert result.crs == "EPSG:4326"

    def test_selective_buffering(self, sample_osm_water):
        """Should buffer known waterways and leave polygons and unknown lines intact."""
        result = buffer_water_features(sample_osm_water)

        # We had 3 features initially, we should still have 3
        assert len(result) == 3

        # 1. 'river' (index 0) was a LineString, should now be a buffered Polygon
        assert result.geom_type.iloc[0] == "Polygon"

        # 2. 'Lake Mead' (index 1) was already a Polygon and had waterway=NA, should remain unchanged
        assert result.geom_type.iloc[1] == "Polygon"
        # Check that the lake wasn't accidentally buffered (area should roughly match original)
        original_area = sample_osm_water.geometry.iloc[1].area
        new_area = result.geometry.iloc[1].area
        assert pytest.approx(original_area) == new_area

        # 3. 'unknown_stream' (index 2) was a LineString not in WATERWAY_DATA, should remain a LineString
        assert result.geom_type.iloc[2] == "LineString"

        # Ensure it returns in EPSG:4326
        assert result.crs == "EPSG:4326"


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
        return {"id": "feat-999", "domain_id": "dom-456", "type": "water"}

    @patch("featureFinder.handlers.water.save_geojson")
    @patch("featureFinder.handlers.water.ox.features_from_polygon")
    def test_happy_path(
        self, mock_osmnx, mock_save, mock_feature, sample_domain_gdf, sample_osm_water
    ):
        """Should fetch, buffer, format, and save the water features."""
        mock_osmnx.return_value = sample_osm_water
        progress = MagicMock()

        result = handle_osm(mock_feature, {}, sample_domain_gdf, progress)

        # Ensure OSMnx was called
        mock_osmnx.assert_called_once()

        # Ensure save_geojson was called
        mock_save.assert_called_once()
        args, _ = mock_save.call_args

        assert args[0] == "dom-456"
        assert args[1] == "feat-999"
        saved_gdf = args[2]

        # Saved GDF should have specific columns subset (geometry and name only)
        assert list(saved_gdf.columns) == ["geometry", "name"]

        # Saved GDF should project back to the domain's native CRS
        assert saved_gdf.crs == sample_domain_gdf.crs

        # Ensure georeference is returned
        assert "georeference" in result
        assert result["georeference"]["crs"] == "EPSG:32610"

    @patch("featureFinder.handlers.water.save_geojson")
    @patch("featureFinder.handlers.water.ox.features_from_polygon")
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

    @patch("featureFinder.handlers.water.save_geojson")
    @patch("featureFinder.handlers.water.ox.features_from_polygon")
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
