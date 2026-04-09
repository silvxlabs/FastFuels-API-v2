"""Tests for feature dispatch."""

from unittest.mock import MagicMock, patch

import geopandas as gpd
import pytest
from feature.dispatch import dispatch_handler
from feature.errors import ProcessingError


def test_unknown_feature_type_raises():
    """Ensure an unsupported top-level feature type raises an error."""
    feature = {"type": "buildings", "source": {"product": "osm"}}
    mock_gdf = MagicMock(spec=gpd.GeoDataFrame)

    with pytest.raises(ProcessingError) as exc_info:
        dispatch_handler(feature, mock_gdf, MagicMock())

    assert exc_info.value.code == "UNKNOWN_FEATURE_TYPE"


def test_unknown_product_for_road_raises():
    """Ensure an unsupported product for a road feature raises an error."""
    feature = {"type": "road", "source": {"product": "tiger"}}
    mock_gdf = MagicMock(spec=gpd.GeoDataFrame)

    with pytest.raises(ProcessingError) as exc_info:
        dispatch_handler(feature, mock_gdf, MagicMock())

    assert exc_info.value.code == "UNKNOWN_PRODUCT"


def test_unknown_product_for_water_raises():
    """Ensure an unsupported product for a water feature raises an error."""
    feature = {"type": "water", "source": {"product": "nhd"}}
    mock_gdf = MagicMock(spec=gpd.GeoDataFrame)

    with pytest.raises(ProcessingError) as exc_info:
        dispatch_handler(feature, mock_gdf, MagicMock())

    assert exc_info.value.code == "UNKNOWN_PRODUCT"


@patch("feature.handlers.road.handle_osm")
def test_road_osm_dispatch(mock_handle_osm):
    """Ensure road+osm correctly routes to the road OSM handler."""
    mock_handle_osm.return_value = {
        "georeference": {"crs": "EPSG:4326", "bounds": [0, 0, 1, 1]}
    }

    feature = {"type": "road", "source": {"product": "osm"}}
    mock_gdf = MagicMock(spec=gpd.GeoDataFrame)
    progress = MagicMock()

    result = dispatch_handler(feature, mock_gdf, progress)

    mock_handle_osm.assert_called_once_with(
        feature, feature["source"], mock_gdf, progress
    )
    assert "georeference" in result


@patch("feature.handlers.water.handle_osm")
def test_water_osm_dispatch(mock_handle_osm):
    """Ensure water+osm correctly routes to the water OSM handler."""
    mock_handle_osm.return_value = {
        "georeference": {"crs": "EPSG:4326", "bounds": [0, 0, 1, 1]}
    }

    feature = {"type": "water", "source": {"product": "osm"}}
    mock_gdf = MagicMock(spec=gpd.GeoDataFrame)
    progress = MagicMock()

    result = dispatch_handler(feature, mock_gdf, progress)

    mock_handle_osm.assert_called_once_with(
        feature, feature["source"], mock_gdf, progress
    )
    assert "georeference" in result
