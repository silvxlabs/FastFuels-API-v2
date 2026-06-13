"""Tests for standgen dispatch."""

from unittest.mock import MagicMock, patch

import geopandas as gpd
import pytest
from standgen.dispatch import dispatch_handler

from lib.errors import ProcessingError


def test_unknown_source_raises():
    inventory = {"source": {"name": "unknown_source"}}
    mock_gdf = MagicMock(spec=gpd.GeoDataFrame)
    with pytest.raises(ProcessingError) as exc_info:
        dispatch_handler(inventory, mock_gdf, MagicMock())
    assert exc_info.value.code == "UNKNOWN_SOURCE"


@patch("standgen.handlers.pim.handle_pim")
def test_pim_dispatch(mock_handle_pim):
    mock_handle_pim.return_value = {"georeference": {}}
    inventory = {"source": {"name": "pim"}}
    mock_gdf = MagicMock(spec=gpd.GeoDataFrame)
    progress = MagicMock()

    result = dispatch_handler(inventory, mock_gdf, progress)

    mock_handle_pim.assert_called_once_with(
        inventory, inventory["source"], mock_gdf, progress
    )
    assert "georeference" in result


@patch("standgen.handlers.chm.handle_chm")
def test_chm_dispatch(mock_handle_chm):
    mock_handle_chm.return_value = {"georeference": {}}
    inventory = {
        "source": {
            "name": "chm",
            "source_chm_grid_id": "test-grid-id",
            "algorithm": {"name": "lmf", "min_height": 2.0, "footprint_size": 3},
        }
    }
    mock_gdf = MagicMock(spec=gpd.GeoDataFrame)
    progress = MagicMock()

    result = dispatch_handler(inventory, mock_gdf, progress)

    mock_handle_chm.assert_called_once_with(
        inventory, inventory["source"], mock_gdf, progress
    )
    assert "georeference" in result


@patch("standgen.handlers.gdam.handle_gdam")
def test_gdam_dispatch(mock_handle_gdam):
    mock_handle_gdam.return_value = {"georeference": {}}
    inventory = {"source": {"name": "gdam", "source_tree_inventory_id": "test-inv-id"}}
    mock_gdf = MagicMock(spec=gpd.GeoDataFrame)
    progress = MagicMock()

    result = dispatch_handler(inventory, mock_gdf, progress)

    mock_handle_gdam.assert_called_once_with(
        inventory, inventory["source"], mock_gdf, progress
    )
    assert "georeference" in result
