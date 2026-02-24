"""Tests for standgen dispatch."""

from unittest.mock import MagicMock, patch

import geopandas as gpd
import pytest
from standgen.dispatch import dispatch_handler
from standgen.errors import ProcessingError


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
