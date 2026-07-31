"""Unit tests for lakitu.dispatch source routing."""

from unittest.mock import MagicMock, patch

import geopandas as gpd
import pytest
from lakitu.dispatch import dispatch_handler
from shapely.geometry import box

from lib.errors import ProcessingError


@pytest.fixture
def domain_gdf():
    return gpd.GeoDataFrame(geometry=[box(0, 0, 100, 100)], crs="EPSG:32612")


def test_routes_3dep_to_its_handler(domain_gdf):
    point_cloud = {"id": "pc-1", "source": {"name": "3dep"}}
    progress = MagicMock()
    with patch("lakitu.dispatch.threedep.handle_3dep") as mock_handler:
        mock_handler.return_value = {"buffer": None}
        result = dispatch_handler(point_cloud, domain_gdf, progress)

    assert result == {"buffer": None}
    mock_handler.assert_called_once_with(
        point_cloud, point_cloud["source"], domain_gdf, progress
    )


def test_unknown_source_raises(domain_gdf):
    point_cloud = {"id": "pc-1", "source": {"name": "carrier-pigeon"}}
    with pytest.raises(ProcessingError) as exc:
        dispatch_handler(point_cloud, domain_gdf, MagicMock())
    assert exc.value.code == "UNKNOWN_SOURCE"


def test_upload_source_is_not_handled_here(domain_gdf):
    """Uploads are ingested by the uploader on a GCS event, not dispatched."""
    point_cloud = {"id": "pc-1", "source": {"name": "upload"}}
    with pytest.raises(ProcessingError) as exc:
        dispatch_handler(point_cloud, domain_gdf, MagicMock())
    assert exc.value.code == "UNKNOWN_SOURCE"
