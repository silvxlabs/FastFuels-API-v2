import geopandas as gpd
import pytest
from shapely.geometry import Point, box
from standgen.region import in_pacific_northwest


def _square_domain(lon: float, lat: float, utm_epsg: int) -> gpd.GeoDataFrame:
    """1 km square domain centred on (lon, lat), in the given UTM CRS."""
    center = gpd.GeoSeries([Point(lon, lat)], crs=4326).to_crs(utm_epsg).iloc[0]
    square = box(center.x - 500, center.y - 500, center.x + 500, center.y + 500)
    return gpd.GeoDataFrame(geometry=[square], crs=utm_epsg)


@pytest.mark.parametrize(
    "lon, lat, utm_epsg",
    [
        (-121.31, 44.06, 32610),  # Bend, OR
        (-120.51, 46.60, 32610),  # Yakima, WA
        (-119.78, 36.74, 32611),  # Fresno, CA
    ],
)
def test_inside_pacific_northwest(lon, lat, utm_epsg):
    assert in_pacific_northwest(_square_domain(lon, lat, utm_epsg)) is True


@pytest.mark.parametrize(
    "lon, lat, utm_epsg",
    [
        (-116.20, 43.62, 32611),  # Boise, ID
        (-113.99, 46.87, 32612),  # Missoula, MT
        (-119.81, 39.53, 32611),  # Reno, NV
    ],
)
def test_outside_pacific_northwest(lon, lat, utm_epsg):
    assert in_pacific_northwest(_square_domain(lon, lat, utm_epsg)) is False


def test_geographic_crs_domain():
    inside = gpd.GeoDataFrame(geometry=[box(-121.32, 44.05, -121.30, 44.07)], crs=4326)
    outside = gpd.GeoDataFrame(geometry=[box(-116.21, 43.61, -116.19, 43.63)], crs=4326)
    assert in_pacific_northwest(inside) is True
    assert in_pacific_northwest(outside) is False
