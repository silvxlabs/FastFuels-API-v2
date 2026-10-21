"""
Unit tests for api/v2/resources/features/disturbance/validate.py

Pure unit tests for ``validate_disturbance`` — no server, no Firestore, no GCS.
The domain comes from ``make_domain_data``, the same factory the shared
``domain_for_testing`` fixture uses: a 1 km square at (500000, 5200000) in
EPSG:32611, with coordinates stored as a JSON string as Firestore stores them.
"""

import pandas as pd
import pytest
from api.resources.features.disturbance.validate import (
    DisturbanceValidationResult,
    validate_disturbance,
)
from fastapi import HTTPException
from pyproj import Transformer

from tests.fixtures import make_domain_data

_UTM_URN = {"type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::32611"}}


def _domain():
    return make_domain_data()


def _square(x, y, size=100.0):
    return [
        [x, y],
        [x + size, y],
        [x + size, y + size],
        [x, y + size],
        [x, y],
    ]


def _feature(coords, **properties):
    return {
        "type": "Feature",
        "properties": {"fdist": 122, **properties},
        "geometry": {"type": "Polygon", "coordinates": [coords]},
    }


def _collection(features, crs=_UTM_URN):
    collection = {"type": "FeatureCollection", "features": features}
    if crs is not None:
        collection["crs"] = crs
    return collection


def _to_lon_lat(coords):
    transformer = Transformer.from_crs("EPSG:32611", "EPSG:4326", always_xy=True)
    return [list(transformer.transform(x, y)) for x, y in coords]


INSIDE = _square(500100.0, 5200100.0)


class TestValidateDisturbance:
    def test_urn_crs_is_normalized_to_epsg(self):
        result = validate_disturbance(_collection([_feature(INSIDE)]), _domain())
        assert isinstance(result, DisturbanceValidationResult)
        assert result.crs_string == "EPSG:32611"

    def test_bare_epsg_crs_is_accepted(self):
        crs = {"type": "name", "properties": {"name": "EPSG:32611"}}
        result = validate_disturbance(_collection([_feature(INSIDE)], crs), _domain())
        assert result.crs_string == "EPSG:32611"

    def test_no_crs_block_is_read_as_lat_lon(self):
        features = [_feature(_to_lon_lat(INSIDE))]
        result = validate_disturbance(_collection(features, crs=None), _domain())
        assert result.crs_string == "EPSG:4326"
        assert result.crs.is_geographic is True

    def test_different_utm_zone_is_compared_in_the_files_crs(self):
        """A file in another CRS from the domain's is still checked against it."""
        to_32612 = Transformer.from_crs("EPSG:32611", "EPSG:32612", always_xy=True)
        coords = [list(to_32612.transform(x, y)) for x, y in INSIDE]
        crs = {"type": "name", "properties": {"name": "EPSG:32612"}}
        result = validate_disturbance(_collection([_feature(coords)], crs), _domain())
        assert result.crs_string == "EPSG:32612"

    def test_unparseable_crs_is_a_422(self):
        crs = {"type": "name", "properties": {"name": "not-a-real-crs"}}
        with pytest.raises(HTTPException) as exc_info:
            validate_disturbance(_collection([_feature(INSIDE)], crs), _domain())
        assert exc_info.value.status_code == 422

    def test_bounds_are_in_the_files_own_crs(self):
        features = [
            _feature(_square(500100.0, 5200100.0)),
            _feature(_square(500500.0, 5200600.0)),
        ]
        result = validate_disturbance(_collection(features), _domain())
        assert result.bounds == (500100.0, 5200100.0, 500600.0, 5200700.0)

    def test_no_polygon_intersecting_the_domain_is_a_422(self):
        far_away = _square(10000.0, 10000.0)
        with pytest.raises(HTTPException) as exc_info:
            validate_disturbance(_collection([_feature(far_away)]), _domain())
        assert exc_info.value.status_code == 422
        assert "intersect" in exc_info.value.detail

    def test_polygon_partly_outside_the_domain_is_accepted(self):
        straddling = _square(500950.0, 5200100.0)  # crosses the east edge
        result = validate_disturbance(_collection([_feature(straddling)]), _domain())
        assert result.bounds[2] == 501050.0

    def test_one_polygon_inside_is_enough(self):
        features = [_feature(_square(10000.0, 10000.0)), _feature(INSIDE)]
        result = validate_disturbance(_collection(features), _domain())
        assert len(result.gdf) == 2

    def test_polygons_are_tagged_with_the_files_crs(self):
        result = validate_disturbance(_collection([_feature(INSIDE)]), _domain())
        assert result.gdf.crs == result.crs
        assert list(result.gdf["fdist"]) == [122]

    def test_year_column_is_a_nullable_integer(self):
        features = [_feature(INSIDE, year=2022), _feature(INSIDE)]
        result = validate_disturbance(_collection(features), _domain())
        assert str(result.gdf["year"].dtype) == "Int64"
        assert result.gdf["year"].iloc[0] == 2022
        assert pd.isna(result.gdf["year"].iloc[1])

    def test_no_year_column_when_no_polygon_has_one(self):
        result = validate_disturbance(_collection([_feature(INSIDE)]), _domain())
        assert "year" not in result.gdf
