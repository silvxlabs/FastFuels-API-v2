"""
Unit tests for api/v2/resources/features/geojson.py

Request validation and the reproject-and-clip step of GeoJSON feature uploads.
Pure unit tests with no external dependencies.
"""

import json

import pytest
from api.resources.features.geojson import (
    CreateGeojsonFeatureRequest,
    GeojsonSource,
    clip_upload_to_domain,
)
from fastapi import HTTPException
from pydantic import ValidationError
from pyproj import Transformer

from lib.testing import SHARED_TEST_DOMAINS_DIR

# Blackfoot domain: EPSG:32612, x 294095–294784, y 5198982–5199749.
_TO_WGS84 = Transformer.from_crs(32612, 4326, always_xy=True)


def square_lonlat(x, y, size=20):
    """A square in WGS84 lon/lat whose corner is (x, y) in UTM 12N."""
    corners = [(x, y), (x + size, y), (x + size, y + size), (x, y + size), (x, y)]
    return [list(_TO_WGS84.transform(cx, cy)) for cx, cy in corners]


INSIDE = square_lonlat(294300, 5199300)
STRADDLING = square_lonlat(294770, 5199300)  # crosses the east edge
OUTSIDE = square_lonlat(299000, 5199300)


def polygon_feature(ring, properties=None):
    return {
        "type": "Feature",
        "properties": properties,
        "geometry": {"type": "Polygon", "coordinates": [ring]},
    }


def make_body(*features, **fields) -> CreateGeojsonFeatureRequest:
    return CreateGeojsonFeatureRequest.model_validate(
        {"type": "FeatureCollection", "features": list(features), **fields}
    )


@pytest.fixture(scope="module")
def blackfoot_domain():
    with open(SHARED_TEST_DOMAINS_DIR / "blackfoot.json") as f:
        domain = json.load(f)
    for feature in domain["features"]:
        feature["geometry"]["coordinates"] = json.dumps(
            feature["geometry"]["coordinates"]
        )
    domain["id"] = "test-blackfoot"
    return domain


class TestGeojsonSource:
    def test_product_is_geojson(self):
        assert GeojsonSource().product == "geojson"

    def test_product_cannot_be_overridden(self):
        with pytest.raises(ValidationError):
            GeojsonSource(product="osm")

    def test_extent_buffer_m_round_trips(self):
        assert GeojsonSource(extent_buffer_m=25).model_dump()["extent_buffer_m"] == 25


class TestCreateGeojsonFeatureRequest:
    def test_minimal_body(self):
        body = make_body(polygon_feature(INSIDE))
        assert body.name == ""
        assert body.tags == []
        assert body.extent_buffer_m == 0
        assert body.crs is None

    def test_empty_collection_rejected(self):
        with pytest.raises(ValidationError, match="at least one Feature"):
            make_body()

    def test_multipolygon_accepted(self):
        feature = {
            "type": "Feature",
            "properties": {},
            "geometry": {"type": "MultiPolygon", "coordinates": [[INSIDE]]},
        }
        make_body(feature)

    @pytest.mark.parametrize(
        "geometry",
        [
            {"type": "LineString", "coordinates": INSIDE},
            {"type": "Point", "coordinates": INSIDE[0]},
            None,
        ],
    )
    def test_non_polygon_geometry_rejected(self, geometry):
        feature = {"type": "Feature", "properties": {}, "geometry": geometry}
        with pytest.raises(ValidationError):
            make_body(feature)

    def test_geometry_property_rejected(self):
        with pytest.raises(ValidationError, match="can't be used as a property"):
            make_body(polygon_feature(INSIDE, {"geometry": "x"}))

    @pytest.mark.parametrize("buffer", [-1, 100.5])
    def test_extent_buffer_m_out_of_range_rejected(self, buffer):
        with pytest.raises(ValidationError):
            make_body(polygon_feature(INSIDE), extent_buffer_m=buffer)


class TestClipUploadToDomain:
    def test_reprojects_to_domain_crs(self, blackfoot_domain):
        gdf, georeference = clip_upload_to_domain(
            make_body(polygon_feature(INSIDE, {"name": "a"})), blackfoot_domain
        )
        assert gdf.crs.to_epsg() == 32612
        assert georeference.crs == "EPSG:32612"
        minx, miny, _, _ = gdf.total_bounds
        assert minx == pytest.approx(294300, abs=0.01)
        assert miny == pytest.approx(5199300, abs=0.01)
        assert gdf["name"].tolist() == ["a"]

    def test_georeference_is_domain_bounds(self, blackfoot_domain):
        _, georeference = clip_upload_to_domain(
            make_body(polygon_feature(INSIDE)), blackfoot_domain
        )
        assert georeference.bounds == pytest.approx(
            (294094.99, 5198981.67, 294784.08, 5199749.48), abs=0.01
        )

    def test_clips_to_domain_and_drops_outside(self, blackfoot_domain):
        body = make_body(
            polygon_feature(INSIDE),
            polygon_feature(STRADDLING),
            polygon_feature(OUTSIDE),
        )
        gdf, _ = clip_upload_to_domain(body, blackfoot_domain)
        assert len(gdf) == 2
        assert gdf.total_bounds[2] <= 294784.08
        assert gdf.geometry.iloc[1].area < 400

    def test_extent_buffer_m_widens_clip(self, blackfoot_domain):
        tight, _ = clip_upload_to_domain(
            make_body(polygon_feature(STRADDLING)), blackfoot_domain
        )
        loose, georeference = clip_upload_to_domain(
            make_body(polygon_feature(STRADDLING), extent_buffer_m=50),
            blackfoot_domain,
        )
        assert loose.area.iloc[0] == pytest.approx(400, rel=0.01)
        assert tight.area.iloc[0] < loose.area.iloc[0]
        assert georeference.bounds[2] == pytest.approx(294784.08 + 50, abs=0.1)

    def test_declared_projected_crs(self, blackfoot_domain):
        ring = [
            [294300, 5199300],
            [294320, 5199300],
            [294320, 5199320],
            [294300, 5199300],
        ]
        body = make_body(
            polygon_feature(ring),
            crs={"type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::32612"}},
        )
        gdf, _ = clip_upload_to_domain(body, blackfoot_domain)
        assert gdf.total_bounds[0] == pytest.approx(294300)

    def test_nothing_intersecting_returns_422(self, blackfoot_domain):
        with pytest.raises(HTTPException) as exc:
            clip_upload_to_domain(make_body(polygon_feature(OUTSIDE)), blackfoot_domain)
        assert exc.value.status_code == 422
        assert "intersect" in exc.value.detail

    def test_invalid_geometry_returns_422(self, blackfoot_domain):
        a, b, c, d, _ = INSIDE
        bowtie = [a, c, b, d, a]
        with pytest.raises(HTTPException) as exc:
            clip_upload_to_domain(
                make_body(polygon_feature(INSIDE), polygon_feature(bowtie)),
                blackfoot_domain,
            )
        assert exc.value.status_code == 422
        assert exc.value.detail.startswith("Feature 1 has an invalid geometry")

    def test_unparseable_crs_returns_422(self, blackfoot_domain):
        body = make_body(
            polygon_feature(INSIDE),
            crs={"type": "name", "properties": {"name": "EPSG:999999"}},
        )
        with pytest.raises(HTTPException) as exc:
            clip_upload_to_domain(body, blackfoot_domain)
        assert exc.value.status_code == 422
