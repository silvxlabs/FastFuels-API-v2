"""
Unit tests for api/v2/resources/domains/validate.py

Each validation function is tested independently to ensure proper error handling
and correct behavior across edge cases.

Uses real GeoJSON test data from tests/v2/resources/domains/data/.
"""

import json
from pathlib import Path

import geopandas as gpd
import pytest
from api.resources.domains.validate import (
    MAX_DOMAIN_AREA_SQ_METERS,
    DomainValidationResult,
    build_domain_features,
    estimate_utm_crs,
    is_crs_geographic,
    pad_bounds_to_resolution,
    parse_geojson_to_gdf,
    validate_area_within_limits,
    validate_crs,
    validate_domain,
    validate_geometry_has_area,
    validate_within_conus,
)
from fastapi import HTTPException
from geopandas import GeoDataFrame
from pyproj import CRS
from shapely.geometry import Point, Polygon

# Path to v2 test data directory
TEST_DATA_DIR = Path(__file__).parent / "data"


# =============================================================================
# Helper Functions
# =============================================================================


def make_feature_collection(
    features: list[dict],
    crs: str = "EPSG:4326",
) -> dict:
    """Create a GeoJSON FeatureCollection dict."""
    return {
        "type": "FeatureCollection",
        "features": features,
        "crs": {"type": "name", "properties": {"name": crs}},
    }


def load_geojson(path: Path, crs: str | None = None) -> dict:
    """Load a GeoJSON file and wrap as FeatureCollection if needed."""
    with open(path) as f:
        data = json.load(f)

    # Handle both Feature and FeatureCollection
    if data.get("type") == "Feature":
        features = [data]
    elif data.get("type") == "FeatureCollection":
        features = data.get("features", [])
    else:
        features = [data]

    # Use file's CRS if present, otherwise use provided or default
    file_crs = data.get("crs", {}).get("properties", {}).get("name")
    final_crs = crs or file_crs or "EPSG:4326"

    return make_feature_collection(features, final_crs)


# =============================================================================
# Test Fixtures - GeoJSON dicts from files
# =============================================================================


@pytest.fixture
def blue_mountain_geojson():
    """Blue Mountain polygon in Montana (valid, CONUS, WGS84)."""
    return load_geojson(TEST_DATA_DIR / "blue_mountain_feature_4326.geojson")


@pytest.fixture
def point_geojson():
    """Point geometry with zero area."""
    return load_geojson(TEST_DATA_DIR / "point.geojson")


@pytest.fixture
def alaska_geojson():
    """Polygon in Alaska (outside CONUS)."""
    return load_geojson(TEST_DATA_DIR / "polygon_in_alaska.geojson")


@pytest.fixture
def italy_geojson():
    """Polygon in Italy (outside CONUS)."""
    return load_geojson(TEST_DATA_DIR / "polygon_in_italy.geojson")


@pytest.fixture
def saint_mary_geojson():
    """Large polygon (> 16 sq km) in EPSG:5070."""
    return load_geojson(TEST_DATA_DIR / "saint_mary_5070.geojson", crs="EPSG:5070")


@pytest.fixture
def utm_polygon_geojson():
    """Polygon already in UTM (EPSG:32611)."""
    return load_geojson(TEST_DATA_DIR / "polygon_utm.geojson", crs="EPSG:32611")


# =============================================================================
# Test Fixtures - Synthetic GeoJSON dicts
# =============================================================================


@pytest.fixture
def valid_polygon_geojson():
    """A valid polygon in WGS84 (small, under 16 sq km)."""
    return make_feature_collection(
        features=[
            {
                "type": "Feature",
                "properties": {},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [-121.50, 38.50],
                            [-121.49, 38.50],
                            [-121.49, 38.51],
                            [-121.50, 38.51],
                            [-121.50, 38.50],
                        ]
                    ],
                },
            }
        ],
        crs="EPSG:4326",
    )


@pytest.fixture
def zero_area_geojson():
    """A point geometry with zero area."""
    return make_feature_collection(
        features=[
            {
                "type": "Feature",
                "properties": {},
                "geometry": {"type": "Point", "coordinates": [-121.5, 38.5]},
            }
        ],
        crs="EPSG:4326",
    )


@pytest.fixture
def large_polygon_geojson():
    """A polygon larger than 16 sq km."""
    return make_feature_collection(
        features=[
            {
                "type": "Feature",
                "properties": {},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [-122.0, 38.0],
                            [-121.0, 38.0],
                            [-121.0, 39.0],
                            [-122.0, 39.0],
                            [-122.0, 38.0],
                        ]
                    ],
                },
            }
        ],
        crs="EPSG:4326",
    )


@pytest.fixture
def invalid_crs_geojson():
    """A valid polygon with invalid CRS."""
    return make_feature_collection(
        features=[
            {
                "type": "Feature",
                "properties": {},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [-121.50, 38.50],
                            [-121.49, 38.50],
                            [-121.49, 38.51],
                            [-121.50, 38.51],
                            [-121.50, 38.50],
                        ]
                    ],
                },
            }
        ],
        crs="INVALID:CRS",
    )


# =============================================================================
# Test Fixtures - GeoDataFrames for lower-level function tests
# =============================================================================


@pytest.fixture
def valid_polygon_gdf():
    """A valid polygon GeoDataFrame in WGS84."""
    polygon = Polygon(
        [
            (-121.50, 38.50),
            (-121.49, 38.50),
            (-121.49, 38.51),
            (-121.50, 38.51),
            (-121.50, 38.50),
        ]
    )
    return GeoDataFrame(geometry=[polygon], crs="EPSG:4326")


@pytest.fixture
def projected_polygon_gdf(valid_polygon_gdf):
    """A valid polygon GeoDataFrame projected to UTM."""
    return valid_polygon_gdf.to_crs("EPSG:32610")


@pytest.fixture
def zero_area_gdf():
    """A GeoDataFrame with zero area (point) in a projected CRS."""
    point = Point(-121.5, 38.5)
    gdf = GeoDataFrame(geometry=[point], crs="EPSG:4326")
    return gdf.to_crs("EPSG:32610")


@pytest.fixture
def alaska_gdf():
    """Polygon in Alaska (outside CONUS)."""
    return gpd.read_file(TEST_DATA_DIR / "polygon_in_alaska.geojson")


# =============================================================================
# parse_geojson_to_gdf Tests
# =============================================================================


class TestParseGeojsonToGdf:
    def test_valid_polygon_parses(self, valid_polygon_geojson):
        """Should parse valid polygon GeoJSON into GeoDataFrame."""
        gdf = parse_geojson_to_gdf(valid_polygon_geojson)

        assert isinstance(gdf, GeoDataFrame)
        assert len(gdf) == 1
        assert gdf.geometry.iloc[0].geom_type == "Polygon"

    def test_multiple_features_parse(self):
        """Should parse multiple features."""
        geojson = make_feature_collection(
            features=[
                {
                    "type": "Feature",
                    "properties": {},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [
                            [
                                [-121.5, 38.5],
                                [-121.4, 38.5],
                                [-121.4, 38.6],
                                [-121.5, 38.5],
                            ]
                        ],
                    },
                },
                {
                    "type": "Feature",
                    "properties": {},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [
                            [
                                [-122.5, 39.5],
                                [-122.4, 39.5],
                                [-122.4, 39.6],
                                [-122.5, 39.5],
                            ]
                        ],
                    },
                },
            ]
        )

        gdf = parse_geojson_to_gdf(geojson)
        assert len(gdf) == 2

    def test_point_geometry_parses(self, zero_area_geojson):
        """Should parse point geometry (validation catches zero area later)."""
        gdf = parse_geojson_to_gdf(zero_area_geojson)

        assert isinstance(gdf, GeoDataFrame)
        assert gdf.geometry.iloc[0].geom_type == "Point"


# =============================================================================
# validate_crs Tests
# =============================================================================


class TestValidateCrs:
    def test_valid_epsg_4326(self):
        """Should accept EPSG:4326."""
        result = validate_crs("EPSG:4326")
        assert isinstance(result, CRS)
        assert result.to_epsg() == 4326

    def test_valid_epsg_32610(self):
        """Should accept projected CRS EPSG:32610."""
        result = validate_crs("EPSG:32610")
        assert isinstance(result, CRS)
        assert result.to_epsg() == 32610

    def test_valid_urn_format(self):
        """Should accept URN format CRS."""
        result = validate_crs("urn:ogc:def:crs:EPSG::4326")
        assert isinstance(result, CRS)

    def test_invalid_crs_raises_422(self):
        """Should raise 422 for invalid CRS."""
        with pytest.raises(HTTPException) as exc:
            validate_crs("INVALID:CRS")

        assert exc.value.status_code == 422
        assert "Invalid CRS" in exc.value.detail

    def test_empty_string_raises_422(self):
        """Should raise 422 for empty string."""
        with pytest.raises(HTTPException) as exc:
            validate_crs("")

        assert exc.value.status_code == 422

    def test_nonsense_string_raises_422(self):
        """Should raise 422 for nonsense string."""
        with pytest.raises(HTTPException) as exc:
            validate_crs("not-a-crs")

        assert exc.value.status_code == 422


# =============================================================================
# validate_geometry_has_area Tests
# =============================================================================


class TestValidateGeometryHasArea:
    def test_valid_polygon_passes(self, projected_polygon_gdf):
        """Should pass for polygon with area."""
        validate_geometry_has_area(projected_polygon_gdf)

    def test_zero_area_raises_422(self, zero_area_gdf):
        """Should raise 422 for zero area geometry."""
        with pytest.raises(HTTPException) as exc:
            validate_geometry_has_area(zero_area_gdf)

        assert exc.value.status_code == 422
        assert "area greater than zero" in exc.value.detail

    def test_multiple_polygons_with_area(self):
        """Should pass for multiple polygons with combined area."""
        polygon1 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])
        polygon2 = Polygon([(2, 2), (3, 2), (3, 3), (2, 3), (2, 2)])
        gdf = GeoDataFrame(geometry=[polygon1, polygon2], crs="EPSG:32610")

        validate_geometry_has_area(gdf)


# =============================================================================
# validate_area_within_limits Tests
# =============================================================================


class TestValidateAreaWithinLimits:
    def test_small_area_passes(self):
        """Should pass for area under the limit."""
        validate_area_within_limits(1e6)  # 1 sq km

    def test_exact_limit_passes(self):
        """Should pass for area exactly at the limit."""
        validate_area_within_limits(MAX_DOMAIN_AREA_SQ_METERS)

    def test_over_limit_raises_422(self):
        """Should raise 422 for area over the limit."""
        with pytest.raises(HTTPException) as exc:
            validate_area_within_limits(MAX_DOMAIN_AREA_SQ_METERS + 1)

        assert exc.value.status_code == 422
        assert "16 square kilometers" in exc.value.detail

    def test_custom_limit(self):
        """Should respect custom max_area parameter."""
        validate_area_within_limits(500, max_area_sq_meters=1000)

        with pytest.raises(HTTPException):
            validate_area_within_limits(1500, max_area_sq_meters=1000)

    def test_zero_area_passes(self):
        """Should pass for zero area (separate validation handles this)."""
        validate_area_within_limits(0)


# =============================================================================
# validate_within_conus Tests
# =============================================================================


class TestValidateWithinConus:
    def test_california_polygon_passes(self, valid_polygon_gdf):
        """Should pass for polygon in California."""
        gdf_utm = valid_polygon_gdf.to_crs("EPSG:32610")
        validate_within_conus(gdf_utm)

    def test_alaska_polygon_raises_422(self, alaska_gdf):
        """Should raise 422 for polygon in Alaska."""
        gdf_utm = alaska_gdf.to_crs("EPSG:32606")

        with pytest.raises(HTTPException) as exc:
            validate_within_conus(gdf_utm)

        assert exc.value.status_code == 422
        assert "within CONUS" in exc.value.detail

    def test_hawaii_polygon_raises_422(self):
        """Should raise 422 for polygon in Hawaii."""
        polygon = Polygon(
            [
                (-155.5, 19.5),
                (-155.4, 19.5),
                (-155.4, 19.6),
                (-155.5, 19.6),
                (-155.5, 19.5),
            ]
        )
        gdf = GeoDataFrame(geometry=[polygon], crs="EPSG:4326")

        with pytest.raises(HTTPException) as exc:
            validate_within_conus(gdf)

        assert exc.value.status_code == 422


# =============================================================================
# estimate_utm_crs Tests
# =============================================================================


class TestEstimateUtmCrs:
    def test_california_returns_utm_10n(self, valid_polygon_gdf):
        """Should return UTM Zone 10N for California."""
        result = estimate_utm_crs(valid_polygon_gdf)
        assert isinstance(result, CRS)
        assert result.to_epsg() == 32610

    def test_new_york_returns_utm_18n(self):
        """Should return UTM Zone 18N for New York."""
        polygon = Polygon(
            [(-74.0, 40.7), (-73.9, 40.7), (-73.9, 40.8), (-74.0, 40.8), (-74.0, 40.7)]
        )
        gdf = GeoDataFrame(geometry=[polygon], crs="EPSG:4326")

        result = estimate_utm_crs(gdf)
        assert isinstance(result, CRS)
        assert result.to_epsg() == 32618


# =============================================================================
# is_crs_geographic Tests
# =============================================================================


class TestIsCrsGeographic:
    def test_epsg_4326_is_geographic(self):
        """EPSG:4326 should be geographic."""
        assert is_crs_geographic(CRS("EPSG:4326")) is True

    def test_epsg_32610_is_not_geographic(self):
        """EPSG:32610 (UTM) should not be geographic."""
        assert is_crs_geographic(CRS("EPSG:32610")) is False

    def test_epsg_3857_is_not_geographic(self):
        """EPSG:3857 (Web Mercator) should not be geographic."""
        assert is_crs_geographic(CRS("EPSG:3857")) is False


# =============================================================================
# DomainValidationResult Tests
# =============================================================================


class TestDomainValidationResult:
    def test_stores_all_attributes(self, valid_polygon_gdf):
        """Should store all provided attributes."""
        crs = CRS("EPSG:32610")
        features = [{"type": "Feature", "geometry": {}, "properties": {}}]
        bbox = (0.0, 0.0, 1000.0, 1000.0)

        result = DomainValidationResult(
            gdf=valid_polygon_gdf,
            crs=crs,
            utm_crs=crs,
            area=1000000.0,
            features=features,
            bbox=bbox,
        )

        assert result.gdf is valid_polygon_gdf
        assert result.crs == crs
        assert result.utm_crs == crs
        assert result.area == 1000000.0
        assert result.features == features
        assert result.bbox == bbox

    def test_utm_crs_can_be_none(self, valid_polygon_gdf):
        """utm_crs can be None when input was already projected."""
        result = DomainValidationResult(
            gdf=valid_polygon_gdf,
            crs=CRS("EPSG:32610"),
            utm_crs=None,
            area=1000000.0,
            features=[],
            bbox=(0.0, 0.0, 1000.0, 1000.0),
        )

        assert result.utm_crs is None


# =============================================================================
# pad_bounds_to_resolution Tests
# =============================================================================


class TestPadBoundsToResolution:
    def test_basic_snapping(self):
        """Should floor mins and ceil maxs to nearest resolution multiple."""
        result = pad_bounds_to_resolution(10.5, 20.3, 30.7, 40.9, 10)
        assert result == (10.0, 20.0, 40.0, 50.0)

    def test_already_aligned_bounds_unchanged(self):
        """Bounds already on a resolution boundary should not move."""
        result = pad_bounds_to_resolution(10.0, 20.0, 30.0, 40.0, 10)
        assert result == (10.0, 20.0, 30.0, 40.0)

    def test_blue_mountain_padded_to_30m(self):
        """Real-world Blue Mountain coordinates padded to 30m."""
        result = pad_bounds_to_resolution(
            720227.9398802927,
            5189763.323999467,
            721533.6406826023,
            5190645.048516054,
            30,
        )
        assert result == (720210, 5189760, 721560, 5190660)

    def test_negative_coords(self):
        """Should handle negative coordinates correctly (e.g., EPSG:5070)."""
        result = pad_bounds_to_resolution(-1500.5, -1500.7, -1000.3, -1000.1, 100)
        assert result == (-1600.0, -1600.0, -1000.0, -1000.0)

    def test_fractional_resolution(self):
        """Should work with fractional resolution (e.g., 0.5m)."""
        result = pad_bounds_to_resolution(10.3, 20.7, 30.1, 40.9, 0.5)
        assert result == (10.0, 20.5, 30.5, 41.0)

    def test_large_resolution(self):
        """Should work with large resolution (e.g., 100m)."""
        result = pad_bounds_to_resolution(1.0, 2.0, 99.0, 101.0, 100)
        assert result == (0.0, 0.0, 100.0, 200.0)


# =============================================================================
# build_domain_features Tests
# =============================================================================


class TestBuildDomainFeatures:
    @pytest.fixture
    def projected_gdf(self):
        """A simple rectangular polygon in UTM (EPSG:32611)."""
        polygon = Polygon(
            [
                (720228.0, 5189763.0),
                (721534.0, 5189763.0),
                (721534.0, 5190645.0),
                (720228.0, 5190645.0),
                (720228.0, 5189763.0),
            ]
        )
        return GeoDataFrame(geometry=[polygon], crs="EPSG:32611")

    def test_returns_two_features_without_padding(self, projected_gdf):
        """Without padding, returns one 'domain' and one 'input' feature."""
        features, bbox = build_domain_features(projected_gdf, pad_to_resolution=None)

        assert len(features) == 2
        assert features[0]["properties"]["name"] == "domain"
        assert features[1]["properties"]["name"] == "input"

    def test_domain_feature_is_first(self, projected_gdf):
        """The 'domain' feature should be at index 0."""
        features, _ = build_domain_features(projected_gdf)
        assert features[0]["properties"]["name"] == "domain"

    def test_domain_bbox_equals_input_bounds_without_padding(self, projected_gdf):
        """Without padding, domain feature bbox should equal input polygon bounds."""
        features, bbox = build_domain_features(projected_gdf, pad_to_resolution=None)

        expected_bounds = tuple(projected_gdf.total_bounds)
        assert bbox == expected_bounds

        # Verify the domain feature geometry matches the bbox
        domain_coords = features[0]["geometry"]["coordinates"][0]
        domain_xs = [c[0] for c in domain_coords]
        domain_ys = [c[1] for c in domain_coords]
        assert (
            min(domain_xs),
            min(domain_ys),
            max(domain_xs),
            max(domain_ys),
        ) == expected_bounds

    def test_padding_snaps_domain_bbox(self, projected_gdf):
        """With padding, domain feature bbox should be snapped to resolution."""
        features, bbox = build_domain_features(projected_gdf, pad_to_resolution=30)

        # Original bounds: (720228, 5189763, 721534, 5190645)
        # Padded to 30: (720210, 5189760, 721560, 5190660)
        assert bbox == (720210, 5189760, 721560, 5190660)

    def test_padding_does_not_change_input_geometry(self, projected_gdf):
        """The 'input' feature geometry should be unchanged by padding."""
        features, _ = build_domain_features(projected_gdf, pad_to_resolution=30)

        input_coords = features[1]["geometry"]["coordinates"][0]
        original_coords = list(projected_gdf.geometry.iloc[0].exterior.coords)
        assert len(input_coords) == len(original_coords)

    def test_domain_feature_bbox_equals_returned_bbox(self, projected_gdf):
        """The 'domain' feature's geometry bounds must equal the returned bbox tuple."""
        features, bbox = build_domain_features(projected_gdf, pad_to_resolution=30)

        domain_coords = features[0]["geometry"]["coordinates"][0]
        domain_xs = [c[0] for c in domain_coords]
        domain_ys = [c[1] for c in domain_coords]
        assert (min(domain_xs), min(domain_ys), max(domain_xs), max(domain_ys)) == bbox

    def test_multi_polygon_input_all_tagged_input(self):
        """If user submits multiple polygons, all are tagged 'input'."""
        polygon1 = Polygon([(0, 0), (100, 0), (100, 100), (0, 100), (0, 0)])
        polygon2 = Polygon([(200, 200), (300, 200), (300, 300), (200, 300), (200, 200)])
        gdf = GeoDataFrame(geometry=[polygon1, polygon2], crs="EPSG:32611")

        features, bbox = build_domain_features(gdf, pad_to_resolution=None)

        # 1 domain + 2 inputs
        assert len(features) == 3
        assert features[0]["properties"]["name"] == "domain"
        assert features[1]["properties"]["name"] == "input"
        assert features[2]["properties"]["name"] == "input"

        # Domain bbox encompasses both input polygons
        assert bbox == (0.0, 0.0, 300.0, 300.0)


# =============================================================================
# validate_domain Integration Tests
# =============================================================================


class TestValidateDomain:
    def test_valid_geographic_crs_succeeds(self, valid_polygon_geojson):
        """Should succeed for valid polygon with geographic CRS."""
        result = validate_domain(valid_polygon_geojson)

        assert isinstance(result, DomainValidationResult)
        assert result.utm_crs is not None
        assert result.area > 0
        assert isinstance(result.features, list)
        assert len(result.features) > 0

    def test_valid_projected_crs_succeeds(self, utm_polygon_geojson):
        """Should succeed for valid polygon with projected CRS."""
        result = validate_domain(utm_polygon_geojson)

        assert isinstance(result, DomainValidationResult)
        assert result.utm_crs is None
        assert result.area > 0

    def test_invalid_crs_raises_422(self, invalid_crs_geojson):
        """Should raise 422 for invalid CRS."""
        with pytest.raises(HTTPException) as exc:
            validate_domain(invalid_crs_geojson)

        assert exc.value.status_code == 422
        assert "Invalid CRS" in exc.value.detail

    def test_zero_area_raises_422(self, zero_area_geojson):
        """Should raise 422 for zero area geometry."""
        with pytest.raises(HTTPException) as exc:
            validate_domain(zero_area_geojson)

        assert exc.value.status_code == 422
        assert "area greater than zero" in exc.value.detail

    def test_oversized_raises_422(self, large_polygon_geojson):
        """Should raise 422 for oversized geometry."""
        with pytest.raises(HTTPException) as exc:
            validate_domain(large_polygon_geojson)

        assert exc.value.status_code == 422
        assert "16 square kilometers" in exc.value.detail

    def test_outside_conus_raises_422(self, alaska_geojson):
        """Should raise 422 for geometry outside CONUS."""
        with pytest.raises(HTTPException) as exc:
            validate_domain(alaska_geojson)

        assert exc.value.status_code == 422
        assert "within CONUS" in exc.value.detail

    def test_projects_geographic_to_utm(self, valid_polygon_geojson):
        """Should project geographic CRS to UTM."""
        result = validate_domain(valid_polygon_geojson)

        assert result.utm_crs is not None
        assert not result.crs.is_geographic

    def test_preserves_projected_crs(self, utm_polygon_geojson):
        """Should preserve already-projected CRS."""
        result = validate_domain(utm_polygon_geojson)

        assert result.utm_crs is None
        assert result.crs.to_epsg() == 32611

    def test_returns_features_list(self, valid_polygon_geojson):
        """Should return two features: 'domain' (bbox) and 'input' (polygon)."""
        result = validate_domain(valid_polygon_geojson)

        assert isinstance(result.features, list)
        assert len(result.features) == 2
        for feature in result.features:
            assert feature["type"] == "Feature"
            assert "geometry" in feature
            assert "properties" in feature

        names = [f["properties"]["name"] for f in result.features]
        assert names == ["domain", "input"]

    def test_features_are_projected(self, valid_polygon_geojson):
        """All features should contain projected coordinates."""
        result = validate_domain(valid_polygon_geojson)

        # Original coordinates were around -121, 38 (WGS84)
        # UTM coordinates should be much larger (hundreds of thousands)
        for feature in result.features:
            coords = feature["geometry"]["coordinates"][0][0]
            assert abs(coords[0]) > 1000  # UTM easting
            assert abs(coords[1]) > 1000  # UTM northing

    def test_pad_to_resolution_snaps_domain_feature_bbox(self, utm_polygon_geojson):
        """pad_to_resolution should snap the domain feature bbox."""
        utm_polygon_geojson["pad_to_resolution"] = 30
        result = validate_domain(utm_polygon_geojson)

        # The bbox tuple should be snapped to multiples of 30
        minx, miny, maxx, maxy = result.bbox
        assert minx % 30 == 0
        assert miny % 30 == 0
        assert maxx % 30 == 0
        assert maxy % 30 == 0

        # The "domain" feature's geometry should match the padded bbox
        domain_feature = result.features[0]
        assert domain_feature["properties"]["name"] == "domain"
        domain_coords = domain_feature["geometry"]["coordinates"][0]
        domain_xs = [c[0] for c in domain_coords]
        domain_ys = [c[1] for c in domain_coords]
        assert (
            min(domain_xs),
            min(domain_ys),
            max(domain_xs),
            max(domain_ys),
        ) == result.bbox

    def test_pad_to_resolution_none_equivalent_to_unpadded(self, utm_polygon_geojson):
        """Omitting pad_to_resolution should produce a tight bbox."""
        result = validate_domain(utm_polygon_geojson)

        # bbox should equal the gdf's total_bounds (the polygon's tight bbox)
        expected = tuple(result.gdf.total_bounds)
        assert result.bbox == expected

    def test_validate_domain_returns_bbox(self, valid_polygon_geojson):
        """validate_domain should return a bbox tuple in the result."""
        result = validate_domain(valid_polygon_geojson)

        assert result.bbox is not None
        assert len(result.bbox) == 4
        # bbox is in projected CRS so values should be large (UTM meters)
        assert all(abs(x) > 1000 for x in result.bbox)

    def test_padded_area_validates_against_padded_extent(self):
        """Area validation should use the padded extent, not the unpadded one.

        A polygon whose unpadded bbox is just under 16 sq km but whose padded
        bbox exceeds 16 sq km should be rejected.
        """
        # Bbox: 3950 x 4050 = 15,997,500 sq m (under)
        # Padded to 100m: 4000 x 4100 = 16,400,000 sq m (over)
        polygon = Polygon(
            [
                (500000, 5000000),
                (503950, 5000000),
                (503950, 5004050),
                (500000, 5004050),
                (500000, 5000000),
            ]
        )
        gdf = GeoDataFrame(geometry=[polygon], crs="EPSG:32611")
        geojson = json.loads(gdf.to_json())
        geojson["crs"] = {"type": "name", "properties": {"name": "EPSG:32611"}}

        # Without padding it should pass
        result = validate_domain(geojson)
        assert result.area < MAX_DOMAIN_AREA_SQ_METERS

        # With padding to 100m it should fail
        geojson["pad_to_resolution"] = 100
        with pytest.raises(HTTPException) as exc:
            validate_domain(geojson)

        assert exc.value.status_code == 422
        assert "16 square kilometers" in exc.value.detail

    def test_defaults_to_epsg_4326_when_no_crs(self):
        """Should default to EPSG:4326 when CRS not specified."""
        geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [
                            [
                                [-121.50, 38.50],
                                [-121.49, 38.50],
                                [-121.49, 38.51],
                                [-121.50, 38.51],
                                [-121.50, 38.50],
                            ]
                        ],
                    },
                }
            ],
            # No CRS specified
        }

        result = validate_domain(geojson)

        # Should have projected to UTM (meaning it treated input as geographic)
        assert result.utm_crs is not None


# =============================================================================
# Integration Tests with Real GeoJSON Data
# =============================================================================


class TestValidateDomainRealData:
    """Tests using real GeoJSON files from the test data directory."""

    def test_blue_mountain_succeeds(self, blue_mountain_geojson):
        """Blue Mountain polygon in Montana should pass all validations."""
        result = validate_domain(blue_mountain_geojson)

        assert isinstance(result, DomainValidationResult)
        assert result.utm_crs is not None
        assert result.area > 0
        assert result.crs.to_epsg() in [32611, 32612]
        assert len(result.features) == 2
        names = [f["properties"]["name"] for f in result.features]
        assert names == ["domain", "input"]

    def test_point_zero_area_raises_422(self, point_geojson):
        """Point geometry should fail with zero area."""
        with pytest.raises(HTTPException) as exc:
            validate_domain(point_geojson)

        assert exc.value.status_code == 422
        assert "area greater than zero" in exc.value.detail

    def test_italy_raises_422(self, italy_geojson):
        """Italy polygon should fail CONUS check."""
        with pytest.raises(HTTPException) as exc:
            validate_domain(italy_geojson)

        assert exc.value.status_code == 422
        assert "within CONUS" in exc.value.detail

    def test_saint_mary_oversized_raises_422(self, saint_mary_geojson):
        """Saint Mary polygon (> 16 sq km) should fail area check."""
        with pytest.raises(HTTPException) as exc:
            validate_domain(saint_mary_geojson)

        assert exc.value.status_code == 422
        assert "16 square kilometers" in exc.value.detail

    def test_utm_polygon_preserves_crs(self, utm_polygon_geojson):
        """UTM polygon should preserve its projected CRS."""
        result = validate_domain(utm_polygon_geojson)

        assert isinstance(result, DomainValidationResult)
        assert result.utm_crs is None
        assert result.crs.to_epsg() == 32611
