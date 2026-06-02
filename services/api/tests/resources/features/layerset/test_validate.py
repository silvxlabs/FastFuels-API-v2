"""
Unit tests for api/v2/resources/features/layerset/validate.py

Pure unit tests for ``validate_layerset`` — no server, no Firestore, no GCS.
These cover the same edge cases the router's old bespoke ``_parse_crs_name`` /
``_require_projected_crs`` / ``_extract_bounds`` functions did before the
refactor onto the Domain validation pattern.
"""

import pytest
from api.resources.features.layerset.validate import (
    LayersetValidationResult,
    validate_layerset,
)
from fastapi import HTTPException

_PROJECTED_URN = {"type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::32612"}}


def _feature(coords):
    """A single-ring Polygon Feature with the given exterior coordinates."""
    return {
        "type": "Feature",
        "properties": {
            "fuel_type": "shrub",
            "fuel_loading": 1.0,
            "fuel_height": 1.0,
            "percent_cover": 50,
            "distribution": "homogeneous",
        },
        "geometry": {"type": "Polygon", "coordinates": [coords]},
    }


_SQUARE = [
    [294000.0, 5199000.0],
    [294100.0, 5199000.0],
    [294100.0, 5199100.0],
    [294000.0, 5199000.0],
]


class TestValidateLayerset:
    def test_projected_urn_crs_normalized_to_epsg(self):
        """A URN-form projected CRS parses and normalizes to EPSG:<code>."""
        result = validate_layerset(
            {
                "type": "FeatureCollection",
                "crs": _PROJECTED_URN,
                "features": [_feature(_SQUARE)],
            }
        )
        assert isinstance(result, LayersetValidationResult)
        assert result.crs_string == "EPSG:32612"
        assert result.crs.is_geographic is False

    def test_bare_epsg_crs_accepted(self):
        """The bare 'EPSG:32612' form parses identically to the URN form."""
        result = validate_layerset(
            {
                "type": "FeatureCollection",
                "crs": {"type": "name", "properties": {"name": "EPSG:32612"}},
                "features": [_feature(_SQUARE)],
            }
        )
        assert result.crs_string == "EPSG:32612"

    def test_union_bounds_across_features(self):
        """Bounds are the union bounding box across every feature geometry."""
        far = [
            [294200.0, 5199050.0],
            [294300.0, 5199050.0],
            [294300.0, 5199200.0],
            [294200.0, 5199050.0],
        ]
        result = validate_layerset(
            {
                "type": "FeatureCollection",
                "crs": _PROJECTED_URN,
                "features": [_feature(_SQUARE), _feature(far)],
            }
        )
        assert result.bounds == pytest.approx(
            (294000.0, 5199000.0, 294300.0, 5199200.0)
        )

    def test_all_empty_geometry_yields_none_bounds(self):
        """When every geometry is empty, bounds is None (not NaN)."""
        result = validate_layerset(
            {
                "type": "FeatureCollection",
                "crs": _PROJECTED_URN,
                "features": [
                    {
                        "type": "Feature",
                        "properties": {},
                        "geometry": {"type": "MultiPolygon", "coordinates": []},
                    }
                ],
            }
        )
        assert result.bounds is None

    def test_missing_crs_block_rejected_as_geographic(self):
        """No crs block defaults to EPSG:4326 (geographic) and is rejected."""
        with pytest.raises(HTTPException) as exc_info:
            validate_layerset(
                {"type": "FeatureCollection", "features": [_feature(_SQUARE)]}
            )
        assert exc_info.value.status_code == 422
        assert "projected" in exc_info.value.detail.lower()

    def test_explicit_geographic_crs_rejected(self):
        """An explicit geographic CRS (EPSG:4326) is rejected with 422."""
        with pytest.raises(HTTPException) as exc_info:
            validate_layerset(
                {
                    "type": "FeatureCollection",
                    "crs": {
                        "type": "name",
                        "properties": {"name": "urn:ogc:def:crs:EPSG::4326"},
                    },
                    "features": [_feature(_SQUARE)],
                }
            )
        assert exc_info.value.status_code == 422
        assert "geographic" in exc_info.value.detail.lower()

    def test_unparseable_crs_rejected(self):
        """A CRS string pyproj can't parse is rejected with 422."""
        with pytest.raises(HTTPException) as exc_info:
            validate_layerset(
                {
                    "type": "FeatureCollection",
                    "crs": {"type": "name", "properties": {"name": "not-a-real-crs"}},
                    "features": [_feature(_SQUARE)],
                }
            )
        assert exc_info.value.status_code == 422
