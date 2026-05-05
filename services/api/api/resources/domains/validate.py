"""
Domain validation functions for API v2.

This module provides modular validation functions for domain resources.
Each function validates a single aspect and can be independently unit tested.
The main validate_domain() function orchestrates all validations.

Validation Checks:
    1. GeoJSON must be parseable into a GeoDataFrame
    2. CRS must be a valid EPSG/authority string
    3. Geometry must have area > 0
    4. Geometry must be within CONUS
    5. Working extent (possibly padded) area must be < 16 sq km
"""

import json
import logging
import math
from pathlib import Path

import geopandas as gpd
from fastapi import HTTPException, status
from geopandas import GeoDataFrame
from pyproj import CRS
from pyproj.exceptions import CRSError

# Maximum domain area in square meters (16 square kilometers)
MAX_DOMAIN_AREA_SQ_METERS = 1.6e7

# Default CRS if not specified
DEFAULT_CRS = "EPSG:4326"

logger = logging.getLogger(__name__)

# Load CONUS boundary GeoDataFrame at module level
_CONUS_GEOJSON_PATH = (
    Path(__file__).parent.parent.parent / "data" / "conus_4326.geojson"
)
_CONUS_GDF: GeoDataFrame | None = None


def _get_conus_gdf() -> GeoDataFrame:
    """Lazily load the CONUS boundary GeoDataFrame."""
    global _CONUS_GDF
    if _CONUS_GDF is None:
        _CONUS_GDF = gpd.read_file(_CONUS_GEOJSON_PATH)
    return _CONUS_GDF


def parse_geojson_to_gdf(geojson: dict) -> GeoDataFrame:
    """Parse a GeoJSON dict into a GeoDataFrame.

    Args:
        geojson: A GeoJSON FeatureCollection dict.

    Returns:
        A GeoDataFrame containing the parsed geometry.

    Raises:
        HTTPException: 422 if GeoJSON cannot be parsed into a valid GeoDataFrame.
    """
    try:
        geojson_str = json.dumps(geojson)
        return gpd.read_file(geojson_str)
    except Exception as e:
        logger.warning("Failed to parse GeoJSON: %s", e)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Invalid GeoJSON. Unable to parse geometry.",
        )


def validate_crs(crs_name: str) -> CRS:
    """Validate that a CRS is a valid authority string.

    Args:
        crs_name: The CRS name to validate (e.g., 'EPSG:4326').

    Returns:
        The validated pyproj CRS object.

    Raises:
        HTTPException: 422 if CRS is invalid.
    """
    try:
        return CRS(crs_name)
    except CRSError:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"Invalid CRS '{crs_name}'. Must be a valid authority string (e.g., 'EPSG:4326').",
        )


def validate_geometry_has_area(gdf: GeoDataFrame) -> None:
    """Validate that geometry has non-zero area.

    Args:
        gdf: GeoDataFrame containing the geometry to validate.

    Raises:
        HTTPException: 422 if geometry has zero or negative area.
    """
    total_area = gdf.area.sum()
    if total_area <= 0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Invalid geometry. The feature must have an area greater than zero.",
        )


def validate_area_within_limits(
    area_sq_meters: float,
    max_area_sq_meters: float = MAX_DOMAIN_AREA_SQ_METERS,
) -> None:
    """Validate that area is within allowed limits.

    Args:
        area_sq_meters: The area to validate in square meters.
        max_area_sq_meters: Maximum allowed area in square meters.
            Defaults to 16 square kilometers.

    Raises:
        HTTPException: 422 if area exceeds the maximum.
    """
    if area_sq_meters > max_area_sq_meters:
        max_sq_km = max_area_sq_meters / 1e6
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"Invalid spatial extent. Area must be less than {max_sq_km:.0f} square kilometers.",
        )


def validate_within_conus(gdf: GeoDataFrame) -> None:
    """Validate that geometry is entirely within CONUS.

    Args:
        gdf: GeoDataFrame containing the geometry to validate.

    Raises:
        HTTPException: 422 if geometry is outside CONUS.
    """
    conus_gdf = _get_conus_gdf()
    gdf_projected = gdf.to_crs(conus_gdf.crs)

    if not gdf_projected.within(conus_gdf.geometry.iloc[0]).all():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Invalid spatial extent. The domain must be entirely within CONUS.",
        )


def estimate_utm_crs(gdf: GeoDataFrame) -> CRS:
    """Estimate the appropriate UTM CRS for a GeoDataFrame.

    Args:
        gdf: GeoDataFrame with a defined CRS.

    Returns:
        The estimated UTM CRS.

    Raises:
        HTTPException: 422 if UTM CRS cannot be estimated.
    """
    try:
        return gdf.estimate_utm_crs()
    except RuntimeError:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Unable to determine UTM CRS. Please provide a valid projected CRS.",
        )


def is_crs_geographic(crs: CRS) -> bool:
    """Check if a CRS is geographic (not projected).

    Args:
        crs: The pyproj CRS object to check.

    Returns:
        True if CRS is geographic, False if projected.
    """
    return crs.is_geographic


def pad_bounds_to_resolution(
    minx: float,
    miny: float,
    maxx: float,
    maxy: float,
    resolution: float,
) -> tuple[float, float, float, float]:
    """Snap bounding box bounds outward to the nearest multiple of resolution.

    Minimums are floored and maximums are ceiled, so the resulting bbox always
    contains the original bbox and is aligned to a grid at the given resolution.

    Args:
        minx, miny, maxx, maxy: Original bounding box coordinates.
        resolution: Grid cell size in the same units as the coordinates
            (meters, for projected CRS).

    Returns:
        Tuple of (snapped_minx, snapped_miny, snapped_maxx, snapped_maxy).
    """
    return (
        math.floor(minx / resolution) * resolution,
        math.floor(miny / resolution) * resolution,
        math.ceil(maxx / resolution) * resolution,
        math.ceil(maxy / resolution) * resolution,
    )


def build_domain_features(
    gdf: GeoDataFrame,
    pad_to_resolution: float | None = None,
) -> tuple[list[dict], tuple[float, float, float, float]]:
    """Build the two-feature list and bbox for a domain.

    Produces a list of GeoJSON features where:
    - features[0] is the "domain" feature: a polygon covering the bounding box
      of the input geometry, optionally padded to a resolution.
    - features[1:] are "input" features: the user's original projected geometry,
      tagged with properties.name = "input".

    Args:
        gdf: Projected GeoDataFrame containing the user's input geometry.
        pad_to_resolution: Optional resolution (meters) to snap the bbox to.

    Returns:
        Tuple of (features_list, bbox_tuple) where bbox_tuple is the
        (minx, miny, maxx, maxy) of the "domain" feature, in the projected CRS.
    """
    minx, miny, maxx, maxy = gdf.total_bounds

    if pad_to_resolution is not None:
        minx, miny, maxx, maxy = pad_bounds_to_resolution(
            minx, miny, maxx, maxy, pad_to_resolution
        )

    # Build the "domain" feature directly as a GeoJSON dict
    domain_feature = {
        "type": "Feature",
        "properties": {"name": "domain"},
        "geometry": {
            "type": "Polygon",
            "coordinates": [
                [
                    [minx, miny],
                    [maxx, miny],
                    [maxx, maxy],
                    [minx, maxy],
                    [minx, miny],
                ]
            ],
        },
    }

    # Build the "input" feature(s) from the projected GeoDataFrame
    input_features = json.loads(gdf.to_json())["features"]
    for feature in input_features:
        if feature.get("properties") is None:
            feature["properties"] = {}
        feature["properties"]["name"] = "input"

    return [domain_feature, *input_features], (minx, miny, maxx, maxy)


def reproject_features(
    features: list[dict],
    source_crs: CRS,
    target_crs: CRS,
) -> list[dict]:
    """Reproject a list of GeoJSON features from source CRS to target CRS.

    Args:
        features: List of GeoJSON Feature dicts with geometry and properties.
        source_crs: The source pyproj CRS.
        target_crs: The target pyproj CRS.

    Returns:
        List of reprojected GeoJSON Feature dicts with original properties preserved.

    Raises:
        HTTPException: 422 if reprojection fails.
    """
    try:
        gdf = gpd.GeoDataFrame.from_features(features, crs=source_crs)
        gdf = gdf.to_crs(target_crs)
        return json.loads(gdf.to_json())["features"]
    except HTTPException:
        raise
    except Exception as e:
        logger.warning("Failed to reproject features: %s", e)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Failed to reproject geometry.",
        )


class DomainValidationResult:
    """Result of domain validation containing processed geometry data.

    Attributes:
        gdf: The geometry as a projected GeoDataFrame.
        crs: The final CRS (always projected).
        utm_crs: The UTM CRS if estimated from geographic input, None otherwise.
        area: The working extent area in square meters (possibly padded).
        features: The two-feature GeoJSON list (domain + input) ready for storage.
        bbox: The (minx, miny, maxx, maxy) of the "domain" feature.
    """

    def __init__(
        self,
        gdf: GeoDataFrame,
        crs: CRS,
        utm_crs: CRS | None,
        area: float,
        features: list[dict],
        bbox: tuple[float, float, float, float],
    ):
        self.gdf = gdf
        self.crs = crs
        self.utm_crs = utm_crs
        self.area = area
        self.features = features
        self.bbox = bbox


def validate_domain(geojson: dict) -> DomainValidationResult:
    """Validate a domain geometry and return processed result.

    Performs all validation checks on a domain:
    1. Parses GeoJSON into a GeoDataFrame
    2. Validates CRS is a valid authority string
    3. Projects to UTM if geographic CRS
    4. Validates geometry has non-zero area
    5. Validates geometry is within CONUS (on the original projected polygon)
    6. Builds the two-feature list (domain bbox + input polygon), optionally
       padding the bbox to pad_to_resolution
    7. Validates the working extent area is within limits (< 16 sq km)

    Args:
        geojson: A FeatureCollection dict. CRS is extracted from
            geojson["crs"]["properties"]["name"], defaulting to EPSG:4326.
            If geojson["pad_to_resolution"] is set, the working extent bbox
            is snapped outward to that resolution (meters).

    Returns:
        DomainValidationResult with processed geometry data and features.

    Raises:
        HTTPException: 422 if any validation fails.
    """
    # 1. Parse GeoJSON into GeoDataFrame
    gdf = parse_geojson_to_gdf(geojson)

    # 2. Extract and validate CRS from geojson
    crs_name = geojson.get("crs", {}).get("properties", {}).get("name", DEFAULT_CRS)
    crs = validate_crs(crs_name)
    gdf = gdf.set_crs(crs, allow_override=True)

    # 3. Handle CRS and projection
    utm_crs = None

    if is_crs_geographic(crs):
        # Geographic CRS - estimate and project to UTM
        utm_crs = estimate_utm_crs(gdf)
        gdf = gdf.to_crs(utm_crs)
        final_crs = utm_crs
    else:
        # Already projected CRS - keep as is
        final_crs = crs

    # 4. Validate geometry has area
    validate_geometry_has_area(gdf)

    # 5. Validate within CONUS (on the original projected polygon — padding
    # near a CONUS border shouldn't cause false rejections).
    validate_within_conus(gdf)

    # 6. Build the two-feature list with optional padding
    pad_to_resolution = geojson.get("pad_to_resolution")
    features, bbox = build_domain_features(gdf, pad_to_resolution)

    # 7. Validate the working extent area against the limit. We use the
    # (possibly padded) bbox because that is the actual processing footprint.
    minx, miny, maxx, maxy = bbox
    area = (maxx - minx) * (maxy - miny)
    validate_area_within_limits(area)

    return DomainValidationResult(
        gdf=gdf,
        crs=final_crs,
        utm_crs=utm_crs,
        area=area,
        features=features,
        bbox=bbox,
    )
