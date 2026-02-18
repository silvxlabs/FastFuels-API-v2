"""
Domain validation functions for API v2.

This module provides modular validation functions for domain resources.
Each function validates a single aspect and can be independently unit tested.
The main validate_domain() function orchestrates all validations.

Validation Checks:
    1. GeoJSON must be parseable into a GeoDataFrame
    2. CRS must be a valid EPSG/authority string
    3. Geometry must have area > 0
    4. Geometry area must be < 16 sq km (1.6e7 sq meters)
    5. Geometry must be within CONUS
"""

import json
import logging
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
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
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
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
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
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
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
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
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
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
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
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
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


class DomainValidationResult:
    """Result of domain validation containing processed geometry data.

    Attributes:
        gdf: The geometry as a projected GeoDataFrame.
        crs: The final CRS (always projected).
        utm_crs: The UTM CRS if estimated from geographic input, None otherwise.
        area: The total area in square meters.
        features: The GeoJSON features list ready for storage.
    """

    def __init__(
        self,
        gdf: GeoDataFrame,
        crs: CRS,
        utm_crs: CRS | None,
        area: float,
        features: list[dict],
    ):
        self.gdf = gdf
        self.crs = crs
        self.utm_crs = utm_crs
        self.area = area
        self.features = features


def validate_domain(geojson: dict) -> DomainValidationResult:
    """Validate a domain geometry and return processed result.

    Performs all validation checks on a domain:
    1. Parses GeoJSON into a GeoDataFrame
    2. Validates CRS is a valid authority string
    3. Projects to UTM if geographic CRS
    4. Validates geometry has non-zero area
    5. Validates area is within limits (< 16 sq km)
    6. Validates geometry is within CONUS

    Args:
        geojson: A GeoJSON FeatureCollection dict. CRS is extracted from
            geojson["crs"]["properties"]["name"], defaulting to EPSG:4326.

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
    gdf = gdf.set_crs(crs)

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

    # 5. Get area and validate limits
    area = gdf.area.sum()
    validate_area_within_limits(area)

    # 6. Validate within CONUS
    validate_within_conus(gdf)

    # 7. Extract features from projected GeoDataFrame
    features = json.loads(gdf.to_json())["features"]

    return DomainValidationResult(
        gdf=gdf,
        crs=final_crs,
        utm_crs=utm_crs,
        area=area,
        features=features,
    )
