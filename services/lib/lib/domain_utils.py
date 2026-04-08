"""
Domain utilities for parsing domain documents into GeoDataFrames.

Handles Firestore serialization quirks:
- Coordinates stored as JSON strings (Firestore doesn't support nested arrays)
- CRS stored as a GeoJSON CRS object: {"properties": {"name": "EPSG:..."}, "type": "name"}
"""

import json

import geopandas as gpd


class EmptyDomainError(Exception):
    """Domain document has no geometry features."""


class InvalidGeometryError(Exception):
    """Domain geometry could not be parsed into a GeoDataFrame."""


def parse_domain_gdf(domain_data: dict) -> gpd.GeoDataFrame:
    """Parse a domain document dict into a GeoDataFrame.

    Loads all features in the FeatureCollection. For v2 domains this is the
    "domain" working-extent rectangle plus the "input" polygon; the bbox
    rectangle contains the input polygon by construction, so
    ``gdf.total_bounds`` equals the working extent (possibly snapped to
    ``pad_to_resolution``) without any filtering.

    Pure function — no I/O. The caller is responsible for loading the
    domain document from Firestore and handling DocumentNotFoundError.

    Args:
        domain_data: Domain document as a dict (from Firestore snapshot.to_dict())

    Returns:
        GeoDataFrame with domain geometry and CRS

    Raises:
        EmptyDomainError: If the domain has no geometry features
        InvalidGeometryError: If the geometry can't be parsed
    """
    features = domain_data.get("features", [])
    if not features:
        raise EmptyDomainError("Domain has no geometry features.")

    try:
        for feature in features:
            coords = feature.get("geometry", {}).get("coordinates")
            if isinstance(coords, str):
                feature["geometry"]["coordinates"] = json.loads(coords)

        crs_field = domain_data.get("crs")
        if isinstance(crs_field, dict):
            crs = crs_field["properties"]["name"]
        else:
            crs = crs_field or "EPSG:4326"

        gdf = gpd.GeoDataFrame.from_features(features)
        if crs != "local":
            gdf = gdf.set_crs(crs)
    except (EmptyDomainError, InvalidGeometryError):
        raise
    except Exception as e:
        raise InvalidGeometryError(f"Failed to parse domain geometry: {e}") from e

    return gdf
