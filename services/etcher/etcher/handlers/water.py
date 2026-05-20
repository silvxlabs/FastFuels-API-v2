"""
Water feature handlers for OSM data.

Queries OpenStreetMap for water bodies and waterways. Leaves existing
polygon features (lakes/reservoirs) intact, dynamically buffers linear
features (rivers/streams) into polygons, and saves the result as GeoParquet.
"""

import logging

import geopandas as gpd
import osmnx as ox
import pandas as pd
from shapely.geometry import box

from etcher.storage import save_features

logger = logging.getLogger(__name__)

# Estimated widths in meters for linear water features
WATERWAY_DATA = {
    "river": 15,
    "canal": 10,
    "stream": 5,
    "drain": 2,
    "ditch": 2,
}


def handle_osm(
    feature: dict, source: dict, domain_gdf: gpd.GeoDataFrame, progress
) -> dict:
    """Process an OSM water feature request.

    Args:
        feature: Full feature document from Firestore
        source: Source dict with OSM-specific fields
        domain_gdf: Domain geometry as GeoDataFrame
        progress: Callback for progress reporting

    Returns:
        Dict with 'georeference' key
    """
    feature_id = feature["id"]
    domain_id = feature["domain_id"]

    progress("Preparing domain boundary...", 10)

    # Store native CRS to project back at the end
    native_crs = domain_gdf.crs

    # OSM requires EPSG:4326 for querying
    domain_gdf_4326 = domain_gdf.to_crs(epsg=4326)
    minx, miny, maxx, maxy = domain_gdf_4326.total_bounds
    query_polygon = box(minx, miny, maxx, maxy)

    # Query both 'natural' (bodies), 'waterway' (rivers/streams),
    # and 'landuse' (man-made bodies)
    tags = {
        "natural": ["water"],
        "waterway": True,
        "landuse": ["reservoir", "basin"],
    }

    progress("Querying OpenStreetMap for water features...", 30)
    try:
        features_gdf = ox.features_from_polygon(
            polygon=query_polygon,
            tags=tags,
        )
    except Exception as e:
        logger.warning(f"OSMnx query failed or returned no data: {e}")
        features_gdf = gpd.GeoDataFrame(geometry=[], crs=4326)

    if features_gdf.empty:
        logger.info(
            f"No water features found for domain {domain_id}",
            extra={"feature_id": feature_id},
        )
        final_gdf = gpd.GeoDataFrame(geometry=[], crs=native_crs)
    else:
        progress("Buffering linear waterways...", 60)
        # Buffer linestrings (rivers) into polygons, leave lakes alone
        features_gdf = buffer_water_features(features_gdf)

        if not features_gdf.empty:
            if "name" not in features_gdf.columns:
                features_gdf["name"] = None

            # Subset for geometry and name (unlike roads, we don't strictly need a 'type'
            # column here as water features originate from several different OSM tags)
            features_gdf = features_gdf[["geometry", "name"]]

            progress("Clipping to domain boundary...", 80)
            # convert back to source crs
            features_gdf = features_gdf.to_crs(crs=native_crs)

            # Clip to domain boundary
            final_gdf = features_gdf.clip(domain_gdf)
            final_gdf = final_gdf.reset_index(drop=True)
        else:
            final_gdf = gpd.GeoDataFrame(geometry=[], crs=native_crs)

    # Write GeoParquet directly to GCS
    progress("Saving features to storage...", 90)
    save_features(domain_id, feature_id, final_gdf)

    # Compute georeference from domain
    progress("Computing georeference...", 95)
    georeference = compute_georeference(domain_gdf)

    progress("Complete", 100)

    return {"georeference": georeference}


def buffer_water_features(water_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Buffers linear water geometries to create polygons representing water widths.
    Polygon features (e.g., lakes, ponds) or features not found in the width
    dictionary are left unmodified.

    Args:
        water_gdf: GeoDataFrame containing water geometries. Expected EPSG:4326.

    Returns:
        GeoDataFrame with linear waterways buffered into polygons.
        CRS is restored to EPSG:4326.
    """
    if water_gdf.empty:
        return water_gdf

    # If 'waterway' column is missing, assume all features are bodies/polygons
    # and return as is.
    if "waterway" not in water_gdf.columns:
        logger.info("No 'waterway' column found. Assuming all features are polygons.")
        return water_gdf

    # 1. Reproject to UTM for accurate metric buffering
    water_gdf = water_gdf.to_crs(water_gdf.estimate_utm_crs())

    # 2. Apply buffer conditionally
    # If the 'waterway' type exists in our dictionary, buffer it.
    # Otherwise (e.g., natural=water polygons), return the geometry as is.
    water_gdf["geometry"] = water_gdf.apply(
        lambda row: (
            row["geometry"].buffer(WATERWAY_DATA[row["waterway"]] / 2.0)
            if pd.notna(row.get("waterway")) and row["waterway"] in WATERWAY_DATA
            else row["geometry"]
        ),
        axis=1,
    )

    # 3. Reproject back to EPSG:4326
    water_gdf = water_gdf.to_crs(epsg=4326)

    return water_gdf


def compute_georeference(domain_gdf: gpd.GeoDataFrame) -> dict:
    """Compute georeference from domain GeoDataFrame (already in native CRS)."""
    bounds = domain_gdf.total_bounds  # [minx, miny, maxx, maxy]
    return {
        "crs": str(domain_gdf.crs),
        "bounds": [float(b) for b in bounds],
    }
