"""
Road feature handlers for OSM data.

Queries OpenStreetMap for road networks, dynamically buffers them based on
classification, and saves the resulting polygons as GeoParquet.
"""

import logging

import geopandas as gpd
import osmnx as ox
from shapely.geometry import box

from etcher.storage import save_features

logger = logging.getLogger(__name__)

# Estimated widths in meters for linear road features
ROAD_DATA = {
    "motorway": 30,
    "motorway_link": 30,
    "trunk": 25,
    "trunk_link": 25,
    "primary": 20,
    "primary_link": 20,
    "secondary": 15,
    "secondary_link": 15,
    "tertiary": 10,
    "tertiary_link": 10,
    "unclassified": 8,
    "residential": 6,
    "service": 5,
    "track": 6,
    "path": 1,
}


def handle_osm(
    feature: dict, source: dict, domain_gdf: gpd.GeoDataFrame, progress
) -> dict:
    """Process an OSM road feature request.

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

    tags = {"highway": True}

    progress("Querying OpenStreetMap for roads...", 30)
    try:
        features_gdf = ox.features_from_polygon(
            polygon=query_polygon,
            tags=tags,
        )
    except Exception as e:
        # osmnx can raise various errors if no data is found depending on the version
        logger.warning(f"OSMnx query failed or returned no data: {e}")
        features_gdf = gpd.GeoDataFrame(geometry=[], crs=4326)

    if features_gdf.empty:
        logger.info(
            f"No road features found for domain {domain_id}",
            extra={"feature_id": feature_id},
        )
        final_gdf = gpd.GeoDataFrame(geometry=[], crs=native_crs)
    else:
        progress("Buffering road centerlines...", 60)
        # buffer linestring into polygons
        features_gdf = buffer_roads(features_gdf)

        if not features_gdf.empty:
            if "name" not in features_gdf.columns:
                features_gdf["name"] = None
            if "highway" not in features_gdf.columns:
                features_gdf["highway"] = None

            # subset for geometry, type and name
            features_gdf = features_gdf[["geometry", "highway", "name"]]
            features_gdf = features_gdf.rename(columns={"highway": "type"})

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


def buffer_roads(road_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Buffers road geometries to create polygons representing road widths.

    Args:
        road_gdf: GeoDataFrame containing road geometries with a 'highway' column.
                  Expected EPSG:4326.

    Returns:
        GeoDataFrame with buffered polygons based on road type widths.
        CRS is restored to EPSG:4326.
    """
    if "highway" not in road_gdf.columns:
        logger.warning("highway column missing. Returning empty GeoDataFrame.")
        return gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs="EPSG:4326")

    # Filter out for the appropriate road tags
    road_gdf = road_gdf[road_gdf["highway"].isin(ROAD_DATA.keys())].copy()

    if road_gdf.empty:
        logger.info("No valid road features found. Returning empty GeoDataFrame.")
        return road_gdf

    # Reproject to UTM for metric buffering
    road_gdf = road_gdf.to_crs(road_gdf.estimate_utm_crs())

    road_gdf["geometry"] = road_gdf.apply(
        lambda row: row["geometry"].buffer(ROAD_DATA.get(row["highway"]) / 2.0), axis=1
    )

    # Reproject back to EPSG:4326
    road_gdf = road_gdf.to_crs(epsg=4326)

    return road_gdf


def compute_georeference(domain_gdf: gpd.GeoDataFrame) -> dict:
    """Compute georeference from domain GeoDataFrame (already in native CRS)."""
    bounds = domain_gdf.total_bounds  # [minx, miny, maxx, maxy]
    return {
        "crs": str(domain_gdf.crs),
        "bounds": [float(b) for b in bounds],
    }
