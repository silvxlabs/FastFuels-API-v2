"""
Handler dispatch for Griddle.

Routes grid requests to the appropriate handler based on source type.
"""

import json
from collections.abc import Callable

import geopandas as gpd
import xarray as xr

from griddle.errors import ProcessingError
from griddle.handlers import chm, landfire, lookup, pim, resample, uniform
from lib.config import DOMAINS_COLLECTION
from lib.firestore import DocumentNotFoundError, get_document


def dispatch_handler(
    grid: dict,
    progress_callback: Callable[[str, int | None], None],
) -> xr.Dataset:
    """Route to appropriate handler based on source type.

    Args:
        grid: Grid document from Firestore
        progress_callback: Function to report progress (message, percent)

    Returns:
        Dataset with processed grid data

    Raises:
        ProcessingError: If source type is unknown or processing fails
    """
    source = grid["source"]
    source_name = source["name"]

    match source_name:
        case "landfire":
            return handle_landfire(grid, source, progress_callback)
        case "lookup":
            return handle_lookup(grid, source, progress_callback)
        case "resample":
            return handle_resample(grid, source, progress_callback)
        case "pim":
            return handle_pim(grid, source, progress_callback)
        case "uniform":
            return handle_uniform(grid, source, progress_callback)
        case "chm":
            return handle_chm(grid, source, progress_callback)
        case _:
            raise ProcessingError(
                code="UNKNOWN_SOURCE",
                message=f"Unknown source type: {source_name}",
                suggestion="Check that the grid source type is supported.",
            )


def handle_landfire(
    grid: dict,
    source: dict,
    progress: Callable[[str, int | None], None],
) -> xr.Dataset:
    """Handle LANDFIRE source grids.

    Args:
        grid: Grid document
        source: Source configuration from grid
        progress: Progress callback

    Returns:
        Dataset with LANDFIRE data
    """
    domain_gdf = load_domain_gdf(grid["domain_id"])

    product = source["product"]
    version = source.get("version", "2022")

    progress(f"Fetching LANDFIRE {product} v{version}...", 10)

    match product:
        case "fbfm40":
            return landfire.fetch_fbfm40(domain_gdf, version)
        case "topography":
            return landfire.fetch_topography(
                domain_gdf, version, source["bands"], progress
            )
        case _:
            raise ProcessingError(
                code="UNKNOWN_PRODUCT",
                message=f"Unknown LANDFIRE product: {product}",
                suggestion="Supported products: fbfm40, topography",
            )


def handle_pim(
    grid: dict,
    source: dict,
    progress: Callable[[str, int | None], None],
) -> xr.Dataset:
    """Handle PIM source grids.

    Args:
        grid: Grid document
        source: Source configuration from grid
        progress: Progress callback

    Returns:
        Dataset with PIM data
    """
    domain_gdf = load_domain_gdf(grid["domain_id"])

    product = source["product"]
    version = source.get("version", "2022")
    bands = source.get("bands", ["tm_id"])

    progress(f"Fetching PIM {product} v{version}...", 10)

    match product:
        case "treemap":
            return pim.fetch_treemap(domain_gdf, version, bands, progress)
        case _:
            raise ProcessingError(
                code="UNKNOWN_PRODUCT",
                message=f"Unknown PIM product: {product}",
                suggestion="Supported products: treemap",
            )


def handle_lookup(
    grid: dict,
    source: dict,
    progress: Callable[[str, int | None], None],
) -> xr.Dataset:
    """Handle lookup source grids.

    Args:
        grid: Grid document
        source: Source configuration from grid
        progress: Progress callback

    Returns:
        Dataset with looked-up fuel parameters
    """
    table = source["table"]

    progress(f"Looking up {table} fuel parameters...", 10)

    match table:
        case "fbfm40":
            return lookup.fbfm40_lookup(
                source_grid_id=source["source_grid_id"],
                bands=grid["bands"],
                progress=progress,
            )
        case _:
            raise ProcessingError(
                code="UNKNOWN_TABLE",
                message=f"Unknown lookup table: {table}",
                suggestion="Supported tables: fbfm40",
            )


def handle_resample(
    grid: dict,
    source: dict,
    progress: Callable[[str, int | None], None],
) -> xr.Dataset:
    """Handle resample source grids.

    Args:
        grid: Grid document
        source: Source configuration from grid
        progress: Progress callback

    Returns:
        Dataset with resampled data
    """
    progress("Resampling grid...", 10)

    return resample.resample_grid(
        source_grid_id=source["source_grid_id"],
        target_resolution=source["target_resolution"],
        method=source["method"],
        method_overrides=source.get("method_overrides", {}),
        progress=progress,
    )


def handle_uniform(
    grid: dict,
    source: dict,
    progress: Callable[[str, int | None], None],
) -> xr.Dataset:
    """Handle uniform source grids.

    Args:
        grid: Grid document
        source: Source configuration from grid
        progress: Progress callback

    Returns:
        Dataset with constant-value bands
    """
    domain_gdf = load_domain_gdf(grid["domain_id"])

    progress("Creating uniform grid...", 10)

    return uniform.create_uniform_grid(
        domain_gdf=domain_gdf,
        bands=source["bands"],
        resolution=source["resolution"],
        progress=progress,
    )


def handle_chm(
    grid: dict,
    source: dict,
    progress: Callable[[str, int | None], None],
) -> xr.Dataset:
    """Handle CHM source grids.

    Args:
        grid: Grid document
        source: Source configuration from grid
        progress: Progress callback

    Returns:
        Dataset with CHM data
    """
    domain_gdf = load_domain_gdf(grid["domain_id"])

    product = source["product"]
    version = source.get("version", "2024")

    progress(f"Fetching CHM {product} v{version}...", 10)

    match product:
        case "meta":
            return chm.fetch_meta_chm(domain_gdf, version, progress)
        case _:
            raise ProcessingError(
                code="UNKNOWN_PRODUCT",
                message=f"Unknown CHM product: {product}",
                suggestion="Supported products: meta",
            )


def load_domain_gdf(domain_id: str) -> gpd.GeoDataFrame:
    """Load domain as a GeoDataFrame.

    Handles Firestore serialization quirks:
    - Coordinates are stored as JSON strings (Firestore doesn't support nested arrays)
    - CRS is stored as a GeoJSON CRS object: {"properties": {"name": "EPSG:..."}, "type": "name"}

    Follows the same pattern as lib.spatial.get_geodataframe_from_domain_data.

    Args:
        domain_id: Domain document ID

    Returns:
        GeoDataFrame with domain geometry

    Raises:
        ProcessingError: If domain not found, has no geometry, or geometry is invalid
    """
    try:
        _, snapshot = get_document(DOMAINS_COLLECTION, domain_id)
        domain = snapshot.to_dict()
    except DocumentNotFoundError:
        raise ProcessingError(
            code="DOMAIN_NOT_FOUND",
            message=f"Domain {domain_id} not found.",
            suggestion="Ensure the domain exists before creating a grid.",
        )

    features = domain.get("features", [])
    if not features:
        raise ProcessingError(
            code="EMPTY_DOMAIN",
            message="Domain has no geometry.",
            suggestion="Create a domain with at least one polygon feature.",
        )

    # Parse stringified coordinates from Firestore (same as lib.spatial.domain_coords_str_to_dict)
    # Extract CRS from GeoJSON CRS object (same as lib.spatial: domain_data["crs"]["properties"]["name"])
    # Build GeoDataFrame from features (same as lib.spatial.get_geodataframe_from_domain_data)
    try:
        for feature in features:
            coords = feature.get("geometry", {}).get("coordinates")
            if isinstance(coords, str):
                feature["geometry"]["coordinates"] = json.loads(coords)

        crs_field = domain.get("crs")
        if isinstance(crs_field, dict):
            crs = crs_field["properties"]["name"]
        else:
            crs = crs_field or "EPSG:4326"

        gdf = gpd.GeoDataFrame.from_features(features)
        if crs != "local":
            gdf = gdf.set_crs(crs)
    except ProcessingError:
        raise
    except Exception as e:
        raise ProcessingError(
            code="INVALID_GEOMETRY",
            message=f"Failed to parse domain geometry: {e}",
            suggestion="Ensure the domain has valid GeoJSON geometry.",
        )

    return gdf
