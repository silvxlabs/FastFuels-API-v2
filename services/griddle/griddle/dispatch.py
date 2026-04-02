"""
Handler dispatch for Griddle.

Routes grid requests to the appropriate handler based on source type.
"""

from collections.abc import Callable
from datetime import date

import geopandas as gpd
import xarray as xr

from griddle.errors import ProcessingError
from griddle.handlers import chm, landfire, lookup, pim, resample, threedep, uniform

META_CHM_ATTRIBUTION = {
    "1": {
        "license_name": "CC-BY-4.0",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "citation": (
            "High Resolution Canopy Height Maps by WRI and Meta was "
            "accessed on {accessed_on} from "
            "https://registry.opendata.aws/dataforgood-fb-forests. "
            "Meta and World Resources Institute (WRI) - 2024. "
            "High Resolution Canopy Height Maps (CHM). "
            "Source imagery for CHM \u00a9 2016 Maxar."
        ),
        "access_url": "https://registry.opendata.aws/dataforgood-fb-forests",
    },
    "2": {
        "license_name": "DINOv3",
        "license_url": "https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md",
        "citation": (
            "Brandt et al. CHMv2: Improvements in Global Canopy Height Mapping using DINOv3. "
            "arXiv:2603.06382. "
            "Data accessed on {accessed_on} from "
            "https://registry.opendata.aws/dataforgood-fb-forests. "
        ),
        "access_url": "https://registry.opendata.aws/dataforgood-fb-forests",
    },
}


def dispatch_handler(
    grid: dict,
    domain_gdf: gpd.GeoDataFrame,
    progress_callback: Callable[[str, int | None], None],
) -> xr.Dataset:
    """Route to appropriate handler based on source type.

    Args:
        grid: Grid document from Firestore
        domain_gdf: Domain geometry as GeoDataFrame
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
            return handle_landfire(domain_gdf, source, progress_callback)
        case "lookup":
            return handle_lookup(grid, source, progress_callback)
        case "resample":
            return handle_resample(source, progress_callback)
        case "pim":
            return handle_pim(domain_gdf, source, progress_callback)
        case "uniform":
            return handle_uniform(domain_gdf, source, progress_callback)
        case "chm":
            return handle_chm(domain_gdf, source, progress_callback)
        case "3dep":
            return handle_3dep(domain_gdf, source, progress_callback)
        case _:
            raise ProcessingError(
                code="UNKNOWN_SOURCE",
                message=f"Unknown source type: {source_name}",
                suggestion="Check that the grid source type is supported.",
            )


def handle_landfire(
    domain_gdf: gpd.GeoDataFrame,
    source: dict,
    progress: Callable[[str, int | None], None],
) -> xr.Dataset:
    """Handle LANDFIRE source grids."""
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
    domain_gdf: gpd.GeoDataFrame,
    source: dict,
    progress: Callable[[str, int | None], None],
) -> xr.Dataset:
    """Handle PIM source grids."""
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
    """Handle lookup source grids."""
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
    source: dict,
    progress: Callable[[str, int | None], None],
) -> xr.Dataset:
    """Handle resample source grids."""
    progress("Resampling grid...", 10)

    return resample.resample_grid(
        source_grid_id=source["source_grid_id"],
        target_resolution=source["target_resolution"],
        method=source["method"],
        method_overrides=source.get("method_overrides", {}),
        progress=progress,
    )


def handle_uniform(
    domain_gdf: gpd.GeoDataFrame,
    source: dict,
    progress: Callable[[str, int | None], None],
) -> xr.Dataset:
    """Handle uniform source grids."""
    progress("Creating uniform grid...", 10)

    return uniform.create_uniform_grid(
        domain_gdf=domain_gdf,
        bands=source["bands"],
        resolution=source["resolution"],
        progress=progress,
    )


def handle_chm(
    domain_gdf: gpd.GeoDataFrame,
    source: dict,
    progress: Callable[[str, int | None], None],
) -> xr.Dataset:
    """Handle CHM source grids."""
    product = source["product"]
    version = source.get("version", "2")

    progress(f"Fetching CHM {product} v{version}...", 10)

    match product:
        case "meta":
            dataset, tile_metadata = chm.fetch_meta_chm(domain_gdf, version, progress)
            source["tile_metadata"] = tile_metadata
            attribution = META_CHM_ATTRIBUTION[version].copy()
            attribution["accessed_on"] = date.today().isoformat()
            attribution["citation"] = attribution["citation"].format(
                accessed_on=date.today()
            )
            source["attribution"] = attribution

            return dataset
        case "naip":
            dataset, tile_metadata = chm.fetch_naip_chm(domain_gdf, progress)
            source["tile_metadata"] = tile_metadata
            return dataset
        case _:
            raise ProcessingError(
                code="UNKNOWN_PRODUCT",
                message=f"Unknown CHM product: {product}",
                suggestion="Supported products: meta",
            )


def handle_3dep(
    domain_gdf: gpd.GeoDataFrame,
    source: dict,
    progress: Callable[[str, int | None], None],
) -> xr.Dataset:
    """Handle 3DEP source grids.

    Unlike other handlers, 3DEP returns tile metadata alongside the dataset.
    The metadata is merged into the source dict so it gets written back to
    Firestore.
    """
    product = source["product"]
    resolution = source.get("resolution", 10)

    progress(f"Fetching 3DEP {product} {resolution}m...", 10)

    match product:
        case "topography":
            dataset, tile_metadata = threedep.fetch_topography(
                domain_gdf, resolution, source["bands"], progress
            )
            source["tile_metadata"] = tile_metadata
            return dataset
        case _:
            raise ProcessingError(
                code="UNKNOWN_PRODUCT",
                message=f"Unknown 3DEP product: {product}",
                suggestion="Supported products: topography",
            )
