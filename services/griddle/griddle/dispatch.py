"""
Handler dispatch for Griddle.

Routes grid requests to the appropriate handler based on source type.
"""

from collections.abc import Callable
from datetime import date

import geopandas as gpd
import xarray as xr

from griddle.handlers import (
    chm,
    compose,
    landfire,
    layerset,
    lookup,
    pim,
    resample,
    threedep,
    uniform,
)
from lib.config import GRIDS_COLLECTION
from lib.errors import ProcessingError
from lib.firestore import DocumentNotFoundError, get_document, update_document

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
            "Source imagery for CHM © 2016 Maxar."
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


def _load_target_grid_doc(alignment: dict | None) -> dict | None:
    """Load the target grid document from Firestore when the alignment uses
    ``target="grid"``. Returns ``None`` for any other target."""
    if not alignment or alignment.get("target") != "grid":
        return None
    grid_id = alignment["grid_id"]
    try:
        _, snapshot = get_document(GRIDS_COLLECTION, grid_id)
    except DocumentNotFoundError:
        raise ProcessingError(
            code="TARGET_GRID_NOT_FOUND",
            message=f"alignment.grid_id '{grid_id}' not found in Firestore.",
            suggestion=(
                "Ensure the alignment target grid still exists. The API "
                "validates this at request time but the grid may have been "
                "deleted before the worker ran."
            ),
        )
    return snapshot.to_dict()


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
        case "layerset":
            return handle_layerset(grid, domain_gdf, source, progress_callback)
        case "lookup":
            return handle_lookup(grid, source, progress_callback)
        case "resample":
            return handle_resample(grid, domain_gdf, source, progress_callback)
        case "pim":
            return handle_pim(domain_gdf, source, progress_callback)
        case "uniform":
            return handle_uniform(domain_gdf, source, progress_callback)
        case "canopy":
            return handle_canopy(domain_gdf, source, progress_callback)
        case "3dep":
            return handle_3dep(domain_gdf, source, progress_callback)
        case "compose":
            return handle_compose(grid, source, progress_callback)
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
    extent_buffer_cells = source.get("extent_buffer_cells", 0)
    alignment = source.get("alignment") or {"target": "domain"}
    target_grid_doc = _load_target_grid_doc(alignment)

    match product:
        case "fbfm13":
            version = source.get("version", "2024")
            progress(f"Fetching LANDFIRE {product} v{version}...", 10)
            remove_non_burnable = source.get("remove_non_burnable")
            return landfire.fetch_fbfm13(
                domain_gdf,
                version,
                remove_non_burnable=remove_non_burnable,
                extent_buffer_cells=extent_buffer_cells,
                alignment=alignment,
                target_grid_doc=target_grid_doc,
            )
        case "fbfm40":
            version = source.get("version", "2024")
            progress(f"Fetching LANDFIRE {product} v{version}...", 10)
            remove_non_burnable = source.get("remove_non_burnable")
            return landfire.fetch_fbfm40(
                domain_gdf,
                version,
                remove_non_burnable=remove_non_burnable,
                extent_buffer_cells=extent_buffer_cells,
                alignment=alignment,
                target_grid_doc=target_grid_doc,
            )
        case "fccs":
            version = source.get("version", "2023")
            progress(f"Fetching LANDFIRE {product} v{version}...", 10)
            remove_bare_ground = source.get("remove_bare_ground", False)
            return landfire.fetch_fccs(
                domain_gdf,
                version,
                remove_bare_ground=remove_bare_ground,
                extent_buffer_cells=extent_buffer_cells,
                alignment=alignment,
                target_grid_doc=target_grid_doc,
            )
        case "topography":
            version = source.get("version", "2020")
            progress(f"Fetching LANDFIRE {product} v{version}...", 10)
            return landfire.fetch_topography(
                domain_gdf,
                version,
                source["bands"],
                progress,
                extent_buffer_cells=extent_buffer_cells,
                alignment=alignment,
                target_grid_doc=target_grid_doc,
            )
        case _:
            raise ProcessingError(
                code="UNKNOWN_PRODUCT",
                message=f"Unknown LANDFIRE product: {product}",
                suggestion="Supported products: fbfm13, fbfm40, fccs, topography",
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
    extent_buffer_cells = source.get("extent_buffer_cells", 0)
    alignment = source.get("alignment") or {"target": "domain"}
    target_grid_doc = _load_target_grid_doc(alignment)

    progress(f"Fetching PIM {product} v{version}...", 10)

    match product:
        case "treemap":
            return pim.fetch_treemap(
                domain_gdf,
                version,
                bands,
                progress,
                extent_buffer_cells=extent_buffer_cells,
                alignment=alignment,
                target_grid_doc=target_grid_doc,
            )
        case _:
            raise ProcessingError(
                code="UNKNOWN_PRODUCT",
                message=f"Unknown PIM product: {product}",
                suggestion="Supported products: treemap",
            )


def handle_layerset(
    grid: dict,
    domain_gdf: gpd.GeoDataFrame,
    source: dict,
    progress: Callable[[str, int | None], None],
) -> xr.Dataset:
    """Handle layerset rasterization source grids.

    The Grid's ``bands`` field cannot be set at API create time because
    the band layout is derived from the input layerset's unique
    ``fuel_type`` values. After rasterization completes here we build the
    real bands list from the output Dataset and persist it to Firestore
    so downstream consumers (exports, the Grid GET endpoint) see the
    actual data layout.
    """
    domain_id = grid["domain_id"]
    layerset_id = source["layerset_id"]
    overlap_method = source.get("overlap_method", "mean")
    extent_buffer_cells = source.get("extent_buffer_cells", 0)
    alignment = source.get("alignment") or {"target": "domain"}
    target_grid_doc = _load_target_grid_doc(alignment)

    progress(f"Rasterizing layerset {layerset_id}...", 10)

    ds = layerset.fetch_layerset(
        domain_gdf=domain_gdf,
        layerset_id=layerset_id,
        domain_id=domain_id,
        overlap_method=overlap_method,
        progress=progress,
        extent_buffer_cells=extent_buffer_cells,
        alignment=alignment,
        target_grid_doc=target_grid_doc,
    )

    # Persist the real bands list — see docstring above.
    bands = layerset.build_layerset_bands(ds)
    update_document(GRIDS_COLLECTION, grid["id"], {"bands": bands})
    grid["bands"] = bands  # keep the in-memory dict consistent for the caller

    return ds


def handle_lookup(
    grid: dict,
    source: dict,
    progress: Callable[[str, int | None], None],
) -> xr.Dataset:
    """Handle lookup source grids."""
    table = source["table"]

    progress(f"Looking up {table} fuel parameters...", 10)

    match table:
        case "fbfm13":
            return lookup.fbfm13_lookup(
                source_grid_id=source["source_grid_id"],
                bands=grid["bands"],
                progress=progress,
            )
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
    domain_gdf: gpd.GeoDataFrame,
    source: dict,
    progress: Callable[[str, int | None], None],
) -> xr.Dataset:
    """Handle resample source grids."""
    progress("Resampling grid...", 10)

    alignment = source["alignment"]
    target_grid_doc = _load_target_grid_doc(alignment)

    # Pull band types off the source grid so resample can apply role-aware
    # method defaults per variable.
    source_grid_id = source["source_grid_id"]
    try:
        _, source_snapshot = get_document(GRIDS_COLLECTION, source_grid_id)
    except DocumentNotFoundError:
        raise ProcessingError(
            code="SOURCE_GRID_NOT_FOUND",
            message=f"Source grid '{source_grid_id}' not found in Firestore.",
        )
    source_grid_doc = source_snapshot.to_dict()
    band_types = {b["key"]: b["type"] for b in source_grid_doc.get("bands", [])}

    # Derive bands from source and write back before summarize runs.
    source_bands = source_grid_doc.get("bands", [])
    bands = [
        {"key": b["key"], "type": b["type"], "unit": b.get("unit"), "index": b["index"]}
        for b in source_bands
    ]
    update_document(GRIDS_COLLECTION, grid["id"], {"bands": bands})
    grid["bands"] = bands

    return resample.resample_grid(
        source_grid_id=source_grid_id,
        alignment=alignment,
        method_overrides=source.get("method_overrides", {}),
        domain_gdf=domain_gdf,
        target_grid_doc=target_grid_doc,
        band_types=band_types,
        progress=progress,
    )


def handle_compose(
    grid: dict,
    source: dict,
    progress: Callable[[str, int | None], None],
) -> xr.Dataset:
    """Handle compose source grids."""
    progress("Composing grid...", 10)
    return compose.compose_grid(grid, source, progress)


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


def handle_canopy(
    domain_gdf: gpd.GeoDataFrame,
    source: dict,
    progress: Callable[[str, int | None], None],
) -> xr.Dataset:
    """Handle canopy source grids."""
    product = source["product"]
    version = source.get("version", "2")
    extent_buffer_cells = source.get("extent_buffer_cells", 0)
    alignment = source.get("alignment") or {"target": "domain"}
    target_grid_doc = _load_target_grid_doc(alignment)

    progress(f"Fetching canopy {product} v{version}...", 10)

    match product:
        case "meta":
            dataset, tile_metadata = chm.fetch_meta_chm(
                domain_gdf,
                version,
                progress,
                extent_buffer_cells=extent_buffer_cells,
                alignment=alignment,
                target_grid_doc=target_grid_doc,
            )
            source["tile_metadata"] = tile_metadata
            attribution = META_CHM_ATTRIBUTION[version].copy()
            attribution["accessed_on"] = date.today().isoformat()
            attribution["citation"] = attribution["citation"].format(
                accessed_on=date.today()
            )
            source["attribution"] = attribution

            return dataset
        case "naip":
            dataset, tile_metadata = chm.fetch_naip_chm(
                domain_gdf,
                progress,
                extent_buffer_cells=extent_buffer_cells,
                alignment=alignment,
                target_grid_doc=target_grid_doc,
            )
            source["tile_metadata"] = tile_metadata
            return dataset
        case "landfire":
            landfire_version = source.get("version", "2024")
            bands = source["bands"]
            progress(f"Fetching LANDFIRE canopy v{landfire_version}...", 10)
            return landfire.fetch_canopy_landfire(
                domain_gdf,
                landfire_version,
                bands,
                progress,
                extent_buffer_cells=extent_buffer_cells,
                alignment=alignment,
                target_grid_doc=target_grid_doc,
            )
        case _:
            raise ProcessingError(
                code="UNKNOWN_PRODUCT",
                message=f"Unknown canopy product: {product}",
                suggestion="Supported products: meta, naip, landfire",
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
    source_resolution = source.get("source_resolution", 10)
    alignment = source.get("alignment") or {"target": "domain"}
    target_grid_doc = _load_target_grid_doc(alignment)

    progress(f"Fetching 3DEP {product} {source_resolution}m...", 10)

    match product:
        case "topography":
            extent_buffer_cells = source.get("extent_buffer_cells", 0)
            dataset, tile_metadata = threedep.fetch_topography(
                domain_gdf,
                source_resolution,
                source["bands"],
                progress,
                extent_buffer_cells=extent_buffer_cells,
                alignment=alignment,
                target_grid_doc=target_grid_doc,
            )
            source["tile_metadata"] = tile_metadata
            return dataset
        case _:
            raise ProcessingError(
                code="UNKNOWN_PRODUCT",
                message=f"Unknown 3DEP product: {product}",
                suggestion="Supported products: topography",
            )
