"""
PIM-CHM fusion inventory handler.

Expands a Plot Imputation Map (PIM) into tree records conditioned on a Canopy
Height Model (CHM). The source names the fused grids (``name == "pim"``,
``fusion == ["chm"]``); ``source.method`` names the algorithm.

Method ``reimputation`` (the v1 fusion algorithm): resample the PIM to
``resolution``, keep each plot only where the CHM's canopy cover above
``min_height`` exceeds ``cover_threshold``, then expand the surviving plots into
trees exactly as ``tree/pim`` does.
"""

import logging

import geopandas as gpd
from fastfuels_core.onramps.hag_pim import (
    compute_cover_from_hag,
    sample_plots_from_hag,
)

from lib.config import GRIDS_COLLECTION
from lib.errors import ProcessingError
from lib.firestore import DocumentNotFoundError, get_document
from standgen.handlers.pim import PIM_PLOT_ID_BANDS, expand_plots
from standgen.storage import load_grid

logger = logging.getLogger(__name__)

# Reimputation defaults, matching the API schema. These are only a fallback for
# a source dict missing the field; the API applies the same defaults at create
# time. resolution/min_height are the v1 production values; cover_threshold is 0.2.
DEFAULT_RESOLUTION = 7.5
DEFAULT_MIN_HEIGHT = 2.0
DEFAULT_COVER_THRESHOLD = 0.2


def handle_pim_chm_fusion(
    inventory: dict, source: dict, domain_gdf: gpd.GeoDataFrame, progress
) -> dict:
    """Process a PIM-CHM fusion inventory request.

    Args:
        inventory: Full inventory document from Firestore.
        source: Source dict with the fused grid ids and the ``method`` object.
        domain_gdf: Domain geometry as GeoDataFrame.
        progress: Callback for progress reporting.

    Returns:
        Dict with 'georeference', 'columns' with per-column summary statistics,
        and 'forestry_metrics' with stand-level forestry scalars or None.
    """
    inventory_id = inventory["id"]
    source_pim_grid_id = source["source_pim_grid_id"]
    source_chm_grid_id = source["source_chm_grid_id"]
    seed = source["seed"]
    point_process = source["point_process"]

    method = source.get("method") or {}
    method_name = method.get("name", "reimputation")
    if method_name != "reimputation":
        raise ProcessingError(
            code="UNSUPPORTED_FUSION_METHOD",
            message=f"Fusion method '{method_name}' is not supported.",
            suggestion="Supported methods: reimputation.",
        )
    resolution = method.get("resolution", DEFAULT_RESOLUTION)
    min_height = method.get("min_height", DEFAULT_MIN_HEIGHT)
    cover_threshold = method.get("cover_threshold", DEFAULT_COVER_THRESHOLD)

    # Load the PIM grid document for its product/version (drives tree-table and
    # plot-id band selection, exactly as handle_pim does).
    try:
        _, pim_snapshot = get_document(GRIDS_COLLECTION, source_pim_grid_id)
        pim_doc = pim_snapshot.to_dict()
    except DocumentNotFoundError:
        raise ProcessingError(
            code="SOURCE_GRID_NOT_FOUND",
            message=f"Source PIM grid '{source_pim_grid_id}' not found.",
            suggestion="Ensure the source PIM grid exists and has been processed.",
        )

    pim_source = pim_doc.get("source", {})
    product = pim_source.get("product", "treemap")
    version = pim_source.get("version", "2022")

    plot_id_band = PIM_PLOT_ID_BANDS.get(product)
    if plot_id_band is None:
        raise ProcessingError(
            code="UNSUPPORTED_PRODUCT",
            message=f"PIM product '{product}' is not supported for inventory expansion.",
            suggestion="Supported products: treemap",
        )

    try:
        get_document(GRIDS_COLLECTION, source_chm_grid_id)
    except DocumentNotFoundError:
        raise ProcessingError(
            code="SOURCE_GRID_NOT_FOUND",
            message=f"Source CHM grid '{source_chm_grid_id}' not found.",
            suggestion="Ensure the source CHM grid exists and has been processed.",
        )

    # Load both grids.
    progress("Loading PIM grid data...", 5)
    pim_ds = load_grid(source_pim_grid_id)
    if plot_id_band not in pim_ds.data_vars:
        available = list(pim_ds.data_vars)
        raise ProcessingError(
            code="MISSING_BAND",
            message=(
                f"Source PIM grid is missing the required '{plot_id_band}' band. "
                f"Available bands: {available}"
            ),
            suggestion=(
                f"Create a PIM grid that includes the '{plot_id_band}' band. "
                f"This band is required for {product} inventory expansion."
            ),
        )

    progress("Loading CHM grid data...", 8)
    chm_ds = load_grid(source_chm_grid_id)
    if "chm" not in chm_ds.data_vars:
        raise ProcessingError(
            code="MISSING_BAND",
            message="Source CHM grid is missing the required 'chm' band.",
            suggestion="Provide a CHM grid that includes the 'chm' band.",
        )

    pim_da = pim_ds[plot_id_band]
    chm_da = chm_ds["chm"]

    # Fuse: resample the PIM to `resolution`, keep a plot only where the CHM's
    # canopy cover above `min_height` exceeds `cover_threshold`. The core leaves
    # below-threshold cells as nodata; `_mask_fused_plots` maps those to
    # PLOT_ID 0 so they act as the zero-density anchors expand_plots relies on.
    progress("Fusing PIM and CHM (reimputation)...", 12)
    try:
        fused = sample_plots_from_hag(
            pim_da,
            chm_da,
            desired_res=resolution,
            min_hag=min_height,
            hag_threshold=cover_threshold,
        )
    except ValueError as e:
        # The API validates CRS/resolution ordering at create time; a ValueError
        # here means the stored grids changed under a queued job. Terminal, not
        # a system fault — surface as a handled failure rather than a 500.
        raise ProcessingError(
            code="INVALID_FUSION_INPUT",
            message=str(e),
            suggestion=(
                "Ensure the PIM and CHM grids share a projected CRS and that the "
                "CHM is at least as fine as the fusion resolution, which must be "
                "at least as fine as the PIM."
            ),
        )

    plots = _mask_fused_plots(fused, pim_da.rio.nodata)
    surviving = int((plots["PLOT_ID"] != 0).sum())
    logger.info(
        f"Fusion kept {surviving} of {len(plots)} resampled cells above "
        f"cover_threshold {cover_threshold}",
        extra={"inventory_id": inventory_id},
    )

    if surviving == 0:
        # Every plot was masked out: report the highest cover actually observed
        # so the caller can tell whether to lower the threshold or the grids
        # simply hold no canopy.
        max_cover = float(compute_cover_from_hag(chm_da, min_height, resolution).max())
        raise ProcessingError(
            code="EMPTY_AFTER_FUSION",
            message=(
                "No PIM plots remained after CHM conditioning. The maximum canopy "
                f"cover observed was {max_cover:.3f}, at or below the "
                f"cover_threshold {cover_threshold}."
            ),
            suggestion=(
                "Lower method.cover_threshold or method.min_height, or verify the "
                "CHM covers vegetated area within the domain."
            ),
        )

    return expand_plots(
        inventory,
        plots,
        version,
        domain_gdf,
        progress,
        seed=seed,
        point_process=point_process,
    )


def _mask_fused_plots(fused_gdf: gpd.GeoDataFrame, nodata) -> gpd.GeoDataFrame:
    """Map masked (below-threshold) fusion cells to PLOT_ID 0.

    ``sample_plots_from_hag`` leaves below-threshold cells at the PIM's nodata
    sentinel (and any cell that resampling could not fill as NaN). Both must read
    as 0 — the zero-density anchor convention shared with ``raster_to_plots_gdf``
    — so the intensity surface stays flat there and the point process does not
    interpolate trees back into the gaps.

    Returns a GeoDataFrame with just PLOT_ID (int) and Point geometry, matching
    the shape ``expand_plots`` expects.
    """
    plot_id = fused_gdf["PLOT_ID"]
    plot_id = plot_id.where(plot_id != nodata).fillna(0).astype(int)
    return gpd.GeoDataFrame(
        {"PLOT_ID": plot_id.values},
        geometry=fused_gdf.geometry.values,
        crs=fused_gdf.crs,
    )
