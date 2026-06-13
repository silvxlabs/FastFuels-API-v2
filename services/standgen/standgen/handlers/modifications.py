"""
In-place inventory modifications for standgen.

Applies the modification delta queued by the most recent
``POST .../{inventory_id}/modifications`` call to the inventory's own current
Parquet data, writing the result back under the same inventory ID.
"""

import logging

import geopandas as gpd

from lib.errors import ProcessingError
from standgen.modifications import (
    _has_spatial_condition,
    apply_modifications,
    referenced_columns,
    resolve_spatial_conditions,
)
from standgen.storage import load_inventory_parquet, save_parquet_replace

logger = logging.getLogger(__name__)


def apply_in_place_modifications(
    inventory: dict, domain_gdf: gpd.GeoDataFrame, progress
) -> dict:
    """Apply the inventory's pending modification delta to its own data, in place.

    The API appends new rules to the cumulative ``modifications`` ledger and
    queues only that delta in ``pending_modifications``. This loads the
    inventory's current Parquet, applies only the delta, and writes it back
    under the same ID. The ``modifications`` ledger and ``georeference`` are
    unchanged — modifications filter/transform rows but never move the grid
    footprint.

    Args:
        inventory: Full inventory document from Firestore.
        domain_gdf: Domain geometry as GeoDataFrame.
        progress: Callback for progress reporting.

    Returns:
        Dict with 'georeference' key (the inventory's existing georeference).
    """
    inventory_id = inventory["id"]
    modifications = inventory.get("pending_modifications", [])

    # Load the inventory's own current data as a dask DataFrame.
    progress("Loading inventory...", 10)
    ddf = load_inventory_parquet(inventory_id)

    # Rules filter/transform tree attributes. The API rejects rules referencing
    # an absent column at request time from the document's column metadata; this
    # checks the actual Parquet schema (no compute) so a stale document fails
    # with an actionable error instead of a mid-write KeyError — or, worse, a
    # `replace` action silently materializing an all-null column (the exact
    # "absence becomes silently wrong data" this guard exists to prevent).
    missing = sorted(referenced_columns(modifications) - set(ddf.columns))
    if missing:
        raise ProcessingError(
            code="MISSING_COLUMNS",
            message=(
                f"Modification references column(s) not present in inventory "
                f"{inventory_id}'s data: {missing}."
            ),
            suggestion=(
                "Modifications can only reference columns the inventory carries. "
                "A position-and-height-only inventory (e.g. from CHM/ITD "
                "extraction) has no dbh, species, status, or crown-ratio columns."
            ),
        )

    # Apply only the new delta. Resolve spatial-condition geometries once here
    # (off the per-partition path) when any are present.
    progress("Applying modifications...", 40)
    if _has_spatial_condition(modifications):
        modifications = resolve_spatial_conditions(
            modifications, inventory["domain_id"], domain_gdf.crs
        )
    ddf = ddf.map_partitions(apply_modifications, modifications)

    # Replace the inventory's Parquet in place (staging swap — see storage.py).
    progress("Writing modified inventory...", 70)
    save_parquet_replace(inventory_id, ddf)

    progress("Complete", 100)

    return {"georeference": inventory.get("georeference")}
