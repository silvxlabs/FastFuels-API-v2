"""
In-place inventory modifications for standgen.

Applies the modification delta queued by the most recent
``POST .../{inventory_id}/modifications`` call to the inventory's own current
Parquet data, writing the result back under the same inventory ID.
"""

import logging

import geopandas as gpd

from standgen.modifications import (
    _has_spatial_condition,
    apply_modifications,
    resolve_spatial_conditions,
)
from standgen.storage import write_changed_partitions

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

    # Resolve spatial-condition geometries once here (off the per-partition path)
    # when any are present.
    progress("Applying modifications...", 40)
    if _has_spatial_condition(modifications):
        modifications = resolve_spatial_conditions(
            modifications, inventory["domain_id"], domain_gdf.crs
        )

    # Apply the pending delta to each partition and rewrite only the partitions
    # whose content changed — a scoped modification touches only the partitions
    # it overlaps. The ``modifications`` ledger and ``georeference`` are unchanged.
    progress("Writing modified inventory...", 70)
    write_changed_partitions(
        inventory_id, lambda df: apply_modifications(df, modifications)
    )

    progress("Complete", 100)

    return {"georeference": inventory.get("georeference")}
