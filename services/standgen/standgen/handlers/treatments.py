"""
In-place inventory treatments for standgen.

Applies the treatment delta queued by the most recent
``POST .../{inventory_id}/treatments`` call to the inventory's own current
Parquet data, writing the result back under the same inventory ID.
"""

import logging

import geopandas as gpd

from lib.errors import ProcessingError
from standgen.modifications import (
    _has_spatial_condition,
    resolve_spatial_conditions,
)
from standgen.storage import (
    load_inventory_parquet,
    write_changed_partitions,
    write_full_partitions,
)
from standgen.treatments import (
    DIA_COLUMN,
    apply_treatments,
    apply_treatments_to_partition,
    has_proportional_basal_area,
    precompute_cutoffs,
)

logger = logging.getLogger(__name__)


def apply_in_place_treatments(
    inventory: dict, domain_gdf: gpd.GeoDataFrame, progress
) -> dict:
    """Apply the inventory's pending treatment delta to its own data, in place.

    The API appends new treatments to the cumulative ``treatments`` ledger and
    queues only that delta in ``pending_treatments``. This loads the inventory's
    current Parquet, applies only the delta, and writes it back under the same
    ID. The ``treatments`` ledger and ``georeference`` are unchanged — thinning
    removes trees but never moves the grid footprint.

    Args:
        inventory: Full inventory document from Firestore.
        domain_gdf: Domain geometry as GeoDataFrame.
        progress: Callback for progress reporting.

    Returns:
        Dict with 'georeference' key (the inventory's existing georeference).
    """
    inventory_id = inventory["id"]
    treatments = inventory.get("pending_treatments", [])
    # Reuse the inventory's stored expansion seed so ``proportional`` thinning is
    # reproducible (diameter/directional thinning consumes no RNG).
    seed = inventory.get("source", {}).get("seed")

    # Read the inventory's schema. Treatments thin against tree diameter; the API
    # rejects dbh-less inventories at request time from the document's column
    # metadata, but this checks the actual Parquet schema (footer only, no data
    # scan) so a stale document fails with an actionable error instead of a
    # KeyError mid-write.
    ddf = load_inventory_parquet(inventory_id)
    if DIA_COLUMN not in ddf.columns:
        raise ProcessingError(
            code="TREATMENTS_REQUIRE_DBH",
            message=(
                f"Silvicultural treatments require a tree diameter "
                f"('{DIA_COLUMN}') column to thin against, but this "
                f"inventory's data has no such column."
            ),
            suggestion=(
                "CHM-derived inventories and uploads without a dbh column "
                "cannot be treated."
            ),
        )

    # Resolve spatial-condition geometries once here (off the per-partition path)
    # when any are present.
    progress("Applying treatments...", 40)
    if _has_spatial_condition(treatments):
        treatments = resolve_spatial_conditions(
            treatments, inventory["domain_id"], domain_gdf.crs
        )

    progress("Writing treated inventory...", 70)
    if has_proportional_basal_area(treatments):
        # Proportional basal-area removes trees at random over the whole treated
        # population, so it cannot be reproduced as a per-partition filter; the
        # result is materialized and the full dataset rewritten.
        result = apply_treatments(ddf, treatments, domain_gdf, seed=seed).compute()
        write_full_partitions(inventory_id, result)
    else:
        # Diameter and directional basal-area thins are per-partition filters —
        # the latter via a diameter cutoff precomputed over the treated
        # population — so rewrite only the partitions that change, including for
        # polygon-scoped thins.
        cutoffs = precompute_cutoffs(ddf, treatments, domain_gdf)
        write_changed_partitions(
            inventory_id,
            lambda df: apply_treatments_to_partition(
                df, treatments, cutoffs, domain_gdf
            ),
        )

    progress("Complete", 100)

    return {"georeference": inventory.get("georeference")}
