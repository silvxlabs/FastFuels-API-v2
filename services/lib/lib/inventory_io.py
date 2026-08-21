"""Tabular tree-inventory I/O — parquet read, filtering, ID assignment.

"Inventory" here means the tabular tree data (the parquet): rows of trees with
`x, y, fia_species_code, fia_status_code, dbh, height, crown_ratio`. Shared by
the worker services that consume an inventory: treevox voxelizes it into a 3D
fuel grid, griddle derives 2D canopy fuel grids from it.

Memory note: on Cloud Run `/tmp` is RAM-backed tmpfs, so downloading a parquet
to a local file *before* reading it holds both the compressed bytes and the
decoded DataFrame in memory simultaneously. We read directly from GCS via
pandas' fsspec integration to avoid that double-resident copy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from lib.config import INVENTORIES_BUCKET
from lib.errors import ProcessingError
from lib.gcs import get_gcsfs_client

REQUIRED_COLUMNS = [
    "x",
    "y",
    "fia_species_code",
    "fia_status_code",
    "dbh",
    "height",
    "crown_ratio",
]

# Optional first-class FIA column (CCLCD, crown class code 1-5). Carried by
# inventories that have it (some uploads); absent from standgen / GDAM output.
# Projected only when present and never used to drop rows: a tree with no crown
# class is still a valid tree — the canopy handler folds a missing code onto
# FuelCalc's Other/none column rather than discarding the stem.
CROWN_CLASS_COLUMN = "fia_crown_class_code"


def _inventory_column_names(inventory_id: str) -> set[str] | None:
    """Column names in an inventory parquet, or None if the schema can't be read.

    Footer-only read of the aggregated `_metadata` schema that both standgen
    and the uploader write (`write_metadata_file=True`); no row data is
    scanned. Lets `read_inventory` tell whether the optional `fia_status_code`
    column is present before projecting it. Returns None on any read failure so
    the caller falls back to its default projection rather than misreporting a
    transport error here.
    """
    path = f"{INVENTORIES_BUCKET}/{inventory_id}/_metadata"
    try:
        fs = get_gcsfs_client()
        with fs.open(path, "rb") as f:
            return set(pq.read_schema(f).names)
    except Exception:
        return None


def read_inventory(
    inventory_id: str,
    biomass_column: str | None = None,
    crown_radius_column: str | None = None,
    include_crown_class: bool = False,
    required_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Read a tree-inventory parquet directly from GCS with column projection
    and, when the column is present, a `fia_status_code == 1` predicate pushdown.

    Only the required columns (plus `biomass_column` and `crown_radius_column`
    if supplied) are decoded; parquet row groups containing only dead trees
    are skipped when statistics permit. This avoids staging the blob on the
    Cloud Run tmpfs, cuts peak memory roughly in half during load, and
    transfers less data over the wire.

    `required_columns` names the morphology columns this consumer actually reads
    (never `fia_status_code`, which is always handled below). It defaults to the
    full `REQUIRED_COLUMNS` set — the voxelization contract — so existing callers
    are unchanged; a consumer that reads fewer (e.g. a canopy request taking fuel
    and crown radius from columns, needing neither `dbh` nor `fia_species_code`)
    passes its own set so the read neither requires nor projects columns it will
    not use. Use `canopy_required_columns` to derive it for a canopy source.

    `include_crown_class` projects `CROWN_CLASS_COLUMN` (FIA CCLCD) when the
    inventory carries it, so the caller can find it in the returned frame; it is
    left out silently when the inventory has no such column (standgen / GDAM
    output, older uploads) or when the schema can't be read, and it never joins
    the live-tree filter. The column is optional, so absence is not an error —
    the consumer decides what a missing crown class means.

    `fia_status_code` is treated as optional and live-by-default. Inventories
    built by CHM extraction or GDAM allometry never record it (GDAM imputes
    dbh / crown_ratio / species, not the live-dead flag), and an upload may omit
    it. When the file has no `fia_status_code` column, every tree is taken as
    live: the live-tree filter is skipped and the column is set to 1 after the
    read. A `required_columns` member that the inventory lacks (e.g. a `dbh`
    absent because the allometry step was skipped) still fails the read early
    with `INVENTORY_MISSING_MORPHOLOGY`.
    """
    gcs_path = f"gs://{INVENTORIES_BUCKET}/{inventory_id}"

    # `status_absent` is True only when the schema was read AND the column is
    # positively missing. A failed probe (None) falls back to the historical
    # behavior — project the column and push the filter — so a transient read
    # error never drops the live-tree filter or overwrites real dead-tree flags.
    available = _inventory_column_names(inventory_id)
    status_absent = available is not None and "fia_status_code" not in available

    # The morphology columns this read requires (never fia_status_code, which is
    # handled separately just below). Default to the full voxelization set.
    if required_columns is None:
        morphology = [c for c in REQUIRED_COLUMNS if c != "fia_status_code"]
    else:
        morphology = [c for c in required_columns if c != "fia_status_code"]

    # A readable schema missing a required morphology column means the inventory
    # skipped the step that produces it (e.g. CHM extraction leaves only position
    # and height). Surface an actionable error instead of the opaque pyarrow
    # "No match for FieldRef.Name(...)" the projection would otherwise raise
    # mid-read.
    if available is not None:
        missing = [c for c in morphology if c not in available]
        if missing:
            raise ProcessingError(
                code="INVENTORY_MISSING_MORPHOLOGY",
                message=(
                    f"Inventory {inventory_id} is missing column(s) {missing} "
                    f"required by the requested operation."
                ),
                suggestion=(
                    "Impute tree morphology (dbh, crown ratio, species) via the "
                    "allometry endpoint (POST /inventories/tree/allometry/gdam), "
                    "or supply the column(s) in the source inventory."
                ),
            )

    if required_columns is None:
        # Preserve the canonical REQUIRED_COLUMNS projection order (status in
        # place), dropping status only when the inventory lacks it.
        columns = [
            c for c in REQUIRED_COLUMNS if c != "fia_status_code" or not status_absent
        ]
    else:
        columns = list(morphology)
        if not status_absent:
            columns.append("fia_status_code")
    for optional in (biomass_column, crown_radius_column):
        if optional and optional not in columns:
            columns.append(optional)

    # Project the crown-class column only when the inventory positively has it
    # (a readable schema listing it). A failed schema probe or a missing column
    # both fall through to "no crown class" rather than raising a projection
    # error mid-read; the consumer treats that as every tree unclassified.
    if (
        include_crown_class
        and available is not None
        and CROWN_CLASS_COLUMN in available
        and CROWN_CLASS_COLUMN not in columns
    ):
        columns.append(CROWN_CLASS_COLUMN)

    filters = None if status_absent else [("fia_status_code", "=", 1)]

    try:
        df = pd.read_parquet(gcs_path, columns=columns, filters=filters)
    except FileNotFoundError as e:
        raise ProcessingError(
            code="INVENTORY_NOT_FOUND",
            message=f"Inventory {inventory_id} not found at {gcs_path}.",
            suggestion="Verify the inventory ID exists and has completed processing.",
        ) from e
    except Exception as e:
        # gcsfs / pyarrow can surface permission or transport errors as
        # arbitrary exception types; treat any I/O failure as missing for
        # user-facing purposes.
        raise ProcessingError(
            code="INVENTORY_NOT_FOUND",
            message=f"Could not read inventory {inventory_id}: {e}",
            suggestion="Verify the inventory ID exists and has completed processing.",
        ) from e

    if status_absent:
        df["fia_status_code"] = 1
    return df


def drop_null_rows(
    df: pd.DataFrame,
    biomass_column: str | None = None,
    crown_radius_column: str | None = None,
    required_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Drop rows with nulls in any required column (plus `biomass_column` and
    `crown_radius_column` when set).

    Parquet's row-group statistics can skip dead-tree groups (the
    `fia_status_code == 1` pushdown lives in `read_inventory`), but can't
    drop individual rows missing `dbh` / `height` / `crown_ratio`. That's
    this function's job.

    `required_columns` must match the set the paired `read_inventory` call used
    (defaults to `REQUIRED_COLUMNS`). Dropping on a column the request does not
    read would silently discard trees — and their canopy fuel — over a value
    that never enters the computation.
    """
    required = list(
        required_columns if required_columns is not None else REQUIRED_COLUMNS
    )
    for optional in (biomass_column, crown_radius_column):
        if optional and optional not in required:
            required.append(optional)
    return df.dropna(subset=required).reset_index(drop=True)


def canopy_required_columns(source: dict) -> set[str]:
    """Morphology columns an inventory-canopy source's methods read from the tree
    inventory.

    Position (`x`, `y`) and the crown interval (`height`, `crown_ratio`) are
    always read; `dbh` and `fia_species_code` only by the methods that consume
    them — allometric crown biomass, the Reinhardt vertical distribution, the
    FuelCalc hardwood exclusion, and the FuelCalc crown-class factors. Fuel and
    crown-radius columns are not returned here: they are supplied to
    `read_inventory` as `biomass_column` / `crown_radius_column`.

    This is the single authority the API router (pre-dispatch column validation)
    and the griddle handler (what to project and require non-null) both use, so
    they cannot disagree about which columns a request needs — the mismatch that
    would otherwise let the API accept a request the worker then fails.
    """
    required = {"x", "y", "height", "crown_ratio"}
    if source["biomass_source"]["type"] == "allometry":
        required |= {"dbh", "fia_species_code"}
    if source["vertical_distribution"] == "reinhardt_2006":
        required.add("fia_species_code")
    if source["species_inclusion"] == "fuelcalc_default":
        required.add("fia_species_code")
    if source["crown_class_adjustment"]["method"] == "fuelcalc_table":
        required.add("fia_species_code")
    return required


def assign_tree_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with a unique int32 `tree_id` column, without
    deep-copying the input.

    `DataFrame.assign` returns a new frame that shares underlying column
    arrays with the caller — so we get non-mutation for free without paying
    for a full block-manager copy of every existing column.
    """
    return df.assign(tree_id=np.arange(len(df), dtype="int32"))
