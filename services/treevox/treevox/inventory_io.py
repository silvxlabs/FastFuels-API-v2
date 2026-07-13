"""Tabular tree-inventory I/O — parquet read, filtering, ID assignment.

"Inventory" here means the tabular tree data (the parquet): rows of trees with
`x, y, fia_species_code, fia_status_code, dbh, height, crown_ratio`. The job
that turns this tabular data into a 3D fuel grid is called *voxelization* and
lives in `treevox.handlers.voxelize`. Keep the distinction when reading/editing
either module.

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
from lib.gcs import get_gcsfs_client
from treevox.errors import ProcessingError

REQUIRED_COLUMNS = [
    "x",
    "y",
    "fia_species_code",
    "fia_status_code",
    "dbh",
    "height",
    "crown_ratio",
]


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
) -> pd.DataFrame:
    """Read a tree-inventory parquet directly from GCS with column projection
    and, when the column is present, a `fia_status_code == 1` predicate pushdown.

    Only `REQUIRED_COLUMNS` (plus `biomass_column` and `crown_radius_column`
    if supplied) are decoded; parquet row groups containing only dead trees
    are skipped when statistics permit. This avoids staging the blob on the
    Cloud Run tmpfs, cuts peak memory roughly in half during load, and
    transfers less data over the wire.

    `fia_status_code` is treated as optional and live-by-default. Inventories
    built by CHM extraction or GDAM allometry never record it (GDAM imputes
    dbh / crown_ratio / species, not the live-dead flag), and an upload may omit
    it. When the file has no `fia_status_code` column, every tree is taken as
    live: the live-tree filter is skipped and the column is set to 1 after the
    read. The other required columns are still projected unconditionally, so an
    inventory missing `dbh` / `crown_ratio` / `fia_species_code` (e.g. one that
    skipped the allometry step) still fails the read on that column.
    """
    gcs_path = f"gs://{INVENTORIES_BUCKET}/{inventory_id}"

    # `status_absent` is True only when the schema was read AND the column is
    # positively missing. A failed probe (None) falls back to the historical
    # behavior — project the column and push the filter — so a transient read
    # error never drops the live-tree filter or overwrites real dead-tree flags.
    available = _inventory_column_names(inventory_id)
    status_absent = available is not None and "fia_status_code" not in available

    # A readable schema missing tree morphology means the inventory skipped the
    # allometry step (it has only position + height). Surface an actionable
    # error instead of the opaque pyarrow "No match for FieldRef.Name(...)" the
    # projection would otherwise raise mid-read.
    if available is not None:
        missing = [
            c for c in REQUIRED_COLUMNS if c != "fia_status_code" and c not in available
        ]
        if missing:
            raise ProcessingError(
                code="INVENTORY_MISSING_MORPHOLOGY",
                message=(
                    f"Inventory {inventory_id} is missing column(s) {missing} "
                    f"required for voxelization."
                ),
                suggestion=(
                    "Impute tree morphology (dbh, crown ratio, species) via the "
                    "allometry endpoint (POST /inventories/tree/allometry/gdam), "
                    "then voxelize the resulting inventory."
                ),
            )

    columns = [
        c for c in REQUIRED_COLUMNS if c != "fia_status_code" or not status_absent
    ]
    for optional in (biomass_column, crown_radius_column):
        if optional and optional not in columns:
            columns.append(optional)

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
) -> pd.DataFrame:
    """Drop rows with nulls in any required column (plus `biomass_column` and
    `crown_radius_column` when set).

    Parquet's row-group statistics can skip dead-tree groups (the
    `fia_status_code == 1` pushdown lives in `read_inventory`), but can't
    drop individual rows missing `dbh` / `height` / `crown_ratio`. That's
    this function's job.
    """
    required = list(REQUIRED_COLUMNS)
    for optional in (biomass_column, crown_radius_column):
        if optional and optional not in required:
            required.append(optional)
    return df.dropna(subset=required).reset_index(drop=True)


def assign_tree_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with a unique int32 `tree_id` column, without
    deep-copying the input.

    `DataFrame.assign` returns a new frame that shares underlying column
    arrays with the caller — so we get non-mutation for free without paying
    for a full block-manager copy of every existing column.
    """
    return df.assign(tree_id=np.arange(len(df), dtype="int32"))
