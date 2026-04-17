"""Tabular tree-inventory I/O — parquet read, filtering, ID assignment.

"Inventory" here means the tabular tree data (the parquet): rows of trees with
`x, y, fia_species_code, fia_status_code, dbh, height, crown_ratio`. The job
that turns this tabular data into a 3D fuel grid is called *voxelization* and
lives in `treevox.orchestrator`. Keep the distinction when reading/editing
either module.

Memory note: on Cloud Run `/tmp` is RAM-backed tmpfs, so downloading a parquet
to a local file *before* reading it holds both the compressed bytes and the
decoded DataFrame in memory simultaneously. We read directly from GCS via
pandas' fsspec integration to avoid that double-resident copy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from lib.config import INVENTORIES_BUCKET
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


def read_inventory(
    inventory_id: str, biomass_column: str | None = None
) -> pd.DataFrame:
    """Read a tree-inventory parquet directly from GCS with column projection
    and a `fia_status_code == 1` predicate pushdown.

    Only `REQUIRED_COLUMNS` (plus `biomass_column` if supplied) are decoded;
    parquet row groups containing only dead trees are skipped when statistics
    permit. This avoids staging the blob on the Cloud Run tmpfs, cuts peak
    memory roughly in half during load, and transfers less data over the wire.
    """
    gcs_path = f"gs://{INVENTORIES_BUCKET}/{inventory_id}"
    columns = list(REQUIRED_COLUMNS)
    if biomass_column and biomass_column not in columns:
        columns.append(biomass_column)

    try:
        return pd.read_parquet(
            gcs_path,
            columns=columns,
            filters=[("fia_status_code", "=", 1)],
        )
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


def drop_null_rows(df: pd.DataFrame, biomass_column: str | None = None) -> pd.DataFrame:
    """Drop rows with nulls in any required column (plus `biomass_column` when set).

    Parquet's row-group statistics can skip dead-tree groups (the
    `fia_status_code == 1` pushdown lives in `read_inventory`), but can't
    drop individual rows missing `dbh` / `height` / `crown_ratio`. That's
    this function's job.
    """
    required = list(REQUIRED_COLUMNS)
    if biomass_column:
        required.append(biomass_column)
    return df.dropna(subset=required).reset_index(drop=True)


def assign_tree_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with a unique int32 `tree_id` column, without
    deep-copying the input.

    `DataFrame.assign` returns a new frame that shares underlying column
    arrays with the caller — so we get non-mutation for free without paying
    for a full block-manager copy of every existing column.
    """
    return df.assign(tree_id=np.arange(len(df), dtype="int32"))
