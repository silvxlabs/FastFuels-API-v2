"""Tabular tree-inventory I/O — parquet download, filtering, ID assignment.

"Inventory" here means the tabular tree data (the parquet): rows of trees with
`x, y, fia_species_code, fia_status_code, dbh, height, crown_ratio`. The job
that turns this tabular data into a 3D fuel grid is called *voxelization* and
lives in `treevox.orchestrator`. Keep the distinction when reading/editing
either module.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from lib.config import INVENTORIES_BUCKET
from lib.gcs import download_file
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


def download_inventory(inventory_id: str, tmpdir: str) -> pd.DataFrame:
    """Download and parse a tree inventory parquet from INVENTORIES_BUCKET."""
    gcs_path = f"gs://{INVENTORIES_BUCKET}/{inventory_id}"
    local_path = os.path.join(tmpdir, "inventory.parquet")
    try:
        download_file(gcs_path, local_path)
    except FileNotFoundError as e:
        raise ProcessingError(
            code="INVENTORY_NOT_FOUND",
            message=f"Inventory {inventory_id} not found at {gcs_path}.",
            suggestion="Verify the inventory ID exists and has completed processing.",
        ) from e
    except Exception as e:
        # gcsfs raises FileNotFoundError or variants; treat any I/O failure as
        # a missing inventory for user-facing purposes.
        raise ProcessingError(
            code="INVENTORY_NOT_FOUND",
            message=f"Could not read inventory {inventory_id}: {e}",
            suggestion="Verify the inventory ID exists and has completed processing.",
        ) from e
    return pd.read_parquet(local_path)


def filter_live(df: pd.DataFrame, biomass_column: str | None = None) -> pd.DataFrame:
    """Drop null required columns and retain only live trees (fia_status_code == 1)."""
    required = list(REQUIRED_COLUMNS)
    if biomass_column:
        required.append(biomass_column)
    df = df.dropna(subset=required)
    df = df[df["fia_status_code"] == 1].reset_index(drop=True)
    return df


def assign_tree_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Assign a unique int32 `tree_id` per row without mutating the input."""
    out = df.copy()
    out["tree_id"] = np.arange(len(out), dtype="int32")
    return out
