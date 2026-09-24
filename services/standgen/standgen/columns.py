"""Column definitions for tree inventories."""

import dask.dataframe as dd

# Column rename mapping: fastfuels-core output → v2 schema
RENAME_MAP = {
    "SPCD": "fia_species_code",
    "STATUSCD": "fia_status_code",
    "DIA": "dbh",
    "HT": "height",
    "CR": "crown_ratio",
    "X": "x",
    "Y": "y",
}

# Columns to drop from the fastfuels-core output (internal to point process)
DROP_COLUMNS = {"TREE_ID", "PLOT_ID", "TPA"}

BASE_COLUMNS = [
    "x",
    "y",
    "fia_species_code",
    "fia_status_code",
    "dbh",
    "height",
    "crown_ratio",
]

TREE_ID_COLUMN = "tree_id"


def generate_tree_ids(ddf: dd.DataFrame) -> dd.DataFrame:
    """Prepend an int32 ``tree_id`` numbering the rows ``1 … N`` in row order.

    Lazy: a cumulative sum over a constant column, so dask computes each
    partition's local sum and carries only a scalar between partitions. The
    result stays in the caller's graph, so the single ``dask.compute`` at write
    time still runs the upstream graph once. Call before any create-time
    modification or treatment, so a tree removed at creation leaves a gap.
    """
    ddf = ddf.assign(**{TREE_ID_COLUMN: 1})
    ddf = ddf.assign(**{TREE_ID_COLUMN: ddf[TREE_ID_COLUMN].cumsum().astype("int32")})
    return ddf[[TREE_ID_COLUMN, *(c for c in ddf.columns if c != TREE_ID_COLUMN)]]
