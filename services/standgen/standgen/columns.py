"""Column definitions for tree inventories."""

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
