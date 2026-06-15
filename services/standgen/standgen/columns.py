"""Column definitions for tree inventories."""

from lib.inventory import CROWN_RATIO, DIAMETER, HEIGHT, SPECIES, STATUS, X, Y

# Column rename mapping: fastfuels-core output → v2 schema
RENAME_MAP = {
    "SPCD": SPECIES,
    "STATUSCD": STATUS,
    "DIA": DIAMETER,
    "HT": HEIGHT,
    "CR": CROWN_RATIO,
    "X": X,
    "Y": Y,
}

# Columns to drop from the fastfuels-core output (internal to point process)
DROP_COLUMNS = {"TREE_ID", "PLOT_ID", "TPA"}

BASE_COLUMNS = [
    X,
    Y,
    SPECIES,
    STATUS,
    DIAMETER,
    HEIGHT,
    CROWN_RATIO,
]
