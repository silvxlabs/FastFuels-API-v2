"""Canonical tree-inventory column names, shared across services.

A tree inventory's measurement roles are a fixed, known vocabulary: each role
has one canonical column name. The upload boundary normalizes arbitrary source
labels into these names (e.g. a file column ``DIA`` becomes ``dbh``), so
downstream consumers can rely on them directly.

Declared here once so the API guards, the voxelizer (treevox), and stand
generation (standgen) share a single source of truth instead of re-asserting
string literals like ``"dbh"`` in each service. Stdlib-only so the GDAL-free API
can import it.
"""

X = "x"
Y = "y"
HEIGHT = "height"
DIAMETER = "dbh"
SPECIES = "fia_species_code"
STATUS = "fia_status_code"
CROWN_RATIO = "crown_ratio"

# Columns required to voxelize a tree inventory: position, the per-tree
# measurements the crown-profile and biomass-allometry models read, and the
# status code used to keep only live trees.
VOXELIZE_REQUIRED_COLUMNS = frozenset(
    {X, Y, HEIGHT, DIAMETER, SPECIES, STATUS, CROWN_RATIO}
)
