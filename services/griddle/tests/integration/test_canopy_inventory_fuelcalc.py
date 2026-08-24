"""Reproduce FuelCalc's tutorial output through the inventory canopy endpoint.

FuelCalc 1.7 ships a two-plot tutorial treelist and, run on it pre- and
post-thinning, prints a Stand Measurements block per plot. This test lays the
two plots out in space as a 1x2 grid of **ten-acre cells** — plot 1 in the west
cell, plot 2 in the east — builds a real v2 tree inventory from them (including
the ``fia_crown_class_code`` column, #521), configures the grid with the API's
``fuelcalc_comparison`` example settings, runs the whole griddle pipeline (GCS
parquet -> fastfuels-core canopy metrics -> zarr), and checks each cell against
``fastfuels-core``'s own FuelCalc-parity numbers.

Where this differs from ``fastfuels-core``'s comparison test
(``tests/canopy_fuel/test_fuelcalc_comparison.py``): that one calls core
directly on a single aspatial ten-acre cell to check the *science*. This one
goes through the *API schema and griddle ETL* on a spatial two-cell lattice to
prove the contract a user actually drives produces the same numbers. The
objective is the ETL reproduction, not a second copy of the science check — so
the reference here is core's computed metrics (which core's own test pins
against FuelCalc's printout), and the tolerances are tight.

**Why ten-acre cells.** FuelCalc has no horizontal structure; it works from
per-acre expansion factors. To represent that stand as whole trees the cell has
to be big enough that ``expansion_factor * cell_acres`` lands on a whole number
— core uses ten acres precisely because the tutorial's factors are quoted to a
tenth, so ``* 10`` is exact with no rounding. A smaller cell (e.g. 30 m) forces
a fractional-stem rounding that, on this treelist's uniform densities, is a pure
tie: which records round up is then decided by ``argsort``'s tie-break, which
differs by platform (arm64 vs amd64) and swings CBD by tens of percent. Matching
core's ten-acre cell removes the quantization entirely and makes the whole test
deterministic across platforms — there is no ``argsort`` here at all.

**Western larch P2 (a FuelCalc bug).** The one place core deviates from
FuelCalc's *printout*: FuelCalc's compiled table carries ``0.745*exp(-0.0632d)``
where Brown 1978 Table 16 and FuelCalc's own User Guide print ``-0.0362``.
fastfuels-core implements the published coefficient, so larch-bearing stands
(both plots) land a little off FuelCalc's printed CBD / CFL / stand height. The
reference values below are core's — which absorb that deviation — so this test
reproduces core exactly; core's own test is where the FuelCalc anchor lives.

Canopy cover reads a few tenths of a percent below core's single-cell value: at
ten acres the finite-cell edge effect on ``crown_overlap`` is nearly gone but
not quite, and stems here are placed at random positions in the cell rather than
all at its center. The RNG seed is fixed so that placement — and therefore cover
— is reproducible; PCG64 is platform-independent, so it is identical on arm64
and amd64.
"""

from __future__ import annotations

import math
from pathlib import Path
from uuid import uuid4

import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest

from lib.config import INVENTORIES_BUCKET
from lib.gcs import get_gcsfs_client

pytestmark = pytest.mark.integration

TREELIST = Path(__file__).parent / "data" / "fuelcalc_tutorial_treelist.csv"

ACRE_M2 = 4046.8564224
FT_TO_M = 0.3048
LB_TO_KG = 0.45359237
CELL_ACRES = 10
# Side of a ten-acre square cell, in metres. Exactly 201.168 m
# (201.168**2 == 10 * ACRE_M2), so a 2*SIDE x SIDE domain resolves to an exact
# 1x2 lattice. Matches ``fastfuels-core``'s comparison test.
SIDE = math.sqrt(CELL_ACRES * ACRE_M2)
SEED = 0

# FuelCalc species mnemonics -> FIA SPCD.
SPECIES_CODES = {
    "PIPO": 122,
    "LAOC": 73,
    "PSME": 202,
    "PICO": 108,
    "ABLA": 19,
    "ABGR": 17,
}
# FuelCalc crown-class letters -> FIA CCLCD, stored raw in the inventory. The
# tutorial carries D/C/I/S; open grown (1) does not appear here. Griddle maps
# these back onto core's letters.
CROWN_CLASS_TO_CCLCD = {"D": 2, "C": 3, "I": 4, "S": 5}

# The domain fixture spans 2*SIDE x SIDE from this origin (EPSG:32611), so the
# ten-acre lattice is one row of two cells: west cell (col 0) is plot 1, east
# cell (col 1) is plot 2. Each plot's stems fill its own cell.
ORIGIN_X = 720000.0
ORIGIN_Y = 5190000.0
PLOT_CELL_X = {
    1: (ORIGIN_X, ORIGIN_X + SIDE),
    2: (ORIGIN_X + SIDE, ORIGIN_X + 2 * SIDE),
}
CELL_Y = (ORIGIN_Y, ORIGIN_Y + SIDE)

# fastfuels-core's FuelCalc-parity metrics for each plot (from its
# test_fuelcalc_comparison.py), in FuelCalc's reporting units: CBD kg/m**3,
# CBH/stand height ft, cover %, canopy fuel load T/ac. griddle reproduces
# CBD/CBH/CHM/CFL exactly on the same stand; cover lands a few tenths low (see
# the module docstring). "post" replays the tutorial's post-thinning expansion
# factors (TPA_POST) directly rather than re-deriving the thinning.
CORE_PARITY = {
    (1, "pre"): dict(cbd=0.0447, cbh_ft=2.0, chm_ft=103.0, cc=48.77, cfl_tac=3.848),
    (1, "post"): dict(cbd=0.0447, cbh_ft=2.0, chm_ft=103.0, cc=43.50, cfl_tac=3.303),
    (2, "pre"): dict(cbd=0.0463, cbh_ft=1.0, chm_ft=125.0, cc=40.99, cfl_tac=2.952),
    (2, "post"): dict(cbd=0.0236, cbh_ft=3.0, chm_ft=129.0, cc=31.54, cfl_tac=2.254),
}

TREATMENTS = {"pre": "TPA_PRE", "post": "TPA_POST"}


def _build_inventory(expansion: str) -> pd.DataFrame:
    """A v2 tree inventory for both plots laid out in their two ten-acre cells.

    Each record's per-acre expansion factor becomes ``round(factor * 10)`` whole
    stems — exact for the tutorial's tenth-precision factors, so there is no
    fractional rounding and no ``argsort`` tie-break to depend on. Stems are
    placed at random positions within their cell (fixed seed).
    """
    trees = pd.read_csv(TREELIST)
    # FuelCalc excludes dead trees from canopy fuel; drop them here too.
    trees = trees[trees["STATUS"] != "D"]
    trees = trees[trees[expansion] > 0]

    rng = np.random.default_rng(SEED)
    frames = []
    for plot, (x0, x1) in PLOT_CELL_X.items():
        p = trees[trees["PLOT"] == plot]
        counts = np.rint(p[expansion].to_numpy() * CELL_ACRES).astype(int)
        idx = np.repeat(np.arange(len(p)), counts)
        n = len(idx)
        frames.append(
            pd.DataFrame(
                {
                    "x": rng.uniform(x0, x1, n),
                    "y": rng.uniform(CELL_Y[0], CELL_Y[1], n),
                    "fia_species_code": p["SPECIES"]
                    .map(SPECIES_CODES)
                    .to_numpy()[idx]
                    .astype("int64"),
                    "fia_status_code": np.ones(n, dtype="int64"),
                    "fia_crown_class_code": p["CROWN_CLASS"]
                    .map(CROWN_CLASS_TO_CCLCD)
                    .to_numpy()[idx]
                    .astype("int64"),
                    "dbh": (p["DBH_IN"].to_numpy() * 2.54)[idx],
                    "height": (p["HEIGHT_FT"].to_numpy() * FT_TO_M)[idx],
                    "crown_ratio": (
                        1.0 - p["CBH_FT"].to_numpy() / p["HEIGHT_FT"].to_numpy()
                    )[idx],
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


@pytest.fixture
def stage_inventory():
    """Write a built inventory to INVENTORIES_BUCKET and clean it up.

    Yields a factory ``stage(df) -> inventory_id``. The griddle handler derives
    the parquet path from the id alone (it never reads the inventory Firestore
    document), so only the GCS dataset is staged. Written as a dask parquet
    dataset with an aggregated ``_metadata`` footer — the layout standgen and
    the uploader produce and ``read_inventory`` reads.
    """
    staged: list[str] = []

    def _stage(df: pd.DataFrame) -> str:
        inventory_id = f"test-{uuid4().hex}"
        path = f"gs://{INVENTORIES_BUCKET}/{inventory_id}"
        dd.from_pandas(df, npartitions=1).to_parquet(
            path, write_metadata_file=True, write_index=False
        )
        staged.append(inventory_id)
        return inventory_id

    yield _stage

    fs = get_gcsfs_client()
    for inventory_id in staged:
        target = f"{INVENTORIES_BUCKET}/{inventory_id}"
        if fs.exists(target):
            fs.rm(target, recursive=True)


def _cell_values(ds, bands: list[str]) -> dict[int, dict[str, float]]:
    """Map each plot to its cell's band values, keyed by cell x position."""
    assert ds.sizes["y"] == 1, f"expected one lattice row, got {ds.sizes['y']}"
    assert ds.sizes["x"] == 2, f"expected two lattice columns, got {ds.sizes['x']}"
    xs = ds["x"].values
    west, east = int(np.argmin(xs)), int(np.argmax(xs))
    return {
        1: {b: float(ds[b].values[0, west]) for b in bands},
        2: {b: float(ds[b].values[0, east]) for b in bands},
    }


def _assert_matches_core(si: dict[str, float], ref: dict, plot: int, label: str):
    """Each band against core's FuelCalc-parity metrics, in FuelCalc's units."""
    where = f"plot {plot} {label}"

    cbd = si["cbd"]
    assert cbd == pytest.approx(ref["cbd"], abs=5e-4), (
        f"{where}: CBD {cbd:.4f} kg/m**3 vs core {ref['cbd']}"
    )

    # Core anchors CBH to the layer bottom; FuelCalc labels a layer by its top,
    # one 1 ft layer higher. core's reported canopy_base_height_ft applies that
    # shift, and griddle reproduces it to the same 1 ft layer.
    cbh_ft = si["cbh"] / FT_TO_M + 1.0
    assert cbh_ft == pytest.approx(ref["cbh_ft"], abs=0.5), (
        f"{where}: CBH {cbh_ft:.1f} ft vs core {ref['cbh_ft']}"
    )

    chm_ft = si["chm"] / FT_TO_M
    assert chm_ft == pytest.approx(ref["chm_ft"], abs=1.0), (
        f"{where}: stand height {chm_ft:.1f} ft vs core {ref['chm_ft']}"
    )

    cfl_tac = si["cfl"] * ACRE_M2 / LB_TO_KG / 2000.0
    assert cfl_tac == pytest.approx(ref["cfl_tac"], abs=0.02), (
        f"{where}: canopy fuel load {cfl_tac:.3f} T/ac vs core {ref['cfl_tac']}"
    )

    # crown_overlap cover reads a few tenths of a percent below core's
    # single-cell value — the residual ten-acre finite-cell edge effect with
    # stems placed off-center. Never high; bounded near equality.
    cc = si["cc"]
    assert cc == pytest.approx(ref["cc"], abs=1.0), (
        f"{where}: canopy cover {cc:.2f}% vs core {ref['cc']}%"
    )


@pytest.mark.parametrize("label", ["pre", "post"])
def test_griddle_reproduces_fuelcalc_tutorial(griddle_runner, stage_inventory, label):
    """The fuelcalc_comparison schema, run through griddle on the two tutorial
    plots as a 1x2 ten-acre grid, reproduces fastfuels-core's FuelCalc-parity
    metrics per plot."""
    inventory_id = stage_inventory(_build_inventory(TREATMENTS[label]))

    result = griddle_runner(
        "fuelcalc_tutorial_2plot.json",
        "canopy_inventory_fuelcalc.json",
        source_overrides={"source_inventory_id": inventory_id},
    )
    bands = ["cbd", "cbh", "chm", "cc", "cfl"]
    cells = _cell_values(result.ds, bands)

    for plot in (1, 2):
        _assert_matches_core(cells[plot], CORE_PARITY[(plot, label)], plot, label)
