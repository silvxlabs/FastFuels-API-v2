"""Reproduce FuelCalc's tutorial output through the inventory canopy endpoint.

FuelCalc 1.7 ships a two-plot tutorial treelist and, run on it pre- and
post-thinning, prints a Stand Measurements block per plot. Those printed
numbers are the reference here. This test lays the two plots out in space as a
1x2 grid of 30 m cells — plot 1 in the west cell, plot 2 in the east — builds a
real v2 tree inventory from them (including the new ``fia_crown_class_code``
column, #521), configures the grid with the API's ``fuelcalc_comparison``
example settings, runs the whole griddle pipeline (GCS parquet -> fastfuels-core
canopy metrics -> zarr), and checks each cell against the plot report FuelCalc
wrote for it.

Where this differs from ``fastfuels-core``'s own comparison test
(``tests/canopy_fuel/test_fuelcalc_comparison.py``): that one calls core
directly on a single aspatial 10-acre cell to check the *science*. This one goes
through the *API schema and griddle ETL* on a spatial 30 m lattice to prove the
contract a user actually drives reproduces FuelCalc. The objective is the schema
reproduction, not a second copy of the science check.

FuelCalc is the reference software, not the definitive answer. Three understood,
measured effects separate this pipeline's numbers from FuelCalc's printout, and
the tolerances below are sized to them:

- **Western larch P2 (a FuelCalc bug).** FuelCalc's compiled table carries
  ``0.745*exp(-0.0632d)``; Brown 1978 Table 16 and FuelCalc's own User Guide
  print ``-0.0362``. fastfuels-core implements the published coefficient, so
  larch-bearing stands (both plots) land a little off FuelCalc's printed CBD /
  CFL / stand height — the same deviation that test names, ~1-2% at the stand
  level here.
- **30 m stem quantization.** FuelCalc works from per-acre expansion factors; a
  30 m cell holds whole stems, so each record's per-acre density is rounded
  (largest-remainder, holding the stand total to <1%). Shifts CBH/CHM by at most
  one 1 ft layer.
- **30 m finite-cell cover.** ``crown_overlap`` cover is cell-size dependent at
  fixed density — it converges to FuelCalc's aspatial value as the cell grows to
  acre scale, and reads a characterized few-to-~16% low at 30 m. So ``cc`` is
  checked as a bounded reduction of FuelCalc's, not an equality.

CBD/CBH/CHM/CFL are invariant to where stems land within a cell (``stem``
distribution drops each tree's fuel in its own cell); only ``cc`` depends on the
placement, so the RNG seed is fixed.
"""

from __future__ import annotations

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
CELL_M = 30.0
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

# The domain fixture is 60 m x 30 m, so the 30 m lattice is one row of two
# cells: west cell (col 0) is plot 1, east cell (col 1) is plot 2. Stems fill
# the whole cell — cover must be normalized against the area the stems occupy.
# numpy's uniform is high-exclusive, so no plot-1 stem reaches the shared
# 720030 boundary, and a plot-2 stem at exactly 720030 falls in the east cell
# (its own), so no stem lands in the wrong cell.
PLOT_CELL_X = {1: (720000.0, 720030.0), 2: (720030.0, 720060.0)}
CELL_Y = (5190000.0, 5190030.0)

# FuelCalc plot-report Stand Measurements (the .rtf reports shipped with the
# tutorial), in FuelCalc's units: CBD kg/m**3, CBH/stand height ft, cover %,
# canopy fuel load T/ac.
FUELCALC_REPORT = {
    (1, "pre"): dict(cbd=0.044, cbh_ft=1.0, chm_ft=103.0, cc=48.77, cfl_tac=3.79),
    (1, "post"): dict(cbd=0.044, cbh_ft=2.0, chm_ft=103.0, cc=43.50, cfl_tac=3.24),
    (2, "pre"): dict(cbd=0.046, cbh_ft=1.0, chm_ft=123.0, cc=40.99, cfl_tac=2.76),
    (2, "post"): dict(cbd=0.023, cbh_ft=3.0, chm_ft=126.0, cc=31.54, cfl_tac=2.07),
}

TREATMENTS = {"pre": "TPA_PRE", "post": "TPA_POST"}


def _largest_remainder(tpa: np.ndarray, area_m2: float) -> np.ndarray:
    """Whole stems per record at the cell's density, preserving the stand total.

    A 30 m cell holds far fewer stems than FuelCalc's nominal acre, so
    ``tpa * cell_acres`` is fractional; rounding each record independently would
    lose ~10% of the stand. Largest-remainder rounding holds the total to the
    nearest whole stem.
    """
    target = tpa * area_m2 / ACRE_M2
    floor = np.floor(target).astype(int)
    remainder = int(round(target.sum())) - int(floor.sum())
    order = np.argsort(-(target - floor))
    counts = floor.copy()
    counts[order[:remainder]] += 1
    return counts


def _build_inventory(expansion: str) -> pd.DataFrame:
    """A v2 tree inventory for both plots laid out in their two cells."""
    trees = pd.read_csv(TREELIST)
    # FuelCalc excludes dead trees from canopy fuel; drop them here too.
    trees = trees[trees["STATUS"] != "D"]
    trees = trees[trees[expansion] > 0]

    rng = np.random.default_rng(SEED)
    frames = []
    for plot, (x0, x1) in PLOT_CELL_X.items():
        p = trees[trees["PLOT"] == plot]
        counts = _largest_remainder(p[expansion].to_numpy(), CELL_M * CELL_M)
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


def _assert_reproduces_fuelcalc(si: dict[str, float], ref: dict, plot: int, label: str):
    """Each band against FuelCalc's report, in FuelCalc's units and tolerances."""
    where = f"plot {plot} {label}"

    cbd = si["cbd"]
    assert cbd == pytest.approx(ref["cbd"], abs=0.004), (
        f"{where}: CBD {cbd:.4f} kg/m**3 vs FuelCalc {ref['cbd']}"
    )

    # Core anchors CBH to the layer bottom; FuelCalc labels a layer by its top,
    # one 1 ft layer higher. Within one layer of FuelCalc's after that shift.
    cbh_ft = si["cbh"] / FT_TO_M + 1.0
    assert abs(cbh_ft - ref["cbh_ft"]) <= 1.0 + 1e-6, (
        f"{where}: CBH {cbh_ft:.1f} ft vs FuelCalc {ref['cbh_ft']}"
    )

    # Stand height. Within a few feet — the larch coefficient sets where the
    # profile crosses the threshold near the canopy top in the larch-tall plot.
    chm_ft = si["chm"] / FT_TO_M
    assert abs(chm_ft - ref["chm_ft"]) <= 4.0 + 1e-6, (
        f"{where}: stand height {chm_ft:.1f} ft vs FuelCalc {ref['chm_ft']}"
    )

    cfl_tac = si["cfl"] * ACRE_M2 / LB_TO_KG / 2000.0
    assert cfl_tac == pytest.approx(ref["cfl_tac"], abs=0.25), (
        f"{where}: canopy fuel load {cfl_tac:.3f} T/ac vs FuelCalc {ref['cfl_tac']}"
    )

    # crown_overlap cover reads a characterized fraction low at 30 m (finite
    # cell), never high. Bound it rather than equate it.
    cc = si["cc"]
    assert 0.80 * ref["cc"] <= cc <= 0.98 * ref["cc"], (
        f"{where}: canopy cover {cc:.2f}% vs FuelCalc {ref['cc']}% "
        f"(expected the 30 m finite-cell reduction, ~10-15% low)"
    )


@pytest.mark.parametrize("label", ["pre", "post"])
def test_griddle_reproduces_fuelcalc_tutorial(griddle_runner, stage_inventory, label):
    """The fuelcalc_comparison schema, run through griddle on the two tutorial
    plots as a 1x2 grid, reproduces FuelCalc's per-plot reports."""
    inventory_id = stage_inventory(_build_inventory(TREATMENTS[label]))

    result = griddle_runner(
        "fuelcalc_tutorial_2plot.json",
        "canopy_inventory_fuelcalc.json",
        source_overrides={"source_inventory_id": inventory_id},
    )
    bands = ["cbd", "cbh", "chm", "cc", "cfl"]
    cells = _cell_values(result.ds, bands)

    for plot in (1, 2):
        _assert_reproduces_fuelcalc(
            cells[plot], FUELCALC_REPORT[(plot, label)], plot, label
        )
