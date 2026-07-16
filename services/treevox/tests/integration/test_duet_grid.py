"""Integration tests for the DUET surface fuel flow.

Deliberately thin, like the voxelization tests: unit tests cover the wiring
with the binary mocked, and these exist for the one thing they cannot fake —
**actually running the DUET binary in the image we ship**.

That matters more here than for any other handler. DUET is the only compiled
binary in v2, so a whole class of failure lives outside Python and outside this
repo: the ELF needs `libgfortran5` present in the image, needs its executable
bit to survive the build, needs linux/amd64, and fails in ways that do not
resemble a Python error. It also has to fit real Cloud Run memory and finish
inside the real timeout. None of that is observable from a process running on a
CI runner or a laptop — only from the deployed service, which is what
``DEPLOYMENT_ENV != "local"`` routes to via Cloud Tasks.

Two things worth knowing if these fail:

- **A DUET failure looks like a `DUET_FAILED` grid error, not an exception.**
  The model's own message is on the grid document's `error.traceback`.
- **`DEPLOYMENT_ENV=local` will not work on an arm64 machine.** The binary is a
  linux/amd64 ELF, so local mode runs it on the developer's own hardware. CI is
  always deployed mode.

Each DUET run costs a voxelization first (~30s) plus the model itself, so the
list stays short.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.integration

# The three bands DUET reads. `spcd` in particular is not voxelized by default.
SOURCE_BANDS = ["bulk_density.foliage.live", "spcd", "fuel_moisture.live"]
SOURCE_MOISTURE = {"live": {"method": "uniform", "value": 100.0}}

# The Blue Mountain PIM inventory carries 12 FIA species — ponderosa, Douglas-fir,
# lodgepole, larch, grand/subalpine fir, Engelmann spruce, limber pine and two
# junipers (coniferous), plus quaking aspen and water birch (deciduous). All 12
# are modelable by DUET and collapse to 8 litter layers, and having both classes
# present is what makes the coniferous/deciduous split testable.
STATIC_INVENTORY = "static-test-blue-mtn-pim-inventory"


@pytest.fixture
def tree_grid(treevox_runner):
    """A real voxelized tree grid carrying everything DUET needs."""
    return treevox_runner(
        static_inventory=STATIC_INVENTORY,
        bands=SOURCE_BANDS,
        moisture_model=SOURCE_MOISTURE,
    )


def test_duet_on_pim_tree_grid(tree_grid, duet_runner):
    """Full happy path: real tree grid → DUET binary in the image → 2D zarr.

    The assertion that matters most is simply that this completes. Everything
    downstream of "the binary ran" is covered by unit tests; what cannot be
    faked is the binary starting at all inside the deployed image.
    """
    result = duet_runner(
        source_grid_id=tree_grid.grid_id,
        domain_id=tree_grid.grid["domain_id"],
        bands=[
            "fuel_load.grass",
            "fuel_load.litter",
            "fuel_load.litter.coniferous",
            "fuel_load.litter.deciduous",
            "fuel_load.total",
            "fuel_depth.litter",
            "fuel_moisture.litter",
        ],
        years_since_burn=25,
    )

    # The surface grid covers the canopy grid's footprint, minus its z axis.
    source_geo = tree_grid.grid["georeference"]
    assert result.grid["georeference"]["shape"] == source_geo["shape"][1:]
    assert result.grid["georeference"]["crs"] == source_geo["crs"]

    for key in result.grid["source"]["bands"]:
        values = result.ds[key].values
        assert np.isfinite(values).all(), f"{key} has non-finite cells"
        assert (values >= 0).all(), f"{key} has negative values"

    # Litter fell somewhere, and grass grew somewhere.
    assert result.ds["fuel_load.litter"].values.sum() > 0
    assert result.ds["fuel_load.grass"].values.sum() > 0

    # Garbage tripwire, not a physics claim. Raw DUET magnitudes are
    # uncalibrated and deliberately not asserted precisely — but they are also
    # not 1e32. When DUET misreads its input it says so only by returning
    # absurd numbers at returncode 0, and every other assertion here (finite,
    # non-negative, sums, aggregate identities) holds just as well on garbage.
    # Measured on this inventory at 25 years: litter max 1.44 kg/m**2, depth
    # 0.07 m; the mac binary gave 1.48 on comparable real grids. The bound is
    # loose enough to leave the model room and tight enough to catch nonsense.
    assert result.ds["fuel_load.litter"].values.max() < 50, (
        "litter loading is implausibly large — DUET likely misread its input"
    )
    assert result.ds["fuel_depth.litter"].values.max() < 10, (
        "litter depth is implausibly deep — DUET likely misread its input"
    )

    # The stand is mostly conifer but carries aspen and birch, so both litter
    # classes should be present — this is the species remap surviving a real
    # 12-species inventory rather than a synthetic one.
    assert result.ds["fuel_load.litter.coniferous"].values.sum() > 0
    assert result.ds["fuel_load.litter.deciduous"].values.sum() > 0

    # Aggregates are consistent with their parts.
    assert result.ds["fuel_load.litter"].values == pytest.approx(
        result.ds["fuel_load.litter.coniferous"].values
        + result.ds["fuel_load.litter.deciduous"].values,
        rel=1e-4,
    )
    assert result.ds["fuel_load.total"].values == pytest.approx(
        result.ds["fuel_load.grass"].values + result.ds["fuel_load.litter"].values,
        rel=1e-4,
    )


def test_duet_calibrated_hits_its_targets(tree_grid, duet_runner):
    """Calibration through the real duet-tools import path.

    Worth its own run because the import path is where duet-tools 1.0.1 crashes
    on species that deposited no litter, and a 12-species stand is exactly where
    that shows up. The shim lives in the handler, so only a real run exercises
    it against real per-species output.
    """
    result = duet_runner(
        source_grid_id=tree_grid.grid_id,
        domain_id=tree_grid.grid["domain_id"],
        bands=["fuel_load.grass", "fuel_load.litter"],
        years_since_burn=25,
        calibration={
            "fuel_load": {
                "grass": {
                    "source": "values",
                    "method": "maxmin",
                    "max": 1.5,
                    "min": 0.0,
                },
                "litter": {
                    "source": "values",
                    "method": "maxmin",
                    "max": 5.0,
                    "min": 0.0,
                },
            }
        },
    )

    grass = result.ds["fuel_load.grass"].values
    litter = result.ds["fuel_load.litter"].values
    assert grass.max() == pytest.approx(1.5, rel=1e-3)
    assert litter.max() == pytest.approx(5.0, rel=1e-3)
    # Calibration rescales only cells that already carry fuel; it never invents
    # fuel where DUET left none.
    assert grass.min() == 0.0
    assert litter.min() == 0.0
