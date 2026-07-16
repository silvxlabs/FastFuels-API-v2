"""DUET surface fuel handler: run the DUET binary over a 3D tree grid.

Routed to by `treevox.dispatch.dispatch_handler` on the
(operation='duet', input='grid', entity='tree') source triple.

DUET (McDanold et al. 2023) drops leaf and needle litter from each tree crown
along wind-driven elliptical trajectories, then grows grass as a function of
shade and litter cover. It reads three 3D `.dat` files from its working
directory, writes layered surface `.dat` files back, and duet-tools turns those
into per-fuel-type 2D arrays.

This is the first subprocess in v2. Two consequences shape the module:

- **The binary is a black box that fails quietly.** It returns 0 on inputs it
  partially ignores (see `treevox.duet_species`), so every guard has to run
  before the call, not after.
- **Its working directory is its interface.** Everything lands in a per-job temp
  dir, never a fixed path. v1 wrote into a directory inside the installed
  package, which survived across warm-instance invocations.

Unlike the voxelize handler, this one is single-process: DUET is one Fortran
call over the whole domain, so there is nothing to chunk or pool.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rioxarray  # noqa: F401 — registers `.rio` accessor on xr.Dataset
import xarray as xr

from lib.config import GRIDS_BUCKET
from lib.zarr_utils import load_zarr, save_zarr
from treevox import duet_species
from treevox.errors import ProcessingError

logger = logging.getLogger(__name__)

DUET_BINARY = Path(__file__).parent.parent / "data" / "duet_v2.1_FF_linux.exe"
DUET_SPECIES_FILE = duet_species.DUET_SPECIES_FILE

# DUET requires its species table under this exact name in the working dir.
DUET_SPECIES_FILENAME = "FIA_FastFuels_fin_fulllist_populated.txt"

# Bands the source tree grid must carry. Mirrors DUET_REQUIRED_SOURCE_BANDS in
# the API's duet schema, which rejects a missing band at create time; this is
# the backstop for a grid whose bands changed in between.
FOLIAGE_BAND = "bulk_density.foliage.live"
SPCD_BAND = "spcd"
MOISTURE_BAND = "fuel_moisture.live"

# duet-tools fuel type per API band suffix.
_FUEL_TYPES = {
    "grass": "grass",
    "litter": "litter",
    "litter.coniferous": "coniferous",
    "litter.deciduous": "deciduous",
    "total": "integrated",
}

# duet-tools fuel parameter per API band prefix.
_PARAMETERS = {
    "fuel_load": "loading",
    "fuel_depth": "depth",
    "fuel_moisture": "moisture",
}

# 2D chunking for the output surface grid, matching griddle's 2D grids.
CHUNK_SHAPE_2D = (512, 512)


@dataclass
class DuetResult:
    """Returned by the DUET job for persistence in Firestore."""

    gcs_path: str
    georeference: dict
    chunk_shape: list[int]


def gcs_path(grid_id: str) -> str:
    """Build the GCS URI for a grid's zarr store."""
    return f"gs://{GRIDS_BUCKET}/{grid_id}"


def _split_band(key: str) -> tuple[str, str]:
    """Split an API band key into its (parameter, fuel-type) duet-tools names."""
    for prefix, parameter in _PARAMETERS.items():
        if key.startswith(f"{prefix}."):
            return parameter, _FUEL_TYPES[key[len(prefix) + 1 :]]
    raise ProcessingError(
        code="UNKNOWN_BAND",
        message=f"Unrecognized DUET band {key!r}.",
        suggestion="Request bands from the DUET band vocabulary.",
    )


def _load_source_grid(source_grid_id: str) -> xr.Dataset:
    """Open the source tree grid's zarr and load the three bands DUET reads.

    Goes through `lib.zarr_utils.load_zarr` rather than `xr.open_zarr` for its
    `decode_coords="all"`, without which `spatial_ref` stays a data variable,
    `.rio.crs` is None, and the output grid loses the CRS it should inherit. Its
    `mask_and_scale=False` also keeps `spcd` an integer instead of promoting it
    to float.

    Loads eagerly: DUET needs whole-domain arrays on disk before it starts, so
    there is nothing to gain from staying lazy, and the peak is the same either
    way.
    """
    path = gcs_path(source_grid_id)
    ds = load_zarr(path)
    missing = [
        band
        for band in (FOLIAGE_BAND, SPCD_BAND, MOISTURE_BAND)
        if band not in ds.data_vars
    ]
    if missing:
        raise ProcessingError(
            code="SOURCE_GRID_MISSING_BANDS",
            message=f"Source grid {source_grid_id} is missing bands: {missing}.",
            suggestion=(
                "Voxelize a tree inventory requesting bulk_density.foliage.live, "
                "spcd, and fuel_moisture.live, then run DUET against that grid."
            ),
        )
    if ds.rio.crs is None:
        raise ProcessingError(
            code="SOURCE_GRID_MISSING_CRS",
            message=f"Source grid {source_grid_id} has no CRS.",
            suggestion=(
                "The grid's zarr store is missing its spatial_ref coordinate. "
                "Re-create the source grid."
            ),
        )
    # Subset after the CRS check: `[[...]]` keeps spatial_ref, but reading it
    # from the full Dataset keeps the failure attributable to the store.
    return ds[[FOLIAGE_BAND, SPCD_BAND, MOISTURE_BAND]].load()


def _remap_species(spcd: np.ndarray, grid_id: str) -> np.ndarray:
    """Rewrite species codes to the representatives DUET and duet-tools share.

    Rejects codes neither tool can handle rather than letting them through: DUET
    ignores a code it does not know and still exits 0, so the failure would
    otherwise surface as a grid that is quietly all grass. See
    `treevox.duet_species` for both drops and their measurements.
    """
    present = {int(code) for code in np.unique(spcd) if code != 0}
    if not present:
        raise ProcessingError(
            code="NO_SPECIES_IN_GRID",
            message="Source grid's spcd band contains no tree species.",
            suggestion=(
                "Verify the source grid was voxelized from an inventory with "
                "populated fia_species_code values."
            ),
        )

    unusable = duet_species.unmappable(present)
    if unusable:
        raise ProcessingError(
            code="UNSUPPORTED_SPECIES",
            message=(
                f"DUET cannot model FIA species code(s) {sorted(unusable)}. Their "
                f"litter would be silently omitted, so the run was stopped instead."
            ),
            suggestion=(
                "Remove or reclassify these species in the source inventory, then "
                "re-voxelize. DUET models 274 FIA species; the codes above are "
                "outside its table or unclassifiable as coniferous/deciduous."
            ),
        )

    mapping = duet_species.remap(present)
    remapped = spcd.copy()
    for original, representative in mapping.items():
        if original != representative:
            remapped[spcd == original] = representative
    layers = len(set(mapping.values()))
    logger.info(
        f"Species: {len(present)} SPCD -> {layers} DUET litter layer(s)",
        extra={"grid_id": grid_id},
    )
    return remapped


def _write_duet_inputs(work: Path, ds: xr.Dataset, source: dict, grid_id: str) -> None:
    """Stage the binary, its species table, and the three input arrays.

    Arrays are written as `(nz, ny, nx)` with no axis juggling. duet-tools'
    `write_array_to_dat` streams them out in that order and DUET's column-major
    read recovers `(nx, ny, nz)`, which is what the Fortran expects. v1's two
    `np.moveaxis` calls transposed the grid.
    """
    from duet_tools import InputFile
    from duet_tools.utils import write_array_to_dat

    shutil.copy(DUET_SPECIES_FILE, work / DUET_SPECIES_FILENAME)
    shutil.copy(DUET_BINARY, work / "duet.exe")
    os.chmod(work / "duet.exe", 0o755)

    foliage = np.ascontiguousarray(ds[FOLIAGE_BAND].values, dtype=np.float32)
    spcd = _remap_species(
        np.ascontiguousarray(ds[SPCD_BAND].values, dtype=np.int32), grid_id
    )
    # v2 grids store moisture as a percentage; DUET works in fractions.
    moisture = np.ascontiguousarray(ds[MOISTURE_BAND].values / 100.0, dtype=np.float32)

    write_array_to_dat(foliage, "treesrhof.dat", work, dtype=np.float32)
    write_array_to_dat(moisture, "treesmoist.dat", work, dtype=np.float32)
    write_array_to_dat(spcd, "treesspcd.dat", work, dtype=np.int32)

    nz, ny, nx = foliage.shape
    hr = float(abs(ds.x.values[1] - ds.x.values[0])) if nx > 1 else 1.0
    vr = float(abs(ds.z.values[1] - ds.z.values[0])) if nz > 1 else 1.0
    InputFile(
        nx=nx,
        ny=ny,
        nz=nz,
        dx=hr,
        dy=hr,
        dz=vr,
        # DUET reads a seed but does not use it — seeds 42 and 1234 produce
        # bit-identical output — so it is not exposed on the API and fixed here.
        random_seed=42,
        # int(), not a cast for tidiness: `InputFile` writes whatever repr it is
        # given, and DUET reads wind with a Fortran *integer* list read. A float
        # renders as "270.0" and aborts the model with "Bad integer for item 1"
        # before it reads a single array. The API types these as int; this keeps
        # a document written before that was true from taking the model down.
        wind_direction=int(source["wind_direction"]),
        wind_variability=int(source["wind_variability"]),
        duration=int(source["years_since_burn"]),
    ).to_file(work)


def _run_binary(work: Path, grid_id: str) -> None:
    """Execute DUET in its working directory.

    No timeout is passed: Cloud Run kills the request at the service timeout
    first, so a second deadline here would only ever fire later than the one
    that matters. v1 set 30 minutes against a 540s service limit.
    """
    start = time.monotonic()
    process = subprocess.run(
        ["./duet.exe"],
        cwd=work,
        capture_output=True,
        text=True,
    )
    elapsed = time.monotonic() - start
    if process.returncode != 0:
        # Capture both streams: DUET's own narration goes to stdout, but a
        # Fortran runtime error ("Fortran runtime error: ...", the `At line N of
        # file ...` trace) goes to stderr, so stdout alone reports the failure
        # without the reason.
        output = "\n".join(
            f"--- {name} ---\n{stream.strip()}"
            for name, stream in (("stdout", process.stdout), ("stderr", process.stderr))
            if stream and stream.strip()
        )
        logger.warning(
            f"DUET exited {process.returncode} after {elapsed:.1f}s:\n{output[-3000:]}",
            extra={"grid_id": grid_id},
        )
        raise ProcessingError(
            code="DUET_FAILED",
            message=f"The DUET model exited with status {process.returncode}.",
            suggestion="Check service logs for the model's output.",
            traceback=output[-4000:],
        )
    logger.info(f"DUET completed in {elapsed:.1f}s", extra={"grid_id": grid_id})


def _import_run(work: Path):
    """Import DUET's outputs, working around duet-tools' degenerate-weight crash.

    `_loading_weighted_average` derives its weights by calling
    `_maxmin_calibration(loading, max=1, min=0)` — a *calibration* routine used
    for weighting. That routine legitimately rejects two inputs a calibration
    can't be defined on, and neither is guarded on the weighting path, so both
    take down the whole import:

    - **No positive loading.** `x1 = x[x > 0]` is empty and `np.max([])` raises
      "zero-size array to reduction operation maximum". `import_duet_manual`
      checks that each litter group *has members*, never that they deposited
      mass, so one hardwood among 36k conifers whose litter never landed in the
      domain is enough. Measured on a real stand.
    - **One distinct positive loading.** `_maxmin_calibration` raises
      explicitly, since it cannot map a single value onto a range. Reachable
      whenever a species' litter lands in exactly one cell.

    Weights are uniform in both cases, which is what upstream's own formula
    tends to: `(x1 - min)/(max - min)` has a zero numerator, and the `weights[
    weights == 0] = 0.01` line then flattens every layer to the same weight. So
    the average degenerates to an unweighted one, and to zeros where there is no
    fuel to have moisture.

    Patched around the call rather than at import so the shim's lifetime is
    visible, and restored in `finally` so a duet-tools release that fixes this
    surfaces here rather than being silently double-guarded.
    """
    import duet_tools.calibration as calibration
    from duet_tools import import_duet

    original = calibration._loading_weighted_average

    def guarded(moisture: np.ndarray, loading: np.ndarray) -> np.ndarray:
        positive = loading[loading > 0]
        if positive.size and np.unique(positive).size > 1:
            return original(moisture, loading)
        if not positive.size:
            return np.zeros(loading.shape[1:])
        masked = np.ma.masked_array(moisture, moisture == 0)
        return np.ma.filled(np.ma.average(masked, axis=0), 0)

    calibration._loading_weighted_average = guarded
    try:
        return import_duet(directory=work, version="v2")
    finally:
        calibration._loading_weighted_average = original


def _build_targets(calibration_config: dict) -> dict:
    """Translate the stored calibration config into duet-tools target objects.

    Returns {duet-tools parameter -> FuelParameterTargets}, ready for
    `calibrate`. The API has already rejected impossible combinations, so this
    only maps names.
    """
    from duet_tools import assign_targets, set_fuel_parameter

    method_kwargs = {
        "maxmin": ("max", "min"),
        "meansd": ("mean", "sd"),
        "constant": ("value",),
    }

    parameter_targets = {}
    for api_parameter, duet_parameter in _PARAMETERS.items():
        fuel_types = calibration_config.get(api_parameter)
        if not fuel_types:
            continue
        assigned = {}
        for fuel_type, target in fuel_types.items():
            method = target["method"]
            kwargs = {
                key: target[key]
                for key in method_kwargs[method]
                if target.get(key) is not None
            }
            assigned[fuel_type] = assign_targets(method=method, **kwargs)
        parameter_targets[duet_parameter] = set_fuel_parameter(
            parameter=duet_parameter, **assigned
        )
    return parameter_targets


def _calibrate(duet_run, calibration_config: dict | None, grid_id: str):
    """Apply calibration targets, if any."""
    if not calibration_config:
        logger.info(
            "No calibration requested; storing raw DUET output",
            extra={"grid_id": grid_id},
        )
        return duet_run

    from duet_tools import calibrate

    targets = _build_targets(calibration_config)
    logger.info(
        f"Calibrating parameters: {sorted(targets)}", extra={"grid_id": grid_id}
    )
    try:
        return calibrate(
            duet_run=duet_run, fuel_parameter_targets=list(targets.values())
        )
    except ValueError as e:
        # duet-tools raises this when a targeted fuel type has no fuel anywhere
        # in the domain — e.g. calibrating deciduous litter in a pure conifer
        # stand. That is a request/domain mismatch, not a system fault.
        raise ProcessingError(
            code="CALIBRATION_FAILED",
            message=str(e),
            suggestion=(
                "Calibration targets must name fuel types DUET actually produced "
                "in this domain. Check the stand's species composition, or drop "
                "the target."
            ),
        ) from e


def _build_dataset(
    duet_run, bands: list[str], y_coords: np.ndarray, x_coords: np.ndarray, crs
) -> xr.Dataset:
    """Extract the requested bands into a 2D (y, x) Dataset carrying the source CRS.

    Moisture is converted back to percent to match every other v2 fuel_moisture
    band.
    """
    data_vars = {}
    for key in bands:
        parameter, fuel_type = _split_band(key)
        values = duet_run.to_numpy(fuel_type=fuel_type, fuel_parameter=parameter)
        if parameter == "moisture":
            values = values * 100.0
        data_vars[key] = (("y", "x"), np.asarray(values, dtype=np.float32))

    ds = xr.Dataset(data_vars, coords={"y": y_coords, "x": x_coords})
    return ds.rio.write_crs(crs)


def duet_grid(
    grid: dict,
    domain_gdf,
    progress: Callable[[str, int | None], None],
) -> DuetResult:
    """Run DUET over a 3D tree grid and write a 2D surface fuel zarr to GCS.

    Stages:
      1. _load_source_grid  — read the three bands DUET needs.
      2. _write_duet_inputs — remap species, stage binary + dat files.
      3. _run_binary        — the subprocess.
      4. _import_run        — duet-tools import, with the empty-species shim.
      5. _calibrate         — optional target application.
      6. _build_dataset     — requested bands as 2D (y, x), then save.

    `domain_gdf` is unused: the source grid already carries the domain's extent
    and CRS, and DUET works in grid space. It stays in the signature to match
    the handler contract `dispatch_handler` calls.
    """
    grid_id = grid["id"]
    source = grid["source"]
    path = gcs_path(grid_id)

    job_start = time.monotonic()
    logger.info(f"Starting DUET for grid {grid_id}", extra={"grid_id": grid_id})

    progress("Loading source grid...", 5)
    source_ds = _load_source_grid(source["source_grid_id"])
    nz, ny, nx = source_ds[FOLIAGE_BAND].shape
    logger.info(
        f"Source grid: {nz}x{ny}x{nx} (z, y, x), "
        f"{source['years_since_burn']} years since burn",
        extra={"grid_id": grid_id},
    )

    # Cloud Run's local writes are tmpfs, so this directory is RAM and counts
    # against the container's memory limit alongside the arrays themselves.
    work = Path(tempfile.mkdtemp(prefix=f"duet_{grid_id}_"))
    try:
        progress("Preparing DUET inputs...", 15)
        _write_duet_inputs(work, source_ds, source, grid_id)
        # The dat files on disk are now the only copy DUET reads. Cloud Run
        # counts those files as RAM, so hold the canopy arrays and the tmpfs
        # copies at once for as short a window as possible.
        y_coords, x_coords = source_ds.y.values, source_ds.x.values
        crs = source_ds.rio.crs
        del source_ds

        progress(f"Running DUET for {source['years_since_burn']} years...", 25)
        _run_binary(work, grid_id)

        progress("Importing DUET output...", 75)
        duet_run = _import_run(work)

        progress("Calibrating...", 85)
        duet_run = _calibrate(duet_run, source.get("calibration"), grid_id)

        progress("Writing grid...", 92)
        ds = _build_dataset(duet_run, source["bands"], y_coords, x_coords, crs)
        save_zarr(path, ds, chunk_shape=CHUNK_SHAPE_2D)
    finally:
        shutil.rmtree(work, ignore_errors=True)

    georeference = {
        "crs": str(crs),
        "transform": list(ds.rio.transform())[:6],
        "shape": [ny, nx],
    }

    logger.info(
        f"DUET job completed in {time.monotonic() - job_start:.2f} seconds",
        extra={"grid_id": grid_id},
    )
    return DuetResult(
        gcs_path=path,
        georeference=georeference,
        chunk_shape=list(CHUNK_SHAPE_2D),
    )
