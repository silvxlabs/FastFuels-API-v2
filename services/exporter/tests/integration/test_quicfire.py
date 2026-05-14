"""
Integration tests for the QUIC-Fire combined export.

One expensive setup per module — stage source grids, run the exporter with
every role populated, download the zip, pre-load expected source bands —
then many small test functions each verify one invariant against the
already-unpacked output. A separate class runs one additional exporter
invocation to verify the `weighted_avg` moisture mode (the only variant
that produces different bytes than the canonical run).

All fixtures sit on the lattice ``origin=(720226, 5190646), shape=(442, 654)``
at 2 m horizontal × 1 m vertical — the canonical Domain-anchored fire grid.
"""

import json
import shutil
import tempfile
import zipfile
from pathlib import Path

import gcsfs
import numpy as np
import pytest
from exporter.filename import sanitize_filename
from exporter.storage import load_grid_zarr
from scipy.io import FortranFile

from lib.config import EXPORTS_BUCKET

# Fire-grid dims for the static fixtures.
_NZ = 37
_NY = 442
_NX = 654
_DX = 2.0
_DZ = 1.0


# --- Helpers ---


def _gcs_path_from_signed_url(signed_url: str) -> str:
    no_scheme = signed_url.split("://", 1)[1]
    host_and_path = no_scheme.split("?", 1)[0]
    _, bucket_and_path = host_and_path.split("/", 1)
    return bucket_and_path


def _gcs_for(export: dict) -> str:
    filename = sanitize_filename(export.get("name", ""), ".zip")
    return f"{EXPORTS_BUCKET}/{export['id']}/{filename}"


def _download_and_extract(export: dict) -> Path:
    fs = gcsfs.GCSFileSystem()
    tmpdir = Path(tempfile.mkdtemp(prefix="qf-export-"))
    local_zip = tmpdir / "export.zip"
    fs.get(_gcs_for(export), str(local_zip))
    with zipfile.ZipFile(local_zip) as zf:
        zf.extractall(tmpdir)
    return tmpdir


def _read_fortran_3d(path: Path) -> np.ndarray:
    with FortranFile(str(path), "r") as ff:
        return ff.read_record(dtype=np.float32).reshape(_NZ, _NY, _NX)


def _read_fortran_2d(path: Path) -> np.ndarray:
    with FortranFile(str(path), "r") as ff:
        return ff.read_record(dtype=np.float32).reshape(_NY, _NX)


def _load_band(grid_id: str, band: str, *, rank: int) -> np.ndarray:
    ds = load_grid_zarr(grid_id)
    dims = ("z", "y", "x") if rank == 3 else ("y", "x")
    return ds[band].transpose(*dims).values.astype(np.float32, copy=False)


def _y_flip_3d(arr: np.ndarray) -> np.ndarray:
    return np.flip(arr, axis=1)


def _y_flip_2d(arr: np.ndarray) -> np.ndarray:
    return np.flipud(arr)


@pytest.fixture(scope="module")
def full_export(quicfire_sources, quicfire_exporter_runner):
    """Run the exporter once with every optional role populated. Yields the
    unpacked output directory."""
    sources = quicfire_sources
    export = quicfire_exporter_runner(
        sources,
        source_overrides={
            "topography": {"grid_id": sources["topography"], "band": "elevation"},
            "canopy_savr": {"grid_id": sources["canopy"], "band": "savr.foliage"},
            "surface_savr": {"grid_id": sources["lookup"], "band": "savr.1hr"},
        },
    )
    out_dir = _download_and_extract(export)
    yield out_dir
    shutil.rmtree(out_dir, ignore_errors=True)


@pytest.fixture(scope="module")
def expected_bands(quicfire_sources) -> dict[str, np.ndarray]:
    """Pre-load every source band the tests compare against (Y-flipped,
    NaN→0, fractions where the API stores percents). One zarr read per
    band per module."""
    sources = quicfire_sources
    return {
        "canopy_rhof": _y_flip_3d(
            np.nan_to_num(
                _load_band(sources["canopy"], "bulk_density.foliage.live", rank=3)
            )
        ),
        "canopy_moist_frac": _y_flip_3d(
            np.nan_to_num(_load_band(sources["canopy"], "fuel_moisture.live", rank=3))
        )
        / 100.0,
        "canopy_savr": _y_flip_3d(
            np.nan_to_num(_load_band(sources["canopy"], "savr.foliage", rank=3))
        ),
        "surf_load": _y_flip_2d(
            np.nan_to_num(_load_band(sources["lookup"], "fuel_load.1hr", rank=2))
        ),
        "surf_depth": _y_flip_2d(
            np.nan_to_num(_load_band(sources["lookup"], "fuel_depth", rank=2))
        ),
        "surf_savr": _y_flip_2d(
            np.nan_to_num(_load_band(sources["lookup"], "savr.1hr", rank=2))
        ),
        "surf_moist_frac": _y_flip_2d(
            np.nan_to_num(
                _load_band(sources["uniform_moisture"], "fuel_moisture.1hr", rank=2)
            )
        )
        / 100.0,
        "topo": _y_flip_2d(
            np.nan_to_num(_load_band(sources["topography"], "elevation", rank=2))
        ),
    }


class TestQuicfireExport:
    """One canonical export with every role populated; each test verifies
    a single invariant of the unpacked output."""

    def test_all_expected_files_present(self, full_export):
        for name in (
            "treesrhof.dat",
            "treesmoist.dat",
            "treesfueldepth.dat",
            "topo.dat",
            "treesss.dat",
            "metadata.json",
            "domain.geojson",
        ):
            assert (full_export / name).exists(), f"{name} missing from output"

    def test_treesrhof_has_fire_grid_shape(self, full_export):
        rhof = _read_fortran_3d(full_export / "treesrhof.dat")
        assert rhof.shape == (_NZ, _NY, _NX)

    def test_treesrhof_above_k0_preserves_canopy(self, full_export, expected_bands):
        rhof = _read_fortran_3d(full_export / "treesrhof.dat")
        np.testing.assert_allclose(
            rhof[1:], expected_bands["canopy_rhof"][1:], atol=1e-6
        )

    def test_treesrhof_at_k0_adds_surface_load_over_dz(
        self, full_export, expected_bands
    ):
        rhof = _read_fortran_3d(full_export / "treesrhof.dat")
        expected = expected_bands["canopy_rhof"][0] + expected_bands["surf_load"] / _DZ
        np.testing.assert_allclose(rhof[0], expected, atol=1e-5)

    def test_treesmoist_above_k0_is_canopy_fraction(self, full_export, expected_bands):
        moist = _read_fortran_3d(full_export / "treesmoist.dat")
        np.testing.assert_allclose(
            moist[1:], expected_bands["canopy_moist_frac"][1:], atol=1e-6
        )

    def test_treesmoist_at_k0_is_max_of_canopy_and_surface(
        self, full_export, expected_bands
    ):
        """Default `moist_merge="max"` — k=0 is max(canopy, surface). v1 parity."""
        moist = _read_fortran_3d(full_export / "treesmoist.dat")
        expected = np.maximum(
            expected_bands["canopy_moist_frac"][0], expected_bands["surf_moist_frac"]
        )
        np.testing.assert_allclose(moist[0], expected, atol=1e-6)

    def test_treesfueldepth_above_k0_is_zero(self, full_export):
        fd = _read_fortran_3d(full_export / "treesfueldepth.dat")
        np.testing.assert_allclose(fd[1:], 0.0, atol=1e-7)

    def test_treesfueldepth_at_k0_is_surface_depth(self, full_export, expected_bands):
        fd = _read_fortran_3d(full_export / "treesfueldepth.dat")
        np.testing.assert_allclose(fd[0], expected_bands["surf_depth"], atol=1e-6)

    def test_topo_matches_source_elevation_y_flipped(self, full_export, expected_bands):
        topo = _read_fortran_2d(full_export / "topo.dat")
        np.testing.assert_allclose(topo, expected_bands["topo"], atol=1e-3)

    def test_treesss_above_k0_is_two_over_canopy_savr(
        self, full_export, expected_bands
    ):
        treesss = _read_fortran_3d(full_export / "treesss.dat")
        canopy_above = expected_bands["canopy_savr"][1:]
        with np.errstate(divide="ignore", invalid="ignore"):
            expected = np.where(canopy_above > 0, 2.0 / canopy_above, 0.0)
        np.testing.assert_allclose(treesss[1:], expected, atol=1e-4)

    def test_treesss_at_k0_is_mass_weighted_savr(self, full_export, expected_bands):
        treesss = _read_fortran_3d(full_export / "treesss.dat")
        surf_rhof = expected_bands["surf_load"] / _DZ
        total = expected_bands["canopy_rhof"][0] + surf_rhof
        with np.errstate(divide="ignore", invalid="ignore"):
            numerator = (
                expected_bands["canopy_rhof"][0] * expected_bands["canopy_savr"][0]
                + surf_rhof * expected_bands["surf_savr"]
            )
            savr_k0 = np.where(total > 0, numerator / total, 0.0)
            expected = np.where(savr_k0 > 0, 2.0 / savr_k0, 0.0)
        np.testing.assert_allclose(treesss[0], expected, atol=1e-4)

    def test_outputs_are_float32(self, full_export):
        # All trees*.dat files are float32 records — read_fortran_* would
        # raise on a dtype mismatch, but assert here for clarity.
        rhof = _read_fortran_3d(full_export / "treesrhof.dat")
        topo = _read_fortran_2d(full_export / "topo.dat")
        assert rhof.dtype == np.float32
        assert topo.dtype == np.float32

    def test_metadata_records_fire_grid_spec(self, full_export):
        meta = json.loads((full_export / "metadata.json").read_text())
        assert meta["format"] == "quicfire"
        assert "exporter_version" in meta
        assert "completed_on" in meta
        fg = meta["fire_grid"]
        assert fg["nx"] == _NX
        assert fg["ny"] == _NY
        assert fg["nz"] == _NZ
        assert fg["dx"] == _DX
        assert fg["dz"] == _DZ
        assert "transform" in fg
        assert "crs" in fg
        assert "source" in meta

    def test_domain_geojson_is_a_feature_collection(self, full_export):
        gj = json.loads((full_export / "domain.geojson").read_text())
        assert gj["type"] == "FeatureCollection"
        assert len(gj["features"]) >= 1
        # Coordinates were JSON-stringified in Firestore; the handler must
        # have parsed them back to nested lists.
        coords = gj["features"][0]["geometry"]["coordinates"]
        assert isinstance(coords, list)
        assert isinstance(coords[0], list)


class TestQuicfireExportWeightedAvgMoist:
    """`moist_merge="weighted_avg"` produces a different treesmoist.dat at
    k=0 than the default `max` mode. Requires its own exporter invocation."""

    @pytest.fixture(scope="class")
    def weighted_export(self, quicfire_sources, quicfire_exporter_runner):
        export = quicfire_exporter_runner(
            quicfire_sources, source_overrides={"moist_merge": "weighted_avg"}
        )
        out_dir = _download_and_extract(export)
        yield out_dir
        shutil.rmtree(out_dir, ignore_errors=True)

    def test_moist_at_k0_is_mass_weighted_average(
        self, weighted_export, expected_bands
    ):
        moist = _read_fortran_3d(weighted_export / "treesmoist.dat")
        surf_rhof = expected_bands["surf_load"] / _DZ
        total = expected_bands["canopy_rhof"][0] + surf_rhof
        with np.errstate(divide="ignore", invalid="ignore"):
            numerator = (
                expected_bands["canopy_rhof"][0]
                * expected_bands["canopy_moist_frac"][0]
                + surf_rhof * expected_bands["surf_moist_frac"]
            )
            expected = np.where(total > 0, numerator / total, 0.0)
        np.testing.assert_allclose(moist[0], expected, atol=1e-6)

    def test_moist_above_k0_unchanged(self, weighted_export, expected_bands):
        """Weighted-avg only affects k=0; canopy data above is preserved."""
        moist = _read_fortran_3d(weighted_export / "treesmoist.dat")
        np.testing.assert_allclose(
            moist[1:], expected_bands["canopy_moist_frac"][1:], atol=1e-6
        )
