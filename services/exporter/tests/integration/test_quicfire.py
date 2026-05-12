"""
Integration tests for the QUIC-Fire combined export.

Drives the exporter against the five real static fixtures in GCS, downloads
the resulting zip, extracts each ``trees*.dat`` file, and verifies it against
the source zarr data. All fixtures share the lattice
``origin=(720226, 5190646), shape=(442, 654)`` at 2 m horizontal × 1 m
vertical — the canonical Domain-anchored fire grid.
"""

import json
import os
import tempfile
import zipfile
from pathlib import Path

import gcsfs
import numpy as np
import pytest
import xarray as xr
from exporter.storage import load_grid_zarr
from scipy.io import FortranFile

from lib.config import EXPORTS_BUCKET

# Lattice every QF role fixture sits on (post-padded-domain e2e).
_NZ = 37
_NY = 442
_NX = 654
_DZ = 1.0


def _download_and_extract(signed_url: str, gcs_path: str | None = None) -> Path:
    """Pull the export zip from GCS and unpack it to a tmpdir."""
    fs = gcsfs.GCSFileSystem()
    tmpdir = Path(tempfile.mkdtemp(prefix="qf-export-"))
    # Prefer the GCS path (avoids signed-URL flakiness in tests).
    bucket_blob = gcs_path or _gcs_path_from_signed_url(signed_url)
    local_zip = tmpdir / "export.zip"
    fs.get(bucket_blob, str(local_zip))
    with zipfile.ZipFile(local_zip) as zf:
        zf.extractall(tmpdir)
    return tmpdir


def _gcs_path_from_signed_url(signed_url: str) -> str:
    """Strip the GCS host/query from a signed URL to get the bucket/path."""
    # Signed URLs look like https://storage.googleapis.com/<bucket>/<path>?<sig>
    no_scheme = signed_url.split("://", 1)[1]
    host_and_path = no_scheme.split("?", 1)[0]
    _, bucket_and_path = host_and_path.split("/", 1)
    return bucket_and_path


def _read_fortran_3d(path: Path, nz: int, ny: int, nx: int) -> np.ndarray:
    with FortranFile(str(path), "r") as ff:
        arr = ff.read_record(dtype=np.float32)
    return arr.reshape(nz, ny, nx)


def _read_fortran_2d(path: Path, ny: int, nx: int) -> np.ndarray:
    with FortranFile(str(path), "r") as ff:
        arr = ff.read_record(dtype=np.float32)
    return arr.reshape(ny, nx)


def _load_band(grid_id: str, band: str, *, rank: int) -> np.ndarray:
    """Load a band from a source grid in GCS and return the array shape
    expected by the handler — ``(nz, ny, nx)`` for 3D, ``(ny, nx)`` for 2D."""
    ds = load_grid_zarr(grid_id)
    dims = ("z", "y", "x") if rank == 3 else ("y", "x")
    return ds[band].transpose(*dims).values.astype(np.float32, copy=False)


def _y_flip_3d(arr: np.ndarray) -> np.ndarray:
    return np.flip(arr, axis=1)


def _y_flip_2d(arr: np.ndarray) -> np.ndarray:
    return np.flipud(arr)


class TestQuicfireExport:
    def test_minimal_required_roles(self, quicfire_sources, quicfire_exporter_runner):
        """5 required roles, default Domain-anchored fire grid. Verify file
        presence, shapes, and merge math against the source data."""
        export = quicfire_exporter_runner(quicfire_sources)

        out_dir = _download_and_extract(export["signed_url"], gcs_path=_gcs_for(export))

        # All required files present; optional ones absent.
        assert (out_dir / "treesrhof.dat").exists()
        assert (out_dir / "treesmoist.dat").exists()
        assert (out_dir / "treesfueldepth.dat").exists()
        assert (out_dir / "metadata.json").exists()
        assert (out_dir / "domain.geojson").exists()
        assert not (out_dir / "topo.dat").exists()
        assert not (out_dir / "treesss.dat").exists()

        # Shapes match the fire grid.
        rhof = _read_fortran_3d(out_dir / "treesrhof.dat", _NZ, _NY, _NX)
        moist = _read_fortran_3d(out_dir / "treesmoist.dat", _NZ, _NY, _NX)
        fueldepth = _read_fortran_3d(out_dir / "treesfueldepth.dat", _NZ, _NY, _NX)
        assert rhof.shape == (_NZ, _NY, _NX)

        # Load the source bands for verification.
        canopy_rhof = _y_flip_3d(
            _load_band(quicfire_sources["canopy"], "bulk_density.foliage.live", rank=3)
        )
        canopy_moist = _y_flip_3d(
            _load_band(quicfire_sources["canopy"], "fuel_moisture.live", rank=3)
        )
        surf_load = _y_flip_2d(
            _load_band(quicfire_sources["lookup"], "fuel_load.1hr", rank=2)
        )
        surf_depth = _y_flip_2d(
            _load_band(quicfire_sources["lookup"], "fuel_depth", rank=2)
        )
        surf_moist = _y_flip_2d(
            _load_band(
                quicfire_sources["uniform_moisture"], "fuel_moisture.1hr", rank=2
            )
        )
        canopy_rhof = np.nan_to_num(canopy_rhof)
        canopy_moist_frac = np.nan_to_num(canopy_moist) / 100.0
        surf_load = np.nan_to_num(surf_load)
        surf_depth = np.nan_to_num(surf_depth)
        surf_moist_frac = np.nan_to_num(surf_moist) / 100.0

        # rhof: canopy preserved above k=0; k=0 = canopy[0] + surf_load/dz.
        np.testing.assert_allclose(rhof[1:], canopy_rhof[1:], rtol=0, atol=1e-6)
        expected_k0 = canopy_rhof[0] + surf_load / _DZ
        np.testing.assert_allclose(rhof[0], expected_k0, rtol=0, atol=1e-5)

        # moist (max merge by default): canopy_moist/100 above k=0; k=0 =
        # max(canopy_moist[0]/100, surf_moist/100).
        np.testing.assert_allclose(moist[1:], canopy_moist_frac[1:], rtol=0, atol=1e-6)
        expected_moist_k0 = np.maximum(canopy_moist_frac[0], surf_moist_frac)
        np.testing.assert_allclose(moist[0], expected_moist_k0, rtol=0, atol=1e-6)

        # fueldepth: zeros above k=0, surface depth at k=0.
        np.testing.assert_allclose(fueldepth[1:], 0.0, atol=1e-7)
        np.testing.assert_allclose(fueldepth[0], surf_depth, rtol=0, atol=1e-6)

    def test_with_topography(self, quicfire_sources, quicfire_exporter_runner):
        """+ topography role writes topo.dat matching the source elevation
        (NaN→0, Y-flipped)."""
        export = quicfire_exporter_runner(
            quicfire_sources,
            source_overrides={
                "topography": {
                    "grid_id": quicfire_sources["topography"],
                    "band": "elevation",
                },
            },
        )

        out_dir = _download_and_extract(export["signed_url"], gcs_path=_gcs_for(export))
        assert (out_dir / "topo.dat").exists()

        topo = _read_fortran_2d(out_dir / "topo.dat", _NY, _NX)
        expected = np.nan_to_num(
            _y_flip_2d(_load_band(quicfire_sources["topography"], "elevation", rank=2))
        )
        np.testing.assert_allclose(topo, expected, rtol=0, atol=1e-3)

    def test_with_savr_pair(self, quicfire_sources, quicfire_exporter_runner):
        """+ canopy/surface SAVR pair writes treesss.dat as 2/SAVR with
        mass-weighted SAVR at k=0."""
        export = quicfire_exporter_runner(
            quicfire_sources,
            source_overrides={
                "canopy_savr": {
                    "grid_id": quicfire_sources["canopy"],
                    "band": "savr.foliage",
                },
                "surface_savr": {
                    "grid_id": quicfire_sources["lookup"],
                    "band": "savr.1hr",
                },
            },
        )

        out_dir = _download_and_extract(export["signed_url"], gcs_path=_gcs_for(export))
        assert (out_dir / "treesss.dat").exists()

        treesss = _read_fortran_3d(out_dir / "treesss.dat", _NZ, _NY, _NX)
        canopy_rhof = np.nan_to_num(
            _y_flip_3d(
                _load_band(
                    quicfire_sources["canopy"],
                    "bulk_density.foliage.live",
                    rank=3,
                )
            )
        )
        canopy_savr = np.nan_to_num(
            _y_flip_3d(_load_band(quicfire_sources["canopy"], "savr.foliage", rank=3))
        )
        surf_load = np.nan_to_num(
            _y_flip_2d(_load_band(quicfire_sources["lookup"], "fuel_load.1hr", rank=2))
        )
        surf_savr = np.nan_to_num(
            _y_flip_2d(_load_band(quicfire_sources["lookup"], "savr.1hr", rank=2))
        )

        # Above k=0: size_scale = 2 / canopy_savr (zero where SAVR<=0).
        canopy_above = canopy_savr[1:]
        expected_above = np.where(
            canopy_above > 0, 2.0 / np.maximum(canopy_above, 1e-12), 0.0
        )
        np.testing.assert_allclose(treesss[1:], expected_above, rtol=0, atol=1e-4)

        # At k=0: mass-weighted SAVR → 2/SAVR.
        surf_rhof_layer = surf_load / _DZ
        eps = 1e-12
        total = canopy_rhof[0] + surf_rhof_layer + eps
        savr_k0 = (
            canopy_rhof[0] * canopy_savr[0] + surf_rhof_layer * surf_savr
        ) / total
        expected_k0 = np.where(savr_k0 > 0, 2.0 / np.maximum(savr_k0, eps), 0.0)
        np.testing.assert_allclose(treesss[0], expected_k0, rtol=0, atol=1e-4)


def _gcs_for(export: dict) -> str:
    """Extract the bucket/blob path the exporter actually wrote to.

    The handler's `_upload_zip` uses `sanitize_filename(export["name"], ".zip")`,
    so we reproduce the same path locally without re-fetching via signed URL.
    """
    from exporter.filename import sanitize_filename

    filename = sanitize_filename(export.get("name", ""), ".zip")
    return f"{EXPORTS_BUCKET}/{export['id']}/{filename}"


# Suppress pyflakes complaint about the unused imports kept for forward use.
_ = (os, json, xr, pytest, Path)
