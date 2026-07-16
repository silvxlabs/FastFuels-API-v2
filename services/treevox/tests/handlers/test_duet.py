"""Unit tests for treevox.handlers.duet — the DUET job and its stages.

The binary and GCS are mocked here; `tests/integration/test_duet_grid.py` runs
the real thing. What these cover is the wiring around the subprocess: the guards
that must fire *before* it runs (because it fails silently, not loudly), the
band/axis/unit translation, and the shim around duet-tools' import crash.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import xarray as xr
from treevox.errors import ProcessingError
from treevox.handlers import duet as handler


def _read_duet_in(work: Path) -> list[str]:
    """Return duet.in's values, dropping its trailing `! description` comments."""
    return [
        line.split("!")[0].strip()
        for line in (work / "duet.in").read_text().splitlines()
        if line.strip()
    ]


def _source_dataset(nz=6, ny=8, nx=10, spcd=122, moisture=100.0) -> xr.Dataset:
    foliage = np.zeros((nz, ny, nx), dtype=np.float32)
    foliage[2:5, 2:6, 2:8] = 0.4
    codes = np.zeros((nz, ny, nx), dtype=np.uint16)
    codes[2:5, 2:6, 2:8] = spcd
    moist = np.zeros((nz, ny, nx), dtype=np.float32)
    moist[2:5, 2:6, 2:8] = moisture
    ds = xr.Dataset(
        {
            handler.FOLIAGE_BAND: (("z", "y", "x"), foliage),
            handler.SPCD_BAND: (("z", "y", "x"), codes),
            handler.MOISTURE_BAND: (("z", "y", "x"), moist),
        },
        coords={
            "z": np.arange(nz, dtype=float) + 0.5,
            "y": np.arange(ny, dtype=float) * 2.0 + 1.0,
            "x": np.arange(nx, dtype=float) * 2.0 + 1.0,
        },
    )
    return ds.rio.write_crs("EPSG:32613")


class TestSplitBand:
    @pytest.mark.parametrize(
        "key,expected",
        [
            ("fuel_load.grass", ("loading", "grass")),
            ("fuel_load.litter", ("loading", "litter")),
            ("fuel_load.litter.coniferous", ("loading", "coniferous")),
            ("fuel_load.litter.deciduous", ("loading", "deciduous")),
            ("fuel_load.total", ("loading", "integrated")),
            ("fuel_depth.grass", ("depth", "grass")),
            ("fuel_moisture.total", ("moisture", "integrated")),
        ],
    )
    def test_maps_api_band_to_duet_tools_names(self, key, expected):
        assert handler._split_band(key) == expected

    def test_unknown_band_raises(self):
        with pytest.raises(ProcessingError) as exc:
            handler._split_band("bulk_density.foliage.live")
        assert exc.value.code == "UNKNOWN_BAND"


class TestRemapSpecies:
    def test_rewrites_codes_to_representatives(self):
        # Two California live oaks that share a bucket.
        spcd = np.array([[0, 801], [805, 0]], dtype=np.int32)
        out = handler._remap_species(spcd, "g1")
        assert set(np.unique(out)) == {0, 314}

    def test_leaves_fill_value_alone(self):
        spcd = np.zeros((3, 3), dtype=np.int32)
        spcd[1, 1] = 122
        out = handler._remap_species(spcd, "g1")
        assert out[0, 0] == 0
        assert (out == 0).sum() == 8

    def test_rejects_species_duet_would_silently_drop(self):
        # Great Basin bristlecone pine. Without this guard DUET returns 0 and
        # the grid is quietly all grass.
        spcd = np.array([[122, 142]], dtype=np.int32)
        with pytest.raises(ProcessingError) as exc:
            handler._remap_species(spcd, "g1")
        assert exc.value.code == "UNSUPPORTED_SPECIES"
        assert "142" in exc.value.message

    def test_rejects_species_duet_tools_would_drop_at_import(self):
        spcd = np.array([[122, 1000]], dtype=np.int32)
        with pytest.raises(ProcessingError) as exc:
            handler._remap_species(spcd, "g1")
        assert exc.value.code == "UNSUPPORTED_SPECIES"
        assert "1000" in exc.value.message

    def test_rejects_empty_species_band(self):
        with pytest.raises(ProcessingError) as exc:
            handler._remap_species(np.zeros((4, 4), dtype=np.int32), "g1")
        assert exc.value.code == "NO_SPECIES_IN_GRID"


class TestLoadSourceGrid:
    def test_missing_band_raises_before_dispatching_work(self):
        ds = _source_dataset().drop_vars(handler.SPCD_BAND)
        with patch.object(handler, "load_zarr", return_value=ds):
            with pytest.raises(ProcessingError) as exc:
                handler._load_source_grid("src")
        assert exc.value.code == "SOURCE_GRID_MISSING_BANDS"
        assert handler.SPCD_BAND in exc.value.message

    def test_missing_crs_raises(self):
        ds = _source_dataset().drop_vars("spatial_ref")
        with patch.object(handler, "load_zarr", return_value=ds):
            with pytest.raises(ProcessingError) as exc:
                handler._load_source_grid("src")
        assert exc.value.code == "SOURCE_GRID_MISSING_CRS"

    def test_reads_the_crs_back_off_a_real_zarr_round_trip(self, tmp_path):
        # xr.open_zarr without decode_coords="all" leaves spatial_ref a data
        # variable and .rio.crs None, so the output grid silently loses its CRS.
        # Only a real round trip catches it — an in-memory Dataset always has it.
        _source_dataset().to_zarr(tmp_path / "src", mode="w", consolidated=True)
        with patch.object(handler, "gcs_path", lambda gid: str(tmp_path / gid)):
            ds = handler._load_source_grid("src")
        assert ds.rio.crs == "EPSG:32613"

    def test_spcd_stays_an_integer_across_a_round_trip(self, tmp_path):
        # mask_and_scale=True would promote the categorical band to float.
        _source_dataset().to_zarr(tmp_path / "src", mode="w", consolidated=True)
        with patch.object(handler, "gcs_path", lambda gid: str(tmp_path / gid)):
            ds = handler._load_source_grid("src")
        assert np.issubdtype(ds[handler.SPCD_BAND].dtype, np.integer)


class TestWriteDuetInputs:
    def test_writes_dat_files_and_input_file(self, tmp_path):
        ds = _source_dataset()
        source = {
            "years_since_burn": 25,
            "wind_direction": 270.0,
            "wind_variability": 30.0,
        }
        handler._write_duet_inputs(tmp_path, ds, source, "g1")

        written = {f.name for f in tmp_path.iterdir()}
        assert {
            "treesrhof.dat",
            "treesmoist.dat",
            "treesspcd.dat",
            "duet.in",
            "duet.exe",
            handler.DUET_SPECIES_FILENAME,
        } <= written

    def test_binary_is_executable(self, tmp_path):
        ds = _source_dataset()
        handler._write_duet_inputs(
            tmp_path,
            ds,
            {"years_since_burn": 5, "wind_direction": 0.0, "wind_variability": 0.0},
            "g1",
        )
        import os

        assert os.access(tmp_path / "duet.exe", os.X_OK)

    def test_input_file_carries_grid_shape_and_wind(self, tmp_path):
        ds = _source_dataset(nz=6, ny=8, nx=10)
        handler._write_duet_inputs(
            tmp_path,
            ds,
            {
                "years_since_burn": 25,
                "wind_direction": 180.0,
                "wind_variability": 45.0,
            },
            "g1",
        )
        values = _read_duet_in(tmp_path)
        # duet.in is one value per line: nx, ny, nz, dx, dy, dz, seed, wind
        # direction, wind variability, duration.
        assert [int(v) for v in values[:3]] == [10, 8, 6]
        assert [float(v) for v in values[3:6]] == [2.0, 2.0, 1.0]
        assert int(values[7]) == 180
        assert int(values[8]) == 45
        assert int(values[9]) == 25

    def test_wind_is_written_as_integers(self, tmp_path):
        # DUET reads wind with a Fortran integer list read. "270.0" aborts it
        # with "Bad integer for item 1" before it reads a single array, so every
        # job would fail. Measured against the real binary.
        ds = _source_dataset()
        handler._write_duet_inputs(
            tmp_path,
            ds,
            {
                "years_since_burn": 25,
                "wind_direction": 270.0,
                "wind_variability": 30.0,
            },
            "g1",
        )
        values = _read_duet_in(tmp_path)
        assert values[7] == "270"
        assert values[8] == "30"
        assert values[9] == "25"

    def test_moisture_is_converted_from_percent_to_fraction(self, tmp_path):
        # v2 grids store 100.0; DUET reads fractions. v1 fed the percent through.
        ds = _source_dataset(moisture=100.0)
        handler._write_duet_inputs(
            tmp_path,
            ds,
            {"years_since_burn": 5, "wind_direction": 0.0, "wind_variability": 0.0},
            "g1",
        )
        written = np.fromfile(tmp_path / "treesmoist.dat", dtype=np.float32)
        assert written.max() == pytest.approx(1.0)

    def test_arrays_are_written_without_transposing(self, tmp_path):
        # duet-tools wants (nz, ny, nx) passed straight through; Fortran's
        # column-major read recovers (nx, ny, nz). v1's double moveaxis
        # transposed the grid.
        ds = _source_dataset(nz=6, ny=8, nx=10)
        handler._write_duet_inputs(
            tmp_path,
            ds,
            {"years_since_burn": 5, "wind_direction": 0.0, "wind_variability": 0.0},
            "g1",
        )
        raw = np.fromfile(tmp_path / "treesrhof.dat", dtype=np.float32)
        expected = ds[handler.FOLIAGE_BAND].values
        # Strip Fortran's record markers, then compare in (nz, ny, nx) order.
        assert raw.size >= expected.size
        assert raw[1 : expected.size + 1].reshape(expected.shape) == pytest.approx(
            expected
        )


class TestRunBinary:
    def test_nonzero_exit_raises_processing_error(self, tmp_path):
        completed = subprocess.CompletedProcess(
            args=["./duet.exe"], returncode=2, stdout="fortran said no", stderr=""
        )
        with patch("subprocess.run", return_value=completed):
            with pytest.raises(ProcessingError) as exc:
                handler._run_binary(tmp_path, "g1")
        assert exc.value.code == "DUET_FAILED"
        assert "fortran said no" in exc.value.traceback

    def test_success_is_quiet(self, tmp_path):
        completed = subprocess.CompletedProcess(
            args=["./duet.exe"], returncode=0, stdout="", stderr=""
        )
        with patch("subprocess.run", return_value=completed):
            handler._run_binary(tmp_path, "g1")


class TestImportRunShim:
    """duet-tools 1.0.1 crashes importing a species that deposited no litter.

    The guard has to leave no trace once the call returns, so a duet-tools
    release that fixes the bug does not end up double-guarded.
    """

    def test_zero_loading_returns_zeros_instead_of_raising(self, tmp_path):
        import duet_tools.calibration as calibration

        captured = {}

        def fake_import_duet(directory, version):
            moisture = np.zeros((2, 4, 4))
            loading = np.zeros((2, 4, 4))
            captured["result"] = calibration._loading_weighted_average(
                moisture, loading
            )
            return "duet_run"

        with patch("duet_tools.import_duet", side_effect=fake_import_duet):
            assert handler._import_run(tmp_path) == "duet_run"
        assert captured["result"].shape == (4, 4)
        assert not captured["result"].any()

    def test_single_distinct_loading_returns_an_unweighted_average(self, tmp_path):
        # _maxmin_calibration raises outright here ("only one positive value"),
        # a separate crash path from the empty case. Weights are uniform, so the
        # answer is the plain average of the layers' moisture.
        import duet_tools.calibration as calibration

        captured = {}

        def fake_import_duet(directory, version):
            moisture = np.stack([np.full((2, 2), 0.4), np.full((2, 2), 0.8)])
            loading = np.full((2, 2, 2), 2.0)
            captured["result"] = calibration._loading_weighted_average(
                moisture, loading
            )
            return "duet_run"

        with patch("duet_tools.import_duet", side_effect=fake_import_duet):
            handler._import_run(tmp_path)
        assert captured["result"] == pytest.approx(np.full((2, 2), 0.6))

    def test_varying_loading_delegates_to_duet_tools(self, tmp_path):
        # The normal path must reach upstream untouched — the shim only covers
        # inputs upstream refuses.
        import duet_tools.calibration as calibration

        moisture = np.stack([np.full((2, 2), 0.4), np.full((2, 2), 0.8)])
        loading = np.stack([np.full((2, 2), 1.0), np.full((2, 2), 3.0)])
        expected = calibration._loading_weighted_average(moisture, loading)

        captured = {}

        def fake_import_duet(directory, version):
            captured["result"] = calibration._loading_weighted_average(
                moisture, loading
            )
            return "duet_run"

        with patch("duet_tools.import_duet", side_effect=fake_import_duet):
            handler._import_run(tmp_path)
        assert captured["result"] == pytest.approx(expected)

    def test_shim_is_removed_after_the_call(self, tmp_path):
        import duet_tools.calibration as calibration

        before = calibration._loading_weighted_average
        with patch("duet_tools.import_duet", return_value="duet_run"):
            handler._import_run(tmp_path)
        assert calibration._loading_weighted_average is before

    def test_shim_is_removed_even_when_import_raises(self, tmp_path):
        import duet_tools.calibration as calibration

        before = calibration._loading_weighted_average
        with patch("duet_tools.import_duet", side_effect=RuntimeError("boom")):
            with pytest.raises(RuntimeError):
                handler._import_run(tmp_path)
        assert calibration._loading_weighted_average is before


class TestBuildTargets:
    def test_builds_one_fuel_parameter_per_configured_parameter(self):
        config = {
            "fuel_load": {
                "grass": {
                    "source": "values",
                    "method": "meansd",
                    "mean": 0.5,
                    "sd": 0.25,
                },
                "litter": {
                    "source": "values",
                    "method": "maxmin",
                    "max": 5.0,
                    "min": 0.0,
                },
            },
            "fuel_depth": {
                "grass": {"source": "values", "method": "constant", "value": 0.3},
            },
        }
        targets = handler._build_targets(config)
        assert set(targets) == {"loading", "depth"}

    def test_ignores_unset_parameters(self):
        assert handler._build_targets({}) == {}

    def test_passes_only_the_fields_the_method_uses(self):
        config = {
            "fuel_load": {
                "grass": {"source": "values", "method": "constant", "value": 1.0},
            }
        }
        with patch("duet_tools.assign_targets") as assign:
            with patch("duet_tools.set_fuel_parameter"):
                handler._build_targets(config)
        assign.assert_called_once_with(method="constant", value=1.0)


class TestCalibrate:
    def test_no_config_returns_the_run_untouched(self):
        run = object()
        assert handler._calibrate(run, None, "g1") is run
        assert handler._calibrate(run, {}, "g1") is run

    def test_missing_fuel_type_becomes_a_processing_error(self):
        config = {
            "fuel_load": {
                "deciduous": {"source": "values", "method": "constant", "value": 1.0},
            }
        }
        with patch.object(
            handler, "_build_targets", return_value={"loading": object()}
        ):
            with patch(
                "duet_tools.calibrate",
                side_effect=ValueError("No fuels present for fuel type deciduous"),
            ):
                with pytest.raises(ProcessingError) as exc:
                    handler._calibrate(object(), config, "g1")
        assert exc.value.code == "CALIBRATION_FAILED"
        assert "deciduous" in exc.value.message


class TestBuildDataset:
    def _duet_run(self):
        run = MagicMock()
        run.to_numpy.side_effect = lambda fuel_type, fuel_parameter: np.full(
            (8, 10), 0.25 if fuel_parameter == "moisture" else 2.0
        )
        return run

    def test_produces_2d_bands_on_the_source_grid(self):
        y = np.arange(8, dtype=float) * 2.0 + 1.0
        x = np.arange(10, dtype=float) * 2.0 + 1.0
        ds = handler._build_dataset(
            self._duet_run(),
            ["fuel_load.grass", "fuel_load.litter"],
            y,
            x,
            "EPSG:32613",
        )
        assert list(ds.data_vars) == ["fuel_load.grass", "fuel_load.litter"]
        assert ds["fuel_load.grass"].dims == ("y", "x")
        assert ds["fuel_load.grass"].shape == (8, 10)
        assert ds.rio.crs == "EPSG:32613"

    def test_moisture_is_converted_back_to_percent(self):
        y = np.arange(8, dtype=float)
        x = np.arange(10, dtype=float)
        ds = handler._build_dataset(
            self._duet_run(), ["fuel_moisture.grass"], y, x, "EPSG:32613"
        )
        # duet-tools works in fractions; every other v2 fuel_moisture band is %.
        assert float(ds["fuel_moisture.grass"].max()) == pytest.approx(25.0)

    def test_load_bands_are_not_rescaled(self):
        y = np.arange(8, dtype=float)
        x = np.arange(10, dtype=float)
        ds = handler._build_dataset(
            self._duet_run(), ["fuel_load.grass"], y, x, "EPSG:32613"
        )
        assert float(ds["fuel_load.grass"].max()) == pytest.approx(2.0)


class TestDuetGrid:
    """Full flow with the binary, GCS, and duet-tools mocked."""

    def _grid(self, bands=None, calibration=None):
        source = {
            "operation": "duet",
            "input": "grid",
            "entity": "tree",
            "source_grid_id": "src",
            "years_since_burn": 25,
            "wind_direction": 270.0,
            "wind_variability": 30.0,
            "bands": bands or ["fuel_load.grass", "fuel_load.litter"],
        }
        if calibration:
            source["calibration"] = calibration
        return {"id": "g1", "domain_id": "d1", "source": source}

    def _run_handler(self, grid, saved):
        duet_run = MagicMock()
        duet_run.to_numpy.return_value = np.full((8, 10), 1.5)
        with (
            patch.object(handler, "_load_source_grid", return_value=_source_dataset()),
            patch.object(handler, "_run_binary") as run_binary,
            patch.object(handler, "_import_run", return_value=duet_run),
            patch.object(
                handler,
                "save_zarr",
                side_effect=lambda p, d, chunk_shape: saved.update(
                    path=p, ds=d, chunk_shape=chunk_shape
                ),
            ),
        ):
            result = handler.duet_grid(grid, MagicMock(), lambda *a, **k: None)
        return result, run_binary

    def test_writes_requested_bands_and_returns_a_2d_georeference(self):
        saved = {}
        result, _ = self._run_handler(self._grid(), saved)

        assert list(saved["ds"].data_vars) == ["fuel_load.grass", "fuel_load.litter"]
        assert saved["chunk_shape"] == handler.CHUNK_SHAPE_2D
        # DUET reads a 3D canopy and writes a 2D surface, so the georeference
        # loses the z axis the source grid had.
        assert result.georeference["shape"] == [8, 10]
        assert len(result.georeference["transform"]) == 6
        assert result.georeference["crs"] == "EPSG:32613"
        assert result.chunk_shape == [512, 512]

    def test_cleans_up_the_working_directory(self):
        captured = {}
        original = handler._write_duet_inputs

        def capture(work, ds, source, grid_id):
            captured["work"] = work
            return original(work, ds, source, grid_id)

        duet_run = MagicMock()
        duet_run.to_numpy.return_value = np.full((8, 10), 1.5)
        with (
            patch.object(handler, "_load_source_grid", return_value=_source_dataset()),
            patch.object(handler, "_write_duet_inputs", side_effect=capture),
            patch.object(handler, "_run_binary"),
            patch.object(handler, "_import_run", return_value=duet_run),
            patch.object(handler, "save_zarr"),
        ):
            handler.duet_grid(self._grid(), MagicMock(), lambda *a, **k: None)
        assert not Path(captured["work"]).exists()

    def test_working_directory_is_removed_when_the_binary_fails(self):
        captured = {}
        original = handler._write_duet_inputs

        def capture(work, ds, source, grid_id):
            captured["work"] = work
            return original(work, ds, source, grid_id)

        with (
            patch.object(handler, "_load_source_grid", return_value=_source_dataset()),
            patch.object(handler, "_write_duet_inputs", side_effect=capture),
            patch.object(
                handler,
                "_run_binary",
                side_effect=ProcessingError(code="DUET_FAILED", message="nope"),
            ),
        ):
            with pytest.raises(ProcessingError):
                handler.duet_grid(self._grid(), MagicMock(), lambda *a, **k: None)
        # A stale working directory is RAM on Cloud Run, not just clutter.
        assert not Path(captured["work"]).exists()

    def test_uncalibrated_run_skips_duet_tools_calibrate(self):
        saved = {}
        with patch("duet_tools.calibrate") as calibrate:
            self._run_handler(self._grid(), saved)
        calibrate.assert_not_called()

    def test_years_since_burn_reaches_the_input_file(self):
        captured = {}
        original = handler._write_duet_inputs

        def capture(work, ds, source, grid_id):
            original(work, ds, source, grid_id)
            captured["duet_in"] = _read_duet_in(work)

        duet_run = MagicMock()
        duet_run.to_numpy.return_value = np.full((8, 10), 1.5)
        with (
            patch.object(handler, "_load_source_grid", return_value=_source_dataset()),
            patch.object(handler, "_write_duet_inputs", side_effect=capture),
            patch.object(handler, "_run_binary"),
            patch.object(handler, "_import_run", return_value=duet_run),
            patch.object(handler, "save_zarr"),
        ):
            handler.duet_grid(self._grid(), MagicMock(), lambda *a, **k: None)
        assert int(captured["duet_in"][9]) == 25
