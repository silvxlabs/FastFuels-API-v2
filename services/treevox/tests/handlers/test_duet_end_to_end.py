"""End-to-end DUET handler test: real binary, real duet-tools, local zarr.

Everything except GCS is real here. This is the only test that proves the
pieces actually compose — the baked binary starts, the dat files it reads are
the ones we wrote, duet-tools can import what it produced, and the result lands
in a readable 2D zarr.

Skipped where the baked binary cannot execute (it is a linux/amd64 ELF, so on a
macOS dev machine). CI runs on ubuntu-latest, and the production path is the
image the Dockerfile builds.
"""

from __future__ import annotations

import platform
import subprocess

import numpy as np
import pytest
import xarray as xr
from treevox.errors import ProcessingError
from treevox.handlers import duet as handler


def _binary_runs() -> bool:
    """Can the baked DUET binary execute here?"""
    if not handler.DUET_BINARY.exists():
        return False
    if platform.system() != "Linux" or platform.machine() not in ("x86_64", "AMD64"):
        return False
    try:
        subprocess.run(
            [str(handler.DUET_BINARY), "--version"], capture_output=True, timeout=30
        )
    except OSError:
        # Most likely libgfortran5 missing — the Dockerfile installs it.
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _binary_runs(),
    reason="DUET binary is a linux/amd64 ELF and cannot run on this platform",
)


# Ponderosa pine (coniferous) and northern red oak (deciduous): distinct
# duet-tools classes, so the litter layers are separable.
PINE, OAK = 122, 833


@pytest.fixture
def source_grid(tmp_path, monkeypatch):
    """Write a small 3D tree grid to a local zarr and point the handler at it.

    Left half pine, right half oak, canopy in the middle of the z column.
    """
    nz, ny, nx = 20, 40, 40
    foliage = np.zeros((nz, ny, nx), dtype=np.float32)
    codes = np.zeros((nz, ny, nx), dtype=np.uint16)
    moisture = np.zeros((nz, ny, nx), dtype=np.float32)
    foliage[8:18, 4:36, :] = 0.5
    moisture[8:18, 4:36, :] = 100.0  # v2 stores percent
    codes[8:18, 4:36, :20] = PINE
    codes[8:18, 4:36, 20:] = OAK

    ds = xr.Dataset(
        {
            handler.FOLIAGE_BAND: (("z", "y", "x"), foliage),
            handler.SPCD_BAND: (("z", "y", "x"), codes),
            handler.MOISTURE_BAND: (("z", "y", "x"), moisture),
        },
        coords={
            "z": np.arange(nz, dtype=float) + 0.5,
            "y": np.arange(ny, dtype=float) * 2.0 + 1.0,
            "x": np.arange(nx, dtype=float) * 2.0 + 1.0,
        },
    )
    ds = ds.rio.write_crs("EPSG:32613")
    stores = tmp_path / "stores"
    stores.mkdir()
    ds.to_zarr(stores / "src", mode="w", consolidated=True)

    monkeypatch.setattr(handler, "gcs_path", lambda grid_id: str(stores / grid_id))
    return {"shape": (ny, nx), "stores": stores}


def _grid(bands, calibration=None, years=25):
    source = {
        "operation": "duet",
        "input": "grid",
        "entity": "tree",
        "source_grid_id": "src",
        "years_since_burn": years,
        "wind_direction": 270.0,
        "wind_variability": 30.0,
        "bands": bands,
    }
    if calibration:
        source["calibration"] = calibration
    return {"id": "out", "domain_id": "d1", "source": source}


class TestUncalibratedRun:
    def test_produces_a_readable_2d_grid(self, source_grid):
        bands = [
            "fuel_load.grass",
            "fuel_load.litter",
            "fuel_load.litter.coniferous",
            "fuel_load.litter.deciduous",
            "fuel_load.total",
            "fuel_depth.litter",
            "fuel_moisture.litter",
        ]
        result = handler.duet_grid(_grid(bands), None, lambda *a, **k: None)

        assert result.georeference["shape"] == list(source_grid["shape"])
        assert result.georeference["crs"] == "EPSG:32613"

        ds = xr.open_zarr(source_grid["stores"] / "out", decode_coords="all")
        # Compared as a set: zarr does not preserve variable insertion order, and
        # band order is carried by the `index` on the grid document's bands.
        assert set(ds.data_vars) == set(bands)
        for key in bands:
            assert ds[key].shape == source_grid["shape"]
            assert np.isfinite(ds[key].values).all()
        assert ds.rio.crs == "EPSG:32613"

    def test_both_litter_classes_are_produced(self, source_grid):
        # Pine on the left, oak on the right, so both duet-tools classes should
        # carry mass. Where that mass ends up is a wind question — see
        # TestLitterLandsUnderItsOwnTrees for the spatial claim.
        handler.duet_grid(
            _grid(["fuel_load.litter.coniferous", "fuel_load.litter.deciduous"]),
            None,
            lambda *a, **k: None,
        )
        ds = xr.open_zarr(source_grid["stores"] / "out")
        assert ds["fuel_load.litter.coniferous"].values.sum() > 0
        assert ds["fuel_load.litter.deciduous"].values.sum() > 0

    def test_litter_is_the_sum_of_its_parts(self, source_grid):
        handler.duet_grid(
            _grid(
                [
                    "fuel_load.litter",
                    "fuel_load.litter.coniferous",
                    "fuel_load.litter.deciduous",
                ]
            ),
            None,
            lambda *a, **k: None,
        )
        ds = xr.open_zarr(source_grid["stores"] / "out")
        assert ds["fuel_load.litter"].values == pytest.approx(
            ds["fuel_load.litter.coniferous"].values
            + ds["fuel_load.litter.deciduous"].values,
            rel=1e-5,
        )

    def test_total_includes_grass(self, source_grid):
        handler.duet_grid(
            _grid(["fuel_load.grass", "fuel_load.litter", "fuel_load.total"]),
            None,
            lambda *a, **k: None,
        )
        ds = xr.open_zarr(source_grid["stores"] / "out")
        assert ds["fuel_load.total"].values == pytest.approx(
            ds["fuel_load.grass"].values + ds["fuel_load.litter"].values, rel=1e-5
        )

    def test_more_years_since_burn_accumulates_more_litter(self, source_grid):
        # The parameter that matters most, and the one v1 hardcoded to 1.
        handler.duet_grid(
            _grid(["fuel_load.litter"], years=1), None, lambda *a, **k: None
        )
        one_year = xr.open_zarr(source_grid["stores"] / "out")[
            "fuel_load.litter"
        ].values.sum()

        handler.duet_grid(
            _grid(["fuel_load.litter"], years=25), None, lambda *a, **k: None
        )
        many_years = xr.open_zarr(source_grid["stores"] / "out")[
            "fuel_load.litter"
        ].values.sum()

        assert many_years > one_year


class TestLitterLandsUnderItsOwnTrees:
    """Proves the (nz, ny, nx) arrays reach DUET un-transposed.

    v1 called `np.moveaxis` twice and fed DUET a transposed grid. A square test
    domain would hide that, so this uses a deliberately non-square one with a
    single off-centre clump: if the axes were swapped, the litter would land at
    the mirrored coordinates and the assertions would fail.

    One year since burn and no wind variability keep the litter near its source.
    """

    @pytest.fixture
    def clump_grid(self, tmp_path, monkeypatch):
        nz, ny, nx = 16, 60, 30  # non-square on purpose
        foliage = np.zeros((nz, ny, nx), dtype=np.float32)
        codes = np.zeros((nz, ny, nx), dtype=np.uint16)
        moisture = np.zeros((nz, ny, nx), dtype=np.float32)
        # One clump, low y and low x, well away from the far edges.
        foliage[6:14, 8:18, 4:10] = 0.6
        moisture[6:14, 8:18, 4:10] = 100.0
        codes[6:14, 8:18, 4:10] = PINE
        ds = xr.Dataset(
            {
                handler.FOLIAGE_BAND: (("z", "y", "x"), foliage),
                handler.SPCD_BAND: (("z", "y", "x"), codes),
                handler.MOISTURE_BAND: (("z", "y", "x"), moisture),
            },
            coords={
                "z": np.arange(nz, dtype=float) + 0.5,
                "y": np.arange(ny, dtype=float) * 2.0 + 1.0,
                "x": np.arange(nx, dtype=float) * 2.0 + 1.0,
            },
        ).rio.write_crs("EPSG:32613")
        stores = tmp_path / "stores"
        stores.mkdir()
        ds.to_zarr(stores / "src", mode="w", consolidated=True)
        monkeypatch.setattr(handler, "gcs_path", lambda gid: str(stores / gid))
        return stores

    def test_output_keeps_the_source_grid_shape(self, clump_grid):
        result = handler.duet_grid(
            _grid(["fuel_load.litter"], years=1), None, lambda *a, **k: None
        )
        # (ny, nx), not (nx, ny) — the shape a transpose would invert.
        assert result.georeference["shape"] == [60, 30]

    def test_litter_is_centred_on_the_canopy(self, clump_grid):
        source = _grid(["fuel_load.litter"], years=1)
        source["source"]["wind_variability"] = 0
        handler.duet_grid(source, None, lambda *a, **k: None)
        litter = xr.open_zarr(clump_grid / "out")["fuel_load.litter"].values
        assert litter.shape == (60, 30)
        assert litter.sum() > 0

        # The clump sits at y 8:18 of 60 and x 4:10 of 30 — both in the lower
        # third of their axis, so the litter's centre of mass must be too.
        ys, xs = np.nonzero(litter)
        weights = litter[ys, xs]
        y_centre = float(np.average(ys, weights=weights))
        x_centre = float(np.average(xs, weights=weights))
        assert y_centre < 30, (
            f"litter centre of mass at y={y_centre} is not near the canopy"
        )
        assert x_centre < 15, (
            f"litter centre of mass at x={x_centre} is not near the canopy"
        )


class TestCalibratedRun:
    def test_maxmin_hits_the_target_maximum(self, source_grid):
        handler.duet_grid(
            _grid(
                ["fuel_load.litter.coniferous"],
                calibration={
                    "fuel_load": {
                        "coniferous": {
                            "source": "values",
                            "method": "maxmin",
                            "max": 5.0,
                            "min": 0.0,
                        }
                    }
                },
            ),
            None,
            lambda *a, **k: None,
        )
        values = xr.open_zarr(source_grid["stores"] / "out")[
            "fuel_load.litter.coniferous"
        ].values
        assert values.max() == pytest.approx(5.0, rel=1e-4)
        assert values.min() >= 0.0

    def test_constant_assigns_one_value_to_fuel_bearing_cells(self, source_grid):
        handler.duet_grid(
            _grid(
                ["fuel_load.grass"],
                calibration={
                    "fuel_load": {
                        "grass": {
                            "source": "values",
                            "method": "constant",
                            "value": 0.75,
                        }
                    }
                },
            ),
            None,
            lambda *a, **k: None,
        )
        values = xr.open_zarr(source_grid["stores"] / "out")["fuel_load.grass"].values
        assert set(np.unique(np.round(values, 6))) <= {0.0, 0.75}
        assert np.isclose(values, 0.75).any()

    def test_calibration_leaves_empty_cells_empty(self, source_grid):
        # Both methods rescale only cells that already carry fuel. Users need to
        # expect this: a domain mean sits below a meansd target where cover is
        # sparse, and that is correct.
        handler.duet_grid(
            _grid(["fuel_load.litter.coniferous"]), None, lambda *a, **k: None
        )
        raw = xr.open_zarr(source_grid["stores"] / "out")[
            "fuel_load.litter.coniferous"
        ].values

        handler.duet_grid(
            _grid(
                ["fuel_load.litter.coniferous"],
                calibration={
                    "fuel_load": {
                        "coniferous": {
                            "source": "values",
                            "method": "maxmin",
                            "max": 5.0,
                            "min": 1.0,
                        }
                    }
                },
            ),
            None,
            lambda *a, **k: None,
        )
        calibrated = xr.open_zarr(source_grid["stores"] / "out")[
            "fuel_load.litter.coniferous"
        ].values
        assert (calibrated[raw == 0] == 0).all()

    def test_targeting_an_absent_fuel_type_is_a_processing_error(
        self, tmp_path, monkeypatch
    ):
        # Pure pine stand, calibrating deciduous litter.
        nz, ny, nx = 20, 30, 30
        foliage = np.zeros((nz, ny, nx), dtype=np.float32)
        codes = np.zeros((nz, ny, nx), dtype=np.uint16)
        moisture = np.zeros((nz, ny, nx), dtype=np.float32)
        foliage[8:18, 4:26, 4:26] = 0.5
        moisture[8:18, 4:26, 4:26] = 100.0
        codes[8:18, 4:26, 4:26] = PINE
        ds = xr.Dataset(
            {
                handler.FOLIAGE_BAND: (("z", "y", "x"), foliage),
                handler.SPCD_BAND: (("z", "y", "x"), codes),
                handler.MOISTURE_BAND: (("z", "y", "x"), moisture),
            },
            coords={
                "z": np.arange(nz, dtype=float) + 0.5,
                "y": np.arange(ny, dtype=float) * 2.0 + 1.0,
                "x": np.arange(nx, dtype=float) * 2.0 + 1.0,
            },
        ).rio.write_crs("EPSG:32613")
        stores = tmp_path / "stores"
        stores.mkdir()
        ds.to_zarr(stores / "src", mode="w", consolidated=True)
        monkeypatch.setattr(handler, "gcs_path", lambda grid_id: str(stores / grid_id))

        with pytest.raises(ProcessingError) as exc:
            handler.duet_grid(
                _grid(
                    ["fuel_load.litter.deciduous"],
                    calibration={
                        "fuel_load": {
                            "deciduous": {
                                "source": "values",
                                "method": "constant",
                                "value": 1.0,
                            }
                        }
                    },
                ),
                None,
                lambda *a, **k: None,
            )
        assert exc.value.code == "CALIBRATION_FAILED"


class TestSpeciesRemapIsInert:
    """The remap must not change what DUET and duet-tools produce.

    Two California live oaks share a (signature, class) bucket, so remapping
    them onto one representative has to be invisible in the output. If this
    fails, the bucket key is wrong.
    """

    def _run(self, tmp_path, monkeypatch, left, right):
        nz, ny, nx = 20, 40, 40
        foliage = np.zeros((nz, ny, nx), dtype=np.float32)
        codes = np.zeros((nz, ny, nx), dtype=np.uint16)
        moisture = np.zeros((nz, ny, nx), dtype=np.float32)
        foliage[8:18, 4:36, :] = 0.5
        moisture[8:18, 4:36, :] = 100.0
        codes[8:18, 4:36, :20] = left
        codes[8:18, 4:36, 20:] = right
        ds = xr.Dataset(
            {
                handler.FOLIAGE_BAND: (("z", "y", "x"), foliage),
                handler.SPCD_BAND: (("z", "y", "x"), codes),
                handler.MOISTURE_BAND: (("z", "y", "x"), moisture),
            },
            coords={
                "z": np.arange(nz, dtype=float) + 0.5,
                "y": np.arange(ny, dtype=float) * 2.0 + 1.0,
                "x": np.arange(nx, dtype=float) * 2.0 + 1.0,
            },
        ).rio.write_crs("EPSG:32613")
        stores = tmp_path / f"stores_{left}_{right}"
        stores.mkdir()
        ds.to_zarr(stores / "src", mode="w", consolidated=True)
        monkeypatch.setattr(handler, "gcs_path", lambda grid_id: str(stores / grid_id))
        handler.duet_grid(
            _grid(["fuel_load.litter.coniferous", "fuel_load.litter.deciduous"]),
            None,
            lambda *a, **k: None,
        )
        out = xr.open_zarr(stores / "out")
        return {k: out[k].values.copy() for k in out.data_vars}

    def test_same_bucket_species_produce_the_same_output(self, tmp_path, monkeypatch):
        # 801 and 805 both remap to 314.
        two_species = self._run(tmp_path, monkeypatch, 801, 805)
        one_species = self._run(tmp_path, monkeypatch, 314, 314)
        for key in two_species:
            assert two_species[key] == pytest.approx(one_species[key])

    def test_juniper_and_oak_stay_in_different_layers(self, tmp_path, monkeypatch):
        # Both are DUET `wo`. A signature-only collapse would put the oak's
        # litter in the coniferous layer and leave deciduous empty.
        out = self._run(tmp_path, monkeypatch, 66, 814)
        assert out["fuel_load.litter.coniferous"].sum() > 0
        assert out["fuel_load.litter.deciduous"].sum() > 0


class TestGuards:
    def test_species_duet_cannot_model_is_rejected_before_running(
        self, tmp_path, monkeypatch
    ):
        nz, ny, nx = 10, 20, 20
        foliage = np.zeros((nz, ny, nx), dtype=np.float32)
        codes = np.zeros((nz, ny, nx), dtype=np.uint16)
        moisture = np.zeros((nz, ny, nx), dtype=np.float32)
        foliage[4:8, 4:16, 4:16] = 0.5
        moisture[4:8, 4:16, 4:16] = 100.0
        codes[4:8, 4:16, 4:16] = 142  # Great Basin bristlecone pine
        ds = xr.Dataset(
            {
                handler.FOLIAGE_BAND: (("z", "y", "x"), foliage),
                handler.SPCD_BAND: (("z", "y", "x"), codes),
                handler.MOISTURE_BAND: (("z", "y", "x"), moisture),
            },
            coords={
                "z": np.arange(nz, dtype=float) + 0.5,
                "y": np.arange(ny, dtype=float) * 2.0 + 1.0,
                "x": np.arange(nx, dtype=float) * 2.0 + 1.0,
            },
        ).rio.write_crs("EPSG:32613")
        stores = tmp_path / "stores"
        stores.mkdir()
        ds.to_zarr(stores / "src", mode="w", consolidated=True)
        monkeypatch.setattr(handler, "gcs_path", lambda grid_id: str(stores / grid_id))

        with pytest.raises(ProcessingError) as exc:
            handler.duet_grid(_grid(["fuel_load.litter"]), None, lambda *a, **k: None)
        assert exc.value.code == "UNSUPPORTED_SPECIES"
        # Without the guard DUET returns 0 and this grid is quietly all grass.
        assert not (stores / "out").exists()
