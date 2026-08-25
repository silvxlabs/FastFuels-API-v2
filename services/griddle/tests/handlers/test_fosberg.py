"""Unit tests for the Fosberg 1-hr dead fuel moisture handler."""

from unittest.mock import patch

import numpy as np
import pytest
import rioxarray  # noqa: F401
import xarray as xr
from fastfuels_core.fuel_moisture.fosberg import calculate_1hr_fuel_moisture
from griddle.handlers.fosberg import OUTPUT_KEY, fosberg_grid

from lib.errors import ProcessingError

CRS = "EPSG:32611"
# 10 m cells anchored at an arbitrary UTM origin.
TRANSFORM = (10.0, 0.0, 500000.0, 0.0, -10.0, 5000000.0)


def _coords(height, width):
    y = 5000000.0 - (np.arange(height) + 0.5) * 10.0
    x = 500000.0 + (np.arange(width) + 0.5) * 10.0
    return y, x


def _affine():
    from affine import Affine

    return Affine(*TRANSFORM)


def _topo_ds(slope, aspect, nodata=None):
    height, width = slope.shape
    y, x = _coords(height, width)
    data_vars = {}
    for key, arr in (("slope", slope), ("aspect", aspect)):
        da = xr.DataArray(arr, dims=("y", "x"), coords={"y": y, "x": x})
        if nodata is not None:
            da = da.rio.write_nodata(nodata)
        data_vars[key] = da
    ds = xr.Dataset(data_vars).rio.write_crs(CRS)
    return ds.rio.write_transform(_affine())


def _surface_ds(surface, threed=False):
    height, width = surface.shape
    y, x = _coords(height, width)
    if threed:
        depth = 4
        grid = np.full((depth, height, width), np.nan, dtype=np.float32)
        grid[0] = surface
        z = np.arange(depth, dtype=float)
        da = xr.DataArray(grid, dims=("z", "y", "x"), coords={"z": z, "y": y, "x": x})
    else:
        da = xr.DataArray(surface, dims=("y", "x"), coords={"y": y, "x": x})
    ds = xr.Dataset({"irradiance.surface.relative": da}).rio.write_crs(CRS)
    return ds.rio.write_transform(_affine())


def _run(topo_ds, irr_ds, *, source_overrides=None):
    source = {
        "name": "fosberg",
        "source_topography_grid_id": "topo",
        "source_irradiance_grid_id": "irr",
        "dry_bulb_temp": 75,
        "relative_humidity": 30,
        "time": 1200,
        "month": "June",
        "elevation": "near",
    }
    source.update(source_overrides or {})

    dss = {"topo": topo_ds, "irr": irr_ds}
    with patch("griddle.handlers.fosberg.load_zarr", side_effect=lambda gid: dss[gid]):
        return fosberg_grid({"id": "grid_out"}, source, lambda *a, **k: None)


def test_produces_single_dead_1hr_band_matching_core():
    slope = np.array([[10.0, 40.0], [0.0, 20.0]], dtype=np.float32)
    aspect = np.array([[0.0, 90.0], [180.0, 270.0]], dtype=np.float32)
    surface = np.array([[1.0, 0.2], [0.8, 0.5]], dtype=np.float32)

    result = _run(_topo_ds(slope, aspect), _surface_ds(surface))

    assert list(result.data_vars) == [OUTPUT_KEY]
    # The handler feeds the core percent slope (the tables' "31%" class split);
    # our slope band is degrees, so the reference converts it explicitly.
    expected = calculate_1hr_fuel_moisture(
        dry_bulb_temp=75,
        relative_humidity=30,
        aspect=aspect.astype(float),
        slope=np.tan(np.radians(slope.astype(float))) * 100,
        time=1200,
        month="June",
        elevation=1,
        shading=1.0 - surface.astype(float),
    )
    np.testing.assert_allclose(result[OUTPUT_KEY].values, expected)
    assert str(result.rio.crs) == CRS


def test_slope_classified_by_percent_grade_not_degrees():
    # A 25 deg slope is a 47% grade -> the tables' steep ("31%") class. Read as
    # raw degrees against the core's `slope > 30` it would fall in the gentle
    # class, so this guards the degrees->percent breakpoint conversion.
    slope = np.full((2, 2), 25.0, dtype=np.float32)
    aspect = np.full((2, 2), 0.0, dtype=np.float32)  # north
    surface = np.full((2, 2), 1.0, dtype=np.float32)  # unshaded

    result = _run(_topo_ds(slope, aspect), _surface_ds(surface))

    steep = calculate_1hr_fuel_moisture(
        dry_bulb_temp=75,
        relative_humidity=30,
        aspect=0.0,
        slope=40.0,
        time=1200,
        month="June",
        elevation=1,
        shading=0.0,
    )
    gentle = calculate_1hr_fuel_moisture(
        dry_bulb_temp=75,
        relative_humidity=30,
        aspect=0.0,
        slope=10.0,
        time=1200,
        month="June",
        elevation=1,
        shading=0.0,
    )
    assert steep != gentle  # the two classes differ for this cell
    np.testing.assert_allclose(result[OUTPUT_KEY].values, steep)


def test_output_carries_topography_georeference():
    slope = np.zeros((3, 3), dtype=np.float32)
    aspect = np.zeros((3, 3), dtype=np.float32)
    surface = np.full((3, 3), 0.5, dtype=np.float32)

    result = _run(_topo_ds(slope, aspect), _surface_ds(surface))

    assert result[OUTPUT_KEY].dims == ("y", "x")
    assert (result.rio.height, result.rio.width) == (3, 3)
    assert tuple(result.rio.transform())[:6] == TRANSFORM


def test_nodata_cells_are_masked_to_nan():
    slope = np.array([[10.0, np.nan], [0.0, 20.0]], dtype=np.float32)
    aspect = np.array([[0.0, 90.0], [180.0, 270.0]], dtype=np.float32)
    surface = np.array([[1.0, 0.2], [0.8, 0.5]], dtype=np.float32)

    result = _run(_topo_ds(slope, aspect), _surface_ds(surface))
    out = result[OUTPUT_KEY].values

    assert np.isnan(out[0, 1])  # slope nodata -> masked
    assert np.isfinite(out[0, 0])
    assert np.isfinite(out[1, 1])


def test_float32_nan_nodata_is_masked():
    # rio.nodata comes back as a numpy float32 NaN here (not a Python float). A
    # NaN aspect cell must still be recognized as nodata; otherwise it reaches
    # the core's table lookup and raises KeyError. Guards the regression where
    # `isinstance(nodata, float)` is False for a numpy float and misfires.
    slope = np.array([[10.0, 40.0], [0.0, 20.0]], dtype=np.float32)
    aspect = np.array([[0.0, np.nan], [180.0, 270.0]], dtype=np.float32)
    surface = np.full((2, 2), 0.5, dtype=np.float32)

    result = _run(
        _topo_ds(slope, aspect, nodata=np.float32("nan")), _surface_ds(surface)
    )
    out = result[OUTPUT_KEY].values
    assert np.isnan(out[0, 1])  # NaN nodata cell masked, no crash
    assert np.isfinite(out[0, 0])


def test_surface_nodata_is_masked():
    slope = np.zeros((2, 2), dtype=np.float32)
    aspect = np.zeros((2, 2), dtype=np.float32)
    surface = np.array([[0.5, np.nan], [0.5, 0.5]], dtype=np.float32)

    result = _run(_topo_ds(slope, aspect), _surface_ds(surface))
    out = result[OUTPUT_KEY].values
    assert np.isnan(out[0, 1])
    assert np.isfinite(out[0, 0])


def test_three_dimensional_surface_is_sliced_at_z0():
    slope = np.array([[10.0, 40.0], [0.0, 20.0]], dtype=np.float32)
    aspect = np.array([[0.0, 90.0], [180.0, 270.0]], dtype=np.float32)
    surface = np.array([[1.0, 0.2], [0.8, 0.5]], dtype=np.float32)

    flat = _run(_topo_ds(slope, aspect), _surface_ds(surface))
    threed = _run(_topo_ds(slope, aspect), _surface_ds(surface, threed=True))

    assert threed[OUTPUT_KEY].dims == ("y", "x")
    np.testing.assert_allclose(
        threed[OUTPUT_KEY].values, flat[OUTPUT_KEY].values, equal_nan=True
    )


def test_irradiance_resampled_onto_topography_lattice():
    # Topography is 4x4; irradiance is a coarser 2x2 on the same extent.
    slope = np.full((4, 4), 10.0, dtype=np.float32)
    aspect = np.full((4, 4), 180.0, dtype=np.float32)
    topo = _topo_ds(slope, aspect)

    from affine import Affine

    coarse = np.full((2, 2), 0.5, dtype=np.float32)
    cy = 5000000.0 - (np.arange(2) + 0.5) * 20.0
    cx = 500000.0 + (np.arange(2) + 0.5) * 20.0
    da = xr.DataArray(coarse, dims=("y", "x"), coords={"y": cy, "x": cx})
    irr = xr.Dataset({"irradiance.surface.relative": da}).rio.write_crs(CRS)
    irr = irr.rio.write_transform(Affine(20.0, 0.0, 500000.0, 0.0, -20.0, 5000000.0))

    result = _run(topo, irr)
    assert (result.rio.height, result.rio.width) == (4, 4)
    assert np.isfinite(result[OUTPUT_KEY].values).all()


def test_elevation_category_selects_correction():
    # `above` (index 2) must differ from `near` (index 1) for a shaded slope,
    # confirming the label -> category-index mapping reaches the core.
    slope = np.full((2, 2), 40.0, dtype=np.float32)
    aspect = np.full((2, 2), 0.0, dtype=np.float32)
    surface = np.full((2, 2), 0.0, dtype=np.float32)  # fully shaded

    near = _run(_topo_ds(slope, aspect), _surface_ds(surface))
    above = _run(
        _topo_ds(slope, aspect),
        _surface_ds(surface),
        source_overrides={"elevation": "above"},
    )
    assert not np.allclose(near[OUTPUT_KEY].values, above[OUTPUT_KEY].values)


def test_unreadable_source_raises_terminal_error():
    def boom(_gid):
        raise FileNotFoundError("no such store")

    with patch("griddle.handlers.fosberg.load_zarr", side_effect=boom):
        with pytest.raises(ProcessingError) as exc:
            fosberg_grid(
                {"id": "grid_out"},
                {
                    "name": "fosberg",
                    "source_topography_grid_id": "topo",
                    "source_irradiance_grid_id": "irr",
                    "dry_bulb_temp": 75,
                    "relative_humidity": 30,
                    "time": 1200,
                    "month": "June",
                    "elevation": "near",
                },
                lambda *a, **k: None,
            )
    assert exc.value.code == "FOSBERG_SOURCE_UNAVAILABLE"
