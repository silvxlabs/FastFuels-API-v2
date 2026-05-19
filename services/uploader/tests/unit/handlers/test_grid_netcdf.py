"""
Unit tests for uploader/handlers/grid.py — netCDF handler.

Tests _build_netcdf_dataset in isolation using local netCDF files written to
tmp_path. No GCP I/O — the function accepts any file-like or path that
xr.open_dataset can open.
"""

import numpy as np
import pytest
import rioxarray  # noqa: F401  registers .rio accessor
import xarray as xr
from uploader.handlers.grid import _build_netcdf_dataset

from lib.errors import ProcessingError

# Blue Mountain domain bounds (EPSG:32611) — same as the geotiff unit tests.
DOMAIN_CRS = "EPSG:32611"
DOMAIN_BOUNDS = (720228, 5189763, 721534, 5190645)  # xmin, ymin, xmax, ymax


class _FakeDomainGdf:
    def __init__(self, xmin, ymin, xmax, ymax):
        self.total_bounds = (xmin, ymin, xmax, ymax)


DOMAIN_GDF = _FakeDomainGdf(*DOMAIN_BOUNDS)

# Dataset bounds — small window inside the domain.
_DEFAULT_BOUNDS = (720400.0, 5190000.0, 721200.0, 5190400.0)


def _build_2d_dataset(
    bands: dict[str, np.ndarray] | None = None,
    crs: str | None = DOMAIN_CRS,
    bounds: tuple[float, float, float, float] = _DEFAULT_BOUNDS,
    shape: tuple[int, int] = (20, 40),
    units: dict[str, str] | None = None,
) -> xr.Dataset:
    """Build a small 2D dataset with named (y, x) data variables."""
    ny, nx = shape
    xmin, ymin, xmax, ymax = bounds
    dx = (xmax - xmin) / nx
    dy = (ymax - ymin) / ny
    x = xmin + dx / 2 + np.arange(nx) * dx
    y = ymax - dy / 2 - np.arange(ny) * dy

    if bands is None:
        bands = {"fbfm": np.full((ny, nx), 101, dtype=np.int32)}

    ds = xr.Dataset(
        {
            name: xr.DataArray(data, dims=("y", "x"), coords={"y": y, "x": x})
            for name, data in bands.items()
        }
    )
    if units:
        for name, unit in units.items():
            ds[name].attrs["units"] = unit
    if crs is not None:
        ds = ds.rio.write_crs(crs)
    return ds


def _build_3d_dataset(
    band_name: str = "bulk_density.foliage",
    crs: str | None = DOMAIN_CRS,
    bounds: tuple[float, float, float, float] = _DEFAULT_BOUNDS,
    shape: tuple[int, int, int] = (5, 20, 40),
    z_coords: np.ndarray | None = None,
    z_positive: str | None = "up",
    units: str | None = "kg/m**3",
) -> xr.Dataset:
    """Build a small 3D dataset with one (z, y, x) data variable."""
    nz, ny, nx = shape
    xmin, ymin, xmax, ymax = bounds
    dx = (xmax - xmin) / nx
    dy = (ymax - ymin) / ny
    x = xmin + dx / 2 + np.arange(nx) * dx
    y = ymax - dy / 2 - np.arange(ny) * dy
    if z_coords is None:
        z_coords = np.arange(nz, dtype=np.float64)

    da = xr.DataArray(
        np.random.rand(nz, ny, nx).astype(np.float32),
        dims=("z", "y", "x"),
        coords={"z": z_coords, "y": y, "x": x},
    )
    ds = xr.Dataset({band_name: da})
    if z_positive is not None:
        ds["z"].attrs["positive"] = z_positive
    if units is not None:
        ds[band_name].attrs["units"] = units
    if crs is not None:
        ds = ds.rio.write_crs(crs)
    return ds


def _write_nc(ds: xr.Dataset, path) -> str:
    """Serialize a Dataset to a .nc file via h5netcdf, return path as str."""
    ds.to_netcdf(path, engine="h5netcdf")
    return str(path)


class TestBuildNetcdfDataset2D:
    def test_happy_path(self, tmp_path):
        ds = _build_2d_dataset()
        path = _write_nc(ds, tmp_path / "ok.nc")

        out = _build_netcdf_dataset(path, DOMAIN_CRS, DOMAIN_GDF, 0)

        assert "fbfm" in out.data_vars
        assert out["fbfm"].dims == ("y", "x")
        assert out.rio.crs is not None
        assert out["fbfm"].encoding.get("grid_mapping") == "spatial_ref"

    def test_unit_passthrough(self, tmp_path):
        ds = _build_2d_dataset(
            bands={"bd": np.zeros((20, 40), dtype=np.float32)},
            units={"bd": "kg/m**3"},
        )
        path = _write_nc(ds, tmp_path / "unit.nc")

        out = _build_netcdf_dataset(path, DOMAIN_CRS, DOMAIN_GDF, 0)
        assert out["bd"].attrs["units"] == "kg/m**3"


class TestBuildNetcdfDataset3D:
    def test_happy_path(self, tmp_path):
        ds = _build_3d_dataset()
        path = _write_nc(ds, tmp_path / "ok3d.nc")

        out = _build_netcdf_dataset(path, DOMAIN_CRS, DOMAIN_GDF, 0)

        assert "z" in out.sizes
        assert out["bulk_density.foliage"].dims == ("z", "y", "x")
        assert out["bulk_density.foliage"].encoding.get("grid_mapping") == "spatial_ref"


class TestWrongDims:
    def test_xy_order_swapped(self, tmp_path):
        # WRONG_DIMS is checked before CRS, so we skip write_crs entirely —
        # avoids xarray's auto grid_mapping stamping which conflicts with our
        # explicit attrs-based stamping.
        da = xr.DataArray(
            np.zeros((40, 20), dtype=np.float32),
            dims=("x", "y"),
            coords={"x": np.arange(40), "y": np.arange(20)},
        )
        ds = xr.Dataset({"swapped": da})
        path = _write_nc(ds, tmp_path / "swapped.nc")

        with pytest.raises(ProcessingError) as exc:
            _build_netcdf_dataset(path, DOMAIN_CRS, DOMAIN_GDF, 0)
        assert exc.value.code == "WRONG_DIMS"

    def test_mixed_rank_rejected(self, tmp_path):
        ny, nx, nz = 20, 40, 5
        ds = xr.Dataset(
            {
                "var_2d": xr.DataArray(
                    np.zeros((ny, nx), dtype=np.float32), dims=("y", "x")
                ),
                "var_3d": xr.DataArray(
                    np.zeros((nz, ny, nx), dtype=np.float32),
                    dims=("z", "y", "x"),
                ),
            }
        )
        path = _write_nc(ds, tmp_path / "mixed.nc")

        with pytest.raises(ProcessingError) as exc:
            _build_netcdf_dataset(path, DOMAIN_CRS, DOMAIN_GDF, 0)
        assert exc.value.code == "WRONG_DIMS"

    def test_empty_dataset(self, tmp_path):
        ds = xr.Dataset()
        path = _write_nc(ds, tmp_path / "empty.nc")

        with pytest.raises(ProcessingError) as exc:
            _build_netcdf_dataset(path, DOMAIN_CRS, DOMAIN_GDF, 0)
        assert exc.value.code == "WRONG_DIMS"


class TestCrs:
    def test_missing_crs(self, tmp_path):
        ds = _build_2d_dataset(crs=None)
        path = _write_nc(ds, tmp_path / "nocrs.nc")

        with pytest.raises(ProcessingError) as exc:
            _build_netcdf_dataset(path, DOMAIN_CRS, DOMAIN_GDF, 0)
        assert exc.value.code == "MISSING_CRS"

    def test_crs_mismatch(self, tmp_path):
        ds = _build_2d_dataset(
            crs="EPSG:4326", bounds=(-114.11, 46.825, -114.07, 46.845)
        )
        path = _write_nc(ds, tmp_path / "wrongcrs.nc")

        with pytest.raises(ProcessingError) as exc:
            _build_netcdf_dataset(path, DOMAIN_CRS, DOMAIN_GDF, 0)
        assert exc.value.code == "CRS_MISMATCH"


class TestUnits:
    def test_invalid_units(self, tmp_path):
        ds = _build_2d_dataset(
            bands={"bd": np.zeros((20, 40), dtype=np.float32)},
            units={"bd": "kg/m3"},  # non-canonical: bare numeric exponent
        )
        path = _write_nc(ds, tmp_path / "badunits.nc")

        with pytest.raises(ProcessingError) as exc:
            _build_netcdf_dataset(path, DOMAIN_CRS, DOMAIN_GDF, 0)
        assert exc.value.code == "INVALID_UNITS"

    def test_no_units_allowed(self, tmp_path):
        ds = _build_2d_dataset()  # no units set
        path = _write_nc(ds, tmp_path / "nounits.nc")

        out = _build_netcdf_dataset(path, DOMAIN_CRS, DOMAIN_GDF, 0)
        assert "units" not in out["fbfm"].attrs


class TestZ:
    def test_positive_down_rejected(self, tmp_path):
        ds = _build_3d_dataset(z_positive="down")
        path = _write_nc(ds, tmp_path / "zdown.nc")

        with pytest.raises(ProcessingError) as exc:
            _build_netcdf_dataset(path, DOMAIN_CRS, DOMAIN_GDF, 0)
        assert exc.value.code == "MISSING_Z_POSITIVE"

    def test_positive_unset_rejected(self, tmp_path):
        ds = _build_3d_dataset(z_positive=None)
        path = _write_nc(ds, tmp_path / "zunset.nc")

        with pytest.raises(ProcessingError) as exc:
            _build_netcdf_dataset(path, DOMAIN_CRS, DOMAIN_GDF, 0)
        assert exc.value.code == "MISSING_Z_POSITIVE"

    def test_nonuniform_z_rejected(self, tmp_path):
        ds = _build_3d_dataset(
            shape=(4, 20, 40),
            z_coords=np.array([0.0, 1.0, 3.0, 7.0]),  # non-uniform spacing
        )
        path = _write_nc(ds, tmp_path / "nonuniformz.nc")

        with pytest.raises(ProcessingError) as exc:
            _build_netcdf_dataset(path, DOMAIN_CRS, DOMAIN_GDF, 0)
        assert exc.value.code == "NONUNIFORM_Z"

    def test_single_z_layer_rejected(self, tmp_path):
        """nz=1 must be rejected — z_resolution cannot be derived from one level.

        Without this check, _build_netcdf_dataset returns successfully and
        handle_grid_netcdf later crashes with IndexError on z_vals[1].
        """
        ds = _build_3d_dataset(shape=(1, 20, 40), z_coords=np.array([0.0]))
        path = _write_nc(ds, tmp_path / "nz1.nc")

        with pytest.raises(ProcessingError) as exc:
            _build_netcdf_dataset(path, DOMAIN_CRS, DOMAIN_GDF, 0)
        assert exc.value.code == "SINGLE_Z_LAYER"


class TestPixelGeometry:
    def test_non_square_pixels_rejected(self, tmp_path):
        """dx != dy must be rejected — the contract assumes square pixels."""
        # bounds=800x400, shape=(40, 40) → dx=20, dy=10 → non-square
        ds = _build_2d_dataset(bounds=_DEFAULT_BOUNDS, shape=(40, 40))
        path = _write_nc(ds, tmp_path / "nonsquare.nc")

        with pytest.raises(ProcessingError) as exc:
            _build_netcdf_dataset(path, DOMAIN_CRS, DOMAIN_GDF, 0)
        assert exc.value.code == "NON_SQUARE_PIXELS"


class TestOverlap:
    def test_no_overlap(self, tmp_path):
        # Place the dataset far north of the domain.
        ds = _build_2d_dataset(bounds=(720400.0, 5300000.0, 721200.0, 5300400.0))
        path = _write_nc(ds, tmp_path / "north.nc")

        with pytest.raises(ProcessingError) as exc:
            _build_netcdf_dataset(path, DOMAIN_CRS, DOMAIN_GDF, 0)
        assert exc.value.code == "NO_OVERLAP"


class TestBuffer:
    def test_buffer_expands_clip(self, tmp_path):
        """num_buffer_cells > 0 retains more pixels around the domain."""
        # 2000m x 1500m bounds with shape=(150, 200) → 10m square pixels.
        extended_bounds = (
            DOMAIN_BOUNDS[0] - 200,
            DOMAIN_BOUNDS[1] - 200,
            DOMAIN_BOUNDS[0] - 200 + 2000,
            DOMAIN_BOUNDS[1] - 200 + 1500,
        )
        ds = _build_2d_dataset(bounds=extended_bounds, shape=(150, 200))
        path = _write_nc(ds, tmp_path / "padded.nc")

        out0 = _build_netcdf_dataset(path, DOMAIN_CRS, DOMAIN_GDF, 0)
        out3 = _build_netcdf_dataset(path, DOMAIN_CRS, DOMAIN_GDF, 3)

        assert out3.rio.width > out0.rio.width
        assert out3.rio.height > out0.rio.height
