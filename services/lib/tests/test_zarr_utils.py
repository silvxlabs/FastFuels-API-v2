"""
Tests for lib.zarr_utils save/load with chunking support.
"""

import numpy as np
import pytest
import rioxarray  # noqa: F401
import xarray as xr
import zarr
from pyproj import CRS
from rasterio.transform import from_bounds

from lib.zarr_utils import load_zarr, save_zarr


def make_spatial_dataset(
    bands: dict[str, np.ndarray] | None = None,
    crs: str = "EPSG:32610",
    shape: tuple[int, int] = (100, 120),
) -> xr.Dataset:
    """Create a Dataset with full spatial metadata (CRS, transform)."""
    ny, nx = shape
    if bands is None:
        bands = {"band_0": np.random.rand(ny, nx).astype(np.float32)}

    transform = from_bounds(
        500000, 4200000, 500000 + nx * 30, 4200000 + ny * 30, nx, ny
    )

    ds = xr.Dataset()
    for name, data in bands.items():
        da = xr.DataArray(
            data, dims=("y", "x"), coords={"y": np.arange(ny), "x": np.arange(nx)}
        )
        ds[name] = da

    ds = ds.rio.write_crs(crs)
    ds = ds.rio.write_transform(transform)
    return ds


class TestSaveZarrChunking:
    """Tests for save_zarr on-disk chunk behavior."""

    def test_on_disk_chunks_match_chunk_shape(self, tmp_path):
        """Zarr arrays have the requested chunk shape."""
        ds = make_spatial_dataset(shape=(100, 120))
        path = str(tmp_path / "chunked.zarr")

        save_zarr(path, ds, chunk_shape=(32, 32))

        store = zarr.open(path, mode="r")
        assert store["band_0"].chunks == (32, 32)
        assert store["band_0"].shape == (100, 120)

    def test_non_square_chunks(self, tmp_path):
        """Non-square chunk shape is applied correctly."""
        ds = make_spatial_dataset(shape=(100, 120))
        path = str(tmp_path / "nonsquare.zarr")

        save_zarr(path, ds, chunk_shape=(64, 32))

        store = zarr.open(path, mode="r")
        assert store["band_0"].chunks == (64, 32)

    def test_chunk_shape_larger_than_data(self, tmp_path):
        """Chunk shape larger than data is clamped to dimension size."""
        ds = make_spatial_dataset(shape=(50, 60))
        path = str(tmp_path / "large_chunks.zarr")

        save_zarr(path, ds, chunk_shape=(512, 512))

        store = zarr.open(path, mode="r")
        assert store["band_0"].chunks == (50, 60)

    def test_round_trip_preserves_values(self, tmp_path):
        """Data values survive a save/load round trip."""
        ds = make_spatial_dataset(shape=(80, 100))
        path = str(tmp_path / "roundtrip.zarr")

        save_zarr(path, ds, chunk_shape=(32, 32))
        loaded = load_zarr(path)

        xr.testing.assert_equal(ds, loaded)

    def test_rejects_dataarray(self, tmp_path):
        """save_zarr rejects DataArray input."""
        da = xr.DataArray(np.zeros((10, 10)), dims=["y", "x"])
        path = str(tmp_path / "bad.zarr")

        with pytest.raises(TypeError, match="save_zarr requires xr.Dataset"):
            save_zarr(path, da, chunk_shape=(32, 32))


class TestSaveZarr3DChunking:
    """save_zarr must accept a 3-tuple chunk_shape and enforce it on disk."""

    @staticmethod
    def _make_3d_dataset(shape: tuple[int, int, int]) -> xr.Dataset:
        nz, ny, nx = shape
        transform = from_bounds(
            500000, 4200000, 500000 + nx * 30, 4200000 + ny * 30, nx, ny
        )
        z = np.arange(nz, dtype=np.float64)
        y = np.arange(ny)
        x = np.arange(nx)
        da = xr.DataArray(
            np.random.rand(nz, ny, nx).astype(np.float32),
            dims=("z", "y", "x"),
            coords={"z": z, "y": y, "x": x},
        )
        ds = xr.Dataset({"density": da}).rio.write_crs("EPSG:32610")
        return ds.rio.write_transform(transform)

    def test_three_tuple_enforces_z_chunking(self, tmp_path):
        """A 3-tuple chunk_shape produces matching on-disk z/y/x chunks."""
        nz = 7
        ds = self._make_3d_dataset(shape=(nz, 100, 120))
        path = str(tmp_path / "z_chunked.zarr")

        save_zarr(path, ds, chunk_shape=(nz, 32, 32))

        store = zarr.open(path, mode="r")
        assert store["density"].chunks == (nz, 32, 32)

    def test_three_tuple_independent_z_chunk(self, tmp_path):
        """The z chunk dimension is honored independently of nz."""
        ds = self._make_3d_dataset(shape=(10, 100, 120))
        path = str(tmp_path / "z_partial.zarr")

        # Ask for z-chunks of 5 (so each var has 2 chunks along z).
        save_zarr(path, ds, chunk_shape=(5, 32, 32))

        store = zarr.open(path, mode="r")
        assert store["density"].chunks == (5, 32, 32)


class TestChunkingWithSpatialMetadata:
    """Tests that chunking preserves CRS, transform, and spatial_ref."""

    def test_crs_preserved(self, tmp_path):
        """CRS survives save/load with chunk_shape."""
        ds = make_spatial_dataset(crs="EPSG:32610")
        path = str(tmp_path / "crs.zarr")

        save_zarr(path, ds, chunk_shape=(32, 32))
        loaded = load_zarr(path)

        assert loaded.rio.crs is not None
        assert loaded.rio.crs.to_epsg() == 32610

    def test_transform_preserved(self, tmp_path):
        """Affine transform survives save/load with chunk_shape."""
        ds = make_spatial_dataset()
        path = str(tmp_path / "transform.zarr")

        save_zarr(path, ds, chunk_shape=(32, 32))
        loaded = load_zarr(path)

        assert ds.rio.transform() == loaded.rio.transform()

    def test_spatial_ref_is_coordinate(self, tmp_path):
        """spatial_ref remains a coordinate (not data var) after chunked round-trip."""
        ds = make_spatial_dataset()
        path = str(tmp_path / "spatialref.zarr")

        save_zarr(path, ds, chunk_shape=(32, 32))
        loaded = load_zarr(path)

        assert "spatial_ref" in loaded.coords
        assert "spatial_ref" not in loaded.data_vars


class TestMultiVariableChunking:
    """Tests that chunk_shape applies to all variables."""

    def test_all_variables_chunked(self, tmp_path):
        """Every data variable gets the specified chunk shape."""
        shape = (100, 120)
        bands = {
            "elevation": np.random.rand(*shape).astype(np.float32),
            "slope": np.random.rand(*shape).astype(np.float32),
            "aspect": np.random.rand(*shape).astype(np.float32),
        }
        ds = make_spatial_dataset(bands=bands, shape=shape)
        path = str(tmp_path / "multi.zarr")

        save_zarr(path, ds, chunk_shape=(32, 32))

        store = zarr.open(path, mode="r")
        for var in ["elevation", "slope", "aspect"]:
            assert store[var].chunks == (32, 32), f"{var} has wrong chunks"

    def test_multi_variable_round_trip_values(self, tmp_path):
        """All variable values survive chunked round-trip."""
        shape = (80, 100)
        bands = {
            "fbfm": np.full(shape, 101, dtype=np.int32),
            "fuel_load.1hr": np.random.rand(*shape).astype(np.float64),
            "fuel_depth": np.random.rand(*shape).astype(np.float64),
        }
        ds = make_spatial_dataset(bands=bands, shape=shape)
        path = str(tmp_path / "multi_rt.zarr")

        save_zarr(path, ds, chunk_shape=(32, 32))
        loaded = load_zarr(path)

        for var in bands:
            np.testing.assert_array_equal(
                ds[var].values, loaded[var].values, err_msg=f"{var} values differ"
            )


class TestFaithfulRoundTrip:
    """load_zarr must return grids exactly as stored — dtype and nodata
    sentinel preserved, not CF-masked to float/NaN. See issue #290."""

    @staticmethod
    def _make_band(data: np.ndarray, nodata, crs: str = "EPSG:5070") -> xr.Dataset:
        ny, nx = data.shape
        transform = from_bounds(0, 0, nx * 30, ny * 30, nx, ny)
        da = xr.DataArray(
            data, dims=("y", "x"), coords={"y": np.arange(ny), "x": np.arange(nx)}
        )
        da = da.rio.write_crs(crs).rio.write_transform(transform)
        da = da.rio.write_nodata(nodata)
        return xr.Dataset({"band": da})

    def test_integer_grid_preserves_dtype_and_sentinel(self, tmp_path):
        """An int grid with a nodata sentinel loads back as int with the
        sentinel intact — not float32/NaN."""
        data = np.array([[101, 102], [103, 32767]], dtype=np.int16)
        ds = self._make_band(data, nodata=32767)
        path = str(tmp_path / "int.zarr")

        save_zarr(path, ds, chunk_shape=(2, 2))
        loaded = load_zarr(path)

        assert loaded["band"].dtype == np.int16
        assert loaded["band"].rio.nodata == 32767
        np.testing.assert_array_equal(loaded["band"].values, data)

    def test_large_integer_ids_survive(self, tmp_path):
        """FIA-CN-scale integer IDs (tm_id/plt_cn) round-trip without the
        precision loss a float promotion would cause."""
        data = np.array([[12345678901234, 0], [98765432109876, 0]], dtype=np.int64)
        ds = self._make_band(data, nodata=0)
        path = str(tmp_path / "ids.zarr")

        save_zarr(path, ds, chunk_shape=(2, 2))
        loaded = load_zarr(path)

        assert loaded["band"].dtype == np.int64
        np.testing.assert_array_equal(loaded["band"].values, data)

    def test_float_grid_preserves_nan_nodata(self, tmp_path):
        """A float grid keeps float dtype and NaN nodata."""
        data = np.array([[1.0, 2.0], [3.0, np.nan]], dtype=np.float32)
        ds = self._make_band(data, nodata=np.float32("nan"))
        path = str(tmp_path / "float.zarr")

        save_zarr(path, ds, chunk_shape=(2, 2))
        loaded = load_zarr(path)

        assert loaded["band"].dtype == np.float32
        assert np.isnan(loaded["band"].rio.nodata)

    def test_crs_preserved_under_faithful_load(self, tmp_path):
        """CRS still survives the faithful (mask_and_scale=False) load."""
        data = np.array([[101, 32767], [102, 103]], dtype=np.int16)
        ds = self._make_band(data, nodata=32767, crs="EPSG:32611")
        path = str(tmp_path / "crs.zarr")

        save_zarr(path, ds, chunk_shape=(2, 2))
        loaded = load_zarr(path)

        assert loaded.rio.crs is not None
        assert loaded.rio.crs.to_epsg() == 32611


class TestChunkedToRaster:
    """Tests that chunked Zarr data can be written to GeoTIFF."""

    def test_to_raster_succeeds(self, tmp_path):
        """Dataset.rio.to_raster() works after chunked save/load."""
        ds = make_spatial_dataset()
        zarr_path = str(tmp_path / "chunked.zarr")
        tif_path = str(tmp_path / "out.tif")

        save_zarr(zarr_path, ds, chunk_shape=(32, 32))
        loaded = load_zarr(zarr_path)

        loaded.rio.to_raster(tif_path)
        assert (tmp_path / "out.tif").exists()

    def test_windowed_write_succeeds(self, tmp_path):
        """windowed=True works on chunked data — the exact exporter operation."""
        ds = make_spatial_dataset()
        zarr_path = str(tmp_path / "chunked.zarr")
        tif_path = str(tmp_path / "windowed.tif")

        save_zarr(zarr_path, ds, chunk_shape=(32, 32))
        loaded = load_zarr(zarr_path)

        loaded.rio.to_raster(tif_path, driver="GTiff", windowed=True)
        assert (tmp_path / "windowed.tif").exists()

    def test_multiband_windowed_write(self, tmp_path):
        """Multi-variable chunked Dataset writes valid multi-band GeoTIFF."""
        shape = (80, 100)
        bands = {
            "fuel_load.1hr": np.random.rand(*shape).astype(np.float64),
            "fuel_load.10hr": np.random.rand(*shape).astype(np.float64),
        }
        ds = make_spatial_dataset(bands=bands, shape=shape)
        zarr_path = str(tmp_path / "multiband.zarr")
        tif_path = str(tmp_path / "multiband.tif")

        save_zarr(zarr_path, ds, chunk_shape=(32, 32))
        loaded = load_zarr(zarr_path)

        loaded.rio.to_raster(tif_path, driver="GTiff", windowed=True)

        result = xr.open_dataset(tif_path, engine="rasterio")
        assert result.sizes["band"] == 2
        result.close()

    def test_geotiff_values_match_original(self, tmp_path):
        """GeoTIFF pixel values match the original Dataset."""
        shape = (40, 50)
        original_data = np.random.rand(*shape).astype(np.float32)
        ds = make_spatial_dataset(bands={"band_0": original_data}, shape=shape)
        zarr_path = str(tmp_path / "chunked.zarr")
        tif_path = str(tmp_path / "out.tif")

        save_zarr(zarr_path, ds, chunk_shape=(16, 16))
        loaded = load_zarr(zarr_path)
        loaded.rio.to_raster(tif_path, driver="GTiff", windowed=True)

        result = xr.open_dataset(tif_path, engine="rasterio")
        np.testing.assert_array_equal(original_data, result["band_data"].values[0])
        result.close()

    def test_geotiff_crs_preserved(self, tmp_path):
        """GeoTIFF CRS matches the original Dataset CRS."""
        ds = make_spatial_dataset(crs="EPSG:32611", shape=(40, 50))
        zarr_path = str(tmp_path / "chunked.zarr")
        tif_path = str(tmp_path / "out.tif")

        save_zarr(zarr_path, ds, chunk_shape=(16, 16))
        loaded = load_zarr(zarr_path)
        loaded.rio.to_raster(tif_path, driver="GTiff", windowed=True)

        result = xr.open_dataset(tif_path, engine="rasterio")
        assert CRS(result.rio.crs) == CRS("EPSG:32611")
        result.close()
