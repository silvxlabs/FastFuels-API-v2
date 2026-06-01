"""
Tests for exporter grid handlers.

Unit tests mock storage to test handler logic.
Integration tests write to the real GCS bucket.
"""

import os
import uuid
from unittest.mock import patch

import gcsfs
import numpy as np
import pytest
import rasterio
import rioxarray  # noqa: F401
import xarray as xr
from exporter.errors import ProcessingError
from exporter.handlers.grid import _nodata_is_uniform, export_geotiff
from pyproj import CRS
from rasterio.transform import from_bounds
from rioxarray.raster_array import RasterArray
from rioxarray.raster_dataset import RasterDataset

from lib.zarr_utils import load_zarr, save_zarr

MOCK_TO_RASTER = patch.object(RasterDataset, "to_raster")


def make_test_dataset(
    bands: dict[str, np.ndarray] | None = None,
    crs: str = "EPSG:32611",
    shape: tuple[int, int] = (10, 10),
    nodatas: dict[str, float | int | None] | None = None,
) -> xr.Dataset:
    """Create a synthetic Dataset for testing.

    Pass ``nodatas`` to tag a per-band nodata value (applied last, after
    CRS/transform, via ``rio.write_nodata``) so tests can exercise the
    faithful-load behavior where grids carry a real ``_FillValue``.
    """
    ny, nx = shape

    if bands is None:
        bands = {
            "fbfm": np.full(shape, 101, dtype=np.int32),
            "fuel_load.1hr": np.random.rand(*shape).astype(np.float64),
        }

    transform = from_bounds(
        500000, 5200000, 500000 + nx * 30, 5200000 + ny * 30, nx, ny
    )

    ds = xr.Dataset()
    for name, data in bands.items():
        da = xr.DataArray(
            data,
            dims=("y", "x"),
            coords={"y": np.arange(ny), "x": np.arange(nx)},
        )
        ds[name] = da

    ds = ds.rio.write_crs(crs)
    ds = ds.rio.write_transform(transform)
    if nodatas:
        for name, nd in nodatas.items():
            ds[name] = ds[name].rio.write_nodata(nd)
    return ds


def noop_progress(message: str, percent: int | None = None):
    pass


class TestExportGeotiffUnit:
    """Unit tests for export_geotiff handler logic.

    These mock load_grid_zarr and to_raster to test band validation,
    error handling, and progress reporting without external dependencies.
    """

    @MOCK_TO_RASTER
    @patch("exporter.handlers.grid.load_grid_zarr")
    def test_loads_correct_grid(self, mock_load, _mock_raster):
        """Calls load_grid_zarr with the grid_id from source."""
        mock_load.return_value = make_test_dataset()

        export_geotiff(
            {"id": "test-export"},
            {"grid_id": "grid-abc", "name": "geotiff"},
            noop_progress,
        )

        mock_load.assert_called_once_with("grid-abc")

    @MOCK_TO_RASTER
    @patch("exporter.handlers.grid.load_grid_zarr")
    def test_returns_gcs_path(self, mock_load, _mock_raster):
        """Returns the expected GCS path."""
        mock_load.return_value = make_test_dataset()

        result = export_geotiff(
            {"id": "export-789"},
            {"grid_id": "grid-abc", "name": "geotiff"},
            noop_progress,
        )

        assert result.endswith("/export-789/export.tif")

    @MOCK_TO_RASTER
    @patch("exporter.handlers.grid.load_grid_zarr")
    def test_returns_gcs_path_with_name(self, mock_load, _mock_raster):
        """Uses sanitized export name in the GCS path."""
        mock_load.return_value = make_test_dataset()

        result = export_geotiff(
            {"id": "export-789", "name": "My Export!"},
            {"grid_id": "grid-abc", "name": "geotiff"},
            noop_progress,
        )

        assert result.endswith("/export-789/My_Export.tif")

    @patch("exporter.handlers.grid.load_grid_zarr")
    def test_missing_band_raises_processing_error(self, mock_load):
        """Missing band raises ProcessingError with BAND_NOT_FOUND code."""
        mock_load.return_value = make_test_dataset()

        with pytest.raises(ProcessingError) as exc_info:
            export_geotiff(
                {"id": "test-export"},
                {
                    "grid_id": "grid-abc",
                    "name": "geotiff",
                    "bands": ["fbfm", "nonexistent"],
                },
                noop_progress,
            )

        assert exc_info.value.code == "BAND_NOT_FOUND"
        assert "nonexistent" in exc_info.value.message

    @patch("exporter.handlers.grid.load_grid_zarr")
    def test_grid_load_failure_raises_processing_error(self, mock_load):
        """Grid load failure raises ProcessingError with GRID_LOAD_ERROR code."""
        mock_load.side_effect = FileNotFoundError("Zarr store not found")

        with pytest.raises(ProcessingError) as exc_info:
            export_geotiff(
                {"id": "test-export"},
                {"grid_id": "missing-grid", "name": "geotiff"},
                noop_progress,
            )

        assert exc_info.value.code == "GRID_LOAD_ERROR"
        assert "missing-grid" in exc_info.value.message

    @MOCK_TO_RASTER
    @patch("exporter.handlers.grid.load_grid_zarr")
    def test_progress_without_band_subset(self, mock_load, _mock_raster):
        """Reports loading and writing progress."""
        mock_load.return_value = make_test_dataset()
        calls = []

        export_geotiff(
            {"id": "test-export"},
            {"grid_id": "grid-abc", "name": "geotiff"},
            lambda msg, pct=None: calls.append((msg, pct)),
        )

        assert calls == [
            ("Loading grid data...", 30),
            ("Writing GeoTIFF...", 70),
        ]

    @MOCK_TO_RASTER
    @patch("exporter.handlers.grid.load_grid_zarr")
    def test_progress_with_band_subset(self, mock_load, _mock_raster):
        """Reports band selection step when bands specified."""
        mock_load.return_value = make_test_dataset()
        calls = []

        export_geotiff(
            {"id": "test-export"},
            {"grid_id": "grid-abc", "name": "geotiff", "bands": ["fbfm"]},
            lambda msg, pct=None: calls.append((msg, pct)),
        )

        assert calls == [
            ("Loading grid data...", 30),
            ("Selecting bands...", 50),
            ("Writing GeoTIFF...", 70),
        ]


class TestNodataHandling:
    """The faithful-load change (mask_and_scale=False, issue #290) means grids
    now reach the exporter carrying real per-band nodata sentinels. These cover
    the resulting GeoTIFF write paths, which the mocked-nodata=None unit tests
    above do not.
    """

    @pytest.mark.parametrize(
        "nodatas, expected",
        [
            ({"a": 32767}, True),  # single band is trivially uniform
            ({"a": None, "b": None}, True),  # both unset
            ({"a": 32767, "b": 32767}, True),  # same sentinel
            ({"a": np.nan, "b": np.nan}, True),  # NaN counts as a match
            ({"a": 32767, "b": 0}, False),  # distinct sentinels
            ({"a": 32767, "b": None}, False),  # sentinel vs unset
            ({"a": np.nan, "b": 32767}, False),  # NaN vs sentinel
        ],
    )
    def test_nodata_is_uniform(self, nodatas, expected):
        """_nodata_is_uniform reflects whether every band shares one nodata."""
        bands = {k: np.ones((2, 2), dtype=np.float32) for k in nodatas}
        ds = make_test_dataset(bands=bands, shape=(2, 2), nodatas=nodatas)
        assert _nodata_is_uniform(ds) is expected

    @patch.object(RasterArray, "to_raster")
    @patch.object(RasterDataset, "to_raster")
    @patch("exporter.handlers.grid.load_grid_zarr")
    def test_distinct_nodata_uses_stacked_array_write(
        self, mock_load, mock_ds_raster, mock_arr_raster
    ):
        """Bands with distinct nodata take the stacked (to_array) write path,
        not the Dataset path — a GeoTIFF carries one file-level nodata."""
        mock_load.return_value = make_test_dataset(
            bands={
                "tm_id": np.array([[5, 6], [7, 2147483647]], dtype=np.int32),
                "plt_cn": np.array([[11, 12], [13, 0]], dtype=np.int32),
            },
            shape=(2, 2),
            nodatas={"tm_id": 2147483647, "plt_cn": 0},
        )

        export_geotiff(
            {"id": "test-export"},
            {"grid_id": "grid-abc", "name": "geotiff"},
            noop_progress,
        )

        mock_arr_raster.assert_called_once()
        mock_ds_raster.assert_not_called()

    @patch.object(RasterArray, "to_raster")
    @patch.object(RasterDataset, "to_raster")
    @patch("exporter.handlers.grid.load_grid_zarr")
    def test_uniform_nodata_uses_dataset_write(
        self, mock_load, mock_ds_raster, mock_arr_raster
    ):
        """Bands sharing one nodata take the plain Dataset write path."""
        mock_load.return_value = make_test_dataset(
            bands={
                "fbfm": np.full((4, 4), 101, dtype=np.int16),
                "fccs": np.full((4, 4), 200, dtype=np.int16),
            },
            shape=(4, 4),
            nodatas={"fbfm": 32767, "fccs": 32767},
        )

        export_geotiff(
            {"id": "test-export"},
            {"grid_id": "grid-abc", "name": "geotiff"},
            noop_progress,
        )

        mock_ds_raster.assert_called_once()
        mock_arr_raster.assert_not_called()

    def test_stacked_write_preserves_sentinels_and_dtype(self, tmp_path):
        """The stacked-array write (the non-uniform-nodata fallback) produces a
        valid multi-band GeoTIFF that preserves integer dtype, CRS, and the raw
        per-band sentinels — they are not floated or masked to NaN. Exercises
        the same to_array -> to_raster path as export_geotiff, against a local
        file so it runs without GCS.
        """
        ds = make_test_dataset(
            bands={
                "tm_id": np.array([[5, 6], [7, 2147483647]], dtype=np.int32),
                "plt_cn": np.array([[11, 12], [13, 0]], dtype=np.int32),
            },
            shape=(2, 2),
            nodatas={"tm_id": 2147483647, "plt_cn": 0},
        )
        assert not _nodata_is_uniform(ds)

        out = tmp_path / "stacked.tif"
        ds.to_array(dim="band").rio.to_raster(str(out), driver="GTiff")

        with rasterio.open(out) as r:
            assert r.count == 2
            assert all(dt == "int32" for dt in r.dtypes)
            assert CRS(r.crs) == CRS("EPSG:32611")
            # Raw sentinels survive in their respective bands.
            assert r.read(1)[1, 1] == 2147483647
            assert r.read(2)[1, 1] == 0


class TestExportGeotiffIntegration:
    """Integration tests that write to the real GCS bucket.

    These mock load_grid_zarr (to avoid needing a real grid in Zarr) but
    let the full write pipeline run against GCS.
    """

    BUCKET = os.environ["EXPORTS_BUCKET"]

    @pytest.fixture
    def export_id(self):
        """Generate a unique export ID and clean up GCS files after test."""
        eid = f"test-{uuid.uuid4().hex[:12]}"
        yield eid
        fs = gcsfs.GCSFileSystem()
        bucket = os.environ["EXPORTS_BUCKET"]
        if bucket.startswith("gs://"):
            bucket = bucket[5:]
        gcs_dir = f"{bucket}/{eid}"
        try:
            fs.rm(gcs_dir, recursive=True)
        except FileNotFoundError:
            pass

    @patch("exporter.handlers.grid.load_grid_zarr")
    def test_writes_valid_geotiff_to_gcs(self, mock_load, export_id):
        """Writes a valid GeoTIFF to GCS and reads it back."""
        crs_str = "EPSG:32611"
        ds = make_test_dataset(crs=crs_str)
        mock_load.return_value = ds

        gcs_path = export_geotiff(
            {"id": export_id},
            {"grid_id": "grid-abc", "name": "geotiff"},
            noop_progress,
        )

        assert gcs_path == f"gs://{self.BUCKET}/{export_id}/export.tif"

        # Read back from GCS and verify
        result = xr.open_dataset(gcs_path, engine="rasterio")
        assert len(result.data_vars) > 0
        assert CRS(result.rio.crs) == CRS(crs_str)
        result.close()

    @patch("exporter.handlers.grid.load_grid_zarr")
    def test_band_subset_writes_correct_bands_to_gcs(self, mock_load, export_id):
        """Band subset writes only selected bands to GCS."""
        ds = make_test_dataset(
            bands={
                "fbfm": np.full((10, 10), 101, dtype=np.int32),
                "fuel_load.1hr": np.random.rand(10, 10).astype(np.float64),
                "fuel_load.10hr": np.random.rand(10, 10).astype(np.float64),
            }
        )
        mock_load.return_value = ds

        gcs_path = export_geotiff(
            {"id": export_id},
            {
                "grid_id": "grid-abc",
                "name": "geotiff",
                "bands": ["fuel_load.1hr", "fuel_load.10hr"],
            },
            noop_progress,
        )

        result = xr.open_dataset(gcs_path, engine="rasterio")
        assert result.sizes["band"] == 2
        result.close()


class TestExportChunkedZarr:
    """Tests that chunked Zarr data exports correctly via windowed writes.

    These save a Dataset as chunked Zarr locally, load it back (so it's
    genuinely chunked), then run the full export pipeline to GCS.
    """

    BUCKET = os.environ["EXPORTS_BUCKET"]

    @pytest.fixture
    def export_id(self):
        """Generate a unique export ID and clean up GCS files after test."""
        eid = f"test-{uuid.uuid4().hex[:12]}"
        yield eid
        fs = gcsfs.GCSFileSystem()
        bucket = os.environ["EXPORTS_BUCKET"]
        if bucket.startswith("gs://"):
            bucket = bucket[5:]
        gcs_dir = f"{bucket}/{eid}"
        try:
            fs.rm(gcs_dir, recursive=True)
        except FileNotFoundError:
            pass

    @patch("exporter.handlers.grid.load_grid_zarr")
    def test_chunked_zarr_exports_valid_geotiff(self, mock_load, export_id, tmp_path):
        """Chunked Zarr data produces a valid GeoTIFF via windowed write."""
        ds = make_test_dataset(
            bands={
                "fbfm": np.full((20, 20), 101, dtype=np.int32),
                "fuel_load.1hr": np.random.rand(20, 20).astype(np.float64),
            },
            shape=(20, 20),
        )

        # Save as chunked Zarr and load back — data is genuinely chunked
        zarr_path = str(tmp_path / "chunked.zarr")
        save_zarr(zarr_path, ds, chunk_shape=(8, 8))
        chunked_ds = load_zarr(zarr_path)
        mock_load.return_value = chunked_ds

        gcs_path = export_geotiff(
            {"id": export_id},
            {"grid_id": "grid-abc", "name": "geotiff"},
            noop_progress,
        )

        # Read back and verify
        result = xr.open_dataset(gcs_path, engine="rasterio")
        assert result.sizes["band"] == 2
        assert CRS(result.rio.crs) == CRS("EPSG:32611")
        result.close()
