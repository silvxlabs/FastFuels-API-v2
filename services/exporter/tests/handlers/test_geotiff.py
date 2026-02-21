"""
Tests for exporter GeoTIFF handler.

Unit tests mock storage to test handler logic.
Integration tests write to the real GCS bucket.
"""

import os
import uuid
from unittest.mock import patch

import gcsfs
import numpy as np
import pytest
import rioxarray  # noqa: F401
import xarray as xr
from exporter.errors import ProcessingError
from exporter.handlers.geotiff import export_geotiff
from pyproj import CRS
from rasterio.transform import from_bounds
from rioxarray.raster_dataset import RasterDataset

from lib.zarr_utils import load_zarr, save_zarr

MOCK_TO_RASTER = patch.object(RasterDataset, "to_raster")


def make_test_dataset(
    bands: dict[str, np.ndarray] | None = None,
    crs: str = "EPSG:32611",
    shape: tuple[int, int] = (10, 10),
) -> xr.Dataset:
    """Create a synthetic Dataset for testing."""
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
    return ds


def noop_progress(message: str, percent: int | None = None):
    pass


class TestExportGeotiffUnit:
    """Unit tests for export_geotiff handler logic.

    These mock load_grid_zarr and to_raster to test band validation,
    error handling, and progress reporting without external dependencies.
    """

    @MOCK_TO_RASTER
    @patch("exporter.handlers.geotiff.load_grid_zarr")
    def test_loads_correct_grid(self, mock_load, _mock_raster):
        """Calls load_grid_zarr with the grid_id from source."""
        mock_load.return_value = make_test_dataset()

        export_geotiff(
            {"id": "test-export"},
            {"grid_ids": ["grid-abc"], "name": "geotiff"},
            noop_progress,
        )

        mock_load.assert_called_once_with("grid-abc")

    @MOCK_TO_RASTER
    @patch("exporter.handlers.geotiff.load_grid_zarr")
    def test_returns_gcs_path(self, mock_load, _mock_raster):
        """Returns the expected GCS path."""
        mock_load.return_value = make_test_dataset()

        result = export_geotiff(
            {"id": "export-789"},
            {"grid_ids": ["grid-abc"], "name": "geotiff"},
            noop_progress,
        )

        assert result.endswith("/export-789/export.tif")

    @MOCK_TO_RASTER
    @patch("exporter.handlers.geotiff.load_grid_zarr")
    def test_returns_gcs_path_with_name(self, mock_load, _mock_raster):
        """Uses sanitized export name in the GCS path."""
        mock_load.return_value = make_test_dataset()

        result = export_geotiff(
            {"id": "export-789", "name": "My Export!"},
            {"grid_ids": ["grid-abc"], "name": "geotiff"},
            noop_progress,
        )

        assert result.endswith("/export-789/My_Export.tif")

    @patch("exporter.handlers.geotiff.load_grid_zarr")
    def test_missing_band_raises_processing_error(self, mock_load):
        """Missing band raises ProcessingError with BAND_NOT_FOUND code."""
        mock_load.return_value = make_test_dataset()

        with pytest.raises(ProcessingError) as exc_info:
            export_geotiff(
                {"id": "test-export"},
                {
                    "grid_ids": ["grid-abc"],
                    "name": "geotiff",
                    "bands": ["fbfm", "nonexistent"],
                },
                noop_progress,
            )

        assert exc_info.value.code == "BAND_NOT_FOUND"
        assert "nonexistent" in exc_info.value.message

    @patch("exporter.handlers.geotiff.load_grid_zarr")
    def test_grid_load_failure_raises_processing_error(self, mock_load):
        """Grid load failure raises ProcessingError with GRID_LOAD_ERROR code."""
        mock_load.side_effect = FileNotFoundError("Zarr store not found")

        with pytest.raises(ProcessingError) as exc_info:
            export_geotiff(
                {"id": "test-export"},
                {"grid_ids": ["missing-grid"], "name": "geotiff"},
                noop_progress,
            )

        assert exc_info.value.code == "GRID_LOAD_ERROR"
        assert "missing-grid" in exc_info.value.message

    @MOCK_TO_RASTER
    @patch("exporter.handlers.geotiff.load_grid_zarr")
    def test_progress_without_band_subset(self, mock_load, _mock_raster):
        """Reports loading and writing progress."""
        mock_load.return_value = make_test_dataset()
        calls = []

        export_geotiff(
            {"id": "test-export"},
            {"grid_ids": ["grid-abc"], "name": "geotiff"},
            lambda msg, pct=None: calls.append((msg, pct)),
        )

        assert calls == [
            ("Loading grid data...", 30),
            ("Writing GeoTIFF...", 70),
        ]

    @MOCK_TO_RASTER
    @patch("exporter.handlers.geotiff.load_grid_zarr")
    def test_progress_with_band_subset(self, mock_load, _mock_raster):
        """Reports band selection step when bands specified."""
        mock_load.return_value = make_test_dataset()
        calls = []

        export_geotiff(
            {"id": "test-export"},
            {"grid_ids": ["grid-abc"], "name": "geotiff", "bands": ["fbfm"]},
            lambda msg, pct=None: calls.append((msg, pct)),
        )

        assert calls == [
            ("Loading grid data...", 30),
            ("Selecting bands...", 50),
            ("Writing GeoTIFF...", 70),
        ]


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

    @patch("exporter.handlers.geotiff.load_grid_zarr")
    def test_writes_valid_geotiff_to_gcs(self, mock_load, export_id):
        """Writes a valid GeoTIFF to GCS and reads it back."""
        crs_str = "EPSG:32611"
        ds = make_test_dataset(crs=crs_str)
        mock_load.return_value = ds

        gcs_path = export_geotiff(
            {"id": export_id},
            {"grid_ids": ["grid-abc"], "name": "geotiff"},
            noop_progress,
        )

        assert gcs_path == f"gs://{self.BUCKET}/{export_id}/export.tif"

        # Read back from GCS and verify
        result = xr.open_dataset(gcs_path, engine="rasterio")
        assert len(result.data_vars) > 0
        assert CRS(result.rio.crs) == CRS(crs_str)
        result.close()

    @patch("exporter.handlers.geotiff.load_grid_zarr")
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
                "grid_ids": ["grid-abc"],
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

    @patch("exporter.handlers.geotiff.load_grid_zarr")
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
            {"grid_ids": ["grid-abc"], "name": "geotiff"},
            noop_progress,
        )

        # Read back and verify
        result = xr.open_dataset(gcs_path, engine="rasterio")
        assert result.sizes["band"] == 2
        assert CRS(result.rio.crs) == CRS("EPSG:32611")
        result.close()
