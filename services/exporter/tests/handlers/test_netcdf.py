"""
Tests for the netCDF exporter handler.

Unit tests mock storage + Firestore to test handler logic.
Integration tests write to the real GCS bucket and read back.
"""

import os
import uuid
from unittest.mock import MagicMock, patch

import gcsfs
import numpy as np
import pytest
import rioxarray  # noqa: F401
import xarray as xr
from exporter.errors import ProcessingError
from exporter.handlers.netcdf import export_netcdf
from pyproj import CRS
from rasterio.transform import from_bounds


def make_test_dataset(
    bands: dict[str, np.ndarray] | None = None,
    crs: str = "EPSG:32611",
    shape: tuple[int, int] = (10, 10),
) -> xr.Dataset:
    """2D Dataset analogous to test_geotiff.make_test_dataset."""
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
        ds[name] = xr.DataArray(
            data,
            dims=("y", "x"),
            coords={"y": np.arange(ny), "x": np.arange(nx)},
        )
    ds = ds.rio.write_crs(crs)
    ds = ds.rio.write_transform(transform)
    return ds


def make_3d_test_dataset(
    crs: str = "EPSG:32611",
    shape: tuple[int, int, int] = (3, 6, 8),
) -> xr.Dataset:
    """3D Dataset with bulk_density.foliage.live band."""
    nz, ny, nx = shape
    transform = from_bounds(500000, 5200000, 500000 + nx * 2, 5200000 + ny * 2, nx, ny)
    ds = xr.Dataset(
        {
            "bulk_density.foliage.live": (
                ("z", "y", "x"),
                np.random.rand(*shape).astype(np.float32),
            ),
        },
        coords={
            "x": np.arange(nx),
            "y": np.arange(ny),
            "z": np.arange(nz),
        },
    )
    ds = ds.rio.write_crs(crs)
    ds = ds.rio.write_transform(transform)
    return ds


def fake_grid_snapshot(bands: list[dict]) -> MagicMock:
    """Mock Firestore snapshot whose .to_dict() returns a doc with these bands."""
    snap = MagicMock()
    snap.to_dict.return_value = {"bands": bands}
    return snap


def noop_progress(message: str, percent: int | None = None) -> None:
    pass


class TestExportNetcdfUnit:
    """Unit tests for the netCDF handler — mock storage, Firestore, and GCS upload."""

    @patch("exporter.handlers.netcdf._upload_file")
    @patch("exporter.handlers.netcdf.get_document")
    @patch("exporter.handlers.grid.load_grid_zarr")
    def test_loads_correct_grid(self, mock_load, mock_get_doc, _mock_upload):
        mock_load.return_value = make_test_dataset()
        mock_get_doc.return_value = (
            None,
            fake_grid_snapshot(
                [
                    {"key": "fbfm", "type": "categorical", "unit": None, "index": 0},
                    {
                        "key": "fuel_load.1hr",
                        "type": "continuous",
                        "unit": "kg/m**2",
                        "index": 1,
                    },
                ]
            ),
        )

        export_netcdf(
            {"id": "test-export"},
            {"grid_id": "grid-abc", "name": "netcdf"},
            noop_progress,
        )

        mock_load.assert_called_once_with("grid-abc")

    @patch("exporter.handlers.netcdf._upload_file")
    @patch("exporter.handlers.netcdf.get_document")
    @patch("exporter.handlers.grid.load_grid_zarr")
    def test_returns_gcs_path_with_sanitized_name(
        self, mock_load, mock_get_doc, _mock_upload
    ):
        mock_load.return_value = make_test_dataset()
        mock_get_doc.return_value = (None, fake_grid_snapshot([]))

        result = export_netcdf(
            {"id": "export-789", "name": "My Export!"},
            {"grid_id": "grid-abc", "name": "netcdf"},
            noop_progress,
        )

        assert result.endswith("/export-789/My_Export.nc")

    @patch("exporter.handlers.netcdf.get_document")
    @patch("exporter.handlers.grid.load_grid_zarr")
    def test_missing_band_raises_processing_error(self, mock_load, mock_get_doc):
        mock_load.return_value = make_test_dataset()
        mock_get_doc.return_value = (None, fake_grid_snapshot([]))

        with pytest.raises(ProcessingError) as exc_info:
            export_netcdf(
                {"id": "test-export"},
                {
                    "grid_id": "grid-abc",
                    "name": "netcdf",
                    "bands": ["fbfm", "nonexistent"],
                },
                noop_progress,
            )

        assert exc_info.value.code == "BAND_NOT_FOUND"
        assert "nonexistent" in exc_info.value.message

    @patch("exporter.handlers.grid.load_grid_zarr")
    def test_grid_load_failure_raises_processing_error(self, mock_load):
        mock_load.side_effect = FileNotFoundError("Zarr store not found")

        with pytest.raises(ProcessingError) as exc_info:
            export_netcdf(
                {"id": "test-export"},
                {"grid_id": "missing-grid", "name": "netcdf"},
                noop_progress,
            )

        assert exc_info.value.code == "GRID_LOAD_ERROR"

    @patch("exporter.handlers.netcdf.get_document")
    @patch("exporter.handlers.grid.load_grid_zarr")
    def test_grid_doc_lookup_failure_raises_processing_error(
        self, mock_load, mock_get_doc
    ):
        mock_load.return_value = make_test_dataset()
        mock_get_doc.side_effect = RuntimeError("Firestore offline")

        with pytest.raises(ProcessingError) as exc_info:
            export_netcdf(
                {"id": "test-export"},
                {"grid_id": "grid-abc", "name": "netcdf"},
                noop_progress,
            )

        assert exc_info.value.code == "GRID_DOC_NOT_FOUND"

    @patch("exporter.handlers.netcdf._upload_file")
    @patch("exporter.handlers.netcdf.get_document")
    @patch("exporter.handlers.grid.load_grid_zarr")
    def test_writes_cf_conformant_netcdf(
        self, mock_load, mock_get_doc, mock_upload, tmp_path
    ):
        """Intercepts the upload step and verifies the local netCDF is CF-conformant."""
        mock_load.return_value = make_test_dataset()
        mock_get_doc.return_value = (
            None,
            fake_grid_snapshot(
                [
                    {"key": "fbfm", "type": "categorical", "unit": None, "index": 0},
                    {
                        "key": "fuel_load.1hr",
                        "type": "continuous",
                        "unit": "kg/m**2",
                        "index": 1,
                    },
                ]
            ),
        )

        # Capture the local path passed to _upload_file by copying the file aside.
        captured = {}

        def capture(local_path: str, gcs_path: str) -> None:
            import shutil

            dest = tmp_path / "captured.nc"
            shutil.copy(local_path, dest)
            captured["path"] = str(dest)

        mock_upload.side_effect = capture

        export_netcdf(
            {"id": "test-export"},
            {"grid_id": "grid-abc", "name": "netcdf"},
            noop_progress,
        )

        # Verify the netCDF is CF-conformant
        ds = xr.open_dataset(captured["path"], decode_coords="all")
        ds_raw = xr.open_dataset(captured["path"], decode_coords=False)

        assert ds.attrs["Conventions"] == "CF-1.13"
        assert CRS(ds.rio.crs) == CRS("EPSG:32611")
        # `grid_mapping` is stamped on every data var in the raw netCDF;
        # xarray hoists it from attrs to encoding on decoded reopen.
        for var in ("fbfm", "fuel_load.1hr"):
            assert ds_raw[var].attrs["grid_mapping"] == "spatial_ref", var
        assert ds["fuel_load.1hr"].attrs.get("units") == "kg/m**2"
        assert ds["x"].attrs.get("axis") == "X"
        assert ds["y"].attrs.get("axis") == "Y"
        ds.close()
        ds_raw.close()

    @patch("exporter.handlers.netcdf._upload_file")
    @patch("exporter.handlers.netcdf.get_document")
    @patch("exporter.handlers.grid.load_grid_zarr")
    def test_3d_stamps_z_axis(self, mock_load, mock_get_doc, mock_upload, tmp_path):
        mock_load.return_value = make_3d_test_dataset()
        mock_get_doc.return_value = (
            None,
            fake_grid_snapshot(
                [
                    {
                        "key": "bulk_density.foliage.live",
                        "type": "continuous",
                        "unit": "kg/m**3",
                        "index": 0,
                    },
                ]
            ),
        )

        captured = {}

        def capture(local_path: str, gcs_path: str) -> None:
            import shutil

            dest = tmp_path / "captured.nc"
            shutil.copy(local_path, dest)
            captured["path"] = str(dest)

        mock_upload.side_effect = capture

        export_netcdf(
            {"id": "test-export"},
            {"grid_id": "grid-abc", "name": "netcdf"},
            noop_progress,
        )

        ds = xr.open_dataset(captured["path"], decode_coords="all")
        assert ds["z"].attrs.get("axis") == "Z"
        assert ds["z"].attrs.get("positive") == "up"
        assert ds["bulk_density.foliage.live"].attrs.get("units") == "kg/m**3"
        ds.close()

    @patch("exporter.handlers.netcdf._upload_file")
    @patch("exporter.handlers.netcdf.get_document")
    @patch("exporter.handlers.grid.load_grid_zarr")
    def test_strips_internal_transform_attrs(
        self, mock_load, mock_get_doc, mock_upload, tmp_path
    ):
        """Internal attrs (`transform`, `z_origin`, `z_resolution`) must not
        leak into the netCDF — h5netcdf can't serialize list-of-numbers attrs
        and CF doesn't define them.
        """
        ds_in = make_3d_test_dataset()
        ds_in.attrs["transform"] = [2.0, 0.0, 500000.0, 0.0, -2.0, 5200000.0]
        ds_in.attrs["z_origin"] = 0.0
        ds_in.attrs["z_resolution"] = 1.0
        mock_load.return_value = ds_in
        mock_get_doc.return_value = (
            None,
            fake_grid_snapshot(
                [
                    {"key": "bulk_density.foliage.live", "unit": "kg/m**3"},
                ]
            ),
        )

        captured = {}

        def capture(local_path: str, gcs_path: str) -> None:
            import shutil

            dest = tmp_path / "captured.nc"
            shutil.copy(local_path, dest)
            captured["path"] = str(dest)

        mock_upload.side_effect = capture

        export_netcdf(
            {"id": "test-export"},
            {"grid_id": "grid-abc", "name": "netcdf"},
            noop_progress,
        )

        ds = xr.open_dataset(captured["path"], decode_coords=False)
        assert "transform" not in ds.attrs
        assert "z_origin" not in ds.attrs
        assert "z_resolution" not in ds.attrs
        ds.close()


class TestExportNetcdfIntegration:
    """Integration tests: write a real netCDF to GCS and read it back."""

    BUCKET = os.environ["EXPORTS_BUCKET"]

    @pytest.fixture
    def export_id(self):
        eid = f"test-{uuid.uuid4().hex[:12]}"
        yield eid
        fs = gcsfs.GCSFileSystem()
        bucket = os.environ["EXPORTS_BUCKET"]
        if bucket.startswith("gs://"):
            bucket = bucket[5:]
        try:
            fs.rm(f"{bucket}/{eid}", recursive=True)
        except FileNotFoundError:
            pass

    @patch("exporter.handlers.netcdf.get_document")
    @patch("exporter.handlers.grid.load_grid_zarr")
    def test_writes_valid_netcdf_to_gcs(self, mock_load, mock_get_doc, export_id):
        crs_str = "EPSG:32611"
        ds = make_test_dataset(crs=crs_str)
        mock_load.return_value = ds
        mock_get_doc.return_value = (
            None,
            fake_grid_snapshot(
                [
                    {"key": "fbfm", "type": "categorical", "unit": None, "index": 0},
                    {
                        "key": "fuel_load.1hr",
                        "type": "continuous",
                        "unit": "kg/m**2",
                        "index": 1,
                    },
                ]
            ),
        )

        gcs_path = export_netcdf(
            {"id": export_id},
            {"grid_id": "grid-abc", "name": "netcdf"},
            noop_progress,
        )

        assert gcs_path == f"gs://{self.BUCKET}/{export_id}/export.nc"

        # Read back from GCS through gcsfs (h5netcdf needs a local-style handle)
        fs = gcsfs.GCSFileSystem()
        with fs.open(gcs_path, "rb") as f:
            result = xr.open_dataset(f, decode_coords="all", engine="h5netcdf")
            assert result.attrs.get("Conventions") == "CF-1.13"
            assert CRS(result.rio.crs) == CRS(crs_str)
            assert "fbfm" in result.data_vars
            assert "fuel_load.1hr" in result.data_vars
            assert result["fuel_load.1hr"].attrs.get("units") == "kg/m**2"
            result.close()

    @patch("exporter.handlers.netcdf.get_document")
    @patch("exporter.handlers.grid.load_grid_zarr")
    def test_band_subset_writes_only_selected_bands(
        self, mock_load, mock_get_doc, export_id
    ):
        ds = make_test_dataset(
            bands={
                "fbfm": np.full((10, 10), 101, dtype=np.int32),
                "fuel_load.1hr": np.random.rand(10, 10).astype(np.float64),
                "fuel_load.10hr": np.random.rand(10, 10).astype(np.float64),
            }
        )
        mock_load.return_value = ds
        mock_get_doc.return_value = (
            None,
            fake_grid_snapshot(
                [
                    {"key": k, "type": "continuous", "unit": None, "index": i}
                    for i, k in enumerate(["fbfm", "fuel_load.1hr", "fuel_load.10hr"])
                ]
            ),
        )

        gcs_path = export_netcdf(
            {"id": export_id},
            {
                "grid_id": "grid-abc",
                "name": "netcdf",
                "bands": ["fuel_load.1hr", "fuel_load.10hr"],
            },
            noop_progress,
        )

        fs = gcsfs.GCSFileSystem()
        with fs.open(gcs_path, "rb") as f:
            result = xr.open_dataset(f, decode_coords="all", engine="h5netcdf")
            assert set(result.data_vars) == {"fuel_load.1hr", "fuel_load.10hr"}
            result.close()
