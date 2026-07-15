"""
Tests for the zipped-Zarr exporter handler.

Unit tests mock storage to test handler logic.
"""

import shutil
import zipfile
from unittest.mock import patch

import numpy as np
import rioxarray  # noqa: F401
import xarray as xr
import zarr
from exporter.handlers.grid import export_zarr
from rasterio.transform import from_bounds


def make_test_dataset(
    crs: str = "EPSG:32611",
    shape: tuple[int, int] = (10, 10),
) -> xr.Dataset:
    """2D Dataset analogous to test_netcdf.make_test_dataset."""
    ny, nx = shape
    transform = from_bounds(
        500000, 5200000, 500000 + nx * 30, 5200000 + ny * 30, nx, ny
    )
    ds = xr.Dataset(
        {"fuel_load.1hr": (("y", "x"), np.random.rand(ny, nx).astype(np.float32))},
        coords={"y": np.arange(ny), "x": np.arange(nx)},
    )
    ds = ds.rio.write_crs(crs)
    ds = ds.rio.write_transform(transform)
    return ds


def noop_progress(message: str, percent: int | None = None) -> None:
    pass


class TestExportZarrUnit:
    """Unit tests for the Zarr handler — mock storage and the GCS upload."""

    @patch("google.cloud.storage.Client")
    @patch("exporter.handlers.grid.load_grid_zarr")
    def test_writes_zarr_format_3(self, mock_load, mock_client, tmp_path):
        """The exported store must be Zarr format v3.

        Users open this store with their own tooling and zarr-python 2.x
        cannot read v3, so the format version is part of the export contract
        and must not drift with the library default.
        """
        mock_load.return_value = make_test_dataset()

        captured = {}

        def capture(local_path: str) -> None:
            dest = tmp_path / "captured.zip"
            shutil.copy(local_path, dest)
            captured["path"] = str(dest)

        blob = mock_client.return_value.bucket.return_value.blob.return_value
        blob.upload_from_filename.side_effect = capture

        export_zarr(
            {"id": "test-export"},
            {"grid_id": "grid-abc", "name": "zarr"},
            noop_progress,
        )

        extract_dir = tmp_path / "extracted"
        with zipfile.ZipFile(captured["path"]) as zf:
            zf.extractall(extract_dir)

        group = zarr.open_group(str(extract_dir / "export.zarr"), mode="r")
        assert group.metadata.zarr_format == 3
        assert group["fuel_load.1hr"].metadata.zarr_format == 3
