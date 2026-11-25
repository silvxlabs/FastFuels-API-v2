"""
Tests for the zipped-Zarr exporter handler.

Unit tests mock the grid load and write to an in-memory GCS stand-in.
"""

import io
import json
import zipfile
from unittest.mock import patch

import numpy as np
import rioxarray  # noqa: F401
import xarray as xr
import zarr
from exporter.handlers.grid import export_zarr
from rasterio.transform import from_bounds
from zarr.codecs import GzipCodec

from tests.fake_gcs import UPLOADS, FakeBlob

MOCK_UPLOAD = patch("exporter.storage._blob", side_effect=FakeBlob)


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

    @MOCK_UPLOAD
    @patch("exporter.handlers.grid.load_grid_zarr")
    def test_writes_zarr_format_3(self, mock_load, _mock_upload, tmp_path):
        """The exported store must be Zarr format v3.

        Users open this store with their own tooling and zarr-python 2.x
        cannot read v3, so the format version is part of the export contract
        and must not drift with the library default.
        """
        mock_load.return_value = make_test_dataset()

        gcs_path = export_zarr(
            {"id": "test-export"},
            {"grid_id": "grid-abc", "name": "zarr"},
            noop_progress,
        )

        extract_dir = tmp_path / "extracted"
        with zipfile.ZipFile(io.BytesIO(UPLOADS[gcs_path])) as zf:
            zf.extractall(extract_dir)

        group = zarr.open_group(str(extract_dir / "export.zarr"), mode="r")
        assert group.metadata.zarr_format == 3
        assert group["fuel_load.1hr"].metadata.zarr_format == 3

    @MOCK_UPLOAD
    @patch("exporter.handlers.grid.load_grid_zarr")
    def test_zip_stores_chunks_and_deflates_metadata(self, mock_load, _mock_upload):
        """Chunks are already zstd-compressed, so the zip stores them as-is."""
        mock_load.return_value = make_test_dataset()

        gcs_path = export_zarr({"id": "test-export"}, {"grid_id": "g"}, noop_progress)

        with zipfile.ZipFile(io.BytesIO(UPLOADS[gcs_path])) as zf:
            infos = {i.filename: i for i in zf.infolist()}
        assert next(iter(infos)) == "export.zarr/"
        files = [n for n in infos if not n.endswith("/")]
        assert "export.zarr/zarr.json" in files
        for name in files:
            expected = (
                zipfile.ZIP_DEFLATED if name.endswith(".json") else zipfile.ZIP_STORED
            )
            assert infos[name].compress_type == expected, name

    @MOCK_UPLOAD
    @patch("exporter.handlers.grid.load_grid_zarr")
    def test_chunk_codec_is_pinned_to_zstd(self, mock_load, _mock_upload, tmp_path):
        """A source grid stored with another codec still exports as zstd."""
        source = tmp_path / "source.zarr"
        ds = make_test_dataset()
        ds.to_zarr(
            source,
            zarr_format=3,
            encoding={"fuel_load.1hr": {"compressors": [GzipCodec()]}},
        )
        mock_load.return_value = xr.open_zarr(source)

        gcs_path = export_zarr({"id": "test-export"}, {"grid_id": "g"}, noop_progress)

        with zipfile.ZipFile(io.BytesIO(UPLOADS[gcs_path])) as zf:
            meta = json.loads(zf.read("export.zarr/fuel_load.1hr/zarr.json"))
        codecs = [c["name"] for c in meta["codecs"]]
        assert codecs == ["bytes", "zstd"]
