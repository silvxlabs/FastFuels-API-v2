"""
Tests for exporter grid handlers.

Unit tests mock storage to test handler logic.
Integration tests write to the real GCS bucket.
"""

import os
import subprocess
import sys
import textwrap
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import gcsfs
import numpy as np
import pytest
import rasterio
import rioxarray  # noqa: F401
import xarray as xr
from exporter.errors import ProcessingError
from exporter.handlers.grid import (
    _nodata_is_uniform,
    _stack_with_shared_nodata,
    export_geotiff,
)
from pyproj import CRS
from rasterio.transform import from_bounds
from rioxarray.raster_array import RasterArray
from rioxarray.raster_dataset import RasterDataset

from lib.zarr_utils import load_zarr, save_zarr
from tests.fake_gcs import UPLOADS, FakeBlob

MOCK_UPLOAD = patch("exporter.storage._blob", side_effect=FakeBlob)

EXPORTER_ROOT = Path(__file__).resolve().parents[2]
NO_UPLOAD = patch("exporter.handlers.grid.upload_file", new=lambda local, gcs: None)


def MOCK_TO_RASTER(test):
    """Mock the Dataset write; nothing is written, so skip the upload too."""
    return patch.object(RasterDataset, "to_raster")(NO_UPLOAD(test))


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

    @NO_UPLOAD
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

    @NO_UPLOAD
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

    @staticmethod
    def _write_stacked(ds, path):
        """Write through the same stacked path export_geotiff uses for bands
        with distinct nodata, against a local file so it runs without GCS."""
        _stack_with_shared_nodata(ds).rio.to_raster(str(path), driver="GTiff")
        return rasterio.open(path)

    def test_stacked_write_maps_sentinels_to_shared_nodata(self, tmp_path):
        """Each band's own sentinel becomes the shared -9999 and the file is
        tagged with it; integer dtype and CRS are preserved."""
        ds = make_test_dataset(
            bands={
                "tm_id": np.array([[5, 6], [7, 2147483647]], dtype=np.int32),
                "plt_cn": np.array([[11, 12], [13, 0]], dtype=np.int32),
            },
            shape=(2, 2),
            nodatas={"tm_id": 2147483647, "plt_cn": 0},
        )
        assert not _nodata_is_uniform(ds)

        with self._write_stacked(ds, tmp_path / "stacked.tif") as r:
            assert r.count == 2
            assert all(dt == "int32" for dt in r.dtypes)
            assert CRS(r.crs) == CRS("EPSG:32611")
            assert r.nodata == -9999
            np.testing.assert_array_equal(r.read(1), [[5, 6], [7, -9999]])
            np.testing.assert_array_equal(r.read(2), [[11, 12], [13, -9999]])

    def test_topography_with_flat_aspect_keeps_a_nodata_tag(self, tmp_path):
        """LANDFIRE topography: elevation nodata 32767, aspect nodata -1
        (flat). Both become -9999 under one file-level tag."""
        ds = make_test_dataset(
            bands={
                "elevation": np.array([[1500, 32767]], dtype=np.int16),
                "aspect": np.array([[-1, 90]], dtype=np.int16),
            },
            shape=(1, 2),
            nodatas={"elevation": 32767, "aspect": -1},
        )

        with self._write_stacked(ds, tmp_path / "topo.tif") as r:
            assert r.dtypes == ("int16", "int16")
            assert r.nodata == -9999
            np.testing.assert_array_equal(r.read(1), [[1500, -9999]])
            np.testing.assert_array_equal(r.read(2), [[-9999, 90]])

    def test_unsigned_bands_widen_to_hold_shared_nodata(self, tmp_path):
        ds = make_test_dataset(
            bands={
                "a": np.array([[1, 255]], dtype=np.uint8),
                "b": np.array([[0, 7]], dtype=np.uint8),
            },
            shape=(1, 2),
            nodatas={"a": 255, "b": 0},
        )

        with self._write_stacked(ds, tmp_path / "uint8.tif") as r:
            assert r.dtypes == ("int16", "int16")
            assert r.nodata == -9999
            np.testing.assert_array_equal(r.read(1), [[1, -9999]])
            np.testing.assert_array_equal(r.read(2), [[-9999, 7]])

    def test_float_stack_uses_nan(self, tmp_path):
        ds = make_test_dataset(
            bands={
                "f": np.array([[0.5, np.nan]], dtype=np.float32),
                "i": np.array([[3, 32767]], dtype=np.int16),
            },
            shape=(1, 2),
            nodatas={"f": np.nan, "i": 32767},
        )

        with self._write_stacked(ds, tmp_path / "float.tif") as r:
            assert r.dtypes == ("float32", "float32")
            assert np.isnan(r.nodata)
            np.testing.assert_array_equal(r.read(1), [[0.5, np.nan]])
            np.testing.assert_array_equal(r.read(2), [[3.0, np.nan]])

    def test_band_without_nodata_is_unchanged(self, tmp_path):
        ds = make_test_dataset(
            bands={
                "a": np.array([[4, 5]], dtype=np.int16),
                "b": np.array([[6, 32767]], dtype=np.int16),
            },
            shape=(1, 2),
            nodatas={"b": 32767},
        )

        with self._write_stacked(ds, tmp_path / "unset.tif") as r:
            np.testing.assert_array_equal(r.read(1), [[4, 5]])
            np.testing.assert_array_equal(r.read(2), [[6, -9999]])


class TestExportGeotiffLocalWrite:
    """Run export_geotiff end to end on a grid saved to and loaded from Zarr,
    with the GCS write redirected to a local file."""

    @pytest.fixture
    def export_local(self, tmp_path):
        """Return a function that exports ``ds`` and opens the written file."""
        local_path = tmp_path / "export.tif"

        def run(ds, bands=None):
            zarr_path = str(tmp_path / "grid.zarr")
            save_zarr(zarr_path, ds, chunk_shape=(1, 2))
            source = {"grid_id": "grid-abc", "name": "geotiff"}
            if bands:
                source["bands"] = bands
            with (
                patch(
                    "exporter.handlers.grid.load_grid_zarr",
                    return_value=load_zarr(zarr_path),
                ),
                MOCK_UPLOAD,
            ):
                gcs_path = export_geotiff({"id": "test-export"}, source, noop_progress)
            local_path.write_bytes(UPLOADS[gcs_path])
            return rasterio.open(local_path)

        return run

    @staticmethod
    def _topography():
        return make_test_dataset(
            bands={
                "elevation": np.array([[1500, 32767]], dtype=np.int16),
                "slope": np.array([[0, 12]], dtype=np.int16),
                "aspect": np.array([[-1, 90]], dtype=np.int16),
            },
            shape=(1, 2),
            nodatas={"elevation": 32767, "slope": 32767, "aspect": -1},
        )

    def test_distinct_nodata_writes_shared_nodata_and_band_names(self, export_local):
        # Zarr loads bands in alphabetical order.
        with export_local(self._topography()) as r:
            assert r.nodata == -9999
            assert r.descriptions == ("aspect", "elevation", "slope")
            np.testing.assert_array_equal(r.read(1), [[-9999, 90]])
            np.testing.assert_array_equal(r.read(2), [[1500, -9999]])
            np.testing.assert_array_equal(r.read(3), [[0, 12]])

    def test_uniform_nodata_keeps_band_nodata_and_names(self, export_local):
        with export_local(self._topography(), bands=["elevation", "slope"]) as r:
            assert r.nodata == 32767
            assert r.descriptions == ("elevation", "slope")
            np.testing.assert_array_equal(r.read(1), [[1500, 32767]])


class TestGeotiffLayout:
    """Band-interleaved 512 x 512 tiles, DEFLATE, written from tile-aligned chunks (#608)."""

    @staticmethod
    def _export_local(ds, tmp_path, chunk_shape):
        zarr_path = str(tmp_path / "grid.zarr")
        save_zarr(zarr_path, ds, chunk_shape=chunk_shape)
        local_path = tmp_path / "export.tif"
        with (
            patch(
                "exporter.handlers.grid.load_grid_zarr",
                return_value=load_zarr(zarr_path),
            ),
            MOCK_UPLOAD,
        ):
            gcs_path = export_geotiff({"id": "e"}, {"grid_id": "g"}, noop_progress)
        local_path.write_bytes(UPLOADS[gcs_path])
        return rasterio.open(local_path)

    def test_multi_tile_float_grid(self, tmp_path):
        """Source chunks that don't line up with the tiles still write exactly."""
        rng = np.random.default_rng(0)
        bands = {
            "a": rng.random((600, 1100), dtype=np.float32),
            "b": rng.random((600, 1100), dtype=np.float32),
        }
        bands["a"][::7, ::5] = np.nan
        ds = make_test_dataset(
            bands=bands, shape=(600, 1100), nodatas={"a": np.nan, "b": np.nan}
        )
        with self._export_local(ds, tmp_path, chunk_shape=(250, 300)) as r:
            structure = r.tags(ns="IMAGE_STRUCTURE")
            assert structure["COMPRESSION"] == "DEFLATE"
            assert structure["INTERLEAVE"] == "BAND"
            assert "PREDICTOR" not in structure
            assert r.block_shapes == [(512, 512), (512, 512)]
            assert r.descriptions == ("a", "b")
            np.testing.assert_array_equal(r.read(1), bands["a"])
            np.testing.assert_array_equal(r.read(2), bands["b"])

    def test_integer_grid_round_trips(self, tmp_path):
        """Integer bands are written with the same layout and read back exactly."""
        data = np.arange(700 * 520, dtype=np.int16).reshape(700, 520) % 300
        ds = make_test_dataset(
            bands={"fbfm": data}, shape=(700, 520), nodatas={"fbfm": 32767}
        )
        with self._export_local(ds, tmp_path, chunk_shape=(700, 520)) as r:
            structure = r.tags(ns="IMAGE_STRUCTURE")
            assert structure["COMPRESSION"] == "DEFLATE"
            assert "PREDICTOR" not in structure
            np.testing.assert_array_equal(r.read(1), data)


GEOTIFF_PEAK_RSS_SCRIPT = textwrap.dedent(
    """
    import resource, sys
    from unittest.mock import patch

    from exporter import storage
    from exporter.handlers import grid
    from lib.zarr_utils import load_zarr

    store = sys.argv[1]

    class DiscardBlob:
        def __init__(self, path):
            pass

        def upload_from_filename(self, local, **kwargs):
            pass

    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    with (
        patch.object(grid, "load_grid_zarr", return_value=load_zarr(store)),
        patch.object(storage, "_blob", DiscardBlob),
    ):
        grid.export_geotiff({"id": "e"}, {"grid_id": "g"}, print)
    after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    scale = 1 if sys.platform == "darwin" else 1024  # bytes vs KiB
    print((after - before) * scale)
    """
)


class TestGeotiffPeakMemory:
    """GeoTIFF memory is bounded by the tile, not the grid (#608)."""

    def test_peak_rss_under_150mb_for_4_bands_of_9m_cells(self, tmp_path):
        rng = np.random.default_rng(0)
        shape = (3000, 3000)
        bands = {f"b{i}": rng.random(shape, dtype=np.float32) for i in range(4)}
        ds = make_test_dataset(
            bands=bands, shape=shape, nodatas={k: np.nan for k in bands}
        )
        store = str(tmp_path / "grid.zarr")
        save_zarr(store, ds, chunk_shape=(500, 500))

        result = subprocess.run(
            [
                sys.executable,
                "-c",
                GEOTIFF_PEAK_RSS_SCRIPT,
                store,
            ],
            cwd=EXPORTER_ROOT,
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert result.returncode == 0, result.stderr
        delta_mb = int(result.stdout.strip().splitlines()[-1]) / 2**20
        assert delta_mb < 150, f"export_geotiff grew peak RSS by {delta_mb:.0f} MB"


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
    def test_concurrent_multi_tile_exports_write_to_gcs(
        self, mock_load, export_id, tmp_path
    ):
        """Concurrent multi-tile exports, as on a Cloud Run instance (concurrency 4).

        Writing straight to /vsigs/ failed here: rioxarray reopens the file in
        r+ mode per chunk, which /vsigs/ refuses once another export's
        rasterio.Env has exited.
        """
        rng = np.random.default_rng(0)
        shape = (600, 1100)
        bands = {"a": rng.random(shape, dtype=np.float32)}
        ds = make_test_dataset(bands=bands, shape=shape, nodatas={"a": np.nan})
        zarr_path = str(tmp_path / "grid.zarr")
        save_zarr(zarr_path, ds, chunk_shape=(250, 300))
        mock_load.side_effect = lambda _id: load_zarr(zarr_path)

        def export(i):
            # Nested under export_id so the fixture cleans every object up.
            eid = f"{export_id}/{i}"
            return export_geotiff({"id": eid}, {"grid_id": "g"}, noop_progress)

        with ThreadPoolExecutor(3) as pool:
            gcs_paths = list(pool.map(export, range(3)))

        fs = gcsfs.GCSFileSystem()
        for i, gcs_path in enumerate(gcs_paths):
            local = tmp_path / f"export-{i}.tif"
            with fs.open(gcs_path, "rb") as f:
                local.write_bytes(f.read())
            with rasterio.open(local) as r:
                assert r.tags(ns="IMAGE_STRUCTURE")["COMPRESSION"] == "DEFLATE"
                np.testing.assert_array_equal(r.read(1), bands["a"])

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
