"""
Tests for exporter inventory handlers.

Unit tests mock the inventory load and write to an in-memory filesystem.
Integration tests write to the real GCS bucket.
"""

import io
import json
import os
import subprocess
import sys
import textwrap
import uuid
import zipfile
from pathlib import Path
from unittest.mock import patch

import gcsfs
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from exporter import storage
from exporter.errors import ProcessingError
from exporter.handlers import inventory
from exporter.handlers.inventory import (
    export_csv,
    export_geojson,
    export_geopackage,
    export_parquet,
)

from tests.fake_gcs import UPLOADS, FakeBlob

EXPORTER_ROOT = Path(__file__).resolve().parents[2]
MOCK_UPLOAD = patch("exporter.storage._blob", side_effect=FakeBlob)


def make_test_dataframe(
    n_rows: int = 100,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Create a synthetic DataFrame for testing."""
    rng = np.random.default_rng(42)
    data = {
        "x": rng.uniform(500000, 501000, n_rows),
        "y": rng.uniform(5200000, 5201000, n_rows),
        "fia_species_code": rng.integers(100, 999, n_rows),
        "fia_status_code": rng.integers(1, 3, n_rows),
        "dbh": rng.uniform(5, 80, n_rows),
        "height": rng.uniform(3, 40, n_rows),
        "crown_ratio": rng.uniform(0.1, 0.9, n_rows),
    }
    df = pd.DataFrame(data)
    if columns:
        df = df[columns]
    return df


def noop_progress(message: str, percent: int | None = None):
    pass


# Unit tests


class TestExportParquetUnit:
    """Unit tests for export_parquet handler logic."""

    @MOCK_UPLOAD
    @patch("exporter.handlers.inventory.load_inventory_parquet")
    def test_loads_correct_inventory(self, mock_load, _mock_upload):
        mock_load.return_value = make_test_dataframe()

        export_parquet(
            {"id": "test-export"},
            {"inventory_id": "inv-abc", "name": "parquet"},
            noop_progress,
        )

        mock_load.assert_called_once_with("inv-abc")

    @MOCK_UPLOAD
    @patch("exporter.handlers.inventory.load_inventory_parquet")
    def test_returns_gcs_path_with_zip_extension(self, mock_load, _mock_upload):
        mock_load.return_value = make_test_dataframe()

        result = export_parquet(
            {"id": "export-789"},
            {"inventory_id": "inv-abc", "name": "parquet"},
            noop_progress,
        )

        assert result.endswith("/export-789/export.zip")

    @MOCK_UPLOAD
    @patch("exporter.handlers.inventory.load_inventory_parquet")
    def test_returns_gcs_path_with_name(self, mock_load, _mock_upload):
        mock_load.return_value = make_test_dataframe()

        result = export_parquet(
            {"id": "export-789", "name": "My Trees!"},
            {"inventory_id": "inv-abc", "name": "parquet"},
            noop_progress,
        )

        assert result.endswith("/export-789/My_Trees.zip")

    @MOCK_UPLOAD
    @patch("exporter.handlers.inventory.load_inventory_parquet")
    def test_column_subset_applied(self, mock_load, _mock_upload):
        mock_load.return_value = make_test_dataframe()

        gcs_path = export_parquet(
            {"id": "test-export"},
            {
                "inventory_id": "inv-abc",
                "name": "parquet",
                "columns": ["x", "y", "dbh"],
            },
            noop_progress,
        )

        # Verify the uploaded data only has selected columns
        with zipfile.ZipFile(io.BytesIO(UPLOADS[gcs_path])) as zf:
            inner_name = zf.namelist()[0]
            df = pd.read_parquet(io.BytesIO(zf.read(inner_name)))
        assert list(df.columns) == ["x", "y", "dbh"]

    @patch("exporter.handlers.inventory.load_inventory_parquet")
    def test_missing_column_raises(self, mock_load):
        mock_load.return_value = make_test_dataframe()

        with pytest.raises(ProcessingError) as exc_info:
            export_parquet(
                {"id": "test-export"},
                {
                    "inventory_id": "inv-abc",
                    "name": "parquet",
                    "columns": ["nonexistent"],
                },
                noop_progress,
            )

        assert exc_info.value.code == "COLUMN_NOT_FOUND"

    def test_load_failure_raises(self):
        with patch(
            "exporter.handlers.inventory.load_inventory_parquet",
            side_effect=FileNotFoundError("Not found"),
        ):
            with pytest.raises(ProcessingError) as exc_info:
                export_parquet(
                    {"id": "test-export"},
                    {"inventory_id": "inv-missing", "name": "parquet"},
                    noop_progress,
                )

            assert exc_info.value.code == "INVENTORY_LOAD_ERROR"

    @MOCK_UPLOAD
    @patch("exporter.handlers.inventory.load_inventory_parquet")
    def test_progress_steps(self, mock_load, _mock_upload):
        mock_load.return_value = make_test_dataframe()
        calls = []

        export_parquet(
            {"id": "test-export"},
            {"inventory_id": "inv-abc", "name": "parquet"},
            lambda msg, pct=None: calls.append((msg, pct)),
        )

        assert calls == [
            ("Loading inventory data...", 20),
            ("Writing Parquet...", 60),
        ]

    @MOCK_UPLOAD
    @patch("exporter.handlers.inventory.load_inventory_parquet")
    def test_progress_with_column_subset(self, mock_load, _mock_upload):
        mock_load.return_value = make_test_dataframe()
        calls = []

        export_parquet(
            {"id": "test-export"},
            {"inventory_id": "inv-abc", "name": "parquet", "columns": ["x", "y"]},
            lambda msg, pct=None: calls.append((msg, pct)),
        )

        assert calls == [
            ("Loading inventory data...", 20),
            ("Selecting columns...", 40),
            ("Writing Parquet...", 60),
        ]


class TestExportCsvUnit:
    """Unit tests for export_csv handler logic."""

    @MOCK_UPLOAD
    @patch("exporter.handlers.inventory.load_inventory_parquet")
    def test_returns_gcs_path_with_csv_extension(self, mock_load, _mock_upload):
        mock_load.return_value = make_test_dataframe()

        result = export_csv(
            {"id": "export-789"},
            {"inventory_id": "inv-abc", "name": "csv"},
            noop_progress,
        )

        assert result.endswith("/export-789/export.csv")

    @MOCK_UPLOAD
    @patch("exporter.handlers.inventory.load_inventory_parquet")
    def test_column_subset_applied(self, mock_load, _mock_upload):
        mock_load.return_value = make_test_dataframe()

        gcs_path = export_csv(
            {"id": "test-export"},
            {"inventory_id": "inv-abc", "name": "csv", "columns": ["dbh", "height"]},
            noop_progress,
        )

        df = pd.read_csv(io.BytesIO(UPLOADS[gcs_path]))
        assert list(df.columns) == ["dbh", "height"]

    @patch("exporter.handlers.inventory.load_inventory_parquet")
    def test_missing_column_raises(self, mock_load):
        mock_load.return_value = make_test_dataframe()

        with pytest.raises(ProcessingError) as exc_info:
            export_csv(
                {"id": "test-export"},
                {"inventory_id": "inv-abc", "name": "csv", "columns": ["bad_col"]},
                noop_progress,
            )

        assert exc_info.value.code == "COLUMN_NOT_FOUND"

    @MOCK_UPLOAD
    @patch("exporter.handlers.inventory.load_inventory_parquet")
    def test_progress_steps(self, mock_load, _mock_upload):
        mock_load.return_value = make_test_dataframe()
        calls = []

        export_csv(
            {"id": "test-export"},
            {"inventory_id": "inv-abc", "name": "csv"},
            lambda msg, pct=None: calls.append((msg, pct)),
        )

        assert calls == [
            ("Loading inventory data...", 20),
            ("Writing CSV...", 70),
        ]


class TestExportGeojsonUnit:
    """Unit tests for export_geojson handler logic."""

    @MOCK_UPLOAD
    @patch("exporter.handlers.inventory.load_inventory_parquet")
    def test_returns_gcs_path_with_geojson_extension(self, mock_load, _mock_upload):
        mock_load.return_value = make_test_dataframe()

        result = export_geojson(
            {"id": "export-789"},
            {"inventory_id": "inv-abc", "name": "geojson", "crs": "EPSG:32611"},
            noop_progress,
        )

        assert result.endswith("/export-789/export.geojson")

    @MOCK_UPLOAD
    @patch("exporter.handlers.inventory.load_inventory_parquet")
    def test_missing_xy_raises(self, mock_load, _mock_upload):
        mock_load.return_value = make_test_dataframe(columns=["dbh", "height"])

        with pytest.raises(ProcessingError) as exc_info:
            export_geojson(
                {"id": "test-export"},
                {
                    "inventory_id": "inv-abc",
                    "name": "geojson",
                    "columns": ["dbh", "height"],
                },
                noop_progress,
            )

        assert exc_info.value.code == "MISSING_COORDINATES"

    @MOCK_UPLOAD
    @patch("exporter.handlers.inventory.load_inventory_parquet")
    def test_column_subset_with_xy(self, mock_load, _mock_upload):
        mock_load.return_value = make_test_dataframe()

        gcs_path = export_geojson(
            {"id": "test-export"},
            {
                "inventory_id": "inv-abc",
                "name": "geojson",
                "crs": "EPSG:32611",
                "columns": ["x", "y", "dbh"],
            },
            noop_progress,
        )

        geojson = json.loads(UPLOADS[gcs_path].decode("utf-8"))
        assert geojson["type"] == "FeatureCollection"
        # Should have dbh in properties (x,y moved to geometry)
        props = geojson["features"][0]["properties"]
        assert "dbh" in props
        assert "x" not in props
        assert "y" not in props

    @MOCK_UPLOAD
    @patch("exporter.handlers.inventory.load_inventory_parquet")
    def test_progress_steps(self, mock_load, _mock_upload):
        mock_load.return_value = make_test_dataframe()
        calls = []

        export_geojson(
            {"id": "test-export"},
            {"inventory_id": "inv-abc", "name": "geojson", "crs": "EPSG:32611"},
            lambda msg, pct=None: calls.append((msg, pct)),
        )

        assert calls == [
            ("Loading inventory data...", 20),
            ("Writing GeoJSON...", 80),
        ]


class TestExportGeopackageUnit:
    """Unit tests for export_geopackage handler logic."""

    @MOCK_UPLOAD
    @patch("exporter.handlers.inventory.load_inventory_parquet")
    def test_returns_gcs_path_with_gpkg_extension(self, mock_load, _mock_upload):
        mock_load.return_value = make_test_dataframe()

        result = export_geopackage(
            {"id": "export-789"},
            {"inventory_id": "inv-abc", "name": "geopackage", "crs": "EPSG:32611"},
            noop_progress,
        )

        assert result.endswith("/export-789/export.gpkg")

    @MOCK_UPLOAD
    @patch("exporter.handlers.inventory.load_inventory_parquet")
    def test_missing_xy_raises(self, mock_load, _mock_upload):
        mock_load.return_value = make_test_dataframe(columns=["dbh", "height"])

        with pytest.raises(ProcessingError) as exc_info:
            export_geopackage(
                {"id": "test-export"},
                {
                    "inventory_id": "inv-abc",
                    "name": "geopackage",
                    "columns": ["dbh", "height"],
                },
                noop_progress,
            )

        assert exc_info.value.code == "MISSING_COORDINATES"

    @MOCK_UPLOAD
    @patch("exporter.handlers.inventory.load_inventory_parquet")
    def test_progress_steps(self, mock_load, _mock_upload):
        mock_load.return_value = make_test_dataframe()
        calls = []

        export_geopackage(
            {"id": "test-export"},
            {"inventory_id": "inv-abc", "name": "geopackage", "crs": "EPSG:32611"},
            lambda msg, pct=None: calls.append((msg, pct)),
        )

        assert calls == [
            ("Loading inventory data...", 20),
            ("Writing GeoPackage...", 80),
        ]


def make_nullable_dataframe(n_rows: int = 23) -> pd.DataFrame:
    """Synthetic inventory with NaN floats and NA integers, non-zero index."""
    df = make_test_dataframe(n_rows)
    df["fia_status_code"] = pd.array(df["fia_status_code"], dtype="Int64")
    df.loc[df.index[::3], "fia_status_code"] = pd.NA
    df.loc[df.index[::4], "crown_ratio"] = np.nan
    df.index = df.index + 100
    return df


class TestBatchedWritesUnit:
    """Batched GeoJSON/GeoPackage writes match the whole-frame output (#608)."""

    @pytest.mark.parametrize("crs", ["EPSG:32611", None])
    @pytest.mark.parametrize("n_rows", [0, 1, 7, 23])
    @patch("exporter.handlers.inventory.WRITE_BATCH_ROWS", 7)
    @MOCK_UPLOAD
    @patch("exporter.handlers.inventory.load_inventory_parquet")
    def test_geojson_matches_to_json(self, mock_load, _mock_upload, n_rows, crs):
        df = make_nullable_dataframe().iloc[:n_rows]
        mock_load.return_value = df

        gcs_path = export_geojson(
            {"id": "test-export"},
            {"inventory_id": "inv-abc", "crs": crs},
            noop_progress,
        )

        expected = gpd.GeoDataFrame(
            df.drop(columns=["x", "y"]),
            geometry=gpd.points_from_xy(df["x"], df["y"]),
            crs=crs,
        ).to_json()
        assert UPLOADS[gcs_path].decode("utf-8") == expected

    @pytest.mark.parametrize("n_rows", [0, 1, 7, 23])
    @patch("exporter.handlers.inventory.WRITE_BATCH_ROWS", 7)
    @MOCK_UPLOAD
    @patch("exporter.handlers.inventory.load_inventory_parquet")
    def test_geopackage_round_trips(self, mock_load, _mock_upload, tmp_path, n_rows):
        df = make_nullable_dataframe().iloc[:n_rows]
        mock_load.return_value = df

        gcs_path = export_geopackage(
            {"id": "test-export"},
            {"inventory_id": "inv-abc", "crs": "EPSG:32611"},
            noop_progress,
        )

        local = tmp_path / "out.gpkg"
        local.write_bytes(UPLOADS[gcs_path])
        gdf = gpd.read_file(local)
        assert gdf.crs == "EPSG:32611"
        assert len(gdf) == n_rows
        np.testing.assert_array_equal(gdf.geometry.x, df["x"])
        np.testing.assert_array_equal(gdf.geometry.y, df["y"])
        for col in ["fia_status_code", "crown_ratio", "dbh"]:
            assert gdf[col].isna().tolist() == df[col].isna().tolist()


class TestFailedWriteUnit:
    """A failure partway through a streamed write uploads nothing."""

    @patch("exporter.handlers.inventory.WRITE_BATCH_ROWS", 7)
    @MOCK_UPLOAD
    @patch("exporter.handlers.inventory.load_inventory_parquet")
    def test_geojson_failure_uploads_nothing(self, mock_load, _mock_upload):
        mock_load.return_value = make_test_dataframe(23)
        real = inventory._to_geodataframe
        calls = []

        def fail_on_second_batch(df, crs):
            calls.append(1)
            if len(calls) == 2:
                raise RuntimeError("boom")
            return real(df, crs)

        with (
            patch.object(inventory, "_to_geodataframe", fail_on_second_batch),
            pytest.raises(RuntimeError, match="boom"),
        ):
            export_geojson(
                {"id": "failed-export"}, {"inventory_id": "i"}, noop_progress
            )

        assert not [p for p in UPLOADS if "/failed-export/" in p]


PEAK_RSS_SCRIPT = textwrap.dedent(
    """
    import resource, sys
    from unittest.mock import patch

    import io

    import numpy as np, pandas as pd

    from exporter import storage
    from exporter.handlers import inventory

    handler, rows = getattr(inventory, sys.argv[1]), int(sys.argv[2])
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "x": rng.uniform(5e5, 5.04e5, rows),
        "y": rng.uniform(5.2e6, 5.204e6, rows),
        "fia_species_code": pd.array(rng.integers(100, 999, rows), dtype="Int64"),
        "fia_status_code": pd.array(rng.integers(1, 3, rows), dtype="Int64"),
        "dbh": rng.uniform(5, 80, rows),
        "height": rng.uniform(3, 40, rows),
        "crown_ratio": rng.uniform(0.1, 0.9, rows),
    })

    class Discard(io.RawIOBase):
        # Write-only and unseekable, like a gcsfs upload stream.
        def writable(self):
            return True

        def write(self, b):
            return len(b)

    class DiscardBlob:
        def __init__(self, path):
            pass

        def open(self, mode, **kwargs):
            return Discard()

        def upload_from_filename(self, local, **kwargs):
            pass

    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    with (
        patch.object(inventory, "load_inventory_parquet", return_value=df),
        patch.object(storage, "_blob", DiscardBlob),
    ):
        handler({"id": "e"}, {"inventory_id": "i", "crs": "EPSG:32611"}, print)
    after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    scale = 1 if sys.platform == "darwin" else 1024  # bytes vs KiB
    print((after - before) * scale)
    """
)


class TestPeakMemory:
    """Handler memory is bounded by the write batch, not the row count (#608)."""

    @pytest.mark.parametrize(
        "handler",
        ["export_geojson", "export_geopackage", "export_csv", "export_parquet"],
    )
    def test_peak_rss_under_400mb_at_1m_rows(self, handler):
        result = subprocess.run(
            [sys.executable, "-c", PEAK_RSS_SCRIPT, handler, "1000000"],
            cwd=EXPORTER_ROOT,
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert result.returncode == 0, result.stderr
        delta_mb = int(result.stdout.strip().splitlines()[-1]) / 2**20
        assert delta_mb < 400, f"{handler} grew peak RSS by {delta_mb:.0f} MB"


# Integration tests


class TestExportParquetIntegration:
    """Integration tests that write to the real GCS bucket."""

    BUCKET = os.environ["EXPORTS_BUCKET"]

    @pytest.fixture
    def export_id(self):
        eid = f"test-{uuid.uuid4().hex[:12]}"
        yield eid
        fs = gcsfs.GCSFileSystem()
        bucket = self.BUCKET
        if bucket.startswith("gs://"):
            bucket = bucket[5:]
        try:
            fs.rm(f"{bucket}/{eid}", recursive=True)
        except FileNotFoundError:
            pass

    @patch("exporter.handlers.inventory.load_inventory_parquet")
    def test_writes_valid_zip_parquet_to_gcs(self, mock_load, export_id):
        mock_load.return_value = make_test_dataframe()

        gcs_path = export_parquet(
            {"id": export_id},
            {"inventory_id": "inv-abc", "name": "parquet"},
            noop_progress,
        )

        assert gcs_path.endswith(".zip")

        fs = gcsfs.GCSFileSystem()
        with fs.open(gcs_path, "rb") as f:
            zip_bytes = f.read()

        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            names = zf.namelist()
            assert len(names) == 1
            assert names[0].endswith(".parquet")
            df = pd.read_parquet(io.BytesIO(zf.read(names[0])))
            assert len(df) == 100
            assert "x" in df.columns


class TestExportCsvIntegration:
    """Integration tests for CSV export."""

    BUCKET = os.environ["EXPORTS_BUCKET"]

    @pytest.fixture
    def export_id(self):
        eid = f"test-{uuid.uuid4().hex[:12]}"
        yield eid
        fs = gcsfs.GCSFileSystem()
        bucket = self.BUCKET
        if bucket.startswith("gs://"):
            bucket = bucket[5:]
        try:
            fs.rm(f"{bucket}/{eid}", recursive=True)
        except FileNotFoundError:
            pass

    @patch("exporter.handlers.inventory.load_inventory_parquet")
    def test_writes_valid_csv_to_gcs(self, mock_load, export_id):
        mock_load.return_value = make_test_dataframe()

        gcs_path = export_csv(
            {"id": export_id},
            {"inventory_id": "inv-abc", "name": "csv"},
            noop_progress,
        )

        assert gcs_path.endswith(".csv")

        fs = gcsfs.GCSFileSystem()
        with fs.open(gcs_path, "rb") as f:
            df = pd.read_csv(f)
        assert len(df) == 100
        assert "x" in df.columns


class TestExportGeojsonIntegration:
    """Integration tests for GeoJSON export."""

    BUCKET = os.environ["EXPORTS_BUCKET"]

    @pytest.fixture
    def export_id(self):
        eid = f"test-{uuid.uuid4().hex[:12]}"
        yield eid
        fs = gcsfs.GCSFileSystem()
        bucket = self.BUCKET
        if bucket.startswith("gs://"):
            bucket = bucket[5:]
        try:
            fs.rm(f"{bucket}/{eid}", recursive=True)
        except FileNotFoundError:
            pass

    @patch("exporter.handlers.inventory.load_inventory_parquet")
    def test_writes_valid_geojson_to_gcs(self, mock_load, export_id):
        mock_load.return_value = make_test_dataframe()

        gcs_path = export_geojson(
            {"id": export_id},
            {"inventory_id": "inv-abc", "name": "geojson", "crs": "EPSG:32611"},
            noop_progress,
        )

        assert gcs_path.endswith(".geojson")

        fs = gcsfs.GCSFileSystem()
        with fs.open(gcs_path, "rb") as f:
            geojson = json.loads(f.read().decode("utf-8"))
        assert geojson["type"] == "FeatureCollection"
        assert len(geojson["features"]) == 100

    @patch("exporter.handlers.inventory.WRITE_BATCH_ROWS", 5_000)
    @patch("exporter.handlers.inventory.load_inventory_parquet")
    def test_failure_mid_upload_leaves_no_object(self, mock_load, export_id):
        """Fail after a resumable chunk has gone to GCS; the upload is cancelled."""
        mock_load.return_value = make_test_dataframe(20_000)
        real_blob, real_to_gdf = storage._blob, inventory._to_geodataframe
        calls = []

        def small_chunks(gcs_path):
            blob = real_blob(gcs_path)
            blob.chunk_size = 256 * 1024  # GCS minimum, so a chunk uploads early
            return blob

        def fail_on_last_batch(df, crs):
            calls.append(1)
            if len(calls) == 4:
                raise RuntimeError("boom")
            return real_to_gdf(df, crs)

        with (
            patch.object(storage, "_blob", small_chunks),
            patch.object(inventory, "_to_geodataframe", fail_on_last_batch),
            pytest.raises(RuntimeError, match="boom"),
        ):
            export_geojson(
                {"id": export_id, "name": "failed"},
                {"inventory_id": "inv-abc", "crs": "EPSG:32611"},
                noop_progress,
            )

        bucket = self.BUCKET.removeprefix("gs://")
        fs = gcsfs.GCSFileSystem()
        assert not fs.exists(f"{bucket}/{export_id}/failed.geojson")


class TestExportGeopackageIntegration:
    """Integration tests for GeoPackage export."""

    BUCKET = os.environ["EXPORTS_BUCKET"]

    @pytest.fixture
    def export_id(self):
        eid = f"test-{uuid.uuid4().hex[:12]}"
        yield eid
        fs = gcsfs.GCSFileSystem()
        bucket = self.BUCKET
        if bucket.startswith("gs://"):
            bucket = bucket[5:]
        try:
            fs.rm(f"{bucket}/{eid}", recursive=True)
        except FileNotFoundError:
            pass

    @patch("exporter.handlers.inventory.load_inventory_parquet")
    def test_writes_valid_geopackage_to_gcs(self, mock_load, export_id):
        import geopandas as gpd

        mock_load.return_value = make_test_dataframe()

        gcs_path = export_geopackage(
            {"id": export_id},
            {"inventory_id": "inv-abc", "name": "geopackage", "crs": "EPSG:32611"},
            noop_progress,
        )

        assert gcs_path.endswith(".gpkg")

        # Download and read with geopandas
        fs = gcsfs.GCSFileSystem()
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "output.gpkg")
            with fs.open(gcs_path, "rb") as f, open(path, "wb") as tmp:
                tmp.write(f.read())
            gdf = gpd.read_file(path)
        assert len(gdf) == 100
        assert gdf.crs is not None
