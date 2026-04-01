"""
Tests for exporter inventory handlers.

Unit tests mock storage to test handler logic.
Integration tests write to the real GCS bucket.
"""

import io
import json
import os
import uuid
import zipfile
from unittest.mock import patch

import gcsfs
import numpy as np
import pandas as pd
import pytest
from exporter.errors import ProcessingError
from exporter.handlers.inventory import (
    export_csv,
    export_geojson,
    export_geopackage,
    export_parquet,
)

MOCK_UPLOAD = patch("exporter.handlers.inventory._upload_bytes")


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
    def test_column_subset_applied(self, mock_load, mock_upload):
        mock_load.return_value = make_test_dataframe()

        export_parquet(
            {"id": "test-export"},
            {
                "inventory_id": "inv-abc",
                "name": "parquet",
                "columns": ["x", "y", "dbh"],
            },
            noop_progress,
        )

        # Verify the uploaded data only has selected columns
        uploaded_bytes = mock_upload.call_args[0][0]
        with zipfile.ZipFile(io.BytesIO(uploaded_bytes)) as zf:
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
            ("Creating ZIP archive...", 80),
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
            ("Creating ZIP archive...", 80),
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
    def test_column_subset_applied(self, mock_load, mock_upload):
        mock_load.return_value = make_test_dataframe()

        export_csv(
            {"id": "test-export"},
            {"inventory_id": "inv-abc", "name": "csv", "columns": ["dbh", "height"]},
            noop_progress,
        )

        uploaded_bytes = mock_upload.call_args[0][0]
        df = pd.read_csv(io.BytesIO(uploaded_bytes))
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
    def test_column_subset_with_xy(self, mock_load, mock_upload):
        mock_load.return_value = make_test_dataframe()

        export_geojson(
            {"id": "test-export"},
            {
                "inventory_id": "inv-abc",
                "name": "geojson",
                "crs": "EPSG:32611",
                "columns": ["x", "y", "dbh"],
            },
            noop_progress,
        )

        uploaded_bytes = mock_upload.call_args[0][0]
        geojson = json.loads(uploaded_bytes.decode("utf-8"))
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
            ("Converting to GeoDataFrame...", 60),
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
            ("Converting to GeoDataFrame...", 60),
            ("Writing GeoPackage...", 80),
        ]


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
