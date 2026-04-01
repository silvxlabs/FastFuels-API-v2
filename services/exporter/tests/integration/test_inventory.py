"""
Integration tests for inventory export processing.

Tests the full exporter pipeline: load inventory parquet -> convert -> export file.
Requires static test data in GCS (created by services/api/tests/e2e/).
"""

import io
import json
import os
import tempfile
import zipfile

import gcsfs
import pandas as pd
import pytest
from exporter.filename import sanitize_filename

from lib.config import EXPORTS_BUCKET

STATIC_INVENTORY = "static-test-blue-mtn-pim-inventory"


class TestParquetExport:
    @pytest.mark.parametrize("source_inventory", [STATIC_INVENTORY], indirect=True)
    def test_all_columns(self, inventory_exporter_runner, source_inventory):
        """Export all columns from inventory as parquet."""
        export = inventory_exporter_runner(source_inventory, "parquet.json")

        filename = sanitize_filename(export.get("name", ""), ".zip")
        gcs_path = f"gs://{EXPORTS_BUCKET}/{export['id']}/{filename}"
        fs = gcsfs.GCSFileSystem()
        with fs.open(gcs_path, "rb") as f:
            zip_bytes = f.read()

        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            names = zf.namelist()
            assert len(names) == 1
            assert names[0].endswith(".parquet")
            df = pd.read_parquet(io.BytesIO(zf.read(names[0])))

        assert len(df) > 0
        assert "x" in df.columns
        assert "y" in df.columns
        assert "dbh" in df.columns

    @pytest.mark.parametrize("source_inventory", [STATIC_INVENTORY], indirect=True)
    def test_column_subset(self, inventory_exporter_runner, source_inventory):
        """Export only selected columns from inventory as parquet."""
        export = inventory_exporter_runner(
            source_inventory,
            "parquet.json",
            source_overrides={"columns": ["x", "y", "dbh", "height"]},
        )

        filename = sanitize_filename(export.get("name", ""), ".zip")
        gcs_path = f"gs://{EXPORTS_BUCKET}/{export['id']}/{filename}"
        fs = gcsfs.GCSFileSystem()
        with fs.open(gcs_path, "rb") as f:
            zip_bytes = f.read()

        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            names = zf.namelist()
            df = pd.read_parquet(io.BytesIO(zf.read(names[0])))

        assert list(df.columns) == ["x", "y", "dbh", "height"]


class TestCsvExport:
    @pytest.mark.parametrize("source_inventory", [STATIC_INVENTORY], indirect=True)
    def test_all_columns(self, inventory_exporter_runner, source_inventory):
        """Export all columns from inventory as CSV."""
        export = inventory_exporter_runner(source_inventory, "csv.json")

        filename = sanitize_filename(export.get("name", ""), ".csv")
        gcs_path = f"gs://{EXPORTS_BUCKET}/{export['id']}/{filename}"
        fs = gcsfs.GCSFileSystem()
        with fs.open(gcs_path, "rb") as f:
            df = pd.read_csv(f)

        assert len(df) > 0
        assert "x" in df.columns
        assert "y" in df.columns
        assert "dbh" in df.columns

    @pytest.mark.parametrize("source_inventory", [STATIC_INVENTORY], indirect=True)
    def test_column_subset(self, inventory_exporter_runner, source_inventory):
        """Export only selected columns from inventory as CSV."""
        export = inventory_exporter_runner(
            source_inventory,
            "csv.json",
            source_overrides={"columns": ["x", "y", "dbh"]},
        )

        filename = sanitize_filename(export.get("name", ""), ".csv")
        gcs_path = f"gs://{EXPORTS_BUCKET}/{export['id']}/{filename}"
        fs = gcsfs.GCSFileSystem()
        with fs.open(gcs_path, "rb") as f:
            df = pd.read_csv(f)

        assert list(df.columns) == ["x", "y", "dbh"]


class TestGeojsonExport:
    @pytest.mark.parametrize("source_inventory", [STATIC_INVENTORY], indirect=True)
    def test_all_columns(self, inventory_exporter_runner, source_inventory):
        """Export inventory as GeoJSON with all columns."""
        export = inventory_exporter_runner(source_inventory, "geojson.json")

        filename = sanitize_filename(export.get("name", ""), ".geojson")
        gcs_path = f"gs://{EXPORTS_BUCKET}/{export['id']}/{filename}"
        fs = gcsfs.GCSFileSystem()
        with fs.open(gcs_path, "rb") as f:
            geojson = json.loads(f.read().decode("utf-8"))

        assert geojson["type"] == "FeatureCollection"
        assert len(geojson["features"]) > 0

        # CRS should be present
        assert "crs" in geojson
        assert "EPSG" in str(geojson["crs"])

        # x/y should be in geometry, not properties
        props = geojson["features"][0]["properties"]
        assert "x" not in props
        assert "y" not in props
        assert "dbh" in props

        # Geometry should be Point
        geom = geojson["features"][0]["geometry"]
        assert geom["type"] == "Point"


class TestGeopackageExport:
    @pytest.mark.parametrize("source_inventory", [STATIC_INVENTORY], indirect=True)
    def test_all_columns(self, inventory_exporter_runner, source_inventory):
        """Export inventory as GeoPackage with all columns."""
        import geopandas as gpd

        export = inventory_exporter_runner(source_inventory, "geopackage.json")

        filename = sanitize_filename(export.get("name", ""), ".gpkg")
        gcs_path = f"gs://{EXPORTS_BUCKET}/{export['id']}/{filename}"

        fs = gcsfs.GCSFileSystem()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "output.gpkg")
            with fs.open(gcs_path, "rb") as f, open(path, "wb") as tmp:
                tmp.write(f.read())
            gdf = gpd.read_file(path)

        assert len(gdf) > 0
        assert gdf.crs is not None
        assert "32611" in str(gdf.crs)
        assert gdf.geometry is not None
        assert "dbh" in gdf.columns
