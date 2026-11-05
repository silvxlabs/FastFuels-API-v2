"""
Every export format carries the export's ``attributions`` (#775).

Each test runs a handler with storage mocked, reads the produced file back,
and checks the attribution metadata. An export without attributions writes
no attribution metadata at all.
"""

import io
import json
import shutil
import zipfile
from pathlib import Path
from unittest.mock import patch

import geopandas as gpd
import numpy as np
import pyarrow.parquet as pq
import pyogrio
import pytest
import rasterio
import xarray as xr
import zarr
from exporter.attribution import attribution_metadata, attribution_notice
from exporter.handlers.grid import export_geotiff, export_zarr
from exporter.handlers.inventory import (
    export_geojson,
    export_geopackage,
    export_parquet,
)
from exporter.handlers.landscape import export_landscape
from exporter.handlers.netcdf import export_netcdf
from exporter.handlers.quicfire import export_quicfire
from rioxarray.raster_array import RasterArray

from tests.handlers.test_geotiff import make_test_dataset
from tests.handlers.test_inventory import make_test_dataframe
from tests.handlers.test_landscape import _build_grids as _landscape_grids
from tests.handlers.test_landscape import _build_source as _landscape_source
from tests.handlers.test_netcdf import fake_grid_snapshot
from tests.handlers.test_quicfire import _build_source as _quicfire_source
from tests.handlers.test_quicfire import (
    _fake_domain_doc,
    _make_2d_dataset,
    _make_3d_dataset,
)

OSM = {
    "dataset": "OpenStreetMap",
    "license_name": "ODbL-1.0",
    "license_url": "https://opendatacommons.org/licenses/odbl/1-0/",
    "citation": "© OpenStreetMap contributors",
    "access_url": "https://www.openstreetmap.org/copyright",
    "accessed_on": "2026-11-05",
}
META = {
    "dataset": "Meta CHM",
    "license_name": "CC-BY-4.0",
    "license_url": "https://creativecommons.org/licenses/by/4.0/",
    "citation": "Meta and World Resources Institute",
    "access_url": "https://registry.opendata.aws/dataforgood-fb-forests/",
    "accessed_on": "2026-10-01",
}
ATTRIBUTIONS = [META, OSM]

# Parametrize every format over: with attributions, and without (empty list
# or the key absent, as on exports created before the field existed).
CASES = [
    pytest.param({"attributions": ATTRIBUTIONS}, ATTRIBUTIONS, id="attributions"),
    pytest.param({"attributions": []}, [], id="empty"),
    pytest.param({}, [], id="absent"),
]


def noop_progress(message: str, percent: int | None = None) -> None:
    pass


def assert_metadata(metadata: dict, expected: list[dict]) -> None:
    """``metadata`` holds the JSON list and the notice, or neither."""
    if expected:
        assert json.loads(metadata["attributions"]) == expected
        assert metadata["license"] == attribution_notice(expected)
    else:
        assert "attributions" not in metadata
        assert "license" not in metadata


class TestAttributionNotice:
    def test_lists_every_field_of_every_entry(self):
        notice = attribution_notice(ATTRIBUTIONS)
        for entry in ATTRIBUTIONS:
            for value in entry.values():
                assert value in notice
        assert notice.index("Meta CHM") < notice.index("OpenStreetMap")

    def test_metadata_empty_for_no_attributions(self):
        assert attribution_metadata([]) == {}

    def test_metadata_keys(self):
        metadata = attribution_metadata([OSM])
        assert json.loads(metadata["attributions"]) == [OSM]
        assert "© OpenStreetMap contributors" in metadata["license"]
        assert (
            "ODbL-1.0 (https://opendatacommons.org/licenses/odbl/1-0/)"
            in (metadata["license"])
        )


class TestGeotiff:
    @pytest.mark.parametrize("fields, expected", CASES)
    @pytest.mark.parametrize("nodatas", [None, {"fbfm": -1}], ids=["uniform", "stack"])
    def test_tags(self, tmp_path, fields, expected, nodatas):
        original = RasterArray.to_raster
        local_path = tmp_path / "export.tif"

        def to_local(self, raster_path, **kwargs):
            return original(self, str(local_path), **kwargs)

        with (
            patch(
                "exporter.handlers.grid.load_grid_zarr",
                return_value=make_test_dataset(nodatas=nodatas),
            ),
            patch.object(RasterArray, "to_raster", to_local),
        ):
            export_geotiff(
                {"id": "e1", **fields},
                {"grid_id": "g", "name": "geotiff"},
                noop_progress,
            )

        with rasterio.open(local_path) as src:
            assert_metadata(src.tags(), expected)


class TestZarr:
    @pytest.mark.parametrize("fields, expected", CASES)
    @patch("google.cloud.storage.Client")
    def test_global_attrs(self, mock_client, tmp_path, fields, expected):
        captured = tmp_path / "captured.zip"
        blob = mock_client.return_value.bucket.return_value.blob.return_value
        blob.upload_from_filename.side_effect = lambda p: shutil.copy(p, captured)

        with patch(
            "exporter.handlers.grid.load_grid_zarr", return_value=make_test_dataset()
        ):
            export_zarr(
                {"id": "e1", **fields}, {"grid_id": "g", "name": "zarr"}, noop_progress
            )

        with zipfile.ZipFile(captured) as zf:
            zf.extractall(tmp_path / "extracted")
        group = zarr.open_group(str(tmp_path / "extracted" / "export.zarr"), mode="r")
        assert_metadata(dict(group.attrs), expected)


class TestNetcdf:
    @pytest.mark.parametrize("fields, expected", CASES)
    @patch("exporter.handlers.netcdf._upload_file")
    @patch("exporter.handlers.netcdf.get_document")
    def test_global_attrs(self, mock_get_doc, mock_upload, tmp_path, fields, expected):
        captured = tmp_path / "captured.nc"
        mock_upload.side_effect = lambda local, _gcs: shutil.copy(local, captured)
        mock_get_doc.return_value = (None, fake_grid_snapshot([]))

        with patch(
            "exporter.handlers.grid.load_grid_zarr", return_value=make_test_dataset()
        ):
            export_netcdf(
                {"id": "e1", **fields},
                {"grid_id": "g", "name": "netcdf"},
                noop_progress,
            )

        with xr.open_dataset(captured) as ds:
            assert ds.attrs["Conventions"] == "CF-1.13"
            assert_metadata(ds.attrs, expected)


class TestInventory:
    @pytest.fixture
    def uploaded(self):
        """Patch the inventory load and upload; collect the uploaded bytes."""
        files: list[bytes] = []
        with (
            patch(
                "exporter.handlers.inventory.load_inventory_parquet",
                return_value=make_test_dataframe(),
            ),
            patch(
                "exporter.handlers.inventory._upload_bytes",
                side_effect=lambda data, _path: files.append(data),
            ),
        ):
            yield files

    @pytest.mark.parametrize("fields, expected", CASES)
    def test_parquet_key_value_metadata(self, uploaded, fields, expected):
        export_parquet(
            {"id": "e1", **fields},
            {"inventory_id": "i", "name": "parquet"},
            noop_progress,
        )

        with zipfile.ZipFile(io.BytesIO(uploaded[0])) as zf:
            parquet_bytes = zf.read(zf.namelist()[0])
        schema = pq.read_schema(io.BytesIO(parquet_bytes))
        metadata = {k.decode(): v.decode() for k, v in schema.metadata.items()}
        # The pandas schema metadata is kept alongside the attributions.
        assert "pandas" in metadata
        assert_metadata(metadata, expected)
        table = pq.read_table(io.BytesIO(parquet_bytes))
        assert table.num_rows == 100

    @pytest.mark.parametrize("fields, expected", CASES)
    def test_geojson_foreign_member(self, uploaded, fields, expected):
        export_geojson(
            {"id": "e1", **fields},
            {"inventory_id": "i", "name": "geojson", "crs": "EPSG:32611"},
            noop_progress,
        )

        collection = json.loads(uploaded[0])
        assert collection["type"] == "FeatureCollection"
        assert len(collection["features"]) == 100
        if expected:
            assert collection["attributions"] == expected
        else:
            assert "attributions" not in collection

    @pytest.mark.parametrize("fields, expected", CASES)
    def test_geopackage_metadata(self, uploaded, tmp_path, fields, expected):
        export_geopackage(
            {"id": "e1", **fields},
            {"inventory_id": "i", "name": "geopackage", "crs": "EPSG:32611"},
            noop_progress,
        )

        path = tmp_path / "export.gpkg"
        path.write_bytes(uploaded[0])
        info = pyogrio.read_info(path)
        assert_metadata(info["layer_metadata"] or {}, expected)
        assert len(gpd.read_file(path)) == 100


class TestQuicfire:
    @pytest.mark.parametrize("fields, expected", CASES)
    def test_attribution_file(self, tmp_path, fields, expected):
        nz, ny, nx = 5, 3, 4
        tree = _make_3d_dataset(
            {
                "bulk_density.foliage.live": np.full((nz, ny, nx), 0.2, np.float32),
                "fuel_moisture.live": np.full((nz, ny, nx), 100.0, np.float32),
            }
        )
        surface = _make_2d_dataset(
            {
                band: np.full((ny, nx), 0.5, np.float32)
                for band in ("fuel_load.1hr", "fuel_depth", "fuel_moisture.1hr")
            }
        )
        grids = {"tree": tree, "uniform": surface}
        captured = tmp_path / "captured.zip"

        def fake_upload(zip_path, export):
            shutil.copy(zip_path, captured)
            return "gs://exports/e1/captured.zip"

        class _Snapshot:
            def to_dict(self):
                return _fake_domain_doc()

        with (
            patch(
                "exporter.handlers.quicfire.load_grid_zarr",
                side_effect=lambda gid: grids[gid],
            ),
            patch("exporter.handlers.quicfire._upload_zip", side_effect=fake_upload),
            patch(
                "exporter.handlers.quicfire.get_document",
                return_value=(None, _Snapshot()),
            ),
        ):
            export_quicfire({"id": "e1", **fields}, _quicfire_source(), noop_progress)

        with zipfile.ZipFile(captured) as zf:
            names = zf.namelist()
            if expected:
                text = zf.read("ATTRIBUTION.txt").decode("utf-8")
                assert text == attribution_notice(expected) + "\n"
            else:
                assert "ATTRIBUTION.txt" not in names


class TestLandscape:
    @pytest.mark.parametrize("fields, expected", CASES)
    def test_tags(self, tmp_path, fields, expected):
        captured = tmp_path / "captured.tif"
        grids = _landscape_grids()

        def fake_upload(tif_path, export):
            captured.write_bytes(Path(tif_path).read_bytes())
            return "gs://exports/e1/captured.tif"

        with (
            patch(
                "exporter.handlers.landscape.load_grid_zarr",
                side_effect=lambda gid: grids[gid],
            ),
            patch("exporter.handlers.landscape._upload_tif", side_effect=fake_upload),
        ):
            export_landscape({"id": "e1", **fields}, _landscape_source(), noop_progress)

        with rasterio.open(captured) as src:
            tags = src.tags()
            assert tags["Creator"] == "FastFuels API"
            assert_metadata(tags, expected)
