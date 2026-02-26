"""
Integration tests for grid export processing (GeoTIFF and Zarr).

Tests the full exporter pipeline: load grid zarr -> convert -> write export.
Requires static test data in GCS (created by services/api/tests/e2e/).

Note: xr.open_dataset(engine="rasterio") reads multi-band GeoTIFFs as a single
``band_data`` variable with a ``band`` dimension. Band count is checked via
``ds.sizes["band"]``, not ``len(ds.data_vars)``.
"""

import zipfile

import gcsfs
import pytest
import rioxarray  # noqa: F401
import xarray as xr
from exporter.filename import sanitize_filename

from lib.config import EXPORTS_BUCKET


class TestGeotiffExport:
    @pytest.mark.parametrize(
        "source_grid", ["static-test-blue-mtn-landfire-fbfm40"], indirect=True
    )
    def test_single_band_all_bands(self, exporter_runner, source_grid):
        """Export all bands from single-band FBFM40 grid."""
        export = exporter_runner(source_grid, "geotiff.json")

        filename = sanitize_filename(export.get("name", ""), ".tif")
        gcs_path = f"gs://{EXPORTS_BUCKET}/{export['id']}/{filename}"
        ds = xr.open_dataset(gcs_path, engine="rasterio")

        assert ds.sizes["band"] == 1
        assert ds.rio.crs is not None
        assert str(ds.rio.crs) == "EPSG:32611"
        assert "x" in ds.dims
        assert "y" in ds.dims

        ds.close()

    @pytest.mark.parametrize(
        "source_grid",
        ["static-test-blue-mtn-landfire-topography"],
        indirect=True,
    )
    def test_multi_band_all_bands(self, exporter_runner, source_grid):
        """Export all 3 bands from topography grid."""
        export = exporter_runner(source_grid, "geotiff.json")

        filename = sanitize_filename(export.get("name", ""), ".tif")
        gcs_path = f"gs://{EXPORTS_BUCKET}/{export['id']}/{filename}"
        ds = xr.open_dataset(gcs_path, engine="rasterio")

        assert ds.sizes["band"] == 3
        assert ds.rio.crs is not None
        assert str(ds.rio.crs) == "EPSG:32611"

        ds.close()

    @pytest.mark.parametrize(
        "source_grid",
        ["static-test-blue-mtn-landfire-topography"],
        indirect=True,
    )
    def test_band_subset(self, exporter_runner, source_grid):
        """Export only elevation+slope from topography grid."""
        export = exporter_runner(
            source_grid,
            "geotiff.json",
            source_overrides={"bands": ["elevation", "slope"]},
        )

        filename = sanitize_filename(export.get("name", ""), ".tif")
        gcs_path = f"gs://{EXPORTS_BUCKET}/{export['id']}/{filename}"
        ds = xr.open_dataset(gcs_path, engine="rasterio")

        assert ds.sizes["band"] == 2
        assert ds.rio.crs is not None
        assert str(ds.rio.crs) == "EPSG:32611"

        ds.close()

    @pytest.mark.parametrize(
        "source_grid",
        ["static-test-blue-mtn-landfire-topography"],
        indirect=True,
    )
    def test_single_band_from_multiband(self, exporter_runner, source_grid):
        """Export single band from multi-band grid."""
        export = exporter_runner(
            source_grid,
            "geotiff.json",
            source_overrides={"bands": ["elevation"]},
        )

        filename = sanitize_filename(export.get("name", ""), ".tif")
        gcs_path = f"gs://{EXPORTS_BUCKET}/{export['id']}/{filename}"
        ds = xr.open_dataset(gcs_path, engine="rasterio")

        assert ds.sizes["band"] == 1
        assert ds.rio.crs is not None
        assert str(ds.rio.crs) == "EPSG:32611"

        ds.close()


class TestZarrExport:
    @pytest.mark.parametrize(
        "source_grid", ["static-test-blue-mtn-landfire-fbfm40"], indirect=True
    )
    def test_single_band_all_bands(self, exporter_runner, source_grid):
        """Export all bands from single-band FBFM40 grid to zarr."""
        export = exporter_runner(source_grid, "zarr.json")

        filename = sanitize_filename(export.get("name", ""), ".zip")
        gcs_path = f"gs://{EXPORTS_BUCKET}/{export['id']}/{filename}"

        # Download and verify it's a valid zip containing a zarr store
        fs = gcsfs.GCSFileSystem()
        with fs.open(gcs_path, "rb") as f:
            with zipfile.ZipFile(f) as zf:
                names = zf.namelist()
                assert any(".zmetadata" in n or ".zattrs" in n for n in names)

    @pytest.mark.parametrize(
        "source_grid",
        ["static-test-blue-mtn-landfire-topography"],
        indirect=True,
    )
    def test_multi_band_all_bands(self, exporter_runner, source_grid):
        """Export all 3 bands from topography grid to zarr."""
        export = exporter_runner(source_grid, "zarr.json")

        filename = sanitize_filename(export.get("name", ""), ".zip")
        gcs_path = f"gs://{EXPORTS_BUCKET}/{export['id']}/{filename}"

        fs = gcsfs.GCSFileSystem()
        with fs.open(gcs_path, "rb") as f:
            with zipfile.ZipFile(f) as zf:
                names = zf.namelist()
                assert any(".zmetadata" in n or ".zattrs" in n for n in names)

    @pytest.mark.parametrize(
        "source_grid",
        ["static-test-blue-mtn-landfire-topography"],
        indirect=True,
    )
    def test_band_subset(self, exporter_runner, source_grid):
        """Export only elevation+slope from topography grid to zarr."""
        export = exporter_runner(
            source_grid,
            "zarr.json",
            source_overrides={"bands": ["elevation", "slope"]},
        )

        filename = sanitize_filename(export.get("name", ""), ".zip")
        gcs_path = f"gs://{EXPORTS_BUCKET}/{export['id']}/{filename}"

        # Download zip and open as zarr to verify band subset
        fs = gcsfs.GCSFileSystem()
        with fs.open(gcs_path, "rb") as f:
            with zipfile.ZipFile(f) as zf:
                names = zf.namelist()
                # Verify elevation and slope dirs exist, aspect does not
                has_elevation = any("elevation/" in n for n in names)
                has_slope = any("slope/" in n for n in names)
                has_aspect = any("aspect/" in n for n in names)
                assert has_elevation
                assert has_slope
                assert not has_aspect
