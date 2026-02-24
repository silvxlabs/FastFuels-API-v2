"""
Integration tests for GeoTIFF export processing.

Tests the full exporter pipeline: load grid zarr -> convert -> write GeoTIFF.
Requires static test data in GCS (created by services/api/tests/e2e/).

Note: xr.open_dataset(engine="rasterio") reads multi-band GeoTIFFs as a single
``band_data`` variable with a ``band`` dimension. Band count is checked via
``ds.sizes["band"]``, not ``len(ds.data_vars)``.
"""

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
