"""
Integration tests for grid export processing (GeoTIFF and Zarr).

Tests the full exporter pipeline: load grid zarr -> convert -> write export.
Requires static test data in GCS (created by services/api/tests/e2e/).

Note: xr.open_dataset(engine="rasterio") reads multi-band GeoTIFFs as a single
``band_data`` variable with a ``band`` dimension. Band count is checked via
``ds.sizes["band"]``, not ``len(ds.data_vars)``.
"""

import os
import tempfile
import zipfile

import gcsfs
import pytest
import rioxarray  # noqa: F401
import xarray as xr
import zarr
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


def _download_and_extract_zarr(gcs_path: str) -> tuple[str, str]:
    """Download a zipped zarr from GCS and extract to a temp directory.

    Returns (extract_dir, zarr_path) for cleanup and opening.
    """
    fs = gcsfs.GCSFileSystem()
    extract_dir = tempfile.mkdtemp()
    zip_path = os.path.join(extract_dir, "export.zip")

    with fs.open(gcs_path, "rb") as f:
        with open(zip_path, "wb") as tmp:
            tmp.write(f.read())

    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)

    zarr_path = os.path.join(extract_dir, "export.zarr")
    return extract_dir, zarr_path


class TestZarrExport:
    @pytest.mark.parametrize(
        "source_grid", ["static-test-blue-mtn-landfire-fbfm40"], indirect=True
    )
    def test_single_band_all_bands(self, exporter_runner, source_grid):
        """Export all bands from single-band FBFM40 grid to zarr."""
        export = exporter_runner(source_grid, "zarr.json")

        filename = sanitize_filename(export.get("name", ""), ".zip")
        gcs_path = f"gs://{EXPORTS_BUCKET}/{export['id']}/{filename}"

        extract_dir, zarr_path = _download_and_extract_zarr(gcs_path)

        group = zarr.open_group(zarr_path, mode="r")
        assert len(list(group.members())) > 0

        ds = xr.open_dataset(zarr_path, engine="zarr", decode_coords="all")
        assert "fbfm" in ds.data_vars
        assert "spatial_ref" in ds.coords
        assert "spatial_ref" not in ds.data_vars
        ds.close()

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

        extract_dir, zarr_path = _download_and_extract_zarr(gcs_path)

        group = zarr.open_group(zarr_path, mode="r")
        assert len(list(group.members())) > 0

        ds = xr.open_dataset(zarr_path, engine="zarr", decode_coords="all")
        assert "elevation" in ds.data_vars
        assert "slope" in ds.data_vars
        assert "aspect" in ds.data_vars
        assert "spatial_ref" in ds.coords
        assert "spatial_ref" not in ds.data_vars
        ds.close()

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

        extract_dir, zarr_path = _download_and_extract_zarr(gcs_path)

        ds = xr.open_dataset(zarr_path, engine="zarr", decode_coords="all")
        assert "elevation" in ds.data_vars
        assert "slope" in ds.data_vars
        assert "aspect" not in ds.data_vars
        assert "spatial_ref" in ds.coords
        assert "spatial_ref" not in ds.data_vars
        ds.close()


def _open_netcdf_from_gcs(gcs_path: str) -> tuple[str, xr.Dataset, xr.Dataset]:
    """Download a netCDF from GCS to a temp file and reopen it twice.

    Returns ``(local_path, decoded_ds, raw_ds)`` — the caller is responsible
    for closing both datasets. h5netcdf needs a local seekable file (it
    cannot stream from a gcsfs handle for HDF5 reads), so we download once
    and reopen with and without ``decode_coords="all"`` to inspect both
    decoded and raw-attribute states.
    """
    fs = gcsfs.GCSFileSystem()
    extract_dir = tempfile.mkdtemp()
    local_path = os.path.join(extract_dir, "export.nc")
    with fs.open(gcs_path, "rb") as src, open(local_path, "wb") as dst:
        dst.write(src.read())
    decoded = xr.open_dataset(local_path, decode_coords="all")
    raw = xr.open_dataset(local_path, decode_coords=False)
    return local_path, decoded, raw


class TestNetcdfExport:
    @pytest.mark.parametrize(
        "source_grid", ["static-test-blue-mtn-landfire-fbfm40"], indirect=True
    )
    def test_single_band_all_bands_2d(self, exporter_runner, source_grid):
        """Export the single-band 2D FBFM40 grid to netCDF and verify CF-1.13."""
        export = exporter_runner(source_grid, "netcdf.json")

        filename = sanitize_filename(export.get("name", ""), ".nc")
        gcs_path = f"gs://{EXPORTS_BUCKET}/{export['id']}/{filename}"
        _, ds, ds_raw = _open_netcdf_from_gcs(gcs_path)
        try:
            assert ds.attrs.get("Conventions") == "CF-1.13"
            assert str(ds.rio.crs) == "EPSG:32611"
            assert "fbfm" in ds.data_vars
            assert "spatial_ref" in ds.coords
            assert ds_raw["fbfm"].attrs["grid_mapping"] == "spatial_ref"
            assert ds["x"].attrs.get("axis") == "X"
            assert ds["y"].attrs.get("axis") == "Y"
            # 2D — no z dim, no z-axis attrs to verify
            assert "z" not in ds.dims
            # Internal attrs must be stripped on netCDF write
            assert "transform" not in ds.attrs
        finally:
            ds.close()
            ds_raw.close()

    @pytest.mark.parametrize(
        "source_grid",
        ["static-test-blue-mtn-landfire-topography"],
        indirect=True,
    )
    def test_multi_band_2d(self, exporter_runner, source_grid):
        """Export all 3 bands from the 2D topography grid; per-band units propagate."""
        export = exporter_runner(source_grid, "netcdf.json")

        filename = sanitize_filename(export.get("name", ""), ".nc")
        gcs_path = f"gs://{EXPORTS_BUCKET}/{export['id']}/{filename}"
        _, ds, ds_raw = _open_netcdf_from_gcs(gcs_path)
        try:
            assert ds.attrs.get("Conventions") == "CF-1.13"
            assert str(ds.rio.crs) == "EPSG:32611"
            for band in ("elevation", "slope", "aspect"):
                assert band in ds.data_vars, band
                assert ds_raw[band].attrs["grid_mapping"] == "spatial_ref", band
            # Stamp_cf falls back to band key when no per-band description exists.
            assert ds["elevation"].attrs.get("long_name") == "elevation"
        finally:
            ds.close()
            ds_raw.close()

    @pytest.mark.parametrize(
        "source_grid",
        ["static-test-blue-mtn-landfire-topography"],
        indirect=True,
    )
    def test_band_subset_2d(self, exporter_runner, source_grid):
        """Subset to elevation+slope; only those bands should appear."""
        export = exporter_runner(
            source_grid,
            "netcdf.json",
            source_overrides={"bands": ["elevation", "slope"]},
        )

        filename = sanitize_filename(export.get("name", ""), ".nc")
        gcs_path = f"gs://{EXPORTS_BUCKET}/{export['id']}/{filename}"
        _, ds, _ = _open_netcdf_from_gcs(gcs_path)
        try:
            assert set(ds.data_vars) == {"elevation", "slope"}
        finally:
            ds.close()

    @pytest.mark.parametrize(
        "source_grid",
        ["static-test-blue-mtn-tree-inventory-voxels"],
        indirect=True,
    )
    def test_3d_voxel_grid(self, exporter_runner, source_grid):
        """Export a 3D voxel grid; z-axis attrs and per-band units must be CF-conformant."""
        export = exporter_runner(source_grid, "netcdf.json", timeout=180)

        filename = sanitize_filename(export.get("name", ""), ".nc")
        gcs_path = f"gs://{EXPORTS_BUCKET}/{export['id']}/{filename}"
        _, ds, ds_raw = _open_netcdf_from_gcs(gcs_path)
        try:
            assert ds.attrs.get("Conventions") == "CF-1.13"
            assert str(ds.rio.crs) == "EPSG:32611"
            assert {"z", "y", "x"}.issubset(ds.dims)

            # CF z-axis: positive="up", axis="Z"
            assert ds["z"].attrs.get("axis") == "Z"
            assert ds["z"].attrs.get("positive") == "up"

            # Tree-voxel bands and units from the Grid document
            for band in (
                "volume_fraction",
                "bulk_density.foliage.live",
                "fuel_moisture.live",
                "savr.foliage",
            ):
                assert band in ds.data_vars, band
                assert ds_raw[band].attrs["grid_mapping"] == "spatial_ref", band

            # Units flow from the Grid document's bands list into the netCDF.
            assert ds["bulk_density.foliage.live"].attrs.get("units") == "kg/m**3"
            assert ds["fuel_moisture.live"].attrs.get("units") == "%"
            assert ds["savr.foliage"].attrs.get("units") == "1/m"

            # Internal-only attrs from treevox storage must not leak.
            assert "transform" not in ds.attrs
            assert "z_origin" not in ds.attrs
            assert "z_resolution" not in ds.attrs
        finally:
            ds.close()
            ds_raw.close()

    @pytest.mark.parametrize(
        "source_grid",
        ["static-test-blue-mtn-tree-inventory-voxels"],
        indirect=True,
    )
    def test_3d_band_subset(self, exporter_runner, source_grid):
        """Subset of a 3D grid keeps just the requested bands; CF metadata still applied."""
        export = exporter_runner(
            source_grid,
            "netcdf.json",
            source_overrides={"bands": ["bulk_density.foliage.live"]},
            timeout=180,
        )

        filename = sanitize_filename(export.get("name", ""), ".nc")
        gcs_path = f"gs://{EXPORTS_BUCKET}/{export['id']}/{filename}"
        _, ds, _ = _open_netcdf_from_gcs(gcs_path)
        try:
            assert set(ds.data_vars) == {"bulk_density.foliage.live"}
            assert ds.attrs.get("Conventions") == "CF-1.13"
            assert str(ds.rio.crs) == "EPSG:32611"
            assert ds["bulk_density.foliage.live"].attrs.get("units") == "kg/m**3"
        finally:
            ds.close()
