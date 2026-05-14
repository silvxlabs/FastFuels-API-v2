"""
Unit tests for uploader/handlers/grid.py

Tests _build_dataset in isolation using local GeoTIFF files written to tmp_path.
No GCP I/O — the handler accepts any path that rioxarray/GDAL can open.
"""

import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from uploader.handlers.grid import _build_dataset

from lib.errors import ProcessingError

# Blue Mountain domain bounds (EPSG:32611)
DOMAIN_CRS = "EPSG:32611"
DOMAIN_BOUNDS = (720228, 5189763, 721534, 5190645)  # xmin, ymin, xmax, ymax


class _FakeDomainGdf:
    """Minimal stand-in for the domain GeoDataFrame (only total_bounds used)."""

    def __init__(self, xmin, ymin, xmax, ymax):
        self.total_bounds = (xmin, ymin, xmax, ymax)


DOMAIN_GDF = _FakeDomainGdf(*DOMAIN_BOUNDS)


_DEFAULT_BOUNDS = (
    720400,
    5190000,
    721200,
    5190400,
)  # sub-window inside blue_mtn domain
_WGS84_BOUNDS = (-114.11, 46.825, -114.07, 46.845)  # same area in EPSG:4326


def _write_geotiff(
    path,
    n_bands=1,
    crs=DOMAIN_CRS,
    set_crs=True,
    width=20,
    height=20,
    bounds=_DEFAULT_BOUNDS,
):
    """Write a small GeoTIFF to path. Each band has constant value equal to the band index."""
    xmin, ymin, xmax, ymax = bounds
    transform = from_bounds(xmin, ymin, xmax, ymax, width, height)
    epsg = int(crs.split(":")[1]) if set_crs else None
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=n_bands,
        crs=CRS.from_epsg(epsg) if set_crs else None,
        transform=transform,
        dtype="float32",
    ) as dst:
        for b in range(1, n_bands + 1):
            dst.write(np.full((height, width), float(b), dtype="float32"), b)


class TestBuildDataset:
    def test_single_band_correct_variable_name(self, tmp_path):
        """1-band GeoTIFF produces Dataset with the key from bands_spec."""
        path = str(tmp_path / "single.tif")
        _write_geotiff(path, n_bands=1)

        bands_spec = [{"key": "fbfm", "type": "categorical", "unit": None}]
        ds = _build_dataset(path, bands_spec, DOMAIN_CRS, DOMAIN_GDF)

        assert "fbfm" in ds.data_vars
        assert len(ds.data_vars) == 1

    def test_multi_band_correct_variable_names(self, tmp_path):
        """2-band GeoTIFF produces Dataset with two variables in order."""
        path = str(tmp_path / "multi.tif")
        _write_geotiff(path, n_bands=2)

        bands_spec = [
            {"key": "bulk_density.foliage", "type": "continuous", "unit": "kg/m3"},
            {"key": "bulk_density.branchwood", "type": "continuous", "unit": "kg/m3"},
        ]
        ds = _build_dataset(path, bands_spec, DOMAIN_CRS, DOMAIN_GDF)

        assert "bulk_density.foliage" in ds.data_vars
        assert "bulk_density.branchwood" in ds.data_vars
        assert len(ds.data_vars) == 2

    def test_unit_stored_as_attribute(self, tmp_path):
        """Band unit is stored in DataArray attrs."""
        path = str(tmp_path / "unit.tif")
        _write_geotiff(path, n_bands=1)

        bands_spec = [
            {"key": "bulk_density.foliage", "type": "continuous", "unit": "kg/m3"}
        ]
        ds = _build_dataset(path, bands_spec, DOMAIN_CRS, DOMAIN_GDF)

        assert ds["bulk_density.foliage"].attrs.get("units") == "kg/m3"

    def test_no_unit_no_attribute(self, tmp_path):
        """Band without unit does not set attrs['units']."""
        path = str(tmp_path / "nounit.tif")
        _write_geotiff(path, n_bands=1)

        bands_spec = [{"key": "fbfm", "type": "categorical", "unit": None}]
        ds = _build_dataset(path, bands_spec, DOMAIN_CRS, DOMAIN_GDF)

        assert "units" not in ds["fbfm"].attrs

    def test_band_count_mismatch_raises(self, tmp_path):
        """1-band GeoTIFF with 2-band spec raises BAND_COUNT_MISMATCH."""
        path = str(tmp_path / "mismatch.tif")
        _write_geotiff(path, n_bands=1)

        bands_spec = [
            {"key": "bulk_density.foliage", "type": "continuous", "unit": None},
            {"key": "bulk_density.branchwood", "type": "continuous", "unit": None},
        ]
        with pytest.raises(ProcessingError) as exc_info:
            _build_dataset(path, bands_spec, DOMAIN_CRS, DOMAIN_GDF)

        assert exc_info.value.code == "BAND_COUNT_MISMATCH"

    def test_missing_crs_raises(self, tmp_path):
        """GeoTIFF without CRS raises MISSING_CRS."""
        path = str(tmp_path / "nocrs.tif")
        _write_geotiff(path, n_bands=1, set_crs=False)

        bands_spec = [{"key": "fbfm", "type": "categorical", "unit": None}]
        with pytest.raises(ProcessingError) as exc_info:
            _build_dataset(path, bands_spec, DOMAIN_CRS, DOMAIN_GDF)

        assert exc_info.value.code == "MISSING_CRS"

    def test_reprojection_when_crs_differs(self, tmp_path):
        """GeoTIFF in WGS84 is reprojected to domain CRS before output."""
        path = str(tmp_path / "reproject.tif")
        _write_geotiff(path, n_bands=1, crs="EPSG:4326", bounds=_WGS84_BOUNDS)

        bands_spec = [{"key": "fbfm", "type": "categorical", "unit": None}]
        ds = _build_dataset(path, bands_spec, DOMAIN_CRS, DOMAIN_GDF)

        assert str(ds.rio.crs) == DOMAIN_CRS
        assert "fbfm" in ds.data_vars

    def test_no_overlap_with_domain_raises(self, tmp_path):
        """GeoTIFF entirely outside domain bounds raises NO_OVERLAP."""
        path = str(tmp_path / "outside.tif")
        # Place the GeoTIFF far north of the blue_mtn domain
        _write_geotiff(path, n_bands=1, bounds=(720400, 5300000, 721200, 5300400))

        bands_spec = [{"key": "fbfm", "type": "categorical", "unit": None}]
        with pytest.raises(ProcessingError) as exc_info:
            _build_dataset(path, bands_spec, DOMAIN_CRS, DOMAIN_GDF)

        assert exc_info.value.code == "NO_OVERLAP"

    def test_result_has_spatial_metadata(self, tmp_path):
        """Output Dataset has CRS and transform accessible via .rio."""
        path = str(tmp_path / "spatial.tif")
        _write_geotiff(path, n_bands=1)

        bands_spec = [{"key": "fbfm", "type": "categorical", "unit": None}]
        ds = _build_dataset(path, bands_spec, DOMAIN_CRS, DOMAIN_GDF)

        assert ds.rio.crs is not None
        assert ds.rio.transform() is not None
        assert ds.rio.height > 0
        assert ds.rio.width > 0
