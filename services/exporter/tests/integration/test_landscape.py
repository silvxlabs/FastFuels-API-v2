"""
Integration tests for the landscape export.

The central assertion is **pixel-exactness against LANDFIRE itself**: every
band of the exported GeoTIFF must equal the corresponding window of the
LANDFIRE CONUS source raster, cell for cell. That is possible because the
fixtures were built with ``alignment.target="native"`` in an EPSG:5070
domain, so they sit on the source rasters' own 30 m pixel lattice and were
never resampled or reprojected (see ``_LANDSCAPE_ROLE_FIXTURES`` in conftest).
This pins the whole encoding contract — band order, the crown x10 / x100
scalings, nodata, int16 rounding — to data that lives outside this codebase,
rather than checking the handler's arithmetic against itself.

One expensive setup per module — stage the three source grids, run the
exporter, download the GeoTIFF — then many small test functions each verify
one invariant against the already downloaded output.
"""

import tempfile
from pathlib import Path

import gcsfs
import numpy as np
import pytest
import rasterio
from exporter.filename import sanitize_filename
from rasterio.windows import from_bounds

from lib.config import EXPORTS_BUCKET, RASTERS_BUCKET

# Landscape lattice dims for the static fixtures.
_NY = 39
_NX = 50
_NODATA = -9999

# LANDFIRE nodata sentinels as they appear in the source rasters. Both map to
# the landscape sentinel on the way out.
_SOURCE_NODATA = (32767, -9999)

# Bands in LANDFIRE order, paired with the CONUS raster each is drawn from.
# The versions match the fixtures' `source.version` (topography LF2020,
# fuels LF2024).
_BANDS = [
    ("Elevation", "LF2020_elevation_CONUS.tif"),
    ("Slope", "LF2020_slope_CONUS.tif"),
    ("Aspect", "LF2020_aspect_CONUS.tif"),
    ("Fuel Model", "LF2024_FBFM40_CONUS.tif"),
    ("Canopy Cover", "LF2024_CC_CONUS.tif"),
    ("Canopy Height", "LF2024_CH_CONUS.tif"),
    ("Canopy Base Height", "LF2024_CBH_CONUS.tif"),
    ("Canopy Bulk Density", "LF2024_CBD_CONUS.tif"),
]


# --- Helpers ---


def _gcs_for(export: dict) -> str:
    filename = sanitize_filename(export.get("name", ""), ".tif")
    return f"{EXPORTS_BUCKET}/{export['id']}/{filename}"


def _download(export: dict) -> Path:
    fs = gcsfs.GCSFileSystem()
    tmpdir = Path(tempfile.mkdtemp(prefix="landscape-export-"))
    local = tmpdir / "landscape.tif"
    fs.get(_gcs_for(export), str(local))
    return local


def _source_window(raster: str, bounds) -> np.ndarray:
    """Read the landscape's footprint out of a LANDFIRE CONUS raster, with
    both source nodata sentinels mapped to the landscape sentinel."""
    with rasterio.open(f"gs://{RASTERS_BUCKET}/{raster}") as src:
        raw = src.read(1, window=from_bounds(*bounds, transform=src.transform))
    is_nodata = np.isin(raw, _SOURCE_NODATA)
    return np.where(is_nodata, _NODATA, raw).astype(np.int16)


# --- Fixtures ---


@pytest.fixture(scope="module")
def landscape_export(landscape_sources, landscape_exporter_runner):
    """Run the exporter once. Yields the downloaded GeoTIFF path."""
    export = landscape_exporter_runner(landscape_sources)
    path = _download(export)
    yield path
    path.unlink(missing_ok=True)


@pytest.fixture(scope="module")
def written(landscape_export) -> np.ndarray:
    with rasterio.open(landscape_export) as src:
        return src.read()


@pytest.fixture(scope="module")
def landfire_bands(landscape_export) -> list[np.ndarray]:
    """The LANDFIRE source pixels under the landscape's footprint, in band
    order. One windowed read per band."""
    with rasterio.open(landscape_export) as src:
        bounds = src.bounds
    return [_source_window(raster, bounds) for _, raster in _BANDS]


class TestLandscapeMatchesLandfire:
    """The export must reproduce LANDFIRE's own pixels exactly."""

    @pytest.mark.parametrize(
        "index,band_name", [(i, name) for i, (name, _) in enumerate(_BANDS)]
    )
    def test_band_is_pixel_identical_to_landfire_source(
        self, written, landfire_bands, index, band_name
    ):
        """Each band equals the LANDFIRE source raster cell for cell.

        A swapped band order, a dropped x10 / x100 scaling, a bad nodata
        mapping, or an off-by-one crop all fail here.
        """
        np.testing.assert_array_equal(
            written[index],
            landfire_bands[index],
            err_msg=f"band {index + 1} ({band_name}) diverges from LANDFIRE",
        )

    def test_crown_bands_carry_landfire_scaled_encodings(self, written):
        """Guard the x10 / x100 encodings from both directions: the values
        must be LANDFIRE's scaled integers, not the grids' physical units.

        Without the scaling, canopy bulk density (order 0.1 kg/m**3) would
        round to all-zero and canopy height (order 10 m) would lose its
        decimal — so a non-trivial spread above the unscaled range is the
        signal that the encoding was applied.
        """
        for index in (5, 6, 7):  # CH, CBH, CBD
            valid = written[index][written[index] != _NODATA]
            assert valid.size > 0
            assert valid.max() > 1, (
                f"band {index + 1} looks unscaled (max={valid.max()})"
            )

    def test_fuel_model_codes_are_categorical_not_rescaled(self, written):
        fbfm = written[3]
        valid = fbfm[fbfm != _NODATA]
        assert valid.size > 0
        assert valid.min() >= 91  # NB1 is the lowest FBFM40 code
        assert valid.max() <= 204


class TestLandscapeFormat:
    """Format invariants the operational tools depend on."""

    def test_is_eight_band_int16_geotiff(self, landscape_export):
        with rasterio.open(landscape_export) as src:
            assert src.driver == "GTiff"
            assert src.count == 8
            assert set(src.dtypes) == {"int16"}

    def test_nodata_is_landfire_sentinel_on_every_band(self, landscape_export):
        with rasterio.open(landscape_export) as src:
            assert src.nodatavals == (_NODATA,) * 8

    def test_georeference_matches_the_source_lattice(self, landscape_export):
        with rasterio.open(landscape_export) as src:
            assert src.width == _NX
            assert src.height == _NY
            assert src.crs.to_epsg() == 5070
            assert list(src.transform)[:6] == pytest.approx(
                [30.0, 0.0, -1379265.0, 0.0, -30.0, 2781015.0]
            )

    def test_storage_layout_is_landfire_style(self, landscape_export):
        """Pixel-interleaved and compressed, matching shipped landscapes."""
        with rasterio.open(landscape_export) as src:
            assert src.profile["interleave"] == "pixel"
            assert src.profile["compress"] == "lzw"

    def test_band_descriptions_are_landfire_layer_names(self, landscape_export):
        with rasterio.open(landscape_export) as src:
            assert src.descriptions == tuple(name for name, _ in _BANDS)

    def test_band_tags_carry_name_and_units(self, landscape_export):
        """Mirrors the mechanism LFPS-produced landscapes use (BandName tag),
        plus an explicit Units tag."""
        with rasterio.open(landscape_export) as src:
            assert src.tags(1) == {"BandName": "Elevation", "Units": "meters"}
            assert src.tags(2)["Units"] == "degrees"
            assert src.tags(3)["Units"] == "degrees"
            assert src.tags(5)["Units"] == "percent"
            assert src.tags(6)["Units"] == "meters * 10"
            assert src.tags(7)["Units"] == "meters * 10"
            assert src.tags(8)["Units"] == "kg/m^3 * 100"

    def test_fuel_model_units_tag_reflects_declared_classification(
        self, landscape_export
    ):
        with rasterio.open(landscape_export) as src:
            assert src.tags(4)["Units"] == "Scott and Burgan Fire Behavior Fuel Models"


class TestLandscapeFbfm13:
    """A second exporter run: the declared fuel model classification is the
    only variant that changes output bytes."""

    def test_fbfm13_declaration_changes_units_tag(
        self, landscape_sources, landscape_exporter_runner
    ):
        export = landscape_exporter_runner(
            landscape_sources,
            source_overrides={"fire_behavior_fuel_model": "fbfm13"},
        )
        path = _download(export)
        try:
            with rasterio.open(path) as src:
                assert src.tags(4)["Units"] == "Anderson Fire Behavior Fuel Models"
        finally:
            path.unlink(missing_ok=True)
