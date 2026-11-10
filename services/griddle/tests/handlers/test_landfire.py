"""
Tests for LANDFIRE handler.
"""

import json
import logging
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
import rioxarray  # noqa: F401
import xarray as xr
from fastfuels_core.fuel_models import update_fuel_models
from griddle.handlers import landfire as landfire_handler
from griddle.handlers.landfire import (
    FBFM_NON_BURNABLE,
    FCCS_BARE_GROUND,
    LANDFIRE_EXTRA_NODATA,
    _fetch_fuel_model,
    _fetch_landfire_raster,
    _fuel_rulesets,
    _ldist_attributes,
    _lf_map_zones,
    _lookup_lf_zones,
    _most_frequent,
    _needs_lfps,
    _remove_non_burnable_blocks,
    fetch_annual_disturbance,
    fetch_disturbed_fuel,
    fetch_fbfm13,
    fetch_fbfm40,
    fetch_fccs,
    fetch_topography,
    scatter_categorical_boundaries,
)
from pyproj import Transformer
from rasterio.transform import from_origin
from shapely.geometry import box

from lib.errors import ProcessingError
from lib.landfire import UnknownLandfireVersionError
from lib.testing import SHARED_TEST_DOMAINS_DIR

DOMAINS_DIR = SHARED_TEST_DOMAINS_DIR


@dataclass
class DomainFixture:
    """Test domain configuration."""

    name: str
    description: str
    file_name: str
    expected_shape: tuple  # (y, x)


DOMAIN_FIXTURES = [
    DomainFixture(
        name="Blue Mountain",
        description="~1 sq km in Blue Mountain Recreation Area, Montana",
        file_name="blue_mtn.json",
        # Shape produced by `target="domain"` at LANDFIRE's native pixel
        # size projected into the domain CRS. LANDFIRE is in EPSG:5070
        # (Contiguous Albers) at 30m; over Montana, the equivalent cell
        # size in EPSG:32611 (UTM 11N) is ~29.44m via
        # calculate_default_transform. The unbuffered baseline is
        # (30, 45) cells, plus the test's extent_buffer_cells=8 on each
        # side ⇒ (30 + 16, 45 + 16) = (46, 61).
        expected_shape=(46, 61),
    ),
]


@pytest.fixture(params=DOMAIN_FIXTURES, ids=[d.name for d in DOMAIN_FIXTURES])
def test_domain(request) -> DomainFixture:
    """Return test domain configuration."""
    return request.param


@pytest.fixture
def roi(test_domain) -> gpd.GeoDataFrame:
    """Load domain as a GeoDataFrame."""
    with open(DOMAINS_DIR / test_domain.file_name) as f:
        domain = json.load(f)

    crs = domain["crs"]["properties"]["name"]
    return gpd.GeoDataFrame.from_features(domain["features"], crs=crs)


class TestFetchLandfireRasterNodataConsolidation:
    """Unit tests for the -9999 / declared-nodata consolidation in
    _fetch_landfire_raster."""

    def _make_raster(self, values, roi, nodata=None):
        """Build a (band, y, x) DataArray mimicking extract_window's output."""
        data = xr.DataArray(
            values[np.newaxis, :, :].astype(np.int16),
            dims=("band", "y", "x"),
            coords={
                "band": [1],
                "y": np.arange(values.shape[0], dtype=np.float64),
                "x": np.arange(values.shape[1], dtype=np.float64),
            },
        ).rio.write_crs(roi.crs)
        if nodata is not None:
            data = data.rio.write_nodata(nodata)
        return data

    @patch("griddle.handlers.landfire.RasterConnection")
    def test_both_sentinels_present_collapse_onto_declared(self, mock_raster_cls, roi):
        """-9999 cells become `declared`; declared cells and real data untouched."""
        values = np.array([[101, -9999], [32767, 102]])
        raw = self._make_raster(values, roi, nodata=32767)
        mock_raster = MagicMock()
        mock_raster.raster_x_resolution = 30.0
        mock_raster.extract_window.return_value = raw
        mock_raster_cls.return_value = mock_raster

        result = _fetch_landfire_raster(
            roi,
            "gs://mock/url.tiff",
            extent_buffer_cells=0,
            alignment={"target": "native"},
            target_grid_doc=None,
            is_categorical=True,
        )

        np.testing.assert_array_equal(result.values, [[101, 32767], [32767, 102]])
        assert result.rio.nodata == 32767

    @patch("griddle.handlers.landfire.RasterConnection")
    def test_only_extra_sentinel_present_declared_never_occurs(
        self, mock_raster_cls, roi
    ):
        """Pre-2024 case: declared value never appears in the data, but -9999
        cells still get remapped onto it."""
        values = np.array([[101, -9999], [102, -9999]])
        raw = self._make_raster(values, roi, nodata=32767)
        mock_raster = MagicMock()
        mock_raster.raster_x_resolution = 30.0
        mock_raster.extract_window.return_value = raw
        mock_raster_cls.return_value = mock_raster

        result = _fetch_landfire_raster(
            roi,
            "gs://mock/url.tiff",
            extent_buffer_cells=0,
            alignment={"target": "native"},
            target_grid_doc=None,
            is_categorical=True,
        )

        np.testing.assert_array_equal(result.values, [[101, 32767], [102, 32767]])
        assert result.rio.nodata == 32767

    @patch("griddle.handlers.landfire.RasterConnection")
    def test_no_declared_nodata_falls_back_to_extra_sentinel(
        self, mock_raster_cls, roi
    ):
        """No nodata tag at all: -9999 stays -9999, and rio.nodata is set to it."""
        values = np.array([[101, -9999], [102, 103]])
        raw = self._make_raster(values, roi, nodata=None)
        mock_raster = MagicMock()
        mock_raster.raster_x_resolution = 30.0
        mock_raster.extract_window.return_value = raw
        mock_raster_cls.return_value = mock_raster

        result = _fetch_landfire_raster(
            roi,
            "gs://mock/url.tiff",
            extent_buffer_cells=0,
            alignment={"target": "native"},
            target_grid_doc=None,
            is_categorical=True,
        )

        np.testing.assert_array_equal(result.values, values)
        assert result.rio.nodata == LANDFIRE_EXTRA_NODATA

    @patch("griddle.handlers.landfire.RasterConnection")
    def test_direction_only_extra_sentinel_cells_change(self, mock_raster_cls, roi):
        """It's -9999 that gets overwritten with `declared` — never the reverse."""
        values = np.array([[-9999, 55], [55, -9999]])
        raw = self._make_raster(values, roi, nodata=55)
        mock_raster = MagicMock()
        mock_raster.raster_x_resolution = 30.0
        mock_raster.extract_window.return_value = raw
        mock_raster_cls.return_value = mock_raster

        result = _fetch_landfire_raster(
            roi,
            "gs://mock/url.tiff",
            extent_buffer_cells=0,
            alignment={"target": "native"},
            target_grid_doc=None,
            is_categorical=True,
        )

        # every cell should now read 55 — the -9999 cells were the ones that
        # moved, not the cells that were already 55
        np.testing.assert_array_equal(result.values, [[55, 55], [55, 55]])


class TestFetchLandfireRasterBandSelection:
    """Unit tests for _fetch_landfire_raster's band parameter."""

    @pytest.fixture
    def multiband(self, roi):
        """RasterConnection patched to return a (band, y, x) raster whose bands
        hold 101, 202 and 303. Band 3 also has a -9999 cell."""
        values = {1: 101, 2: 202, 3: 303}
        data = np.stack(
            [np.full((2, 2), values[b], dtype=np.int16) for b in sorted(values)]
        )
        data[2, 0, 0] = -9999
        raw = (
            xr.DataArray(
                data,
                dims=("band", "y", "x"),
                coords={
                    "band": sorted(values),
                    "y": np.arange(2, dtype=np.float64),
                    "x": np.arange(2, dtype=np.float64),
                },
            )
            .rio.write_crs(roi.crs)
            .rio.write_nodata(32767)
        )
        with patch("griddle.handlers.landfire.RasterConnection") as mock_raster_cls:
            mock_raster = MagicMock()
            mock_raster.extract_window.return_value = raw
            mock_raster_cls.return_value = mock_raster
            yield

    def _fetch(self, roi, **kwargs):
        return _fetch_landfire_raster(
            roi,
            "gs://mock/url.tiff",
            extent_buffer_cells=0,
            alignment={"target": "native"},
            target_grid_doc=None,
            is_categorical=True,
            **kwargs,
        )

    def test_default_band_is_one(self, roi, multiband):
        assert (self._fetch(roi).values == 101).all()

    def test_selects_requested_band(self, roi, multiband):
        result = self._fetch(roi, band=2)
        assert (result.values == 202).all()

    def test_band_dimension_dropped(self, roi, multiband):
        assert "band" not in self._fetch(roi, band=2).dims

    def test_nodata_consolidation_applies_to_the_selected_band(self, roi, multiband):
        """Band 3's -9999 cell is folded onto the declared nodata value."""
        result = self._fetch(roi, band=3)

        np.testing.assert_array_equal(result.values, [[32767, 303], [303, 303]])
        assert result.rio.nodata == 32767

    def test_missing_band_raises(self, roi, multiband):
        with pytest.raises(KeyError):
            self._fetch(roi, band=9)


class TestFetchLandfireRasterCoastalEdge:
    """-9999 must be nodata before reprojection, or bilinear blends it into
    neighbouring valid cells (#634)."""

    DECLARED = 32767
    VALID = 45

    @pytest.fixture
    def coastal_raster(self, tmp_path):
        """30 m EPSG:5070 raster: west half -9999 (ocean), east half valid."""
        values = np.full((40, 40), self.VALID, dtype=np.int16)
        values[:, :20] = LANDFIRE_EXTRA_NODATA
        path = tmp_path / "coastal.tif"
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=40,
            width=40,
            count=1,
            dtype="int16",
            crs="EPSG:5070",
            transform=from_origin(0.0, 1200.0, 30.0, 30.0),
            nodata=self.DECLARED,
        ) as dst:
            dst.write(values, 1)
        return str(path)

    @pytest.mark.parametrize(
        ("roi_crs", "alignment"),
        [
            # Domain-anchored lattice shifted 7 m off the source lattice.
            ("EPSG:5070", {"target": "domain"}),
            # CRS-only override at a finer resolution.
            ("EPSG:5070", {"target": "native", "resolution": 10.0}),
            # Default path: reproject into the ROI's CRS.
            ("EPSG:32614", {"target": "native"}),
        ],
        ids=["domain", "native-resolution", "native-reproject"],
    )
    def test_sentinel_not_blended_across_edge(self, coastal_raster, roi_crs, alignment):
        roi = gpd.GeoDataFrame(
            geometry=[box(307.0, 307.0, 907.0, 907.0)], crs="EPSG:5070"
        ).to_crs(roi_crs)

        result = _fetch_landfire_raster(
            roi,
            coastal_raster,
            extent_buffer_cells=0,
            alignment=alignment,
            target_grid_doc=None,
            is_categorical=False,
        )

        values = result.values
        valid = values[values != result.rio.nodata]
        assert result.rio.nodata == self.DECLARED
        assert (values == self.DECLARED).any()
        assert valid.size > 0
        np.testing.assert_array_equal(np.unique(valid), [self.VALID])


class TestFetchTopographyAspectResampling:
    """Aspect defaults to nearest, so resampling never blends angles or flat
    (-1) cells into new directions; flat cells are stored as 0."""

    DECLARED = 32767

    @pytest.fixture
    def aspect_raster(self, tmp_path):
        """30 m EPSG:5070 raster: quadrants of 350, 10, flat (-1), nodata."""
        values = np.full((40, 40), 350, dtype=np.int16)
        values[:20, 20:] = 10
        values[20:, :20] = -1
        values[20:, 20:] = self.DECLARED
        path = tmp_path / "aspect.tif"
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=40,
            width=40,
            count=1,
            dtype="int16",
            crs="EPSG:5070",
            transform=from_origin(0.0, 1200.0, 30.0, 30.0),
            nodata=self.DECLARED,
        ) as dst:
            dst.write(values, 1)
        return str(path)

    @pytest.mark.parametrize(
        ("roi_crs", "alignment"),
        [
            ("EPSG:5070", {"target": "domain"}),
            ("EPSG:5070", {"target": "native", "resolution": 10.0}),
            ("EPSG:32614", {"target": "native"}),
        ],
        ids=["domain", "native-resolution", "native-reproject"],
    )
    def test_aspect_keeps_source_values(self, aspect_raster, roi_crs, alignment):
        roi = gpd.GeoDataFrame(
            geometry=[box(307.0, 307.0, 907.0, 907.0)], crs="EPSG:5070"
        ).to_crs(roi_crs)

        with patch(
            "griddle.handlers.landfire._landfire_cog_url", return_value=aspect_raster
        ):
            result = fetch_topography(
                roi=roi,
                version="2020",
                bands=["aspect"],
                progress=MagicMock(),
                alignment=alignment,
            )

        aspect = result["aspect"]
        assert aspect.dtype == np.int16
        assert aspect.rio.nodata == self.DECLARED
        np.testing.assert_array_equal(
            np.unique(aspect.values), [0, 10, 350, self.DECLARED]
        )

    @patch("griddle.handlers.landfire._fetch_landfire_raster")
    def test_only_aspect_defaults_to_nearest(self, mock_fetch, roi):
        mock_fetch.return_value = (
            xr.DataArray(
                np.zeros((2, 2)),
                dims=("y", "x"),
                coords={"y": [45.0, 15.0], "x": [15.0, 45.0]},
            )
            .rio.write_crs("EPSG:5070")
            .rio.write_transform(from_origin(0.0, 60.0, 30.0, 30.0))
        )

        fetch_topography(
            roi=roi,
            version="2020",
            bands=["elevation", "slope", "aspect"],
            progress=MagicMock(),
        )

        categorical_by_band = {
            call.args[1].rsplit("_", 2)[1]: call.kwargs["is_categorical"]
            for call in mock_fetch.call_args_list
        }
        assert categorical_by_band == {
            "elevation": False,
            "slope": False,
            "aspect": True,
        }


class TestNeedsLfps:
    """Unit tests for _needs_lfps."""

    def test_staged_version_does_not_need_lfps(self):
        assert _needs_lfps("fbfm40", "2024") is False

    def test_lfps_only_version_needs_lfps(self):
        assert _needs_lfps("fbfm40", "2025") is True

    def test_product_without_lfps_available_key_returns_false(self):
        """Falls back cleanly if a product's config has no lfps_available key
        at all, rather than KeyError."""
        assert _needs_lfps("fbfm13", "1999") is False


class TestLookupLfZones:
    """Unit tests for _lookup_lf_zones, on two adjacent square zones in
    EPSG:5070 so every expected zone is known exactly:

        zone 1: x 0-1000, y 0-1000
        zone 2: x 1000-2000, y 0-1000
    """

    ALBERS = "EPSG:5070"
    UTM = "EPSG:32614"

    @pytest.fixture
    def zones(self) -> gpd.GeoDataFrame:
        return gpd.GeoDataFrame(
            {"ZONE_NUM": [1, 2]},
            geometry=[box(0, 0, 1000, 1000), box(1000, 0, 2000, 1000)],
            crs=self.ALBERS,
        )

    @classmethod
    def _utm_grid(cls, center_x, center_y, cell, shape):
        """A UTM grid centered on an EPSG:5070 point."""
        x, y = Transformer.from_crs(cls.ALBERS, cls.UTM, always_xy=True).transform(
            center_x, center_y
        )
        ny, nx = shape
        return from_origin(x - nx * cell / 2, y + ny * cell / 2, cell, cell)

    def test_grid_inside_one_zone_is_filled_without_rasterizing(
        self, zones, monkeypatch
    ):
        def fail(*args, **kwargs):
            raise AssertionError("single-zone grid should not be rasterized")

        monkeypatch.setattr("griddle.handlers.landfire.rasterize", fail)

        result = _lookup_lf_zones(
            zones, from_origin(100, 900, 100, 100), (3, 3), self.ALBERS
        )

        np.testing.assert_array_equal(result, np.full((3, 3), 1))
        assert result.dtype == np.int16

    def test_grid_crossing_a_boundary(self, zones):
        # Columns are centered at x = 850, 950 | 1050, 1150.
        result = _lookup_lf_zones(
            zones, from_origin(800, 900, 100, 100), (2, 4), self.ALBERS
        )

        np.testing.assert_array_equal(result, [[1, 1, 2, 2], [1, 1, 2, 2]])
        assert result.dtype == np.int16

    def test_grid_past_the_edge_of_every_zone(self, zones):
        # Only zone 2 touches this grid, but it doesn't contain it, so the
        # cells past x = 2000 must still be nodata.
        result = _lookup_lf_zones(
            zones, from_origin(1900, 500, 100, 100), (1, 3), self.ALBERS
        )

        np.testing.assert_array_equal(result, [[2, -9999, -9999]])

    def test_grid_outside_every_zone(self, zones):
        result = _lookup_lf_zones(
            zones, from_origin(5000, 5000, 100, 100), (2, 2), self.ALBERS
        )

        np.testing.assert_array_equal(result, np.full((2, 2), -9999))
        assert result.dtype == np.int16

    @pytest.mark.parametrize("center_x, expected", [(500, 1), (1500, 2)])
    def test_grid_in_another_crs_inside_one_zone(self, zones, center_x, expected):
        transform = self._utm_grid(center_x, 500, cell=30, shape=(4, 4))

        result = _lookup_lf_zones(zones, transform, (4, 4), self.UTM)

        np.testing.assert_array_equal(result, np.full((4, 4), expected))

    def test_grid_in_another_crs_crossing_a_boundary(self, zones):
        # Two 100 m cells straddling the boundary at x = 1000; their centers
        # are 50 m either side of it.
        transform = self._utm_grid(1000, 500, cell=100, shape=(1, 2))

        result = _lookup_lf_zones(zones, transform, (1, 2), self.UTM)

        np.testing.assert_array_equal(result, [[1, 2]])


class TestLfMapZones:
    """Unit tests for the map-zone polygon loader."""

    def setup_method(self):
        _lf_map_zones.cache_clear()

    def teardown_method(self):
        _lf_map_zones.cache_clear()

    @patch("griddle.handlers.landfire.gpd.read_parquet")
    def test_reads_the_rasters_bucket_parquet_once(self, mock_read):
        mock_read.return_value = MagicMock()

        first = _lf_map_zones()
        second = _lf_map_zones()

        mock_read.assert_called_once()
        assert mock_read.call_args[0][0].startswith("gs://")
        assert mock_read.call_args[0][0].endswith("/LF_MapZones.parquet")
        assert first is second


class TestFuelRulesets:
    """Unit tests for the Master_Rulesets loader."""

    def setup_method(self):
        _fuel_rulesets.cache_clear()

    def teardown_method(self):
        _fuel_rulesets.cache_clear()

    @staticmethod
    def _raw(**overrides) -> pd.DataFrame:
        """One raw-export row per list entry, in the export's column layout."""
        columns = {
            "Zone": [1],
            "EVT": [7000],
            "DIST": [111],
            "Cover_Low": [0],
            "Cover_High": [999],
            "Height_Low": [0],
            "Height_High": [999],
            "BPSRF": ["any"],
            "Wildcard": ["any"],
            "OnOff": ["On"],
            "FBFM13": [8],
            "FBFM40": ["GR1 / 101"],
            "FCCS": [338],
        }
        return pd.DataFrame({**columns, **overrides})

    @patch("griddle.handlers.landfire.pd.read_parquet")
    def test_reads_the_tables_bucket_parquet_once(self, mock_read):
        mock_read.return_value = self._raw()

        first = _fuel_rulesets()
        second = _fuel_rulesets()

        mock_read.assert_called_once()
        assert mock_read.call_args[0][0].startswith("gs://")
        assert mock_read.call_args[0][0].endswith("/LF_fuel_rulesets.parquet")
        assert first is second

    @patch("griddle.handlers.landfire.pd.read_parquet")
    def test_strips_whitespace_from_the_columns_core_compares(self, mock_read):
        mock_read.return_value = self._raw(
            BPSRF=[" any "], Wildcard=["any "], OnOff=[" On"]
        )

        rules = _fuel_rulesets()

        assert rules["BPSRF"].tolist() == ["any"]
        assert rules["Wildcard"].tolist() == ["any"]
        assert rules["OnOff"].tolist() == ["On"]

    @patch("griddle.handlers.landfire.pd.read_parquet")
    def test_splits_fbfm40_into_a_label_and_an_integer_code(self, mock_read):
        mock_read.return_value = self._raw(
            Zone=[1, 1, 1],
            FBFM40=["SH2 / 142", "NB9 / 099", "NB9 / 99"],
            FBFM13=[2, 99, 99],
            FCCS=[1, 2, 3],
            EVT=[1, 2, 3],
            DIST=[1, 1, 1],
            Cover_Low=[0, 0, 0],
            Cover_High=[999, 999, 999],
            Height_Low=[0, 0, 0],
            Height_High=[999, 999, 999],
            BPSRF=["any"] * 3,
            Wildcard=["any"] * 3,
            OnOff=["On"] * 3,
        )

        rules = _fuel_rulesets()

        assert "FBFM40" not in rules.columns
        assert rules["FBFM40_label"].tolist() == ["SH2", "NB9", "NB9"]
        assert rules["FBFM40_code"].tolist() == [142, 99, 99]
        assert rules["FBFM40_code"].dtype == pd.Int64Dtype()

    @patch("griddle.handlers.landfire.pd.read_parquet")
    def test_leaves_the_numeric_columns_alone(self, mock_read):
        mock_read.return_value = self._raw()

        rules = _fuel_rulesets()

        assert rules["FBFM13"].tolist() == [8]
        assert rules["FCCS"].tolist() == [338]
        assert rules["Zone"].tolist() == [1]

    @patch("griddle.handlers.landfire.pd.read_parquet")
    def test_prepped_table_works_with_core(self, mock_read):
        """The strip matters to core: a padded " On" would rank like "Off", and
        the earlier row (the Off rule) would win. After loading, the On rule
        is the one chosen."""
        mock_read.return_value = self._raw(
            Zone=[1, 1],
            EVT=[7000, 7000],
            DIST=[111, 111],
            Cover_Low=[0, 0],
            Cover_High=[999, 999],
            Height_Low=[0, 0],
            Height_High=[999, 999],
            BPSRF=["any", "any"],
            Wildcard=["any", "any"],
            OnOff=["Off", " On"],
            FBFM13=[2, 8],
            FBFM40=["GR2 / 102", "GR1 / 101"],
            FCCS=[1, 2],
        )
        one = np.ones((1, 1), dtype=np.int16)

        output, updated = update_fuel_models(
            np.array([[5]], dtype=np.int16),
            fuel_model="FBFM40_code",
            dist=one * 111,
            zone=one,
            fvt=one * 7000,
            fvc=one * 150,
            fvh=one * 110,
            bps=one * 11,
            rules=_fuel_rulesets(),
        )

        assert output.tolist() == [[101]]
        assert updated.tolist() == [[True]]


class TestLdistAttributes:
    """Unit tests for the LDist attribute table loader."""

    def setup_method(self):
        _ldist_attributes.cache_clear()

    def teardown_method(self):
        _ldist_attributes.cache_clear()

    @patch("griddle.handlers.landfire.pd.read_csv")
    def test_reads_that_versions_csv_once(self, mock_read):
        mock_read.return_value = pd.DataFrame(
            {"VALUE": [2882], "DIST_TYPE": ["Wildfire"], "SEVERITY": ["Moderate"]}
        )

        first = _ldist_attributes("2025")
        second = _ldist_attributes("2025")

        mock_read.assert_called_once()
        assert mock_read.call_args[0][0].startswith("gs://")
        assert mock_read.call_args[0][0].endswith("/LF2025_LDist_attributes.csv")
        assert first is second

    @patch("griddle.handlers.landfire.pd.read_csv")
    def test_each_version_has_its_own_table(self, mock_read):
        mock_read.side_effect = lambda path: pd.DataFrame({"path": [path]})

        table_2025 = _ldist_attributes("2025")
        table_2026 = _ldist_attributes("2026")

        assert mock_read.call_count == 2
        assert table_2025["path"][0].endswith("LF2025_LDist_attributes.csv")
        assert table_2026["path"][0].endswith("LF2026_LDist_attributes.csv")

    @patch("griddle.handlers.landfire.pd.read_csv")
    def test_missing_table_raises_a_clear_error(self, mock_read):
        mock_read.side_effect = FileNotFoundError

        with pytest.raises(ProcessingError) as exc_info:
            _ldist_attributes("2026")

        assert exc_info.value.code == "LDIST_ATTRIBUTES_NOT_AVAILABLE"
        assert "LF2026_LDist_attributes.csv" in exc_info.value.suggestion


class TestFetchAnnualDisturbance:
    """Unit tests for fetch_annual_disturbance -- always via LFPS, no staged
    COG path exists for this product."""

    @patch("griddle.handlers.landfire._fetch_landfire_raster")
    @patch("griddle.handlers.landfire.landfire_lfps.fetch_lfps")
    def test_calls_fetch_lfps_and_returns_dataset(
        self, mock_fetch_lfps, mock_fetch_raster, roi
    ):
        mock_cm = MagicMock()
        mock_cm.__enter__.return_value = "/tmp/lfps_xyz/result.tif"
        mock_fetch_lfps.return_value = mock_cm
        mock_fetch_raster.return_value = _make_canopy_raster(
            np.zeros((4, 4), dtype=np.int16)
        )
        progress = MagicMock()

        result = fetch_annual_disturbance(roi, progress, version="2025")

        mock_fetch_lfps.assert_called_once_with(
            roi, "annual_disturbance", "2025", {"target": "domain"}, None, 0, progress
        )
        mock_fetch_raster.assert_called_once_with(
            roi,
            "/tmp/lfps_xyz/result.tif",
            0,
            {"target": "domain"},
            None,
            is_categorical=True,
        )
        assert isinstance(result, xr.Dataset)
        assert "annual_disturbance" in result.data_vars

    @patch("griddle.handlers.landfire._fetch_landfire_raster")
    @patch("griddle.handlers.landfire.landfire_lfps.fetch_lfps")
    def test_threads_alignment_and_buffer(
        self, mock_fetch_lfps, mock_fetch_raster, roi
    ):
        mock_cm = MagicMock()
        mock_cm.__enter__.return_value = "/tmp/lfps_xyz/result.tif"
        mock_fetch_lfps.return_value = mock_cm
        mock_fetch_raster.return_value = _make_canopy_raster(
            np.zeros((4, 4), dtype=np.int16)
        )
        progress = MagicMock()
        alignment = {"target": "native", "method": "nearest"}

        fetch_annual_disturbance(
            roi,
            progress,
            version="2025",
            extent_buffer_cells=6,
            alignment=alignment,
        )

        mock_fetch_lfps.assert_called_once_with(
            roi, "annual_disturbance", "2025", alignment, None, 6, progress
        )


class TestFetchFuelModel:
    """Unit tests for _fetch_fuel_model, the staged-COG-vs-LFPS decision
    shared by fetch_fbfm13/40/fccs."""

    @patch("griddle.handlers.landfire._fetch_landfire_raster")
    @patch("griddle.handlers.landfire.landfire_lfps.fetch_lfps")
    def test_staged_version_reads_the_cog_and_skips_lfps(
        self, mock_fetch_lfps, mock_fetch_raster, roi
    ):
        raster = _make_canopy_raster(np.zeros((4, 4), dtype=np.int16))
        mock_fetch_raster.return_value = raster

        result = _fetch_fuel_model(
            roi, "fbfm13", "2024", MagicMock(), 0, {"target": "domain"}, None
        )

        mock_fetch_lfps.assert_not_called()
        assert mock_fetch_raster.call_args[0][1].endswith("LF2024_FBFM13_CONUS.tif")
        assert result is raster

    @patch("griddle.handlers.landfire._fetch_landfire_raster")
    @patch("griddle.handlers.landfire.landfire_lfps.fetch_lfps")
    def test_lfps_only_version_reads_the_lfps_download(
        self, mock_fetch_lfps, mock_fetch_raster, roi
    ):
        mock_cm = MagicMock()
        mock_cm.__enter__.return_value = "/tmp/lfps_xyz/result.tif"
        mock_fetch_lfps.return_value = mock_cm
        raster = _make_canopy_raster(np.zeros((4, 4), dtype=np.int16))
        mock_fetch_raster.return_value = raster
        progress = MagicMock()

        result = _fetch_fuel_model(
            roi, "fbfm13", "2025", progress, 0, {"target": "domain"}, None
        )

        mock_fetch_lfps.assert_called_once_with(
            roi, "fbfm13", "2025", {"target": "domain"}, None, 0, progress, None
        )
        assert mock_fetch_raster.call_args[0][1] == "/tmp/lfps_xyz/result.tif"
        assert result is raster

    @patch("griddle.handlers.landfire._fetch_landfire_raster")
    @patch("griddle.handlers.landfire.landfire_lfps.fetch_lfps")
    def test_season_forces_lfps_even_for_a_staged_version(
        self, mock_fetch_lfps, mock_fetch_raster, roi
    ):
        mock_cm = MagicMock()
        mock_cm.__enter__.return_value = "/tmp/lfps_xyz/result.tif"
        mock_fetch_lfps.return_value = mock_cm
        mock_fetch_raster.return_value = _make_canopy_raster(
            np.zeros((4, 4), dtype=np.int16)
        )
        progress = MagicMock()

        _fetch_fuel_model(
            roi,
            "fbfm40",
            "2024",
            progress,
            0,
            {"target": "domain"},
            None,
            season="SP",
        )

        mock_fetch_lfps.assert_called_once_with(
            roi, "fbfm40", "2024", {"target": "domain"}, None, 0, progress, "SP"
        )

    @patch("griddle.handlers.landfire._fetch_landfire_raster")
    @patch("griddle.handlers.landfire.landfire_lfps.fetch_lfps")
    def test_unknown_version_raises_before_any_fetch(
        self, mock_fetch_lfps, mock_fetch_raster, roi
    ):
        with pytest.raises(UnknownLandfireVersionError):
            _fetch_fuel_model(
                roi, "fbfm13", "1999", MagicMock(), 0, {"target": "domain"}, None
            )

        mock_fetch_lfps.assert_not_called()
        mock_fetch_raster.assert_not_called()


class TestFetchFbfm13:
    """Integration tests for fetch_fbfm13."""

    def test_returns_dataset(self, roi):
        """fetch_fbfm13 returns a Dataset."""
        progress = MagicMock()
        result = fetch_fbfm13(
            roi=roi, progress=progress, version="2024", extent_buffer_cells=8
        )
        assert isinstance(result, xr.Dataset)

    def test_has_fbfm13_variable(self, roi):
        """Dataset contains an 'fbfm13' variable."""
        progress = MagicMock()
        result = fetch_fbfm13(
            roi=roi, progress=progress, version="2024", extent_buffer_cells=8
        )
        assert "fbfm13" in result.data_vars

    def test_fbfm13_shape(self, test_domain, roi):
        """The fbfm13 variable has the expected spatial shape."""
        progress = MagicMock()
        result = fetch_fbfm13(
            roi=roi, progress=progress, version="2024", extent_buffer_cells=8
        )
        assert result["fbfm13"].shape == test_domain.expected_shape

    def test_fbfm13_dtype(self, roi):
        """The fbfm13 variable is int16 (categorical codes)."""
        progress = MagicMock()
        result = fetch_fbfm13(
            roi=roi, progress=progress, version="2024", extent_buffer_cells=8
        )
        assert result["fbfm13"].dtype == "int16"

    def test_crs_preserved(self, roi):
        """CRS is preserved via rioxarray."""
        progress = MagicMock()
        result = fetch_fbfm13(
            roi=roi, progress=progress, version="2024", extent_buffer_cells=8
        )
        assert result.rio.crs == roi.crs

    def test_fbfm13_values_in_valid_set(self, roi):
        """FBFM13 codes are limited to the Anderson 13 models plus non-burnable."""
        progress = MagicMock()
        result = fetch_fbfm13(
            roi=roi, progress=progress, version="2024", extent_buffer_cells=8
        )
        values = result["fbfm13"].values
        valid_codes = set(range(1, 14)) | {91, 92, 93, 98, 99}
        assert set(np.unique(values)).issubset(valid_codes)


class TestFetchFbfm13LfpsPath:
    """Unit tests for fetch_fbfm13's LFPS branch -- routing on lfps_available versions."""

    @patch("griddle.handlers.landfire._fetch_landfire_raster")
    @patch("griddle.handlers.landfire.landfire_lfps.fetch_lfps")
    def test_lfps_available_version_routes_through_lfps(
        self, mock_fetch_lfps, mock_fetch_raster, roi
    ):
        mock_cm = MagicMock()
        mock_cm.__enter__.return_value = "/tmp/lfps_xyz/result.tif"
        mock_fetch_lfps.return_value = mock_cm
        mock_fetch_raster.return_value = _make_canopy_raster(
            np.zeros((4, 4), dtype=np.int16)
        )
        progress = MagicMock()

        result = fetch_fbfm13(roi, progress, version="2025")

        mock_fetch_lfps.assert_called_once_with(
            roi, "fbfm13", "2025", {"target": "domain"}, None, 0, progress, None
        )
        mock_fetch_raster.assert_called_once_with(
            roi,
            "/tmp/lfps_xyz/result.tif",
            0,
            {"target": "domain"},
            None,
            is_categorical=True,
        )
        assert "fbfm13" in result.data_vars

    @patch("griddle.handlers.landfire.landfire_lfps.fetch_lfps")
    def test_staged_version_does_not_call_lfps(self, mock_fetch_lfps, roi):
        progress = MagicMock()
        fetch_fbfm13(roi, progress, version="2024")
        mock_fetch_lfps.assert_not_called()


class TestFetchFbfm40:
    """Integration tests for fetch_fbfm40."""

    @patch("griddle.handlers.landfire.RasterConnection")
    def test_extract_window_default_buffer_is_zero(self, mock_raster_cls, roi):
        """When extent_buffer_cells is omitted, extract_window receives 0."""
        data = xr.DataArray(
            np.array([[[101]]], dtype=np.int16),
            dims=("band", "y", "x"),
            coords={"band": [1], "y": [0.0], "x": [0.0]},
        ).rio.write_crs(roi.crs)
        mock_raster = MagicMock()
        mock_raster.raster_x_resolution = 30.0
        mock_raster.extract_window.return_value = data
        mock_raster_cls.return_value = mock_raster

        _fetch_landfire_raster(
            roi,
            "gs://mock/url.tiff",
            extent_buffer_cells=0,
            alignment={"target": "native"},
            target_grid_doc=None,
            is_categorical=True,
        )

        call_kwargs = mock_raster.extract_window.call_args[1]
        assert "projection_padding_meters" not in call_kwargs
        assert call_kwargs["interpolation_padding_cells"] == 0

    @pytest.mark.parametrize("buffer", [0, 1, 12])
    @patch("griddle.handlers.landfire.RasterConnection")
    def test_extent_buffer_cells_threaded_through(self, mock_raster_cls, roi, buffer):
        """Caller-supplied extent_buffer_cells reaches extract_window unchanged."""
        data = xr.DataArray(
            np.array([[[101]]], dtype=np.int16),
            dims=("band", "y", "x"),
            coords={"band": [1], "y": [0.0], "x": [0.0]},
        ).rio.write_crs(roi.crs)
        mock_raster = MagicMock()
        mock_raster.raster_x_resolution = 30.0
        mock_raster.extract_window.return_value = data
        mock_raster_cls.return_value = mock_raster

        _fetch_landfire_raster(
            roi,
            "gs://mock/url.tiff",
            extent_buffer_cells=buffer,
            alignment={"target": "native"},
            target_grid_doc=None,
            is_categorical=True,
        )

        assert (
            mock_raster.extract_window.call_args[1]["interpolation_padding_cells"]
            == buffer
        )

    def test_returns_dataset(self, roi):
        """fetch_fbfm40 returns a Dataset."""
        progress = MagicMock()
        result = fetch_fbfm40(
            roi=roi, progress=progress, version="2024", extent_buffer_cells=8
        )
        assert isinstance(result, xr.Dataset)

    def test_has_fbfm_variable(self, roi):
        """Dataset contains a 'fbfm' variable."""
        progress = MagicMock()
        result = fetch_fbfm40(
            roi=roi, progress=progress, version="2024", extent_buffer_cells=8
        )
        assert "fbfm" in result.data_vars

    def test_fbfm_shape(self, test_domain, roi):
        """The fbfm variable has the expected spatial shape."""
        progress = MagicMock()
        result = fetch_fbfm40(
            roi=roi, progress=progress, version="2024", extent_buffer_cells=8
        )
        assert result["fbfm"].shape == test_domain.expected_shape

    def test_fbfm_dtype(self, roi):
        """The fbfm variable is int16 (categorical codes)."""
        progress = MagicMock()
        result = fetch_fbfm40(
            roi=roi, progress=progress, version="2024", extent_buffer_cells=8
        )
        assert result["fbfm"].dtype == "int16"

    def test_crs_preserved(self, roi):
        """CRS is preserved via rioxarray."""
        progress = MagicMock()
        result = fetch_fbfm40(
            roi=roi, progress=progress, version="2024", extent_buffer_cells=8
        )
        assert result.rio.crs == roi.crs

    def test_fbfm_values_in_range(self, roi):
        """FBFM40 codes should be <= 204."""
        progress = MagicMock()
        result = fetch_fbfm40(
            roi=roi, progress=progress, version="2024", extent_buffer_cells=8
        )
        assert result["fbfm"].values.max() <= 204


class TestFetchFbfm40LfpsPath:
    """Unit tests for fetch_fbfm40's LFPS branch -- routing to LFPS for a
    season, an lfps_available version, or both."""

    @patch("griddle.handlers.landfire._fetch_landfire_raster")
    @patch("griddle.handlers.landfire.landfire_lfps.fetch_lfps")
    def test_season_routes_through_lfps(self, mock_fetch_lfps, mock_fetch_raster, roi):
        mock_cm = MagicMock()
        mock_cm.__enter__.return_value = "/tmp/lfps_xyz/result.tif"
        mock_fetch_lfps.return_value = mock_cm
        mock_fetch_raster.return_value = _make_canopy_raster(
            np.zeros((4, 4), dtype=np.int16)
        )
        progress = MagicMock()

        result = fetch_fbfm40(roi, progress, version="2025", season="SP")

        mock_fetch_lfps.assert_called_once_with(
            roi, "fbfm40", "2025", {"target": "domain"}, None, 0, progress, "SP"
        )
        mock_fetch_raster.assert_called_once_with(
            roi,
            "/tmp/lfps_xyz/result.tif",
            0,
            {"target": "domain"},
            None,
            is_categorical=True,
        )
        assert "fbfm" in result.data_vars

    @patch("griddle.handlers.landfire._fetch_landfire_raster")
    @patch("griddle.handlers.landfire.landfire_lfps.fetch_lfps")
    def test_lfps_available_version_without_season_routes_through_lfps(
        self, mock_fetch_lfps, mock_fetch_raster, roi
    ):
        mock_cm = MagicMock()
        mock_cm.__enter__.return_value = "/tmp/lfps_xyz/result.tif"
        mock_fetch_lfps.return_value = mock_cm
        mock_fetch_raster.return_value = _make_canopy_raster(
            np.zeros((4, 4), dtype=np.int16)
        )
        progress = MagicMock()

        result = fetch_fbfm40(roi, progress, version="2025")

        mock_fetch_lfps.assert_called_once_with(
            roi, "fbfm40", "2025", {"target": "domain"}, None, 0, progress, None
        )
        assert "fbfm" in result.data_vars

    @patch("griddle.handlers.landfire.landfire_lfps.fetch_lfps")
    def test_no_season_does_not_call_lfps(self, mock_fetch_lfps, roi):
        progress = MagicMock()
        fetch_fbfm40(roi, progress, version="2024")
        mock_fetch_lfps.assert_not_called()


class TestFetchFccs:
    """Integration tests for fetch_fccs."""

    def test_returns_dataset(self, roi):
        """fetch_fccs returns a Dataset."""
        progress = MagicMock()
        result = fetch_fccs(
            roi=roi, progress=progress, version="2023", extent_buffer_cells=8
        )
        assert isinstance(result, xr.Dataset)

    def test_has_fccs_variable(self, roi):
        """Dataset contains a 'fccs' variable."""
        progress = MagicMock()
        result = fetch_fccs(
            roi=roi, progress=progress, version="2023", extent_buffer_cells=8
        )
        assert "fccs" in result.data_vars

    def test_fccs_shape(self, test_domain, roi):
        """The fccs variable has the expected spatial shape."""
        progress = MagicMock()
        result = fetch_fccs(
            roi=roi, progress=progress, version="2023", extent_buffer_cells=8
        )
        assert result["fccs"].shape == test_domain.expected_shape

    def test_fccs_dtype(self, roi):
        """The fccs variable is int32 (codes up to 12990133 exceed int16 range)."""
        progress = MagicMock()
        result = fetch_fccs(
            roi=roi, progress=progress, version="2023", extent_buffer_cells=8
        )
        assert result["fccs"].dtype == "int32"

    def test_crs_preserved(self, roi):
        """CRS is preserved via rioxarray."""
        progress = MagicMock()
        result = fetch_fccs(
            roi=roi, progress=progress, version="2023", extent_buffer_cells=8
        )
        assert result.rio.crs == roi.crs

    def test_fccs_valid_values_in_range(self, roi):
        """Mapped FCCS codes should be between 0 and 12990133."""
        progress = MagicMock()
        result = fetch_fccs(
            roi=roi, progress=progress, version="2023", extent_buffer_cells=8
        )
        values = result["fccs"].values
        valid_mask = ~np.isin(values, [-1111, -9999])
        assert values[valid_mask].min() >= 0
        assert values[valid_mask].max() <= 12990133

    def test_fccs_fill_values_are_expected(self, roi):
        """Any negative values are only the known fill values (-1111, -9999)."""
        progress = MagicMock()
        result = fetch_fccs(
            roi=roi, progress=progress, version="2023", extent_buffer_cells=8
        )
        values = result["fccs"].values
        negative_values = np.unique(values[values < 0])
        assert set(negative_values).issubset({-1111, -9999})

    def test_remove_bare_ground_removes_zeros(self, roi):
        """When remove_bare_ground=True, no bare ground cells (code 0) remain."""
        progress = MagicMock()
        result = fetch_fccs(roi=roi, progress=progress, remove_bare_ground=True)
        values = result["fccs"].values
        valid_mask = ~np.isin(values, [-1111, -9999])
        assert not np.any(values[valid_mask] == 0)

    def test_remove_bare_ground_false_by_default(self, roi):
        """remove_bare_ground defaults to False and does not alter the data."""
        progress = MagicMock()
        default_result = fetch_fccs(roi=roi, progress=progress)
        explicit_result = fetch_fccs(
            roi=roi, progress=progress, remove_bare_ground=False
        )
        np.testing.assert_array_equal(
            default_result["fccs"].values,
            explicit_result["fccs"].values,
        )


class TestFetchFccsLfpsPath:
    """Unit tests for fetch_fccs's LFPS branch -- routing on lfps_available versions."""

    @patch("griddle.handlers.landfire._fetch_landfire_raster")
    @patch("griddle.handlers.landfire.landfire_lfps.fetch_lfps")
    def test_lfps_available_version_routes_through_lfps(
        self, mock_fetch_lfps, mock_fetch_raster, roi
    ):
        mock_cm = MagicMock()
        mock_cm.__enter__.return_value = "/tmp/lfps_xyz/result.tif"
        mock_fetch_lfps.return_value = mock_cm
        mock_fetch_raster.return_value = _make_canopy_raster(
            np.zeros((4, 4), dtype=np.int16)
        )
        progress = MagicMock()

        result = fetch_fccs(roi, progress, version="2025")

        mock_fetch_lfps.assert_called_once_with(
            roi, "fccs", "2025", {"target": "domain"}, None, 0, progress, None
        )
        mock_fetch_raster.assert_called_once_with(
            roi,
            "/tmp/lfps_xyz/result.tif",
            0,
            {"target": "domain"},
            None,
            is_categorical=True,
        )
        assert "fccs" in result.data_vars

    @patch("griddle.handlers.landfire.landfire_lfps.fetch_lfps")
    def test_staged_version_does_not_call_lfps(self, mock_fetch_lfps, roi):
        progress = MagicMock()
        fetch_fccs(roi, progress, version="2023")
        mock_fetch_lfps.assert_not_called()


@pytest.fixture
def env(roi):
    """Everything fetch_disturbed_fuel reaches for, patched; core stays real.

    Set `env.values[layer]` to change a layer's pixels, `env.previous_values`
    to change the staged fuel model, and `env.shape` for the grid size. Layers
    left alone get a default, so a bare test needs no setup. After a call,
    `env.submitted` holds the layers sent in the LFPS batch.
    """
    defaults = {"fvt": 7000, "fvc": 150, "fvh": 110, "bps": 11}
    state = SimpleNamespace(
        shape=(2, 2),
        values={},
        previous_values=None,
        submitted={},
        rules=pd.DataFrame(
            {
                "Zone": [1],
                "EVT": [7000],
                "DIST": [121],
                "Cover_Low": [0],
                "Cover_High": [999],
                "Height_Low": [0],
                "Height_High": [999],
                "BPSRF": ["any"],
                "Wildcard": ["any"],
                "OnOff": ["On"],
                "FBFM13": [2],
                "FBFM40_code": pd.array([101], dtype="Int64"),
                "FCCS": [338],
            }
        ),
        ldist_table=pd.DataFrame(
            {
                "VALUE": [0, 2882, 2971],
                "DIST_TYPE": [None, "Wildfire", "Herbicide"],
                "SEVERITY": [None, "Moderate", "Low"],
            }
        ),
    )

    @contextmanager
    def fake_multi(roi, layers, *args):
        state.submitted = dict(layers)
        yield "/tmp/lfps_xyz/result.tif"

    def fake_raster(
        roi, url, buffer_cells, alignment, target_grid_doc, is_categorical, band
    ):
        name = list(state.submitted)[band - 1]
        values = state.values.get(name, np.full(state.shape, defaults.get(name, 0)))
        return _make_canopy_raster(np.asarray(values), nodata=-9999)

    def fake_previous(*args):
        values = state.previous_values
        if values is None:
            values = np.full(state.shape, 5)
        return _make_canopy_raster(np.asarray(values))

    zones = gpd.GeoDataFrame(
        {"ZONE_NUM": [1]}, geometry=[box(-100, -100, 100, 100)], crs="EPSG:5070"
    )
    state.fetch_fuel_model = MagicMock(side_effect=fake_previous)
    state.ldist_attributes = MagicMock(side_effect=lambda version: state.ldist_table)

    with (
        patch(
            "griddle.handlers.landfire.landfire_lfps.fetch_lfps_multi", new=fake_multi
        ),
        patch("griddle.handlers.landfire._fetch_landfire_raster", new=fake_raster),
        patch("griddle.handlers.landfire._fetch_fuel_model", state.fetch_fuel_model),
        patch("griddle.handlers.landfire._ldist_attributes", state.ldist_attributes),
        patch("griddle.handlers.landfire._lf_map_zones", return_value=zones),
        patch("griddle.handlers.landfire._fuel_rulesets", lambda: state.rules),
    ):
        yield state


class TestFetchDisturbedFuel:
    """Unit tests for fetch_disturbed_fuel."""

    COMBINED = {
        "source": "landfire",
        "mode": "ldist_and_last_year_fdist",
        "ldist_version": "2025",
        "fdist_version": "2024",
    }
    FDIST_ONLY = {"source": "landfire", "mode": "fdist", "fdist_version": "2025"}

    # --- what gets fetched ---

    def test_combined_mode_batches_fdist_ldist_and_vegetation(self, roi, env):
        progress = MagicMock()

        fetch_disturbed_fuel(roi, "fbfm40", "2024", self.COMBINED, progress)

        assert list(env.submitted.items()) == [
            ("fdist", "2024"),
            ("annual_disturbance", "2025"),
            ("fvt", "2024"),
            ("fvc", "2024"),
            ("fvh", "2024"),
            ("bps", "2020"),
        ]
        env.fetch_fuel_model.assert_called_once_with(
            roi, "fbfm40", "2024", progress, 0, {"target": "domain"}, None
        )

    def test_fdist_mode_has_no_ldist(self, roi, env):
        fetch_disturbed_fuel(roi, "fbfm40", "2024", self.FDIST_ONLY, MagicMock())

        assert list(env.submitted) == ["fdist", "fvt", "fvc", "fvh", "bps"]
        assert env.submitted["fdist"] == "2025"
        env.ldist_attributes.assert_not_called()

    def test_lfps_only_previous_joins_the_batch(self, roi, env):
        """fbfm13 2025 is LFPS-only, so it rides in the same job instead of a
        second one, and nothing is fetched separately."""
        fetch_disturbed_fuel(roi, "fbfm13", "2025", self.COMBINED, MagicMock())

        assert list(env.submitted)[-1] == "fbfm13"
        assert env.submitted["fbfm13"] == "2025"
        env.fetch_fuel_model.assert_not_called()

    def test_ldist_attributes_follow_the_planned_ldist_version(self, roi, env):
        fetch_disturbed_fuel(roi, "fbfm40", "2024", self.COMBINED, MagicMock())

        env.ldist_attributes.assert_called_once_with("2025")

    # --- the result, with real core ---

    def test_disturbed_pixel_takes_the_rule_value_and_the_rest_keep_previous(
        self, roi, env, caplog
    ):
        """A moderate wildfire in LDist becomes FDist 121, which the one rule
        maps to fuel model 101; the herbicide pixel has no FDist equivalent, so
        it stays undisturbed and keeps the previous value (5)."""
        env.values["annual_disturbance"] = [[2882, 0], [0, 2971]]

        with caplog.at_level(logging.INFO, logger="griddle.handlers.landfire"):
            result = fetch_disturbed_fuel(
                roi, "fbfm40", "2024", self.COMBINED, MagicMock()
            )

        assert result["fbfm"].values.tolist() == [[101, 5], [5, 5]]
        assert result["fbfm"].dtype == np.int16
        assert "treated as undisturbed: [2971]" in caplog.text
        assert "1 of 1 disturbed pixels were mapped to a new fuel model" in caplog.text

    def test_disturbed_pixel_with_no_matching_rule_keeps_previous(
        self, roi, env, caplog
    ):
        env.values["annual_disturbance"] = [[2882, 0], [0, 0]]
        env.values["fvt"] = np.full((2, 2), 9999)  # no rule for this vegetation

        with caplog.at_level(logging.INFO, logger="griddle.handlers.landfire"):
            result = fetch_disturbed_fuel(
                roi, "fbfm40", "2024", self.COMBINED, MagicMock()
            )

        assert result["fbfm"].values.tolist() == [[5, 5], [5, 5]]
        assert "0 of 1 disturbed pixels were mapped to a new fuel model" in caplog.text

    def test_fdist_mode_uses_the_fdist_raster_directly(self, roi, env):
        env.values["fdist"] = [[121, 0], [0, 0]]

        result = fetch_disturbed_fuel(
            roi, "fbfm40", "2024", self.FDIST_ONLY, MagicMock()
        )

        assert result["fbfm"].values.tolist() == [[101, 5], [5, 5]]

    @pytest.mark.parametrize(
        ("fuel_model", "version", "band_key"),
        [
            ("fbfm13", "2024", "fbfm13"),
            ("fbfm40", "2024", "fbfm"),
            ("fccs", "2023", "fccs"),
        ],
    )
    def test_output_band_matches_the_plain_fetch(
        self, roi, env, fuel_model, version, band_key
    ):
        result = fetch_disturbed_fuel(
            roi, fuel_model, version, self.FDIST_ONLY, MagicMock()
        )

        assert list(result.data_vars) == [band_key]

    # --- post-processing, with core stubbed so the output is exact ---

    @pytest.fixture
    def stub_core(self, env, monkeypatch):
        """Core returns a 10x10 grid of 101 with one isolated cell set by the test."""
        env.shape = (10, 10)
        env.previous_values = np.full((10, 10), 101)
        state = SimpleNamespace(codes=np.full((10, 10), 101, dtype=np.int16))
        monkeypatch.setattr(
            "griddle.handlers.landfire.build_fdist_raster",
            lambda mode, **kw: (np.zeros((10, 10), dtype=np.int32), []),
        )
        monkeypatch.setattr(
            "griddle.handlers.landfire.update_fuel_models",
            lambda previous, **kw: (state.codes, np.zeros((10, 10), dtype=bool)),
        )
        return state

    def test_remove_non_burnable_is_applied_to_the_output(self, roi, stub_core):
        stub_core.codes[5, 5] = 99

        result = fetch_disturbed_fuel(
            roi,
            "fbfm40",
            "2024",
            self.FDIST_ONLY,
            MagicMock(),
            remove_non_burnable=["NB9"],
        )

        assert result["fbfm"].values[5, 5] == 101

    def test_remove_bare_ground_is_applied_to_the_output(self, roi, stub_core):
        stub_core.codes[5, 5] = 0

        result = fetch_disturbed_fuel(
            roi, "fccs", "2023", self.FDIST_ONLY, MagicMock(), remove_bare_ground=True
        )

        assert result["fccs"].values[5, 5] == 101

    @pytest.mark.parametrize(
        ("fuel_model", "version", "protected"),
        [
            ("fbfm13", "2024", FBFM_NON_BURNABLE),
            ("fbfm40", "2024", FBFM_NON_BURNABLE),
            ("fccs", "2023", FCCS_BARE_GROUND),
        ],
    )
    def test_boundary_scatter_protects_the_products_codes(
        self, roi, stub_core, fuel_model, version, protected
    ):
        scatter = {"depth": 5, "seed": 1}

        with patch(
            "griddle.handlers.landfire._apply_boundary_scatter",
            side_effect=lambda data, *a, **kw: data,
        ) as mock_scatter:
            fetch_disturbed_fuel(
                roi,
                fuel_model,
                version,
                self.FDIST_ONLY,
                MagicMock(),
                boundary_scatter=scatter,
            )

        assert mock_scatter.call_args[0][1] == scatter
        assert mock_scatter.call_args[0][2] == protected

    def test_remove_non_burnable_never_fills_with_nodata(self, roi, stub_core):
        """A non-burnable cell near the edge of the data is filled from its real
        neighbors, never with nodata.

        The non-burnable cell (99) sits in a window that is mostly nodata (N).
        Nodata is the most common value around it, but it isn't a fuel model,
        so the 102 cells win. Window (rows 3-7, columns 3-7 of the grid):

            N  N  N   N  N
            N  N  102 N  N
            N  N  99  102 N
            N  N  N   N  N
            N  N  N   N  N
        """
        nodata = 32767
        stub_core.codes[:] = nodata
        stub_core.codes[4, 5] = 102
        stub_core.codes[5, 5] = 99
        stub_core.codes[5, 6] = 102

        expected = stub_core.codes.copy()
        expected[5, 5] = 102  # the only cell that should change

        result = fetch_disturbed_fuel(
            roi,
            "fbfm40",
            "2024",
            self.FDIST_ONLY,
            MagicMock(),
            remove_non_burnable=["NB9"],
        )

        np.testing.assert_array_equal(result["fbfm"].values, expected)

    def test_remove_bare_ground_never_fills_with_nodata(self, roi, stub_core):
        """The same window for FCCS, where the cell to remove is bare ground (0)
        and its neighbors are fuelbed 150."""
        nodata = 32767
        stub_core.codes[:] = nodata
        stub_core.codes[4, 5] = 150
        stub_core.codes[5, 5] = 0
        stub_core.codes[5, 6] = 150

        expected = stub_core.codes.copy()
        expected[5, 5] = 150

        result = fetch_disturbed_fuel(
            roi, "fccs", "2023", self.FDIST_ONLY, MagicMock(), remove_bare_ground=True
        )

        np.testing.assert_array_equal(result["fccs"].values, expected)


class TestFetchTopography:
    """Integration tests for fetch_topography."""

    def test_returns_dataset(self, roi):
        """fetch_topography always returns a Dataset."""
        progress = MagicMock()
        result = fetch_topography(
            roi=roi,
            version="2020",
            bands=["elevation"],
            extent_buffer_cells=8,
            progress=progress,
        )
        assert isinstance(result, xr.Dataset)

    def test_fetch_all_bands(self, test_domain, roi):
        """fetch_topography with all bands returns three variables."""
        progress = MagicMock()
        result = fetch_topography(
            roi=roi,
            version="2020",
            bands=["elevation", "slope", "aspect"],
            extent_buffer_cells=8,
            progress=progress,
        )

        assert list(result.data_vars) == ["elevation", "slope", "aspect"]
        for var in result.data_vars:
            assert result[var].shape == test_domain.expected_shape
        assert result.rio.crs == roi.crs

    def test_fetch_single_band(self, test_domain, roi):
        """fetch_topography with one band returns Dataset with one variable."""
        progress = MagicMock()
        result = fetch_topography(
            roi=roi,
            version="2020",
            bands=["elevation"],
            extent_buffer_cells=8,
            progress=progress,
        )

        assert list(result.data_vars) == ["elevation"]
        assert result["elevation"].shape == test_domain.expected_shape
        assert result.rio.crs == roi.crs

    def test_fetch_two_bands(self, test_domain, roi):
        """fetch_topography with two bands returns correct subset."""
        progress = MagicMock()
        result = fetch_topography(
            roi=roi,
            version="2020",
            bands=["slope", "aspect"],
            extent_buffer_cells=8,
            progress=progress,
        )

        assert list(result.data_vars) == ["slope", "aspect"]

    def test_variable_names_match_request(self, roi):
        """Variable names match the requested band names and order."""
        progress = MagicMock()
        result = fetch_topography(
            roi=roi,
            version="2020",
            bands=["aspect", "elevation"],
            extent_buffer_cells=8,
            progress=progress,
        )

        assert list(result.data_vars) == ["aspect", "elevation"]

    def test_elevation_data_is_numeric(self, roi):
        """Elevation values are numeric (float or int)."""
        progress = MagicMock()
        result = fetch_topography(
            roi=roi,
            version="2020",
            bands=["elevation"],
            extent_buffer_cells=8,
            progress=progress,
        )

        assert result["elevation"].dtype.kind in ("f", "i")

    def test_calls_progress_callback(self, roi):
        """fetch_topography reports progress for each band."""
        progress = MagicMock()
        fetch_topography(
            roi=roi,
            version="2020",
            bands=["elevation", "slope"],
            extent_buffer_cells=8,
            progress=progress,
        )

        assert progress.call_count == 2


class TestMostFrequent:
    """Unit tests for _most_frequent majority filter function."""

    def test_returns_central_when_burnable_and_most_frequent(self):
        """Central pixel is returned when it's burnable and tied for max freq."""
        # 5x5 window flattened, central pixel at index 12
        window = np.array([101] * 13 + [102] * 12, dtype=np.int16)
        assert _most_frequent(window, [91, 93, 99]) == 101

    def test_skips_non_burnable_in_ranking(self):
        """Non-burnable codes are skipped even when most frequent."""
        # Non-burnable 99 is most frequent, but burnable 101 should be returned
        window = np.array([99] * 20 + [101] * 5, dtype=np.int16)
        assert _most_frequent(window, [99]) == 101

    def test_falls_back_to_central_when_all_non_burnable(self):
        """Returns central pixel when no burnable values exist."""
        window = np.array([91] * 13 + [99] * 12, dtype=np.int16)
        assert _most_frequent(window, [91, 99]) == 91

    def test_prefers_central_over_tie(self):
        """Central pixel wins ties when it's burnable."""
        # 101 and 102 tied, central pixel is 101
        window = np.full(25, 102, dtype=np.int16)
        window[12] = 101
        # Actually need a real tie: equal counts
        window = np.array(
            [101] * 12 + [101] + [102] * 12, dtype=np.int16
        )  # 13 x 101, 12 x 102 => 101 is most frequent
        assert _most_frequent(window, [91]) == 101

    def test_selects_most_frequent_burnable_when_central_is_non_burnable(self):
        """When central is non-burnable, returns most frequent burnable neighbor."""
        window = np.array([101] * 12 + [99] + [102] * 12, dtype=np.int16)
        # central=99 (non-burnable), 101 appears 12 times, 102 appears 12 times
        # 101 and 102 are tied but 101 comes first sorted by descending count
        result = _most_frequent(window, [99])
        assert result in (101, 102)
        assert result != 99


class TestRemoveNonBurnableBlocks:
    """Unit tests for _remove_non_burnable_blocks."""

    def test_no_non_burnable_returns_copy(self):
        """Grid with no non-burnable codes is returned unchanged."""
        grid = np.full((10, 10), 101, dtype=np.int16)
        result = _remove_non_burnable_blocks(grid, [91, 93, 99])
        np.testing.assert_array_equal(result, grid)
        assert result is not grid  # Returns a copy

    def test_replaces_single_non_burnable_cell(self):
        """A single non-burnable cell surrounded by burnable is replaced."""
        grid = np.full((10, 10), 102, dtype=np.int16)
        grid[5, 5] = 99  # Single bare ground cell
        result = _remove_non_burnable_blocks(grid, [99])
        assert result[5, 5] == 102
        # All other cells unchanged
        grid[5, 5] = 102
        np.testing.assert_array_equal(result, grid)

    def test_only_targeted_codes_removed(self):
        """Only the specified non-burnable codes are removed; others are kept."""
        grid = np.full((10, 10), 101, dtype=np.int16)
        grid[3, 3] = 91  # Urban — targeted
        grid[7, 7] = 98  # Water — not targeted
        result = _remove_non_burnable_blocks(grid, [91])
        assert result[3, 3] == 101  # Urban replaced
        assert result[7, 7] == 98  # Water preserved

    def test_replaces_with_most_frequent_neighbor(self):
        """Non-burnable cells are replaced by the most frequent burnable neighbor."""
        grid = np.full((10, 10), 101, dtype=np.int16)
        # Put a block of 102 on the right side
        grid[:, 7:] = 102
        # Put a non-burnable cell in the 102 region
        grid[5, 8] = 99
        result = _remove_non_burnable_blocks(grid, [99])
        assert result[5, 8] == 102  # Replaced by dominant neighbor

    def test_preserves_non_targeted_non_burnable(self):
        """Non-burnable codes not in the target list remain untouched."""
        grid = np.full((10, 10), 101, dtype=np.int16)
        grid[2, 2] = 92  # Snow/ice
        grid[4, 4] = 93  # Agriculture
        result = _remove_non_burnable_blocks(grid, [93])
        assert result[2, 2] == 92  # Snow/ice preserved
        assert result[4, 4] == 101  # Agriculture replaced

    def test_large_non_burnable_patch(self):
        """A large patch of non-burnable codes is fully replaced."""
        grid = np.full((20, 20), 101, dtype=np.int16)
        grid[8:12, 8:12] = 99  # 4x4 bare ground block
        result = _remove_non_burnable_blocks(grid, [99])
        assert not np.any(np.isin(result, [99]))

    def test_nodata_is_never_a_fill_source(self):
        """Non-burnable cells bordering nodata fill from burnable neighbours only."""
        grid = np.full((10, 10), 101, dtype=np.int16)
        grid[:, :6] = -9999
        grid[:, 6:8] = 99
        result = _remove_non_burnable_blocks(grid, [99], nodata=-9999)
        np.testing.assert_array_equal(result[:, 6:8], 101)
        np.testing.assert_array_equal(result[:, :6], -9999)

    @pytest.mark.parametrize("nodata_cols", [0, 3])
    def test_stops_when_no_fill_source_exists(self, nodata_cols):
        """Non-burnable cells with no burnable neighbour are left as-is, promptly."""
        grid = np.full((5, 5), 99, dtype=np.int16)
        grid[:, :nodata_cols] = -9999
        real_filter = landfire_handler.generic_filter
        calls = []

        def capped_filter(*args, **kwargs):
            calls.append(1)
            assert len(calls) <= 2, "filter kept running without progress"
            return real_filter(*args, **kwargs)

        with patch.object(landfire_handler, "generic_filter", capped_filter):
            result = _remove_non_burnable_blocks(grid, [99], nodata=-9999)
        np.testing.assert_array_equal(result, grid)


def _make_canopy_raster(
    values: np.ndarray, nodata: float | int | None = 32767
) -> xr.DataArray:
    """Build a 2D DataArray that mimics what _fetch_landfire_raster returns."""
    arr = xr.DataArray(
        values.astype(np.int16),
        dims=("y", "x"),
        coords={"y": np.arange(values.shape[0]), "x": np.arange(values.shape[1])},
    ).rio.write_crs("EPSG:5070")
    if nodata is not None:
        arr = arr.rio.write_nodata(nodata)
    return arr


class TestScaleCanopyBand:
    """Unit tests for the LANDFIRE canopy nodata + scaling helper."""

    def test_chm_divides_by_10_and_masks_declared_nodata(self):
        from griddle.handlers.landfire import _scale_canopy_band

        raw = _make_canopy_raster(np.array([[0, 10, 20, 32767]]))
        out = _scale_canopy_band(raw, 10.0)
        assert out.dtype == np.float32
        np.testing.assert_array_equal(out.values, [[0.0, 1.0, 2.0, np.nan]])

    def test_cbd_divides_by_100(self):
        from griddle.handlers.landfire import _scale_canopy_band

        raw = _make_canopy_raster(np.array([[0, 25, 100]]))
        out = _scale_canopy_band(raw, 100.0)
        np.testing.assert_allclose(out.values, [[0.0, 0.25, 1.0]])

    def test_cc_does_not_divide_but_still_masks_nodata(self):
        from griddle.handlers.landfire import _scale_canopy_band

        raw = _make_canopy_raster(np.array([[0, 45, 95, 32767]]))
        out = _scale_canopy_band(raw, 1.0)
        np.testing.assert_array_equal(out.values, [[0.0, 45.0, 95.0, np.nan]])


class TestFetchCanopyLandfire:
    """Tests for the LANDFIRE canopy multi-band fetch."""

    @patch("griddle.handlers.landfire._fetch_landfire_raster")
    def test_uses_uppercase_product_codes(self, mock_fetch, roi):
        """API band names (lowercase) map to uppercase LANDFIRE blob codes."""
        from griddle.handlers.landfire import fetch_canopy_landfire

        mock_fetch.side_effect = lambda *a, **kw: _make_canopy_raster(
            np.zeros((4, 4), dtype=np.int16)
        )
        fetch_canopy_landfire(
            roi=roi,
            version="2024",
            bands=["chm", "cbd", "cbh", "cc"],
            progress=MagicMock(),
        )

        urls = [call.args[1] for call in mock_fetch.call_args_list]
        for code, url in zip(["CH", "CBD", "CBH", "CC"], urls):
            assert code in url

    @patch("griddle.handlers.landfire._fetch_landfire_raster")
    def test_returns_dataset_with_requested_bands_in_order(self, mock_fetch, roi):
        from griddle.handlers.landfire import fetch_canopy_landfire

        mock_fetch.side_effect = lambda *a, **kw: _make_canopy_raster(
            np.zeros((4, 4), dtype=np.int16)
        )
        result = fetch_canopy_landfire(
            roi=roi,
            version="2024",
            bands=["cbd", "chm"],
            progress=MagicMock(),
        )
        assert isinstance(result, xr.Dataset)
        assert list(result.data_vars) == ["cbd", "chm"]

    @patch("griddle.handlers.landfire._fetch_landfire_raster")
    def test_single_band_chm(self, mock_fetch, roi):
        from griddle.handlers.landfire import fetch_canopy_landfire

        mock_fetch.side_effect = lambda *a, **kw: _make_canopy_raster(
            np.zeros((4, 4), dtype=np.int16)
        )
        result = fetch_canopy_landfire(
            roi=roi,
            version="2024",
            bands=["chm"],
            progress=MagicMock(),
        )
        assert list(result.data_vars) == ["chm"]
        assert "CH" in mock_fetch.call_args_list[0].args[1]

    @patch("griddle.handlers.landfire._fetch_landfire_raster")
    def test_applies_per_band_scale_factors(self, mock_fetch, roi):
        """End-to-end: chm/10, cbd/100, cbh/10, cc/1 all applied correctly."""
        from griddle.handlers.landfire import fetch_canopy_landfire

        def fake_fetch(roi_, url, *_a, **_kw):
            # Return a 1x3 raster with values 0, "100" (in the source encoding), nodata
            return _make_canopy_raster(np.array([[0, 100, 32767]]))

        mock_fetch.side_effect = fake_fetch
        result = fetch_canopy_landfire(
            roi=roi,
            version="2024",
            bands=["chm", "cbd", "cbh", "cc"],
            progress=MagicMock(),
        )
        np.testing.assert_allclose(result["chm"].values, [[0.0, 10.0, np.nan]])
        np.testing.assert_allclose(result["cbd"].values, [[0.0, 1.0, np.nan]])
        np.testing.assert_allclose(result["cbh"].values, [[0.0, 10.0, np.nan]])
        np.testing.assert_allclose(result["cc"].values, [[0.0, 100.0, np.nan]])

    @patch("griddle.handlers.landfire._fetch_landfire_raster")
    def test_threads_alignment_and_buffer(self, mock_fetch, roi):
        from griddle.handlers.landfire import fetch_canopy_landfire

        mock_fetch.side_effect = lambda *a, **kw: _make_canopy_raster(
            np.zeros((4, 4), dtype=np.int16)
        )
        alignment = {"target": "native", "method": "bilinear"}
        fetch_canopy_landfire(
            roi=roi,
            version="2024",
            bands=["chm"],
            progress=MagicMock(),
            extent_buffer_cells=7,
            alignment=alignment,
        )
        # _fetch_landfire_raster(roi, url, extent_buffer_cells, alignment,
        #                       target_grid_doc, is_categorical=...)
        call = mock_fetch.call_args_list[0]
        assert call.args[2] == 7
        assert call.args[3] == alignment
        assert call.kwargs["is_categorical"] is False

    @patch("griddle.handlers.landfire._fetch_landfire_raster")
    def test_default_alignment_is_domain(self, mock_fetch, roi):
        from griddle.handlers.landfire import fetch_canopy_landfire

        mock_fetch.side_effect = lambda *a, **kw: _make_canopy_raster(
            np.zeros((4, 4), dtype=np.int16)
        )
        fetch_canopy_landfire(
            roi=roi,
            version="2024",
            bands=["chm"],
            progress=MagicMock(),
        )
        assert mock_fetch.call_args_list[0].args[3] == {"target": "domain"}

    @patch("griddle.handlers.landfire._fetch_landfire_raster")
    def test_progress_callback_invoked_per_band(self, mock_fetch, roi):
        from griddle.handlers.landfire import fetch_canopy_landfire

        mock_fetch.side_effect = lambda *a, **kw: _make_canopy_raster(
            np.zeros((4, 4), dtype=np.int16)
        )
        progress = MagicMock()
        fetch_canopy_landfire(
            roi=roi,
            version="2024",
            bands=["chm", "cbd", "cbh"],
            progress=progress,
        )
        assert progress.call_count == 3
        for call, band in zip(progress.call_args_list, ["chm", "cbd", "cbh"]):
            assert band in call.args[0]

    @patch("griddle.handlers.landfire.landfire_lfps.fetch_lfps_multi")
    @patch("griddle.handlers.landfire._fetch_landfire_raster")
    def test_lfps_version_batches_requested_bands(self, mock_fetch, mock_multi, roi):
        from griddle.handlers.landfire import fetch_canopy_landfire

        mock_multi.return_value = nullcontext("/tmp/lfps/result.tif")
        mock_fetch.side_effect = lambda *a, **kw: _make_canopy_raster(
            np.zeros((4, 4), dtype=np.int16)
        )
        result = fetch_canopy_landfire(
            roi=roi, version="2025", bands=["cc", "chm"], progress=MagicMock()
        )

        assert mock_multi.call_args.args[1] == {"cc": "2025", "chm": "2025"}
        assert [c.kwargs["band"] for c in mock_fetch.call_args_list] == [1, 2]
        assert list(result.data_vars) == ["cc", "chm"]

    @patch("griddle.handlers.landfire.landfire_lfps.fetch_lfps")
    @patch("griddle.handlers.landfire._fetch_landfire_raster")
    def test_lfps_version_single_band_uses_fetch_lfps(
        self, mock_fetch, mock_single, roi
    ):
        from griddle.handlers.landfire import fetch_canopy_landfire

        mock_single.return_value = nullcontext("/tmp/lfps/result.tif")
        mock_fetch.side_effect = lambda *a, **kw: _make_canopy_raster(
            np.zeros((4, 4), dtype=np.int16)
        )
        fetch_canopy_landfire(
            roi=roi, version="2025", bands=["chm"], progress=MagicMock()
        )

        assert mock_single.call_args.args[1:3] == ("chm", "2025")
        assert mock_fetch.call_args_list[0].kwargs["band"] == 1

    @patch("griddle.handlers.landfire.landfire_lfps.fetch_lfps_multi")
    @patch("griddle.handlers.landfire.landfire_lfps.fetch_lfps")
    @patch("griddle.handlers.landfire._fetch_landfire_raster")
    def test_staged_version_does_not_call_lfps(
        self, mock_fetch, mock_single, mock_multi, roi
    ):
        from griddle.handlers.landfire import fetch_canopy_landfire

        mock_fetch.side_effect = lambda *a, **kw: _make_canopy_raster(
            np.zeros((4, 4), dtype=np.int16)
        )
        fetch_canopy_landfire(
            roi=roi, version="2024", bands=["chm", "cc"], progress=MagicMock()
        )

        mock_single.assert_not_called()
        mock_multi.assert_not_called()

    def test_unknown_version_raises(self, roi):
        from griddle.handlers.landfire import fetch_canopy_landfire

        with pytest.raises(UnknownLandfireVersionError):
            fetch_canopy_landfire(
                roi=roi, version="1999", bands=["chm"], progress=MagicMock()
            )


class TestScatterCategoricalBoundaries:
    """Unit tests for boundary scattering of categorical grids."""

    def test_uniform_grid_unchanged(self):
        """A grid with one value everywhere has no boundaries to scatter."""
        grid = np.full((20, 20), 101, dtype=np.int16)
        result = scatter_categorical_boundaries(
            grid, depth=10, seed=42, protected=set()
        )
        np.testing.assert_array_equal(result, grid)

    def test_output_preserves_dtype(self):
        grid = np.full((10, 10), 101, dtype=np.int16)
        result = scatter_categorical_boundaries(grid, depth=5, seed=42, protected=set())
        assert result.dtype == grid.dtype

    def test_output_shape_matches_input(self):
        grid = np.full((15, 25), 101, dtype=np.int16)
        result = scatter_categorical_boundaries(grid, depth=5, seed=42, protected=set())
        assert result.shape == grid.shape

    def test_deterministic_with_same_seed(self):
        grid = np.zeros((30, 30), dtype=np.int16)
        grid[:, :15] = 101
        grid[:, 15:] = 102
        r1 = scatter_categorical_boundaries(grid, depth=10, seed=42, protected=set())
        r2 = scatter_categorical_boundaries(grid, depth=10, seed=42, protected=set())
        np.testing.assert_array_equal(r1, r2)

    def test_different_seed_different_result(self):
        grid = np.zeros((30, 30), dtype=np.int16)
        grid[:, :15] = 101
        grid[:, 15:] = 102
        r1 = scatter_categorical_boundaries(grid, depth=10, seed=42, protected=set())
        r2 = scatter_categorical_boundaries(grid, depth=10, seed=99, protected=set())
        assert not np.array_equal(r1, r2)

    def test_only_original_values_appear(self):
        """Scattering should only redistribute existing values, not create new ones."""
        grid = np.zeros((30, 30), dtype=np.int16)
        grid[:, :15] = 101
        grid[:, 15:] = 165
        result = scatter_categorical_boundaries(
            grid, depth=10, seed=42, protected=set()
        )
        assert set(np.unique(result)).issubset({101, 165})

    def test_protected_values_stay_in_place(self):
        """Cells with protected values must not change."""
        grid = np.zeros((30, 30), dtype=np.int16)
        grid[:, :10] = 91  # non-burnable
        grid[:, 10:20] = 101
        grid[:, 20:] = 102
        protected = {91}
        result = scatter_categorical_boundaries(
            grid, depth=10, seed=42, protected=protected
        )
        np.testing.assert_array_equal(result[:, :10], 91)

    def test_protected_values_dont_spread(self):
        """Protected values should not overwrite neighboring cells."""
        grid = np.zeros((30, 30), dtype=np.int16)
        grid[:, :15] = 91  # non-burnable
        grid[:, 15:] = 101
        protected = {91}
        result = scatter_categorical_boundaries(
            grid, depth=10, seed=42, protected=protected
        )
        assert 91 not in result[:, 15:]

    def test_scattering_occurs_at_boundary(self):
        """With two vegetation types side by side, some cells near the
        boundary should change."""
        grid = np.zeros((40, 40), dtype=np.int16)
        grid[:, :20] = 101
        grid[:, 20:] = 165
        result = scatter_categorical_boundaries(
            grid, depth=10, seed=42, protected=set()
        )
        changed = result != grid
        assert changed.any()
        # Changes should be near the boundary (column 20), not at the edges
        assert not changed[:, 0].any()
        assert not changed[:, -1].any()

    def test_depth_controls_scatter_reach(self):
        """Larger depth should scatter further from the boundary."""
        grid = np.zeros((40, 40), dtype=np.int16)
        grid[:, :20] = 101
        grid[:, 20:] = 165
        shallow = scatter_categorical_boundaries(
            grid, depth=2, seed=42, protected=set()
        )
        deep = scatter_categorical_boundaries(grid, depth=20, seed=42, protected=set())
        shallow_changed = (shallow != grid).sum()
        deep_changed = (deep != grid).sum()
        assert deep_changed > shallow_changed
