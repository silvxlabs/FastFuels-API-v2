"""
Cross-handler tests for the alignment behavior added in #205.

These tests verify that fetcher handlers thread the alignment specification
through ``RasterConnection.extract_window`` correctly, and that the
domain-anchored / native / target-grid variants produce the expected
output transforms.

The handlers are exercised with a mocked ``RasterConnection`` so the tests
don't depend on remote COG access; the assertions focus on what destination
kwargs the handler builds, which is the actual contract added by this
change.
"""

from unittest.mock import MagicMock, patch

import geopandas as gpd
import numpy as np
import pytest
import xarray as xr
from affine import Affine
from griddle.handlers.landfire import fetch_fbfm40
from griddle.handlers.pim import fetch_treemap
from rasterio.transform import from_bounds
from shapely.geometry import box


def _domain_gdf(crs="EPSG:32611"):
    return gpd.GeoDataFrame(
        geometry=[box(720000.0, 5190000.0, 720600.0, 5190400.0)],
        crs=crs,
    )


def _mock_raster(crs="EPSG:32611", source_resolution=30.0, dtype=np.int16):
    """Mock RasterConnection that mimics a 30m LANDFIRE COG covering the test
    domain. The ``extract_window`` mock returns a small DataArray; the test
    inspects the destination kwargs the handler passed in."""
    height, width = 20, 25
    transform = from_bounds(
        720000.0,
        5190000.0,
        720000.0 + width * source_resolution,
        5190000.0 + height * source_resolution,
        width,
        height,
    )
    da = xr.DataArray(
        np.zeros((1, height, width), dtype=dtype),
        dims=("band", "y", "x"),
        coords={"band": [1]},
    )
    da = da.rio.write_crs(crs)
    da = da.rio.write_transform(transform)

    raster = MagicMock()
    raster.raster_x_resolution = source_resolution
    raster.extract_window.return_value = da
    return raster


class TestLandfireAlignmentDomain:
    @patch("griddle.handlers.landfire.RasterConnection")
    def test_domain_target_passes_destination_kwargs(self, mock_cls):
        mock_cls.return_value = _mock_raster()
        roi = _domain_gdf()

        fetch_fbfm40(
            roi,
            version="2024",
            alignment={"target": "domain", "resolution": 2.0},
        )

        kwargs = mock_cls.return_value.extract_window.call_args[1]
        # Destination transform anchored at domain lower-left, cell size 2m.
        transform = kwargs["destination_transform"]
        assert transform.c == pytest.approx(720000.0)  # minx
        assert transform.f == pytest.approx(5190400.0)  # maxy
        assert abs(transform.a) == pytest.approx(2.0)
        # Shape from ceil(600 / 2) x ceil(400 / 2) = 300 x 200.
        assert kwargs["destination_shape"] == (200, 300)
        # Categorical default: nearest.
        assert kwargs["resampling"].value == 0  # rasterio.enums.Resampling.nearest

    @patch("griddle.handlers.landfire.RasterConnection")
    def test_domain_target_default_uses_source_native_resolution(self, mock_cls):
        mock_cls.return_value = _mock_raster(source_resolution=30.0)
        roi = _domain_gdf()

        fetch_fbfm40(roi, version="2024", alignment={"target": "domain"})

        kwargs = mock_cls.return_value.extract_window.call_args[1]
        transform = kwargs["destination_transform"]
        assert abs(transform.a) == pytest.approx(30.0)


class TestLandfireAlignmentNative:
    @patch("griddle.handlers.landfire.RasterConnection")
    def test_native_target_no_resolution_passes_no_destination(self, mock_cls):
        mock_cls.return_value = _mock_raster()
        roi = _domain_gdf()

        fetch_fbfm40(roi, version="2024", alignment={"target": "native"})

        kwargs = mock_cls.return_value.extract_window.call_args[1]
        assert "destination_transform" not in kwargs
        assert "destination_shape" not in kwargs
        # destination_resolution falls through as None for native + no resolution
        assert kwargs.get("destination_resolution") is None
        assert kwargs.get("destination_crs") is None

    @patch("griddle.handlers.landfire.RasterConnection")
    def test_native_target_with_resolution_passes_resolution(self, mock_cls):
        mock_cls.return_value = _mock_raster()
        roi = _domain_gdf()

        fetch_fbfm40(
            roi,
            version="2024",
            alignment={"target": "native", "resolution": 5.0},
        )

        kwargs = mock_cls.return_value.extract_window.call_args[1]
        # CRS-only override path: destination_crs set, no transform/shape.
        assert kwargs["destination_crs"] == roi.crs
        assert kwargs["destination_resolution"] == 5.0
        assert "destination_transform" not in kwargs


class TestLandfireAlignmentGrid:
    @patch("griddle.handlers.landfire.RasterConnection")
    def test_grid_target_exact_match(self, mock_cls):
        mock_cls.return_value = _mock_raster()
        roi = _domain_gdf()
        target_grid_doc = {
            "georeference": {
                "crs": "EPSG:32611",
                "transform": (5.0, 0.0, 720100.0, 0.0, -5.0, 5190300.0),
                "shape": (40, 60),
            }
        }

        fetch_fbfm40(
            roi,
            version="2024",
            alignment={"target": "grid", "grid_id": "x"},
            target_grid_doc=target_grid_doc,
        )

        kwargs = mock_cls.return_value.extract_window.call_args[1]
        assert kwargs["destination_transform"] == Affine(
            5.0, 0.0, 720100.0, 0.0, -5.0, 5190300.0
        )
        assert kwargs["destination_shape"] == (40, 60)
        assert kwargs["destination_crs"] == "EPSG:32611"

    @patch("griddle.handlers.landfire.RasterConnection")
    def test_grid_target_with_resolution_recomputes_shape(self, mock_cls):
        mock_cls.return_value = _mock_raster()
        roi = _domain_gdf()
        # Target grid: 30m cells, 10x10, anchored lower-left at (720100, 5190200).
        target_grid_doc = {
            "georeference": {
                "crs": "EPSG:32611",
                "transform": (30.0, 0.0, 720100.0, 0.0, -30.0, 5190500.0),
                "shape": (10, 10),
            }
        }

        fetch_fbfm40(
            roi,
            version="2024",
            alignment={"target": "grid", "grid_id": "x", "resolution": 1.0},
            target_grid_doc=target_grid_doc,
        )

        kwargs = mock_cls.return_value.extract_window.call_args[1]
        transform = kwargs["destination_transform"]
        assert transform.c == pytest.approx(720100.0)  # target's lower-left x
        assert transform.f == pytest.approx(5190500.0)  # target's upper-left y
        assert abs(transform.a) == pytest.approx(1.0)
        # Target bounds are 300x300 m at 1m -> 300 x 300 shape.
        assert kwargs["destination_shape"] == (300, 300)


class TestPimAlignmentMethodDefault:
    """PIM/TreeMap is categorical (tm_id, plt_cn). Verify the default method
    resolves to ``nearest`` even when alignment.method is unset."""

    @patch("griddle.handlers.pim.RasterConnection")
    def test_default_method_is_nearest(self, mock_cls):
        mock_cls.return_value = _mock_raster(dtype=np.int32)
        roi = _domain_gdf()
        progress = MagicMock()

        fetch_treemap(
            roi,
            version="2022",
            bands=["tm_id"],
            progress=progress,
            alignment={"target": "domain", "resolution": 2.0},
        )

        kwargs = mock_cls.return_value.extract_window.call_args[1]
        assert kwargs["resampling"].value == 0  # Resampling.nearest

    @patch("griddle.handlers.pim.RasterConnection")
    def test_explicit_method_wins(self, mock_cls):
        mock_cls.return_value = _mock_raster(dtype=np.int32)
        roi = _domain_gdf()
        progress = MagicMock()

        fetch_treemap(
            roi,
            version="2022",
            bands=["tm_id"],
            progress=progress,
            alignment={"target": "domain", "resolution": 2.0, "method": "mode"},
        )

        kwargs = mock_cls.return_value.extract_window.call_args[1]
        # rasterio.enums.Resampling.mode == 7
        from rasterio.enums import Resampling

        assert kwargs["resampling"] == Resampling.mode
