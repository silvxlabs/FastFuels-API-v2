from unittest.mock import patch

import geopandas as gpd
import pytest
from rasterio.crs import CRS
from shapely.geometry import box

from lib.raster import REPROJECTION_GUARD_CELLS, RasterConnection


def _make_connection(
    raster_crs: str,
    resolution: tuple[float, float],
) -> RasterConnection:
    conn = object.__new__(RasterConnection)
    conn.raster_crs = CRS.from_string(raster_crs)
    conn.raster_x_resolution = abs(resolution[0])
    conn.raster_y_resolution = abs(resolution[1])
    conn.raster_resolution = conn.raster_x_resolution
    return conn


def _make_roi(
    crs: str,
    bounds: tuple[float, float, float, float],
) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(geometry=[box(*bounds)], crs=crs)


def test_source_clip_bounds_same_crs_uses_roi_bounds_plus_guard():
    roi = _make_roi("EPSG:32611", (100.0, 200.0, 300.0, 400.0))
    conn = _make_connection("EPSG:32611", (30.0, -30.0))

    with patch("lib.raster.transform_bounds") as mock_transform_bounds:
        result = conn._source_clip_bounds(roi)

    guard = REPROJECTION_GUARD_CELLS * 30.0
    assert result == (
        100.0 - guard,
        200.0 - guard,
        300.0 + guard,
        400.0 + guard,
    )
    mock_transform_bounds.assert_not_called()


def test_source_clip_bounds_includes_output_padding():
    roi = _make_roi("EPSG:32611", (100.0, 200.0, 300.0, 400.0))
    conn = _make_connection("EPSG:32611", (30.0, -30.0))

    result = conn._source_clip_bounds(roi, interpolation_padding_cells=8)

    output_padding = 8 * 30.0
    guard = REPROJECTION_GUARD_CELLS * 30.0
    assert result == (
        100.0 - output_padding - guard,
        200.0 - output_padding - guard,
        300.0 + output_padding + guard,
        400.0 + output_padding + guard,
    )


def test_source_clip_bounds_different_crs_uses_transform_bounds_default_densification():
    roi = _make_roi("EPSG:32611", (100.0, 200.0, 300.0, 400.0))
    conn = _make_connection("EPSG:4326", (0.5, -1.0))

    with patch(
        "lib.raster.transform_bounds",
        return_value=(10.0, 20.0, 30.0, 40.0),
    ) as mock_transform_bounds:
        result = conn._source_clip_bounds(roi)

    x_guard = REPROJECTION_GUARD_CELLS * 0.5
    y_guard = REPROJECTION_GUARD_CELLS * 1.0
    assert result == (
        10.0 - x_guard,
        20.0 - y_guard,
        30.0 + x_guard,
        40.0 + y_guard,
    )
    mock_transform_bounds.assert_called_once_with(
        roi.crs,
        conn.raster_crs,
        *roi.total_bounds,
    )


def test_source_clip_bounds_geographic_source_guard_uses_source_raster_units():
    roi = _make_roi("EPSG:4326", (-110.0, 45.0, -109.0, 46.0))
    conn = _make_connection("EPSG:4326", (0.0001, -0.0002))

    result = conn._source_clip_bounds(roi)

    x_guard = REPROJECTION_GUARD_CELLS * 0.0001
    y_guard = REPROJECTION_GUARD_CELLS * 0.0002
    assert result == pytest.approx(
        (
            -110.0 - x_guard,
            45.0 - y_guard,
            -109.0 + x_guard,
            46.0 + y_guard,
        )
    )


def test_source_clip_bounds_non_square_resolution_expands_axes_independently():
    roi = _make_roi("EPSG:32611", (100.0, 200.0, 300.0, 400.0))
    conn = _make_connection("EPSG:32611", (10.0, -30.0))

    result = conn._source_clip_bounds(roi)

    x_guard = REPROJECTION_GUARD_CELLS * 10.0
    y_guard = REPROJECTION_GUARD_CELLS * 30.0
    assert result == (
        100.0 - x_guard,
        200.0 - y_guard,
        300.0 + x_guard,
        400.0 + y_guard,
    )
