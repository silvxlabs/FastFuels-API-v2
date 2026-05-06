from unittest.mock import patch

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin
from rasterio.warp import transform_bounds as rio_transform_bounds
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


def _write_test_raster(
    path,
    *,
    crs: str,
    transform,
    width: int,
    height: int,
) -> None:
    data = np.arange(width * height, dtype=np.float32).reshape(height, width)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype=data.dtype,
        crs=crs,
        transform=transform,
    ) as dataset:
        dataset.write(data, 1)


@pytest.mark.parametrize(
    ("interpolation_padding_cells", "expected_shape"),
    [
        (0, (10, 10)),
        (1, (12, 12)),
        (10, (30, 30)),
    ],
)
def test_extract_window_buffer_cells_expand_same_crs_shape(
    tmp_path,
    interpolation_padding_cells,
    expected_shape,
):
    raster_path = tmp_path / "same_crs.tif"
    _write_test_raster(
        raster_path,
        crs="EPSG:32611",
        transform=from_origin(0.0, 100.0, 1.0, 1.0),
        width=100,
        height=100,
    )
    roi = _make_roi("EPSG:32611", (10.0, 70.0, 20.0, 80.0))
    conn = RasterConnection(str(raster_path))

    result = conn.extract_window(
        roi,
        interpolation_padding_cells=interpolation_padding_cells,
    )

    assert (result.sizes["y"], result.sizes["x"]) == expected_shape


def test_extract_window_buffer_cells_expand_projected_shape_with_geographic_source(
    tmp_path,
):
    raster_path = tmp_path / "geographic_source.tif"
    source_crs = "EPSG:4326"
    target_crs = "EPSG:32610"
    _write_test_raster(
        raster_path,
        crs=source_crs,
        transform=from_origin(-121.0, 37.0, 0.0001, 0.0001),
        width=400,
        height=400,
    )
    source_roi_bounds = (-120.990, 36.988, -120.988, 36.990)
    roi = _make_roi(
        target_crs,
        rio_transform_bounds(source_crs, target_crs, *source_roi_bounds),
    )
    conn = RasterConnection(str(raster_path))

    original = conn.extract_window(roi, interpolation_padding_cells=0)
    one_cell = conn.extract_window(roi, interpolation_padding_cells=1)
    ten_cells = conn.extract_window(roi, interpolation_padding_cells=10)

    original_shape = (original.sizes["y"], original.sizes["x"])
    one_cell_shape = (one_cell.sizes["y"], one_cell.sizes["x"])
    ten_cell_shape = (ten_cells.sizes["y"], ten_cells.sizes["x"])

    assert one_cell_shape[0] > original_shape[0]
    assert one_cell_shape[1] > original_shape[1]
    assert ten_cell_shape[0] > one_cell_shape[0]
    assert ten_cell_shape[1] > one_cell_shape[1]


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


def test_source_clip_bounds_includes_output_padding_per_axis():
    roi = _make_roi("EPSG:32611", (100.0, 200.0, 300.0, 400.0))
    conn = _make_connection("EPSG:32611", (10.0, -30.0))

    result = conn._source_clip_bounds(roi, interpolation_padding_cells=2)

    x_output_padding = 2 * 10.0
    y_output_padding = 2 * 30.0
    x_guard = REPROJECTION_GUARD_CELLS * 10.0
    y_guard = REPROJECTION_GUARD_CELLS * 30.0
    assert result == (
        100.0 - x_output_padding - x_guard,
        200.0 - y_output_padding - y_guard,
        300.0 + x_output_padding + x_guard,
        400.0 + y_output_padding + y_guard,
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
