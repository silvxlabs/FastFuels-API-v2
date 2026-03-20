"""
Unit tests for api/v2/resources/grids/data/schema.py

Tests compute_chunk_metadata() as a pure function and response models.
"""

import pytest
from api.resources.grids.data.schema import (
    GridDataChunkMetadata,
    GridDataFormat,
    GridDataOrder,
    GridDataResponse,
    compute_chunk_metadata,
)

# Helpers


def _georef(shape, pixel_size=30.0, origin_x=500000.0, origin_y=5200000.0):
    """Build a minimal georeference dict."""
    return {
        "shape": shape,
        "transform": (pixel_size, 0.0, origin_x, 0.0, -pixel_size, origin_y),
        "crs": "EPSG:32611",
    }


class TestComputeChunkMetadata:
    """Tests for compute_chunk_metadata()."""

    def test_standard_grid_4_chunks(self):
        """1024x1024 grid with 512x512 chunks -> 4 chunks."""
        georef = _georef((1024, 1024))
        chunk_shape = [512, 512]

        # Chunk 0: top-left
        m = compute_chunk_metadata(georef, chunk_shape, 0)
        assert m.index == 0
        assert m.shape == (512, 512)
        assert m.offset == (0, 0)
        assert m.transform == georef["transform"]

        # Chunk 1: top-right
        m = compute_chunk_metadata(georef, chunk_shape, 1)
        assert m.index == 1
        assert m.shape == (512, 512)
        assert m.offset == (0, 512)
        assert m.transform[2] == 500000.0 + 30.0 * 512  # c shifted

        # Chunk 2: bottom-left
        m = compute_chunk_metadata(georef, chunk_shape, 2)
        assert m.index == 2
        assert m.shape == (512, 512)
        assert m.offset == (512, 0)
        assert m.transform[5] == 5200000.0 + (-30.0) * 512  # f shifted

        # Chunk 3: bottom-right
        m = compute_chunk_metadata(georef, chunk_shape, 3)
        assert m.index == 3
        assert m.shape == (512, 512)
        assert m.offset == (512, 512)

    def test_edge_chunks_smaller_shape(self):
        """1000x800 grid with 512x512 chunks -> edge chunks are smaller."""
        georef = _georef((1000, 800))
        chunk_shape = [512, 512]

        # 2 rows x 2 cols = 4 chunks
        # Chunk 0: (512, 512)
        m0 = compute_chunk_metadata(georef, chunk_shape, 0)
        assert m0.shape == (512, 512)

        # Chunk 1: top-right, width = 800 - 512 = 288
        m1 = compute_chunk_metadata(georef, chunk_shape, 1)
        assert m1.shape == (512, 288)

        # Chunk 2: bottom-left, height = 1000 - 512 = 488
        m2 = compute_chunk_metadata(georef, chunk_shape, 2)
        assert m2.shape == (488, 512)

        # Chunk 3: bottom-right, both edges
        m3 = compute_chunk_metadata(georef, chunk_shape, 3)
        assert m3.shape == (488, 288)

    def test_single_chunk(self):
        """300x400 grid fits in one 512x512 chunk."""
        georef = _georef((300, 400))
        chunk_shape = [512, 512]

        m = compute_chunk_metadata(georef, chunk_shape, 0)
        assert m.index == 0
        assert m.shape == (300, 400)
        assert m.offset == (0, 0)
        assert m.transform == georef["transform"]

    def test_out_of_range_raises(self):
        """Out-of-range chunk index raises ValueError."""
        georef = _georef((300, 400))
        chunk_shape = [512, 512]

        with pytest.raises(ValueError, match="out of range"):
            compute_chunk_metadata(georef, chunk_shape, 1)

    def test_negative_index_raises(self):
        georef = _georef((1024, 1024))
        with pytest.raises(ValueError, match="out of range"):
            compute_chunk_metadata(georef, [512, 512], -1)

    def test_transform_c_offset(self):
        """Verify c offset uses pixel_width * col_offset."""
        georef = _georef((1024, 1024), pixel_size=10.0, origin_x=100.0)
        m = compute_chunk_metadata(georef, [512, 512], 1)
        expected_c = 100.0 + 10.0 * 512
        assert m.transform[2] == pytest.approx(expected_c)

    def test_transform_f_offset(self):
        """Verify f offset uses -pixel_height * row_offset."""
        georef = _georef((1024, 1024), pixel_size=10.0, origin_y=50000.0)
        m = compute_chunk_metadata(georef, [512, 512], 2)
        expected_f = 50000.0 + (-10.0) * 512
        assert m.transform[5] == pytest.approx(expected_f)

    def test_exact_chunk_boundary(self):
        """Grid shape exactly divisible by chunk shape."""
        georef = _georef((512, 512))
        chunk_shape = [512, 512]

        # Only 1 chunk
        m = compute_chunk_metadata(georef, chunk_shape, 0)
        assert m.shape == (512, 512)

        with pytest.raises(ValueError):
            compute_chunk_metadata(georef, chunk_shape, 1)


class TestResponseModels:
    def test_chunk_metadata_model(self):
        m = GridDataChunkMetadata(
            index=0,
            shape=(100, 200),
            offset=(0, 0),
            transform=(30.0, 0.0, 500000.0, 0.0, -30.0, 5200000.0),
        )
        assert m.index == 0
        assert m.shape == (100, 200)

    def test_data_response_model(self):
        r = GridDataResponse(
            shape=[47, 61],
            order="C",
            data=[1, 2, 3],
        )
        assert r.shape == [47, 61]
        assert r.order == "C"

    def test_format_enum(self):
        assert GridDataFormat.json == "json"
        assert GridDataFormat.binary == "binary"

    def test_order_enum(self):
        assert GridDataOrder.C == "C"
        assert GridDataOrder.F == "F"
