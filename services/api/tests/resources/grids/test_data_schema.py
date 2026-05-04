"""
Unit tests for grid data schema models and compute_chunk_metadata().
"""

import pytest
from api.resources.grids.schema import (
    DenseGridData,
    GridDataArrayFormat,
    GridDataChunkMetadata,
    GridDataOrder,
    GridDataResponse,
    GridDataResponseFormat,
    SparseGridData,
)
from api.resources.grids.utils import compute_chunk_metadata, compute_chunk_slices

# Helpers


def _georef(shape, pixel_size=30.0, origin_x=500000.0, origin_y=5200000.0):
    """Build a minimal georeference dict."""
    georef = {
        "shape": shape,
        "transform": (pixel_size, 0.0, origin_x, 0.0, -pixel_size, origin_y),
        "crs": "EPSG:32611",
    }
    if len(shape) == 3:
        georef["z_origin"] = 10.0
        georef["z_resolution"] = 0.5
    return georef


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

    def test_3d_single_chunk(self):
        georef = _georef((5, 300, 400))
        m = compute_chunk_metadata(georef, [10, 512, 512], 0)

        assert m.index == 0
        assert m.shape == (5, 300, 400)
        assert m.offset == (0, 0, 0)
        assert m.transform == georef["transform"]
        assert m.z_origin == 10.0
        assert m.z_resolution == 0.5

    def test_3d_chunk_index_order_is_z_y_x(self):
        georef = _georef((5, 1000, 800))
        chunk_shape = [2, 512, 512]

        m1 = compute_chunk_metadata(georef, chunk_shape, 1)
        assert m1.offset == (0, 0, 512)
        assert m1.shape == (2, 512, 288)

        m2 = compute_chunk_metadata(georef, chunk_shape, 2)
        assert m2.offset == (0, 512, 0)
        assert m2.shape == (2, 488, 512)

        m4 = compute_chunk_metadata(georef, chunk_shape, 4)
        assert m4.offset == (2, 0, 0)
        assert m4.shape == (2, 512, 512)
        assert m4.z_origin == 11.0

    def test_3d_edge_chunk_smaller_in_all_dimensions(self):
        georef = _georef((5, 1000, 800))
        m = compute_chunk_metadata(georef, [2, 512, 512], 11)

        assert m.offset == (4, 512, 512)
        assert m.shape == (1, 488, 288)
        assert m.z_origin == 12.0
        assert m.transform[2] == pytest.approx(500000.0 + 30.0 * 512)
        assert m.transform[5] == pytest.approx(5200000.0 + (-30.0) * 512)

    def test_3d_out_of_range_raises(self):
        georef = _georef((5, 1000, 800))
        with pytest.raises(ValueError, match="out of range"):
            compute_chunk_metadata(georef, [2, 512, 512], 12)

    def test_shape_and_chunk_shape_rank_mismatch_raises(self):
        georef = _georef((5, 1000, 800))
        with pytest.raises(ValueError, match="2D or both be 3D"):
            compute_chunk_metadata(georef, [512, 512], 0)


class TestComputeChunkSlices:
    """Tests for compute_chunk_slices()."""

    def test_origin_chunk(self):
        """Chunk at offset (0, 0) produces slices starting at 0."""
        meta = GridDataChunkMetadata(
            index=0,
            shape=(512, 512),
            offset=(0, 0),
            transform=(30.0, 0.0, 500000.0, 0.0, -30.0, 5200000.0),
        )
        row_slice, col_slice = compute_chunk_slices(meta)
        assert row_slice == slice(0, 512)
        assert col_slice == slice(0, 512)

    def test_offset_chunk(self):
        """Chunk with non-zero offset produces correctly shifted slices."""
        meta = GridDataChunkMetadata(
            index=3,
            shape=(512, 512),
            offset=(512, 512),
            transform=(30.0, 0.0, 515360.0, 0.0, -30.0, 5184640.0),
        )
        row_slice, col_slice = compute_chunk_slices(meta)
        assert row_slice == slice(512, 1024)
        assert col_slice == slice(512, 1024)

    def test_edge_chunk_smaller_shape(self):
        """Edge chunk with reduced shape produces shorter slices."""
        meta = GridDataChunkMetadata(
            index=1,
            shape=(512, 288),
            offset=(0, 512),
            transform=(30.0, 0.0, 515360.0, 0.0, -30.0, 5200000.0),
        )
        row_slice, col_slice = compute_chunk_slices(meta)
        assert row_slice == slice(0, 512)
        assert col_slice == slice(512, 800)

    def test_corner_edge_chunk(self):
        """Bottom-right edge chunk with both dimensions smaller."""
        meta = GridDataChunkMetadata(
            index=3,
            shape=(488, 288),
            offset=(512, 512),
            transform=(30.0, 0.0, 515360.0, 0.0, -30.0, 5184640.0),
        )
        row_slice, col_slice = compute_chunk_slices(meta)
        assert row_slice == slice(512, 1000)
        assert col_slice == slice(512, 800)

    def test_single_chunk_grid(self):
        """Grid that fits entirely in one chunk."""
        meta = GridDataChunkMetadata(
            index=0,
            shape=(47, 61),
            offset=(0, 0),
            transform=(30.0, 0.0, 500000.0, 0.0, -30.0, 5200000.0),
        )
        row_slice, col_slice = compute_chunk_slices(meta)
        assert row_slice == slice(0, 47)
        assert col_slice == slice(0, 61)

    def test_roundtrip_with_compute_chunk_metadata(self):
        """Slices from compute_chunk_slices match the metadata from compute_chunk_metadata."""
        georef = _georef((1000, 800))
        chunk_shape = [512, 512]

        for chunk_index in range(4):
            meta = compute_chunk_metadata(georef, chunk_shape, chunk_index)
            row_slice, col_slice = compute_chunk_slices(meta)

            assert row_slice.start == meta.offset[0]
            assert col_slice.start == meta.offset[1]
            assert row_slice.stop - row_slice.start == meta.shape[0]
            assert col_slice.stop - col_slice.start == meta.shape[1]

    def test_3d_chunk(self):
        meta = GridDataChunkMetadata(
            index=11,
            shape=(1, 488, 288),
            offset=(4, 512, 512),
            transform=(30.0, 0.0, 515360.0, 0.0, -30.0, 5184640.0),
            z_origin=12.0,
            z_resolution=0.5,
        )
        z_slice, row_slice, col_slice = compute_chunk_slices(meta)
        assert z_slice == slice(4, 5)
        assert row_slice == slice(512, 1000)
        assert col_slice == slice(512, 800)


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

    def test_3d_chunk_metadata_model(self):
        m = GridDataChunkMetadata(
            index=0,
            shape=(4, 100, 200),
            offset=(8, 0, 0),
            transform=(30.0, 0.0, 500000.0, 0.0, -30.0, 5200000.0),
            z_origin=14.0,
            z_resolution=0.5,
        )
        assert m.shape == (4, 100, 200)
        assert m.offset == (8, 0, 0)
        assert m.z_origin == 14.0

    def test_dense_data_response_model(self):
        r = GridDataResponse(
            shape=[47, 61],
            order="C",
            data={"format": "dense", "values": [1, 2, 3]},
        )
        assert r.shape == [47, 61]
        assert r.order == "C"
        assert isinstance(r.data, DenseGridData)
        assert r.data.format == "dense"
        assert r.data.values == [1, 2, 3]

    def test_sparse_data_response_model(self):
        r = GridDataResponse(
            shape=[2, 3],
            order="C",
            data={
                "format": "sparse",
                "fill_value": 0,
                "indices": [0, 4],
                "values": [7, 9],
            },
        )
        assert isinstance(r.data, SparseGridData)
        assert r.data.format == "sparse"
        assert r.data.fill_value == 0
        assert r.data.indices == [0, 4]
        assert r.data.values == [7, 9]

    def test_response_format_enum(self):
        assert GridDataResponseFormat.json == "json"
        assert GridDataResponseFormat.binary == "binary"

    def test_array_format_enum(self):
        assert GridDataArrayFormat.dense == "dense"
        assert GridDataArrayFormat.sparse == "sparse"

    def test_order_enum(self):
        assert GridDataOrder.C == "C"
        assert GridDataOrder.F == "F"
