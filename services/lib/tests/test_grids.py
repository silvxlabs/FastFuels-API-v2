"""Tests for lib.grids.compute_chunks_doc."""

import pytest

from lib.grids import compute_chunks_doc


class TestComputeChunksDoc2D:
    def test_standard_grid_4_chunks(self):
        doc = compute_chunks_doc((1000, 800), (512, 512))
        assert doc == {
            "shape": [512, 512],
            "count": 4,
            "count_by_axis": {"y": 2, "x": 2},
        }

    def test_exact_divisibility(self):
        doc = compute_chunks_doc((1024, 1024), (512, 512))
        assert doc["count"] == 4
        assert doc["count_by_axis"] == {"y": 2, "x": 2}

    def test_single_chunk(self):
        doc = compute_chunks_doc((300, 400), (512, 512))
        assert doc == {
            "shape": [512, 512],
            "count": 1,
            "count_by_axis": {"y": 1, "x": 1},
        }

    def test_axis_ordering_is_y_then_x(self):
        doc = compute_chunks_doc((600, 200), (256, 100))
        assert list(doc["count_by_axis"].keys()) == ["y", "x"]
        assert doc["count_by_axis"] == {"y": 3, "x": 2}


class TestComputeChunksDoc3D:
    def test_standard_3d_grid(self):
        doc = compute_chunks_doc((5, 1000, 800), (2, 512, 512))
        assert doc == {
            "shape": [2, 512, 512],
            "count": 12,
            "count_by_axis": {"z": 3, "y": 2, "x": 2},
        }

    def test_axis_ordering_is_z_y_x(self):
        doc = compute_chunks_doc((10, 600, 400), (5, 300, 200))
        assert list(doc["count_by_axis"].keys()) == ["z", "y", "x"]
        assert doc["count_by_axis"] == {"z": 2, "y": 2, "x": 2}

    def test_3d_count_is_product(self):
        doc = compute_chunks_doc((6, 1000, 1000), (3, 500, 250))
        assert doc["count_by_axis"] == {"z": 2, "y": 2, "x": 4}
        assert doc["count"] == 16


class TestComputeChunksDocErrors:
    def test_rank_mismatch_raises(self):
        with pytest.raises(ValueError, match="2D or both be 3D"):
            compute_chunks_doc((1000, 800), (2, 512, 512))

    def test_unsupported_rank_raises(self):
        with pytest.raises(ValueError, match="2D or both be 3D"):
            compute_chunks_doc((10,), (5,))

    def test_4d_raises(self):
        with pytest.raises(ValueError, match="2D or both be 3D"):
            compute_chunks_doc((1, 2, 3, 4), (1, 1, 1, 1))
