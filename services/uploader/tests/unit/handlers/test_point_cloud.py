"""
Unit tests for uploader/handlers/point_cloud.py

Exercises the pure helpers (_open_cloud, _require_crs, _census, _rewrite) on
local files — no GCS or Firestore. Test files are synthesized with laspy so
every assertion is against known-by-construction ground truth.
"""

import laspy
import numpy as np
import pyproj
import pytest
from pyproj import Transformer
from uploader.handlers.point_cloud import (
    _census,
    _open_cloud,
    _require_crs,
    _rewrite,
)

from lib.errors import ProcessingError
from tests.pointcloud_helpers import make_test_las


class TestOpenCloud:
    def test_opens_valid_laz(self, tmp_path):
        path = tmp_path / "cloud.laz"
        make_test_las(str(path), n=50)
        with _open_cloud(str(path)) as reader:
            assert reader.header.point_count == 50

    def test_garbage_bytes_raise_unreadable(self, tmp_path):
        path = tmp_path / "garbage.laz"
        path.write_bytes(b"\x00\x01\x02 this is not a point cloud" * 100)
        with pytest.raises(ProcessingError) as exc:
            _open_cloud(str(path))
        assert exc.value.code == "UNREADABLE_POINT_CLOUD"


class TestRequireCrs:
    def test_returns_crs(self, tmp_path):
        path = tmp_path / "cloud.laz"
        make_test_las(str(path), epsg=32612)
        with laspy.open(str(path)) as reader:
            crs = _require_crs(reader.header)
        assert crs.to_epsg() == 32612

    def test_missing_crs_raises(self, tmp_path):
        path = tmp_path / "cloud.laz"
        make_test_las(str(path), with_srs=False)
        with laspy.open(str(path)) as reader:
            with pytest.raises(ProcessingError) as exc:
                _require_crs(reader.header)
        assert exc.value.code == "MISSING_CRS"

    def test_compound_crs_resolves_to_horizontal(self, tmp_path):
        path = tmp_path / "cloud.laz"
        header = laspy.LasHeader(version="1.4", point_format=6)
        header.offsets = [500000.0, 4300000.0, 1800.0]
        header.scales = [0.01, 0.01, 0.01]
        # UTM 12N + NAVD88 height.
        header.add_crs(pyproj.CRS.from_user_input("EPSG:32612+5703"))
        las = laspy.LasData(header)
        las.x = np.array([500100.0])
        las.y = np.array([4300100.0])
        las.z = np.array([1850.0])
        las.write(str(path))

        with laspy.open(str(path)) as reader:
            crs = _require_crs(reader.header)
        assert crs.to_epsg() == 32612


class TestCensus:
    def test_counts_and_classes(self, tmp_path):
        path = tmp_path / "cloud.laz"
        truth = make_test_las(str(path), n=100, classes=(1, 2, 5))
        with laspy.open(str(path)) as reader:
            stats = _census(reader)
        assert stats["rewritten"] is False
        assert stats["point_count"] == 100
        assert stats["point_classes"] == [1, 2, 5]
        assert stats["density"] == pytest.approx(100 / truth["xy_area"], rel=1e-6)


class TestRewrite:
    def test_recompresses_las_without_transform(self, tmp_path):
        path = tmp_path / "cloud.las"
        truth = make_test_las(str(path), n=80, classes=(2, 5))
        with laspy.open(str(path)) as reader:
            assert not reader.header.are_points_compressed
            buf, stats, bounds = _rewrite(reader, pyproj.CRS.from_epsg(32612), None)

        out = laspy.read(buf)
        assert out.header.are_points_compressed
        assert out.header.point_count == 80
        assert out.header.parse_crs().to_epsg() == 32612
        np.testing.assert_allclose(np.asarray(out.x), truth["x"], atol=0.011)
        np.testing.assert_allclose(np.asarray(out.z), truth["z"], atol=0.011)
        assert stats["rewritten"] is True
        assert stats["point_count"] == 80
        assert stats["point_classes"] == [2, 5]
        assert bounds[2] == pytest.approx(truth["min_z"], abs=0.011)
        assert bounds[5] == pytest.approx(truth["max_z"], abs=0.011)

    def test_reprojects_to_target_crs(self, tmp_path):
        path = tmp_path / "cloud.laz"
        truth = make_test_las(str(path), n=120, epsg=32613, classes=(1, 2))

        src = pyproj.CRS.from_epsg(32613)
        dst = pyproj.CRS.from_epsg(32612)
        transformer = Transformer.from_crs(src, dst, always_xy=True)
        expected_x, expected_y = transformer.transform(truth["x"], truth["y"])

        with laspy.open(str(path)) as reader:
            buf, stats, bounds = _rewrite(reader, dst, transformer)

        out = laspy.read(buf)
        assert out.header.parse_crs().to_epsg() == 32612
        # Max error is half the coordinate scale quantum (0.01 m).
        np.testing.assert_allclose(np.asarray(out.x), expected_x, atol=0.011)
        np.testing.assert_allclose(np.asarray(out.y), expected_y, atol=0.011)
        # Elevations pass through untouched.
        np.testing.assert_allclose(np.asarray(out.z), truth["z"], atol=0.011)
        assert stats["point_count"] == 120
        assert stats["point_classes"] == [1, 2]
        assert bounds[0] == pytest.approx(expected_x.min(), abs=0.011)
        assert bounds[3] == pytest.approx(expected_x.max(), abs=0.011)

    def test_chunked_rewrite_matches_single_read(self, tmp_path, monkeypatch):
        """Multiple chunks produce one coherent LAZ (chunk boundary safety)."""
        path = tmp_path / "cloud.laz"
        make_test_las(str(path), n=1000, classes=(1, 2, 5))
        monkeypatch.setattr("uploader.handlers.point_cloud._CHUNK_POINTS", 64)
        with laspy.open(str(path)) as reader:
            buf, stats, _ = _rewrite(reader, pyproj.CRS.from_epsg(32612), None)

        out = laspy.read(buf)
        assert out.header.point_count == 1000
        assert stats["point_count"] == 1000
        src = laspy.read(str(path))
        np.testing.assert_allclose(np.asarray(out.x), np.asarray(src.x), atol=0.011)
        np.testing.assert_array_equal(
            np.asarray(out.classification), np.asarray(src.classification)
        )
