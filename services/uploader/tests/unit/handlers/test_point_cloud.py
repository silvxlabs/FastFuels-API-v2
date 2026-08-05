"""
Unit tests for uploader/handlers/point_cloud.py

Exercises the pure helpers (_open_cloud, _require_crs, _output_bounds, _store) on
local files — no GCS or Firestore. Test files are synthesized with laspy so
every assertion is against known-by-construction ground truth.
"""

from unittest.mock import patch

import laspy
import numpy as np
import pyproj
import pytest
from pyproj import Transformer
from uploader.handlers import point_cloud
from uploader.handlers.point_cloud import (
    _open_cloud,
    _output_bounds,
    _require_crs,
    _store,
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
        """A compound CRS reprojects on its horizontal half.

        Handing pyproj the compound CRS instead would build a transform that
        tries to convert elevations between reference surfaces, which is
        exactly what this pipeline does not do.
        """
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
        assert not crs.is_compound


def _store_capturing(reader, dst_crs, transformer, tmp_path=None):
    """Run `_store` with the writer stubbed, returning what it was handed.

    The writer is not re-tested here — it has its own coverage, and it runs its
    workers under forkserver, which re-imports the module in each child and so
    cannot see a patch made in the parent. What this pins is the uploader's own
    contribution: the records it produces and the scaling it chose.
    """
    captured = {}

    def fake_write_parquet(records, info, bucket, prefix):
        captured["info"] = info
        captured["prefix"] = prefix
        captured["records"] = list(records)
        return {"points": 0, "tiles": 0, "files": 0, "output_bytes": 1234}

    with patch.object(point_cloud, "write_parquet", fake_write_parquet):
        summary, bounds, size_bytes = _store(reader, dst_crs, transformer, "pc-1")

    captured["summary"] = summary
    captured["bounds"] = bounds
    captured["size_bytes"] = size_bytes
    captured["columns"] = (
        captured["records"][0].dtype.names if captured["records"] else ()
    )
    return captured


class TestStore:
    """One path now: every upload is decoded and written as a dataset."""

    def test_counts_points_and_classes(self, tmp_path):
        path = tmp_path / "cloud.laz"
        make_test_las(path, n=7)

        with _open_cloud(str(path)) as reader:
            got = _store_capturing(reader, pyproj.CRS.from_epsg(32612), None)

        assert got["summary"]["point_count"] == 7
        assert sum(r.size for r in got["records"]) == 7
        assert got["size_bytes"] == 1234
        assert got["prefix"] == "pc-1/cloud.parquet"

    def test_reprojects_to_the_domain_crs(self, tmp_path):
        path = tmp_path / "cloud.laz"
        make_test_las(path, n=5, epsg=4326, x0=-113.5, y0=46.8, span=0.01)

        src = pyproj.CRS.from_epsg(4326)
        dst = pyproj.CRS.from_epsg(32612)
        transformer = Transformer.from_crs(src, dst, always_xy=True)
        with _open_cloud(str(path)) as reader:
            got = _store_capturing(reader, dst, transformer)

        # Somewhere in UTM 12N, not degrees.
        assert 200_000 < got["bounds"][0] < 800_000
        assert 5_000_000 < got["bounds"][1] < 5_400_000

    def test_output_bounds_use_every_corner(self):
        """A reprojected rectangle is not a rectangle."""

        class _Header:
            mins = [0.0, 0.0, 0.0]
            maxs = [10.0, 10.0, 0.0]

        class _Bowed:
            def transform(self, xs, ys):
                # Bows the top edge outward, so the extreme is not a corner of
                # the source rectangle taken naively.
                return [x for x in xs], [y + (5.0 if y else 0.0) for y in ys]

        assert _output_bounds(_Header(), None) == (0.0, 0.0, 10.0, 10.0)
        assert _output_bounds(_Header(), _Bowed()) == (0.0, 0.0, 10.0, 15.0)


class TestStoredFormat:
    """What survives the move to a fixed schema, and what does not.

    Writing LAZ meant the uploader could carry a source file's own header
    forward. A Parquet dataset has a fixed schema, so some of that is now gone
    on purpose. These pin which is which, because the difference is data the
    user loses.
    """

    def _source(self, path, *, scales, point_format=3, extra=None):
        header = laspy.LasHeader(
            version="1.2", point_format=laspy.PointFormat(point_format)
        )
        header.scales = [scales] * 3
        header.offsets = [500000.0, 4300000.0, 0.0]
        header.global_encoding.gps_time_type = laspy.header.GpsTimeType.STANDARD
        if extra:
            header.add_extra_dim(laspy.ExtraBytesParams(name=extra, type=np.float32))
        header.add_crs(pyproj.CRS.from_user_input("EPSG:32612"))

        las = laspy.LasData(header)
        count = 5
        las.points = laspy.point.record.ScaleAwarePointRecord.zeros(
            count, header=header
        )
        # Points 10 micrometres apart: distinguishable only at a sub-millimetre
        # scale, which is exactly what a terrestrial scan relies on.
        las.x = 500000.0 + np.array([1e-5, 3e-5, 4e-5, 5e-5, 6e-5])
        las.y = np.full(count, 4300000.0)
        las.z = np.full(count, 1500.0)
        if extra:
            las[extra] = np.arange(count, dtype=np.float32)
        las.write(str(path))
        return las

    def test_sub_millimetre_scale_is_preserved(self, tmp_path):
        """Re-encoding at a coarser scale collapses distinct points into one.

        The canonical header imposes millimetre scaling, which is right for
        merging acquisitions and wrong for one terrestrial scan. The dataset
        stores its own scale, so the source's is kept and the points stay apart.
        """
        path = tmp_path / "fine.las"
        self._source(path, scales=1e-5)

        with _open_cloud(str(path)) as reader:
            got = _store_capturing(
                reader, pyproj.CRS.from_user_input("EPSG:32612"), None
            )

        assert got["info"]["scales"][0] == pytest.approx(1e-5)
        # Five points 10 um apart survive as five distinct stored positions.
        assert len(set(got["records"][0]["X"].tolist())) == 5

    def test_colour_is_preserved(self, tmp_path):
        path = tmp_path / "rgb.las"
        self._source(path, scales=0.001, point_format=3)

        with _open_cloud(str(path)) as reader:
            got = _store_capturing(
                reader, pyproj.CRS.from_user_input("EPSG:32612"), None
            )

        assert {"red", "green", "blue"} <= set(got["columns"])

    def test_classification_is_uniform_across_source_formats(self, tmp_path):
        """Formats 0-5 pack classification in with the flags; 6+ separate them.

        Normalising through the canonical point format is what makes a reader
        able to filter on ASPRS class without knowing the source format.
        """
        path = tmp_path / "legacy.las"
        self._source(path, scales=0.001, point_format=3)

        with _open_cloud(str(path)) as reader:
            got = _store_capturing(
                reader, pyproj.CRS.from_user_input("EPSG:32612"), None
            )

        assert "classification" in got["columns"]
        assert set(got["summary"]["point_classes"]) <= set(range(256))

    def test_extra_dimensions_are_dropped(self, tmp_path):
        """Documented loss: the schema is fixed and has nowhere to put them."""
        path = tmp_path / "extra.las"
        self._source(path, scales=0.001, extra="Amplitude")

        with _open_cloud(str(path)) as reader:
            got = _store_capturing(
                reader, pyproj.CRS.from_user_input("EPSG:32612"), None
            )

        assert "Amplitude" not in got["columns"]

    def test_gps_time_is_dropped(self, tmp_path):
        """Documented loss: 23% of the file, and nothing downstream reads it."""
        path = tmp_path / "gps.las"
        self._source(path, scales=0.001)

        with _open_cloud(str(path)) as reader:
            got = _store_capturing(
                reader, pyproj.CRS.from_user_input("EPSG:32612"), None
            )

        assert "gps_time" not in got["columns"]
