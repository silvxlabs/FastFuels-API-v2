"""
Unit tests for uploader/handlers/point_cloud.py

Exercises the pure metadata/transcode logic with real PDAL on local files (no
GCS, no Firestore). Mocking PDAL's metadata would defeat the purpose — the
metadata-key shapes are the thing most likely to drift — so we run it for real.
"""

import json

import pdal
import pytest
from pyproj.crs import CRS, CompoundCRS
from uploader.handlers.point_cloud import (
    _inspect_and_transcode,
    _require_crs_match,
    _wkt_to_epsg,
)

from lib.errors import ProcessingError
from tests.pointcloud_helpers import make_test_las


class TestInspectAndTranscode:
    def test_extracts_crs_count_classes_density(self, tmp_path):
        las = tmp_path / "cloud.las"
        truth = make_test_las(str(las), n=100, epsg=32612, classes=(1, 2, 5))

        result = _inspect_and_transcode(str(las), None)

        assert result["crs"] == "EPSG:32612"
        assert result["point_count"] == 100
        assert result["point_classes"] == [1, 2, 5]
        assert len(result["bounds"]) == 6
        # min_z / max_z land in the 3D bounds (index 2 and 5)
        assert result["bounds"][2] == pytest.approx(truth["min_z"], abs=0.01)
        assert result["bounds"][5] == pytest.approx(truth["max_z"], abs=0.01)
        assert result["density"] == pytest.approx(100 / truth["xy_area"], rel=1e-6)

    def test_transcode_writes_readable_copc(self, tmp_path):
        las = tmp_path / "cloud.laz"
        make_test_las(str(las), n=120, epsg=32612, classes=(2, 5))
        copc = tmp_path / "cloud.copc.laz"

        result = _inspect_and_transcode(str(las), str(copc))

        assert result["point_count"] == 120
        assert copc.exists()
        # The COPC re-reads with the COPC reader and round-trips the point count.
        count = pdal.Pipeline(
            json.dumps([{"type": "readers.copc", "filename": str(copc)}])
        ).execute()
        assert count == 120

    def test_missing_crs_raises(self, tmp_path):
        las = tmp_path / "no_crs.las"
        make_test_las(str(las), n=50, with_srs=False)

        with pytest.raises(ProcessingError) as exc:
            _inspect_and_transcode(str(las), None)
        assert exc.value.code == "MISSING_CRS"


class TestWktToEpsg:
    def test_plain_projected_crs(self):
        wkt = CRS.from_epsg(32612).to_wkt()
        assert _wkt_to_epsg(wkt) == "EPSG:32612"

    def test_compound_crs_resolves_horizontal(self):
        # UTM 12N (32612) + NAVD88 height (5703): the horizontal code wins.
        compound = CompoundCRS(
            name="UTM12N + NAVD88",
            components=[CRS.from_epsg(32612), CRS.from_epsg(5703)],
        )
        assert _wkt_to_epsg(compound.to_wkt()) == "EPSG:32612"

    def test_empty_wkt_raises_missing_crs(self):
        with pytest.raises(ProcessingError) as exc:
            _wkt_to_epsg("")
        assert exc.value.code == "MISSING_CRS"

    def test_non_epsg_crs_raises_unresolvable(self):
        # A bespoke transverse-mercator that matches no EPSG entry.
        wkt = CRS.from_proj4(
            "+proj=tmerc +lat_0=17 +lon_0=-100.3 +k=0.9998 "
            "+x_0=12345 +y_0=67890 +datum=WGS84 +units=m +no_defs"
        ).to_wkt()
        with pytest.raises(ProcessingError) as exc:
            _wkt_to_epsg(wkt)
        assert exc.value.code == "UNRESOLVABLE_CRS"


class TestRequireCrsMatch:
    def test_match_passes(self):
        _require_crs_match("EPSG:32612", "EPSG:32612")

    def test_mismatch_raises(self):
        with pytest.raises(ProcessingError) as exc:
            _require_crs_match("EPSG:32613", "EPSG:32612")
        assert exc.value.code == "CRS_MISMATCH"
