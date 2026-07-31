"""
Integration tests for the 3DEP point cloud handler.

These run the worker end to end: real USGS 3DEP reads, real Firestore, real
GCS. They need network access and GCP credentials.

Domain fixtures with known, opposite coverage:

- ``bondurant.json`` — fully covered by one acquisition, ~1.7M points.
- ``threedep_ept_seam.json`` — sits on the boundary between two acquisitions,
  so a fetch has to merge them.
- ``blue_mtn.json`` — no 3DEP lidar at all.

Run with: uv run pytest tests/integration/ -v
"""

import laspy
import numpy as np
import pytest
from lakitu.main import process_point_cloud_request
from lakitu.storage import cloud_path

from lib.gcs import get_gcsfs_client

from .conftest import MockRequest

pytestmark = pytest.mark.integration


def run_worker(point_cloud_id: str):
    """Invoke the worker the way Cloud Tasks would."""
    return process_point_cloud_request(MockRequest({"id": point_cloud_id}))


def read_written_cloud(point_cloud_id: str) -> laspy.LasData:
    """Read back the LAZ the worker stored."""
    with get_gcsfs_client().open(cloud_path(point_cloud_id), "rb") as stream:
        return laspy.read(stream)


class TestSingleAcquisition:
    """The common case: one acquisition covers the whole domain."""

    def test_fetches_clips_and_reprojects(
        self, seeded_domain, seeded_point_cloud, read_point_cloud
    ):
        domain_id = seeded_domain("bondurant.json")
        point_cloud_id = seeded_point_cloud(domain_id)

        _, code = run_worker(point_cloud_id)
        assert code == 200

        doc = read_point_cloud(point_cloud_id)
        assert doc["status"] == "completed", doc.get("error")

        # Stored in the domain's CRS.
        georeference = doc["georeference"]
        assert georeference["crs"] == "EPSG:32612"

        summary = doc["summary"]
        assert summary["point_count"] > 1_000_000
        # Ground and unclassified are present in any 3DEP tile.
        assert {1, 2} <= set(summary["point_classes"])
        assert summary["density"] > 0

        # Storage quota aggregates this; without it the cloud counts as free.
        assert doc["size_bytes"] > 0

        source = doc["source"]
        assert source["name"] == "3dep"
        assert source["datasets"] == ["WY_Southwest_1_2020"]
        assert source["coverage_fraction"] == pytest.approx(1.0, abs=1e-3)

    def test_written_points_match_the_domain(
        self, seeded_domain, seeded_point_cloud, read_point_cloud
    ):
        """Every stored point must lie inside the domain, in the domain's CRS."""
        domain_id = seeded_domain("bondurant.json")
        point_cloud_id = seeded_point_cloud(domain_id)
        run_worker(point_cloud_id)

        doc = read_point_cloud(point_cloud_id)
        assert doc["status"] == "completed", doc.get("error")

        las = read_written_cloud(point_cloud_id)
        assert las.header.parse_crs().to_epsg() == 32612
        assert len(las.points) == doc["summary"]["point_count"]

        # The bondurant fixture extent.
        min_x, min_y, max_x, max_y = 522800, 4720400, 523300, 4720900
        x, y = np.asarray(las.x), np.asarray(las.y)
        assert x.min() >= min_x - 0.01 and x.max() <= max_x + 0.01
        assert y.min() >= min_y - 0.01 and y.max() <= max_y + 0.01

    def test_checksum_survives_processing(
        self, seeded_domain, seeded_point_cloud, read_point_cloud
    ):
        """Derived resources compare against the creation-time checksum."""
        domain_id = seeded_domain("bondurant.json")
        point_cloud_id = seeded_point_cloud(domain_id)
        before = read_point_cloud(point_cloud_id)["checksum"]

        run_worker(point_cloud_id)

        assert read_point_cloud(point_cloud_id)["checksum"] == before

    def test_pinned_acquisition_is_honored(
        self, seeded_domain, seeded_point_cloud, read_point_cloud
    ):
        domain_id = seeded_domain("bondurant.json")
        point_cloud_id = seeded_point_cloud(
            domain_id,
            source={
                "name": "3dep",
                "requested_datasets": ["WY_Southwest_1_2020"],
            },
        )

        run_worker(point_cloud_id)

        doc = read_point_cloud(point_cloud_id)
        assert doc["status"] == "completed", doc.get("error")
        assert doc["source"]["datasets"] == ["WY_Southwest_1_2020"]


class TestAcquisitionMerge:
    """A domain no single acquisition covers, so two must be combined."""

    def test_merges_two_acquisitions_without_duplicating_points(
        self, seeded_domain, seeded_point_cloud, read_point_cloud
    ):
        """The seam is the risk: overlapping acquisitions must not double up.

        Each acquisition is read only within its own disjoint contribution, so
        no ground is covered twice.
        """
        domain_id = seeded_domain("threedep_ept_seam.json")
        point_cloud_id = seeded_point_cloud(domain_id)

        _, code = run_worker(point_cloud_id)
        assert code == 200

        doc = read_point_cloud(point_cloud_id)
        assert doc["status"] == "completed", doc.get("error")
        assert len(doc["source"]["datasets"]) >= 2
        assert doc["source"]["coverage_fraction"] == pytest.approx(1.0, abs=1e-2)

        las = read_written_cloud(point_cloud_id)
        identity = np.stack(
            [
                np.asarray(las.x),
                np.asarray(las.y),
                np.asarray(las.z),
                np.asarray(las.gps_time),
            ],
            axis=1,
        )
        unique = np.unique(identity, axis=0).shape[0]
        assert unique == len(las.points)

    def test_merged_output_is_a_single_readable_cloud(
        self, seeded_domain, seeded_point_cloud, read_point_cloud
    ):
        """Acquisitions need not share a point format; the output is one file.

        laspy refuses to write records whose point format differs from the
        file's, so a merge only works because every source is normalized first.
        """
        domain_id = seeded_domain("threedep_ept_seam.json")
        point_cloud_id = seeded_point_cloud(domain_id)
        run_worker(point_cloud_id)

        doc = read_point_cloud(point_cloud_id)
        assert doc["status"] == "completed", doc.get("error")

        las = read_written_cloud(point_cloud_id)
        assert las.header.point_format.id >= 6
        assert len(las.points) == doc["summary"]["point_count"]


class TestNoCoverage:
    """A domain 3DEP does not reach."""

    def test_uncovered_domain_fails_with_a_structured_error(
        self, seeded_domain, seeded_point_cloud, read_point_cloud
    ):
        domain_id = seeded_domain("blue_mtn.json")
        point_cloud_id = seeded_point_cloud(domain_id)

        _, code = run_worker(point_cloud_id)
        # Terminal outcome: acknowledged so Cloud Tasks does not retry it.
        assert code == 200

        doc = read_point_cloud(point_cloud_id)
        assert doc["status"] == "failed"
        assert doc["error"]["code"] == "COVERAGE_ERROR"
        assert doc["error"]["message"]

    def test_no_artifact_is_left_behind(self, seeded_domain, seeded_point_cloud):
        """A failed fetch must not leave bytes counting against storage quota."""
        domain_id = seeded_domain("blue_mtn.json")
        point_cloud_id = seeded_point_cloud(domain_id)

        run_worker(point_cloud_id)

        assert not get_gcsfs_client().exists(cloud_path(point_cloud_id))


class TestUnknownSource:
    """Sources this worker does not serve."""

    def test_unknown_source_fails_terminally(
        self, seeded_domain, seeded_point_cloud, read_point_cloud
    ):
        domain_id = seeded_domain("bondurant.json")
        point_cloud_id = seeded_point_cloud(domain_id, source={"name": "upload"})

        _, code = run_worker(point_cloud_id)
        assert code == 200

        doc = read_point_cloud(point_cloud_id)
        assert doc["status"] == "failed"
        assert doc["error"]["code"] == "UNKNOWN_SOURCE"
