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

import numpy as np
import pyarrow.dataset as ds
import pytest
from lakitu.main import process_point_cloud_request
from lakitu.parquet_writer import COLUMNS
from lakitu.storage import cloud_prefix

from lib.gcs import get_gcsfs_client

from .conftest import MockRequest

pytestmark = pytest.mark.integration

# Every stored attribute except the LOD level, which the writer assigns.
POINT_COLUMNS = tuple(c for c in COLUMNS if c != "lod")


def run_worker(point_cloud_id: str):
    """Invoke the worker the way Cloud Tasks would."""
    return process_point_cloud_request(MockRequest({"id": point_cloud_id}))


def read_written_cloud(point_cloud_id: str):
    """Read back the Parquet dataset the worker stored, as real-world coords.

    Coordinates are stored as LAS scaled int32s, so they are decoded here the
    same way any reader has to decode them.
    """
    fs = get_gcsfs_client()
    prefix = cloud_prefix(point_cloud_id)
    dataset = ds.dataset(prefix, filesystem=fs, format="parquet", partitioning="hive")
    table = dataset.to_table()
    scales, offsets = _scaling(fs, prefix)
    xyz = (
        np.stack([table.column(c).to_numpy() for c in ("X", "Y", "Z")], axis=1) * scales
        + offsets
    )
    return table, xyz


def _scaling(fs, prefix):
    """Read the scale/offset the dataset was written with."""
    import json

    with fs.open(f"{prefix}/_manifest.json", "rb") as stream:
        manifest = json.load(stream)
    return np.asarray(manifest["scales"]), np.asarray(manifest["offsets"])


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

        table, xyz = read_written_cloud(point_cloud_id)
        assert table.num_rows == doc["summary"]["point_count"]
        # The dataset carries no CRS of its own; the resource is what georeferences
        # it, and that is asserted above.

        # The bondurant fixture extent.
        min_x, min_y, max_x, max_y = 522800, 4720400, 523300, 4720900
        x, y = xyz[:, 0], xyz[:, 1]
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

        table, _ = read_written_cloud(point_cloud_id)
        # A point's identity is every attribute it carries, not its position.
        # Coordinates alone are not unique: 5,697 pairs of genuinely distinct
        # returns share a cubic millimetre in this fixture, differing in
        # intensity, classification and source id. gps_time used to supply the
        # discrimination and the schema no longer stores it, so the remaining
        # attributes have to. `lod` is excluded because it is assigned here
        # rather than read, and a duplicated point could land on two levels --
        # which would hide exactly what this test looks for.
        identity = np.stack(
            [table.column(c).to_numpy().astype(np.int64) for c in POINT_COLUMNS],
            axis=1,
        )
        unique = np.unique(identity, axis=0).shape[0]
        assert unique == table.num_rows

    def test_merged_output_is_a_single_readable_cloud(
        self, seeded_domain, seeded_point_cloud, read_point_cloud
    ):
        """Acquisitions need not share a point format; the output is one schema.

        Sources carrying different LAS point formats have different dimensions,
        so a merge only works because every source is normalized first.
        """
        domain_id = seeded_domain("threedep_ept_seam.json")
        point_cloud_id = seeded_point_cloud(domain_id)
        run_worker(point_cloud_id)

        doc = read_point_cloud(point_cloud_id)
        assert doc["status"] == "completed", doc.get("error")

        table, _ = read_written_cloud(point_cloud_id)
        # tile_x/tile_y come from the hive partitioning, not the point schema.
        assert set(COLUMNS) <= set(table.column_names)
        assert table.num_rows == doc["summary"]["point_count"]


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

        assert not get_gcsfs_client().exists(cloud_prefix(point_cloud_id))


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
