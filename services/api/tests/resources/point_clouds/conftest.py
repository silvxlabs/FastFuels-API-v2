"""
Shared fixtures for point cloud router tests.

These fixtures seed documents directly in Firestore — the same way the
list/get/patch/delete endpoints are exercised.
"""

import pytest

from lib.config import POINT_CLOUDS_COLLECTION
from tests.fixtures import make_point_cloud_data


@pytest.fixture(scope="session")
def point_cloud_in_firestore(firestore_client, domain_for_testing):
    """A completed ALS point cloud (with a georeference) in the test domain."""
    pc_data = make_point_cloud_data(
        domain_id=domain_for_testing["id"],
        name="Test Point Cloud for GET",
        description="Created by fixture for GET endpoint tests",
        status="completed",
        tags=["test", "fixture"],
        georeference={
            "crs": "EPSG:32612",
            "bounds": [500000.0, 5060000.0, 1800.0, 501000.0, 5061000.0, 1980.0],
        },
    )
    doc_ref = firestore_client.collection(POINT_CLOUDS_COLLECTION).document(
        pc_data["id"]
    )
    doc_ref.set(pc_data)
    yield pc_data
    doc_ref.delete()


@pytest.fixture(scope="session")
def point_cloud_with_different_owner(firestore_client, domain_with_different_owner):
    """A point cloud owned by a different user, for ownership validation tests."""
    pc_data = make_point_cloud_data(
        domain_id=domain_with_different_owner["id"],
        owner_id="different-owner",
        name="Other User's Point Cloud",
    )
    doc_ref = firestore_client.collection(POINT_CLOUDS_COLLECTION).document(
        pc_data["id"]
    )
    doc_ref.set(pc_data)
    yield pc_data
    doc_ref.delete()
