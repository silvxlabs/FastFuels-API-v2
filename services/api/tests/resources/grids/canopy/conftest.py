"""
Point cloud fixtures for the point-cloud CHM endpoint.

The point_clouds test package has its own fixtures, but they are not visible
here, so the source clouds this endpoint validates against are seeded locally.
Each covers one rejection the endpoint is responsible for.
"""

import pytest

from lib.config import POINT_CLOUDS_COLLECTION
from tests.fixtures import make_point_cloud_data


def _seed(firestore_client, **kwargs):
    """Write a point cloud document and remove it on teardown."""
    data = make_point_cloud_data(**kwargs)
    doc_ref = firestore_client.collection(POINT_CLOUDS_COLLECTION).document(data["id"])
    doc_ref.set(data)
    yield data
    doc_ref.delete()


@pytest.fixture(scope="session")
def als_point_cloud(firestore_client, domain_for_testing):
    """A completed airborne cloud carrying ground classification."""
    yield from _seed(
        firestore_client,
        domain_id=domain_for_testing["id"],
        name="ALS cloud for CHM",
        status="completed",
        point_cloud_type="als",
        summary={
            "point_count": 10_496_309,
            "point_classes": [1, 2, 7, 9, 18],
            "density": 19.8,
        },
    )


@pytest.fixture(scope="session")
def tls_point_cloud(firestore_client, domain_for_testing):
    """A terrestrial cloud — no landscape canopy surface to rasterize."""
    yield from _seed(
        firestore_client,
        domain_id=domain_for_testing["id"],
        name="TLS cloud",
        status="completed",
        point_cloud_type="tls",
    )


@pytest.fixture(scope="session")
def pending_point_cloud(firestore_client, domain_for_testing):
    """A cloud still being produced — its LAZ does not exist yet."""
    yield from _seed(
        firestore_client,
        domain_id=domain_for_testing["id"],
        name="Pending cloud",
        status="pending",
    )


@pytest.fixture(scope="session")
def point_cloud_in_other_domain(firestore_client, second_domain):
    """A cloud belonging to a different domain of the same owner."""
    yield from _seed(
        firestore_client,
        domain_id=second_domain["id"],
        name="Cloud in another domain",
        status="completed",
    )


@pytest.fixture(scope="session")
def point_cloud_of_other_owner(firestore_client, domain_for_testing):
    """A completed cloud owned by somebody else."""
    yield from _seed(
        firestore_client,
        domain_id=domain_for_testing["id"],
        owner_id="a-different-owner",
        name="Someone else's cloud",
        status="completed",
    )
