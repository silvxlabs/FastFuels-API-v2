"""
Shared fixtures for grid router tests.

Provides target-grid fixtures used by every router that reads or aligns
to an existing grid (resample plus every source-grid endpoint that
accepts ``alignment.target='grid'``). Each fixture seeds a different
combination of (owner, domain, status, georeference) so tests can probe
each rejection path of the shared validator.
"""

import pytest

from lib.config import GRIDS_COLLECTION
from tests.fixtures import make_grid_data


@pytest.fixture(scope="session")
def complete_grid(firestore_client, domain_for_testing):
    """A complete grid with bands and georeference for use as a resample
    source or alignment target."""
    grid_data = make_grid_data(
        domain_id=domain_for_testing["id"],
        name="Source grid for resample tests",
        status="completed",
    )
    doc_ref = firestore_client.collection(GRIDS_COLLECTION).document(grid_data["id"])
    doc_ref.set(grid_data)
    yield grid_data
    doc_ref.delete()


@pytest.fixture(scope="session")
def pending_grid(firestore_client, domain_for_testing):
    """A grid with status "pending" (not yet complete)."""
    grid_data = make_grid_data(
        domain_id=domain_for_testing["id"],
        name="Pending grid for resample tests",
        status="pending",
    )
    doc_ref = firestore_client.collection(GRIDS_COLLECTION).document(grid_data["id"])
    doc_ref.set(grid_data)
    yield grid_data
    doc_ref.delete()


@pytest.fixture(scope="session")
def complete_grid_no_georeference(firestore_client, domain_for_testing):
    """A complete grid without a georeference."""
    grid_data = make_grid_data(
        domain_id=domain_for_testing["id"],
        name="Complete grid without georeference",
        status="completed",
    )
    grid_data["georeference"] = None
    doc_ref = firestore_client.collection(GRIDS_COLLECTION).document(grid_data["id"])
    doc_ref.set(grid_data)
    yield grid_data
    doc_ref.delete()


@pytest.fixture(scope="session")
def grid_in_different_domain(firestore_client, second_domain):
    """A complete grid with georeference in a different domain owned by
    the same user."""
    grid_data = make_grid_data(
        domain_id=second_domain["id"],
        name="Grid in different domain",
        status="completed",
    )
    doc_ref = firestore_client.collection(GRIDS_COLLECTION).document(grid_data["id"])
    doc_ref.set(grid_data)
    yield grid_data
    doc_ref.delete()


@pytest.fixture(scope="session")
def grid_owned_by_other_user(firestore_client, domain_with_different_owner):
    """A complete grid owned by a different user, in their domain."""
    grid_data = make_grid_data(
        owner_id="different-owner",
        domain_id=domain_with_different_owner["id"],
        name="Grid owned by other user",
        status="completed",
    )
    doc_ref = firestore_client.collection(GRIDS_COLLECTION).document(grid_data["id"])
    doc_ref.set(grid_data)
    yield grid_data
    doc_ref.delete()
