"""
Shared fixtures and helpers for lookup endpoint tests (FBFM13, FBFM40, FCCS, ...).

Fixtures here (second_domain_for_lookup, grid_without_fbfm_band) are
auto-discovered by pytest for every test module under this directory --
no import needed. The plain helper functions (_make_landfire_grid_data,
_persist_and_cleanup, _make_cross_domain_lookup_grid) are not fixtures;
each product's test_router.py imports them explicitly since they're
called directly by name.
"""

import pytest

from lib.config import DOMAINS_COLLECTION, GRIDS_COLLECTION
from tests.fixtures import make_domain_data, make_grid_data


def _make_landfire_grid_data(
    domain_id, *, product, version, band_key, description, **overrides
):
    """Build grid data for a completed/pending LANDFIRE source grid."""
    return make_grid_data(
        domain_id=domain_id,
        source={
            "name": "landfire",
            "product": product,
            "version": version,
            "description": description,
        },
        bands=[{"key": band_key, "type": "categorical", "unit": None, "index": 0}],
        **overrides,
    )


def _persist_and_cleanup(firestore_client, collection, data):
    """Write a Firestore doc, yield it, delete it on teardown."""
    doc_ref = firestore_client.collection(collection).document(data["id"])
    doc_ref.set(data)
    yield data
    doc_ref.delete()


def _make_cross_domain_lookup_grid(
    firestore_client,
    second_domain_for_lookup,
    *,
    product,
    version,
    band_key,
    description,
    name,
):
    """Build and persist a completed LANDFIRE grid in
    `second_domain_for_lookup`, for a product's cross-domain lookup test.
    Yields the grid data; deletes it on teardown."""
    grid_data = _make_landfire_grid_data(
        second_domain_for_lookup["id"],
        product=product,
        version=version,
        band_key=band_key,
        description=description,
        name=name,
        status="completed",
    )
    yield from _persist_and_cleanup(firestore_client, GRIDS_COLLECTION, grid_data)


@pytest.fixture(scope="session")
def second_domain_for_lookup(firestore_client):
    """A second domain owned by test-owner, used for cross-domain tests --
    shared by lookup tests across products."""
    domain_data = make_domain_data(name="Second Domain for Lookup Tests")
    doc_ref = firestore_client.collection(DOMAINS_COLLECTION).document(
        domain_data["id"]
    )
    doc_ref.set(domain_data)
    yield domain_data
    doc_ref.delete()


@pytest.fixture(scope="session")
def grid_without_fbfm_band(firestore_client, domain_for_testing):
    """A complete grid with neither an 'fbfm' nor 'fbfm13' band -- shared
    by lookup tests across products for the missing-band case."""
    grid_data = make_grid_data(
        domain_id=domain_for_testing["id"],
        name="3DEP elevation grid",
        status="completed",
        source={"name": "3dep", "resolution": "10m", "version": "2023"},
        bands=[
            {"key": "elevation", "type": "continuous", "unit": "m", "index": 0},
        ],
    )
    yield from _persist_and_cleanup(firestore_client, GRIDS_COLLECTION, grid_data)
