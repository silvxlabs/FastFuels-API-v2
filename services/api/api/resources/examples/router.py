"""
api/resources/examples/router.py

Discovery for the shared, prebuilt example (#582).

``GET /examples`` returns the canonical example domain id and its completed
grid id(s) so the web client can deep-link into the 3D viewer without any
hardcoded ids. The example itself is read through the normal domain/grid
endpoints, whose read paths admit it for any authenticated caller (see
``api/access.py``); this endpoint only advertises which ids those are.

Listing is derived from Firestore, not a static config, so the day the example
is re-seeded under new ids the discovery output follows automatically. An
example is any resource owned by :data:`EXAMPLE_OWNER_ID` and flagged
``is_example`` — the exact predicate the read path authorizes against, so this
endpoint can never advertise something a caller would then be denied.
"""

from fastapi import APIRouter, status
from google.cloud.firestore import FieldFilter

from api.db.documents import firestore_client
from api.resources.examples.schema import (
    Example,
    ExampleGrid,
    ListExamplesResponse,
)
from lib.config import DOMAINS_COLLECTION, EXAMPLE_OWNER_ID, GRIDS_COLLECTION

router = APIRouter()


def _example_query(collection: str):
    """Firestore query for the flagged example resources in ``collection``."""
    return (
        firestore_client.collection(collection)
        .where(filter=FieldFilter("owner_id", "==", EXAMPLE_OWNER_ID))
        .where(filter=FieldFilter("is_example", "==", True))
    )


@router.get(
    "",
    response_model=ListExamplesResponse,
    status_code=status.HTTP_200_OK,
    summary="Discover the shared example domain(s) and grid(s)",
)
async def list_examples() -> ListExamplesResponse:
    """
    # List Examples Endpoint

    Returns the canonical, prebuilt example(s) — a Blue Mountain domain and its
    completed NAIP canopy grid — that any authenticated caller, guests included,
    may read without owning them or spending quota.

    Use this to discover the domain id and grid id(s) to deep-link into (for
    example, to open the finished grid in the 3D viewer) instead of hardcoding
    them: the ids are resolved from the live database, so they survive a
    re-seed of the example.

    ## Response

    - **examples**: (array) One entry per example domain:
      - **domain_id**: (string) The example domain's id — read it via
        `GET /domains/{domain_id}`.
      - **name**: (string) The domain's display name.
      - **grids**: (array) The domain's completed, readable grids, each with
        `id`, `name`, and `domain_id`. Read grid data via
        `GET /domains/{domain_id}/grids/{grid_id}/data/{band}/{chunk_index}`.

    An empty `examples` array means no example has been seeded yet.

    ## Access

    Read-only and available to any authenticated caller, guests included. There
    are no write, export, or per-example quota operations here — those run
    against a caller's own resources, never the example.
    """
    domain_snaps = await _example_query(DOMAINS_COLLECTION).get()
    example_domain_ids = {snap.id for snap in domain_snaps}
    if not example_domain_ids:
        return ListExamplesResponse(examples=[])

    grids_by_domain: dict[str, list[ExampleGrid]] = {
        domain_id: [] for domain_id in example_domain_ids
    }
    grid_snaps = await _example_query(GRIDS_COLLECTION).get()
    for snap in grid_snaps:
        grid = snap.to_dict() or {}
        domain_id = grid.get("domain_id")
        # An example grid whose domain is not itself a listed example is a
        # seeding inconsistency; skip it rather than advertise an unreadable id.
        if domain_id not in grids_by_domain:
            continue
        # Only advertise finished grids — the point is a completed example.
        if grid.get("status") != "completed":
            continue
        grids_by_domain[domain_id].append(
            ExampleGrid(
                id=snap.id,
                name=grid.get("name", ""),
                domain_id=domain_id,
            )
        )

    examples = [
        Example(
            domain_id=snap.id,
            name=(snap.to_dict() or {}).get("name", ""),
            grids=grids_by_domain[snap.id],
        )
        for snap in domain_snaps
    ]
    return ListExamplesResponse(examples=examples)
