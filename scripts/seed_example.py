"""
Seed the shared, prebuilt example (#582): the Blue Mountain domain + one
completed NAIP canopy grid, owned by the example owner and readable by any
guest.

This builds the example ONCE. Guests then read it without owning it or spending
quota (see ``api/access.py`` and ``GET /examples``); walle never reaps it (the
example owner is exempt in ``walle/cleanup.py``).

## How it works

There is no credential that authenticates *as* the example owner — it is only
an ownership tag. So the script builds the example through the normal API using
a real seeding key (spending that key's quota exactly once), waits for the NAIP
grid to finish, then PROMOTES the two Firestore docs to the example owner:

1. ``POST /domains`` with the Blue Mountain geometry.
2. ``POST /domains/{id}/grids/canopy/naip`` to dispatch the build.
3. Poll the grid until ``status == "completed"`` (this is the slow, real build).
4. Set ``owner_id = EXAMPLE_OWNER_ID`` and ``is_example = True`` on both docs
   directly in Firestore.

Step 4 is safe for the grid's GCS artifact: artifacts are keyed by grid id, not
owner (see ``walle/layouts.artifact_path``), so re-tagging the owner never
orphans the blob.

Idempotent: if a completed example grid already exists it exits without doing
anything (pass ``--force`` to build another).

## Usage

    cd services/api
    uv run --active python ../../scripts/seed_example.py \\
        --base-url https://api.fastfuels.silvxlabs.com/v2 \\
        --api-key "$SEED_API_KEY"

The API key must belong to a real owner with enough quota for one domain and
one grid dispatch. Point ``--base-url`` and the ambient GCP credentials
(``GCP_PROJECT`` / ADC) at the SAME environment — the script writes Firestore
directly, so a mismatch would promote a doc the API never created.

Set ``--timeout`` (seconds) for how long to wait on the NAIP build; the default
is generous but a cold build can exceed it, in which case re-run with
``--grid-id`` to resume polling an already-dispatched grid.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import httpx
from google.cloud import firestore

from lib.config import (
    DOMAINS_COLLECTION,
    EXAMPLE_OWNER_ID,
    GRIDS_COLLECTION,
)

_GEOMETRY = Path(__file__).parent / "data" / "blue_mountain_feature_4326.geojson"
_POLL_INTERVAL_SECONDS = 15


def _domain_body() -> dict:
    feature = json.loads(_GEOMETRY.read_text())
    return {
        "type": "FeatureCollection",
        "features": [feature],
        "name": "Blue Mountain Example",
        "description": (
            "Shared read-only example domain near Missoula, Montana. Built once "
            "and readable by any guest via GET /examples (#582)."
        ),
        "tags": ["example"],
    }


def _existing_example_grid(db: firestore.Client) -> str | None:
    """Return an existing completed example grid id, or None."""
    query = (
        db.collection(GRIDS_COLLECTION)
        .where("owner_id", "==", EXAMPLE_OWNER_ID)
        .where("is_example", "==", True)
        .where("status", "==", "completed")
        .limit(1)
    )
    for snap in query.stream():
        return snap.id
    return None


def _create_domain(client: httpx.Client) -> str:
    resp = client.post("/domains", json=_domain_body())
    resp.raise_for_status()
    domain_id = resp.json()["id"]
    print(f"created domain {domain_id}")
    return domain_id


def _dispatch_naip(client: httpx.Client, domain_id: str) -> str:
    resp = client.post(
        f"/domains/{domain_id}/grids/canopy/naip",
        json={
            "name": "NAIP Canopy Height",
            "description": "NAIP-derived canopy height model for the example.",
            "tags": ["example"],
        },
    )
    resp.raise_for_status()
    grid_id = resp.json()["id"]
    print(f"dispatched NAIP canopy grid {grid_id}")
    return grid_id


def _wait_for_grid(
    client: httpx.Client, domain_id: str, grid_id: str, timeout: int
) -> None:
    deadline = time.monotonic() + timeout
    while True:
        resp = client.get(f"/domains/{domain_id}/grids/{grid_id}")
        resp.raise_for_status()
        state = resp.json().get("status")
        print(f"grid {grid_id} status: {state}")
        if state == "completed":
            return
        if state == "failed":
            raise SystemExit(f"grid {grid_id} failed to build")
        if time.monotonic() >= deadline:
            raise SystemExit(
                f"timed out after {timeout}s waiting for grid {grid_id}; re-run "
                f"with --domain-id {domain_id} --grid-id {grid_id} to resume."
            )
        time.sleep(_POLL_INTERVAL_SECONDS)


def _promote(db: firestore.Client, domain_id: str, grid_id: str) -> None:
    """Hand the domain and grid to the example owner and flag them public."""
    patch = {"owner_id": EXAMPLE_OWNER_ID, "is_example": True}
    db.collection(DOMAINS_COLLECTION).document(domain_id).update(patch)
    db.collection(GRIDS_COLLECTION).document(grid_id).update(patch)
    print(
        f"promoted domain {domain_id} and grid {grid_id} to owner "
        f"'{EXAMPLE_OWNER_ID}' with is_example=True"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed the shared example (#582).")
    parser.add_argument("--base-url", required=True, help="API base URL (…/v2).")
    parser.add_argument("--api-key", required=True, help="Seeding owner's API key.")
    parser.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="Seconds to wait for the NAIP build (default 3600).",
    )
    parser.add_argument(
        "--domain-id",
        help="Resume: an already-created domain id (skips domain creation).",
    )
    parser.add_argument(
        "--grid-id",
        help="Resume: an already-dispatched grid id (skips domain + dispatch).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Build even if a completed example grid already exists.",
    )
    args = parser.parse_args()

    db = firestore.Client()

    if not args.force:
        existing = _existing_example_grid(db)
        if existing:
            print(
                f"example already seeded (grid {existing}); nothing to do. "
                f"Pass --force to build another."
            )
            return

    client = httpx.Client(
        base_url=args.base_url.rstrip("/"),
        headers={"api-key": args.api_key},
        timeout=60.0,
    )
    with client:
        if args.grid_id:
            if not args.domain_id:
                sys.exit("--grid-id requires --domain-id to resume polling.")
            domain_id, grid_id = args.domain_id, args.grid_id
        else:
            domain_id = args.domain_id or _create_domain(client)
            grid_id = _dispatch_naip(client, domain_id)
        _wait_for_grid(client, domain_id, grid_id, args.timeout)

    _promote(db, domain_id, grid_id)
    print("done. Verify with GET /examples.")


if __name__ == "__main__":
    main()
