"""
api/v2/resources/inventories/modifications/router.py

Router for the in-place inventory modifications endpoint.

POST /domains/{domain_id}/inventories/{inventory_id}/modifications
Applies modifications to the existing inventory in place (same ID), asynchronously.
"""

import uuid
from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Body, HTTPException, Request, status
from google.cloud import firestore

from api.db.documents import firestore_client
from api.dependencies import VerifiedDomain
from api.resources.inventories.modifications.examples import (
    APPLY_MODIFICATIONS_OPENAPI_EXAMPLES,
)
from api.resources.inventories.modifications.schema import ApplyModificationsRequest
from api.resources.inventories.schema import Inventory
from api.resources.inventories.utils import validate_feature_conditions
from api.resources.modifications import stringify_modification_coordinates
from api.schema import JobStatus
from api.tasks import create_http_task_async
from lib.config import INVENTORIES_COLLECTION, STANDGEN_QUEUE, STANDGEN_SERVICE

router = APIRouter()

COLLECTION = INVENTORIES_COLLECTION


@router.post(
    "",
    response_model=Inventory,
    status_code=status.HTTP_200_OK,
    summary="Apply modifications to an inventory in place",
)
async def apply_modifications(
    request: Request,
    domain: VerifiedDomain,
    inventory_id: str,
    body: Annotated[
        ApplyModificationsRequest,
        Body(openapi_examples=APPLY_MODIFICATIONS_OPENAPI_EXAMPLES),
    ],
):
    """
    # Apply Modifications to an Inventory (in place)

    Applies modifications to **this** inventory in place — the inventory keeps
    its ID and the submitted rules are appended to its cumulative
    `modifications` list, then the tree data is re-derived asynchronously. To
    keep the original data instead, duplicate the inventory first
    (`POST .../{inventory_id}/duplicate`) and modify the copy.

    Modifications filter trees by conditions and apply actions (remove,
    multiply, divide, add, subtract, replace) to matching rows. Conditions
    within a single rule are ANDed together; multiple rules are evaluated
    independently in order.

    ## Conditions

    **Attribute conditions** compare a single tree attribute against a value:
    - `attribute`: one of `dbh`, `height`, `crown_ratio`, `fia_species_code`
    - `operator`: `eq`, `ne`, `gt`, `lt`, `ge`, `le`
      (`fia_species_code` only supports `eq`/`ne`)
    - `value`: number, string, or list for `eq`/`ne`
    - `unit`: (optional) pint-compatible unit string (e.g., `"in"`, `"ft"`)

    **Expression conditions** use a boolean expression:
    - `expression`: e.g., `"dbh < 5 and height < 2"`
    - Only `dbh`, `height`, `crown_ratio` are allowed in expressions
    - Expressions always use native units (cm, m, 0-1 fraction)

    **Spatial conditions** test each tree's location (a point) against a
    geometry. Two variants discriminated by the required `source` field:

    - `source: "geometry"` — supply GeoJSON directly via `geometry` (plus
      optional `crs`; defaults to the domain CRS).
    - `source: "feature"` — reference a persisted Feature resource by
      `feature_id` (road, water, layerset). The Feature must belong to the
      same domain as this inventory and be in `completed` status;
      cross-domain, missing, or unfinished references are rejected with 422.

    Both spatial variants accept:
    - `operator`: `within`, `outside`, or `intersects`
    - `buffer_m`: (optional, meters) expands the geometry outward in the
      domain's projected CRS before testing. Effectively required for
      linestring features (e.g. roads) because a tree point almost never
      intersects a bare linestring.

    Spatial conditions have **no `target` field** — trees are points, so
    the test is always point-in-(optionally-buffered)-geometry.

    Spatial and attribute conditions can be combined in a single rule
    (AND semantics). For example: `{conditions: [feature within road
    buffer, dbh > 30], actions: [remove]}` removes only large trees that
    fall inside the buffered road.

    ## Actions

    - `{"modifier": "remove"}` — remove matching trees (must be sole action)
    - `{"attribute": "...", "modifier": "multiply|divide|add|subtract|replace", "value": ...}`
    - `unit` on actions converts the value before applying

    ## Response

    Returns this inventory (same ID) with status `"pending"`. Its `checksum`
    changes immediately, so any resource derived from it can detect that the
    source has changed. The submitted rules appear in the inventory's
    `modifications` list once processing completes — poll the inventory until
    status returns to `"completed"`.

    If processing fails, the inventory's status becomes `"failed"` with error
    details, the stored data is unchanged, and the queued rules are retained —
    submit another POST to retry (the new rules are applied together with the
    retained ones).

    ## Error Responses

    - **404 Not Found**: The inventory does not exist, is not owned by the
      caller, or is not in this domain.
    - **422 Unprocessable Content**: The inventory is not in `completed` status
      (and is not a retryable failed modification); or a referenced `feature_id`
      is missing, cross-domain, or not completed.
    """
    owner_id = request.state.id
    domain_id = domain["id"]

    await validate_feature_conditions(body.modifications, owner_id, domain_id)

    new_modifications = stringify_modification_coordinates(
        [m.model_dump() for m in body.modifications]
    )
    new_checksum = uuid.uuid4().hex
    ref = firestore_client.collection(COLLECTION).document(inventory_id)

    @firestore.async_transactional
    async def _append_pending(transaction) -> dict:
        """Read-validate-append atomically so concurrent POSTs can't drop a
        rule: the loser's transaction retries, re-reads the now-`pending`
        status, and is rejected instead of overwriting the winner's delta."""
        snapshot = await ref.get(transaction=transaction)
        inventory_data = snapshot.to_dict() if snapshot.exists else None
        if (
            inventory_data is None
            or inventory_data.get("owner_id") != owner_id
            or inventory_data.get("domain_id") != domain_id
        ):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document not found: inventories/{inventory_id}",
            )

        pending = inventory_data.get("pending_modifications") or []
        inventory_status = inventory_data.get("status")
        retryable_failed = inventory_status == JobStatus.failed.value and pending
        if inventory_status != JobStatus.completed.value and not retryable_failed:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=(
                    f"inventories/{inventory_id} status is '{inventory_status}', "
                    f"expected 'completed'."
                ),
            )

        # Queue only the delta; standgen merges pending_modifications into the
        # cumulative `modifications` ledger atomically with status=completed, so
        # the ledger always equals the applied data (#319). The checksum rotates
        # now so derivatives become detectably stale (#304).
        update = {
            "pending_modifications": pending + new_modifications,
            "checksum": new_checksum,
            "status": JobStatus.pending.value,
            "progress": None,
            "error": None,
            "modified_on": datetime.now(),
        }
        transaction.update(ref, update)
        return {**inventory_data, **update}

    inventory_data = await _append_pending(firestore_client.transaction())

    # The task name embeds the fresh checksum: Cloud Tasks tombstones reused
    # task names, so re-using the bare inventory_id (the create task's name)
    # would silently drop this task.
    await create_http_task_async(
        STANDGEN_QUEUE,
        STANDGEN_SERVICE,
        inventory_id,
        task_name=f"{inventory_id}-{new_checksum}",
    )

    # pending_modifications is an internal work-queue field, not part of the
    # Inventory schema; Pydantic ignores it (and owner_id) on construction.
    return Inventory(**inventory_data)
