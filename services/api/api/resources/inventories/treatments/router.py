"""
api/v2/resources/inventories/treatments/router.py

Router for the in-place inventory treatments endpoint.

POST /domains/{domain_id}/inventories/{inventory_id}/treatments
Applies silvicultural treatments to the existing inventory in place (same ID),
asynchronously.
"""

import uuid
from datetime import datetime
from typing import Annotated

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Body,
    HTTPException,
    Request,
    Response,
    status,
)
from google.cloud import firestore

from api.db.documents import firestore_client
from api.dependencies import VerifiedDomain
from api.quota import QUOTA_429_RESPONSE, enforce_create_quotas, register_dispatch
from api.resources.inventories.schema import Inventory
from api.resources.inventories.treatments.examples import (
    APPLY_TREATMENTS_OPENAPI_EXAMPLES,
)
from api.resources.inventories.treatments.schema import ApplyTreatmentsRequest
from api.resources.inventories.utils import (
    validate_feature_conditions,
    validate_inventory_wide_treatment_area,
    validate_treatment_columns,
)
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
    summary="Apply treatments to an inventory in place",
    responses=QUOTA_429_RESPONSE,
)
async def apply_treatments(
    request: Request,
    response: Response,
    background_tasks: BackgroundTasks,
    domain: VerifiedDomain,
    inventory_id: str,
    body: Annotated[
        ApplyTreatmentsRequest,
        Body(openapi_examples=APPLY_TREATMENTS_OPENAPI_EXAMPLES),
    ],
):
    """
    # Apply Treatments to an Inventory (in place)

    Applies silvicultural treatments to **this** inventory in place — the
    inventory keeps its ID and the submitted treatments are appended to its
    cumulative `treatments` list, then the tree data is re-derived
    asynchronously. To keep the original data instead, duplicate the inventory
    first (`POST .../{inventory_id}/duplicate`) and treat the copy.

    A treatment thins the stand toward a target metric. Treatments compose: each
    is applied to the result of the previous.

    ## Metrics

    Each treatment is discriminated by its `metric`:

    - `diameter` — thin to a diameter-at-breast-height limit (in cm unless
      `unit` is set). `from_below` removes trees smaller than the limit;
      `from_above` removes trees larger than it.
    - `basal_area` — thin to a residual basal area (in `m**2/ha` unless `unit`
      is set). `from_below`/`from_above` remove the smallest/largest trees first
      until the target is reached; `proportional` removes across all diameter
      classes, preserving the diameter distribution.

    - `crown_spacing` — thin to a minimum gap between retained crowns, crown
      edge to crown edge (in `m` unless `unit` is set). Trees are visited from
      the largest `rank_by` value down (`dbh` by default, or `height`), and each
      is kept only if its crown clears every crown already kept. Crown radii
      come from `max_crown_radius_source`: the `purves` allometry by default,
      `crookston_stage`, or a radius column such as the `crown_radius` added by
      CHM crown segmentation. Trees with no value in the column use their
      `purves` radius.

    `proportional` is only valid for a basal-area target — it is not an option
    for a diameter limit.

    ## Units

    `value` uses the metric's native unit (`cm` for diameter, `m**2/ha` for
    basal area, `m` for crown spacing) unless an optional `unit` is supplied. A supplied `unit` must be
    canonical and dimensionally compatible with the native unit; it is converted
    before the treatment is applied.

    ## Spatial scoping

    An optional `conditions` list restricts the treatment to a region
    (`within`/`outside`/`intersects` a geometry or a referenced Feature, with an
    optional `buffer_m`). An empty/omitted list treats the entire inventory. A
    referenced Feature must belong to the same domain as this inventory and be
    in `completed` status; cross-domain, missing, or unfinished references are
    rejected with 422.

    A tree is in the region when its stem is. A spatially scoped crown-spacing
    treatment removes only trees in the region, but trees in the region must
    also clear the crowns of the trees around it.

    Basal-area and crown-spacing treatments hold their entire treated population
    in memory at once, so an inventory-wide one over a very large domain is
    rejected with 422 — scope it with a spatial condition.

    ## Requirements

    The inventory must have every column a treatment reads. Diameter and
    basal-area treatments read `dbh`. A crown-spacing treatment reads its
    `rank_by` column and either its radius column or the columns of its radius
    allometry (`fia_species_code`, `dbh`, `height`, and, for `purves`,
    `crown_base_height`). A CHM-derived inventory with crown segmentation can be
    thinned with `rank_by: height` and the `crown_radius` column.

    ## Response

    Returns this inventory (same ID) with status `"pending"`. Its `checksum`
    changes immediately, so any resource derived from it can detect that the
    source has changed. The submitted treatments appear in the inventory's
    `treatments` list once processing completes — poll the inventory until
    status returns to `"completed"`.

    If processing fails, the inventory's status becomes `"failed"` with error
    details, the stored data is unchanged, and the queued treatments are
    retained — submit another POST to retry (the new treatments are applied
    together with the retained ones).

    ## Error Responses

    - **404 Not Found**: The inventory does not exist, is not owned by the
      caller, or is not in this domain.
    - **422 Unprocessable Content**: The inventory is not in `completed` status
      (and is not a retryable failed treatment); the inventory lacks a column a
      treatment reads (e.g. `dbh` on a CHM-derived inventory); an
      inventory-wide basal-area or crown-spacing treatment over a very large
      domain; or a referenced `feature_id` is missing, cross-domain, or not
      completed.
    - **429 Too Many Requests**: You have too many active inventory jobs in
      progress (your `max_active_inventories` quota). Wait for jobs to complete
      or delete unneeded inventories, then retry. The response detail names the
      exact `quota` and includes a `Retry-After` header.
    """
    owner_id = request.state.id
    domain_id = domain["id"]

    await enforce_create_quotas(COLLECTION, request)

    await validate_feature_conditions(body.treatments, owner_id, domain_id)
    validate_inventory_wide_treatment_area(domain, body.treatments)

    new_treatments = stringify_modification_coordinates(
        [t.model_dump() for t in body.treatments]
    )
    new_checksum = uuid.uuid4().hex
    ref = firestore_client.collection(COLLECTION).document(inventory_id)

    @firestore.async_transactional
    async def _append_pending(transaction) -> dict:
        """Read-validate-append atomically so concurrent POSTs can't drop a
        treatment: the loser's transaction retries, re-reads the now-`pending`
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

        # Reject a treatment that reads a column this inventory lacks (e.g. a
        # diameter limit on a CHM-derived inventory) rather than failing the
        # async job later.
        validate_treatment_columns(
            body.treatments,
            {column["key"] for column in inventory_data.get("columns", [])},
        )

        pending = inventory_data.get("pending_treatments") or []
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

        # Queue only the delta; standgen merges pending_treatments into the
        # cumulative `treatments` ledger atomically with status=completed, so the
        # ledger always equals the applied data (#319). The checksum rotates now
        # so derivatives become detectably stale (#304).
        update = {
            "pending_treatments": pending + new_treatments,
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
    register_dispatch(request, response, background_tasks)

    # pending_treatments is an internal work-queue field, not part of the
    # Inventory schema; Pydantic ignores it (and owner_id) on construction.
    return Inventory(**inventory_data)
