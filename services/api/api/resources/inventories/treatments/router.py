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

from fastapi import APIRouter, Body, HTTPException, Request, status

from api.db.documents import get_document_async, set_document_async
from api.dependencies import VerifiedDomain
from api.resources.inventories.schema import Inventory
from api.resources.inventories.treatments.examples import (
    APPLY_TREATMENTS_OPENAPI_EXAMPLES,
)
from api.resources.inventories.treatments.schema import ApplyTreatmentsRequest
from api.resources.inventories.utils import (
    validate_feature_conditions,
    validate_inventory_wide_treatment_area,
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
)
async def apply_treatments(
    request: Request,
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

    A treatment thins the stand toward a target metric using a tree-selection
    method. Treatments compose: each is applied to the result of the previous.

    ## Metrics

    Each treatment is discriminated by its `metric`:

    - `diameter` — thin to a diameter-at-breast-height limit (in cm unless
      `unit` is set). `from_below` removes trees smaller than the limit;
      `from_above` removes trees larger than it.
    - `basal_area` — thin to a residual basal area (in `m**2/ha` unless `unit`
      is set). `from_below`/`from_above` remove the smallest/largest trees first
      until the target is reached; `proportional` removes across all diameter
      classes, preserving the diameter distribution.

    `proportional` is only valid for a basal-area target — it is not an option
    for a diameter limit.

    ## Units

    `value` uses the metric's native unit (`cm` for diameter, `m**2/ha` for
    basal area) unless an optional `unit` is supplied. A supplied `unit` must be
    canonical and dimensionally compatible with the native unit; it is converted
    before the treatment is applied.

    ## Spatial scoping

    An optional `conditions` list restricts the treatment to a region
    (`within`/`outside`/`intersects` a geometry or a referenced Feature, with an
    optional `buffer_m`). An empty/omitted list treats the entire inventory. A
    referenced Feature must belong to the same domain as this inventory and be
    in `completed` status; cross-domain, missing, or unfinished references are
    rejected with 422.

    Because a basal-area treatment holds its entire treated population in memory
    at once, an inventory-wide basal-area treatment over a very large domain is
    rejected with 422 — scope it with a spatial condition.

    ## Requirements

    Treatments thin against tree diameter, so the inventory must have a `dbh`
    column. Inventories derived from a canopy height model (CHM) carry only
    height and position, so treatments cannot be applied to them (422).

    ## Response

    Returns this inventory (same ID) with status `"pending"` and the submitted
    treatments appended to `treatments`. Its `checksum` changes, so any resource
    derived from it can detect that the source has changed. Poll the inventory
    until status returns to `"completed"`.
    """
    owner_id = request.state.id
    domain_id = domain["id"]

    await validate_feature_conditions(body.treatments, owner_id, domain_id)
    validate_inventory_wide_treatment_area(domain, body.treatments)

    # Inventory must exist, be owned, in this domain, and completed.
    _, snapshot = await get_document_async(
        COLLECTION,
        inventory_id,
        owner_id=owner_id,
        domain_id=domain_id,
        document_status="completed",
    )
    inventory_data = snapshot.to_dict()

    # Treatments thin against diameter; an inventory without a dbh column (e.g. a
    # CHM-derived inventory) has nothing to thin. Reject in the same spirit as
    # the create-time CHM rejection rather than failing the async job later.
    columns = inventory_data.get("columns", [])
    if not any(column.get("key") == "dbh" for column in columns):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=(
                "Silvicultural treatments require a tree diameter (`dbh`) to thin "
                "against, but this inventory has no `dbh` column. CHM-derived "
                "inventories carry only height and position, so treatments cannot "
                "be applied to them."
            ),
        )

    new_treatments = stringify_modification_coordinates(
        [t.model_dump() for t in body.treatments]
    )

    # Append the new treatments to the cumulative ledger, and queue only this
    # delta for standgen to apply to the current data (pending_treatments). The
    # checksum is re-assigned here so derivatives become detectably stale (#304).
    existing = inventory_data.get("treatments", [])
    inventory_data["treatments"] = existing + new_treatments
    inventory_data["pending_treatments"] = new_treatments
    inventory_data["checksum"] = uuid.uuid4().hex
    inventory_data["status"] = JobStatus.pending.value
    inventory_data["progress"] = None
    inventory_data["error"] = None
    inventory_data["modified_on"] = datetime.now()

    await set_document_async(COLLECTION, inventory_id, inventory_data)
    await create_http_task_async(STANDGEN_QUEUE, STANDGEN_SERVICE, inventory_id)

    # pending_treatments is an internal work-queue field, not part of the
    # Inventory schema; Pydantic ignores it (and owner_id) on construction.
    return Inventory(**inventory_data)
