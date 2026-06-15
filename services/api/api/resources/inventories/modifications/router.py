"""
api/v2/resources/inventories/modifications/router.py

Router for the in-place inventory modifications endpoint.

POST /domains/{domain_id}/inventories/{inventory_id}/modifications
Applies modifications to the existing inventory in place (same ID), asynchronously.
"""

import uuid
from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Body, Request, status

from api.db.documents import get_document_async, set_document_async
from api.dependencies import VerifiedDomain
from api.resources.inventories.modification_models import (
    modification_referenced_columns,
)
from api.resources.inventories.modifications.examples import (
    APPLY_MODIFICATIONS_OPENAPI_EXAMPLES,
)
from api.resources.inventories.modifications.schema import ApplyModificationsRequest
from api.resources.inventories.schema import Inventory
from api.resources.inventories.utils import (
    inventory_column_keys,
    require_inventory_columns,
    validate_feature_conditions,
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

    Returns this inventory (same ID) with status `"pending"` and the submitted
    rules appended to `modifications`. Its `checksum` changes, so any resource
    derived from it can detect that the source has changed. Poll the inventory
    until status returns to `"completed"`.
    """
    owner_id = request.state.id
    domain_id = domain["id"]

    await validate_feature_conditions(body.modifications, owner_id, domain_id)

    # Inventory must exist, be owned, in this domain, and completed.
    _, snapshot = await get_document_async(
        COLLECTION,
        inventory_id,
        owner_id=owner_id,
        domain_id=domain_id,
        document_status="completed",
    )
    inventory_data = snapshot.to_dict()

    # Reject rules that reference a column this inventory doesn't have (e.g.
    # `dbh > 30` on an upload or CHM inventory with no dbh). Absence is loud now
    # that the uploader no longer pads missing columns with nulls.
    require_inventory_columns(
        inventory_column_keys(inventory_data),
        modification_referenced_columns(body.modifications),
        detail="A modification references column(s) this inventory doesn't have.",
    )

    new_modifications = stringify_modification_coordinates(
        [m.model_dump() for m in body.modifications]
    )

    # Append the new rules to the cumulative ledger, and queue only this delta
    # for standgen to apply to the current data (pending_modifications). The
    # checksum is re-assigned here so derivatives become detectably stale (#304).
    existing = inventory_data.get("modifications", [])
    inventory_data["modifications"] = existing + new_modifications
    inventory_data["pending_modifications"] = new_modifications
    inventory_data["checksum"] = uuid.uuid4().hex
    inventory_data["status"] = JobStatus.pending.value
    inventory_data["progress"] = None
    inventory_data["error"] = None
    inventory_data["modified_on"] = datetime.now()

    await set_document_async(COLLECTION, inventory_id, inventory_data)
    await create_http_task_async(STANDGEN_QUEUE, STANDGEN_SERVICE, inventory_id)

    # pending_modifications is an internal work-queue field, not part of the
    # Inventory schema; Pydantic ignores it (and owner_id) on construction.
    return Inventory(**inventory_data)
