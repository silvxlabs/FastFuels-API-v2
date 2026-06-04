"""
api/v2/resources/inventories/modifications/router.py

Router for standalone inventory modifications endpoint.

POST /domains/{domain_id}/inventories/{inventory_id}/modifications
Creates a new inventory by applying modifications to an existing one.
"""

import uuid
from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Body, Request, status

from api.db.documents import get_document_async, set_document_async
from api.dependencies import VerifiedDomain
from api.resources.inventories.modifications.examples import (
    APPLY_MODIFICATIONS_OPENAPI_EXAMPLES,
)
from api.resources.inventories.modifications.schema import (
    ApplyModificationsRequest,
    ModificationsInventorySource,
)
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
    status_code=status.HTTP_201_CREATED,
    summary="Create a modified inventory",
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
    # Apply Modifications to an Inventory

    Creates a **new** inventory by applying modifications to an existing
    completed inventory. The source inventory is not changed.

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
      same domain as the source inventory; cross-domain references are
      rejected.

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

    Returns the new Inventory resource with status `"pending"`. The original
    inventory is unchanged.
    """
    owner_id = request.state.id
    domain_id = domain["id"]

    await validate_feature_conditions(body.modifications, owner_id, domain_id)

    # Validate source inventory exists, is owned, belongs to domain, is completed
    _, source_snapshot = await get_document_async(
        COLLECTION,
        inventory_id,
        owner_id=owner_id,
        domain_id=domain_id,
        document_status="completed",
    )
    source_data = source_snapshot.to_dict()

    new_inventory_id = uuid.uuid4().hex
    request_time = datetime.now()

    source = ModificationsInventorySource(
        source_inventory_id=inventory_id,
        modifications=stringify_modification_coordinates(
            [m.model_dump() for m in body.modifications]
        ),
    )

    inventory_data = {
        "id": new_inventory_id,
        "domain_id": domain_id,
        "type": source_data["type"],
        "name": body.name,
        "description": body.description,
        "status": JobStatus.pending.value,
        "progress": None,
        "created_on": request_time,
        "modified_on": request_time,
        "source": source.model_dump(),
        "modifications": [],
        "columns": source_data.get("columns", []),
        "georeference": None,
        "error": None,
        "tags": body.tags,
        "owner_id": owner_id,
    }

    await set_document_async(COLLECTION, new_inventory_id, inventory_data)
    await create_http_task_async(STANDGEN_QUEUE, STANDGEN_SERVICE, new_inventory_id)

    return Inventory(**inventory_data)
