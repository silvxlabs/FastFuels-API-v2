"""
api/v2/resources/inventories/tree/allometry/crown_base_height/router.py

Router for crown base height inventory creation.
"""

import json
import uuid
from datetime import UTC, datetime
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

from api.db.documents import get_document_async, set_document_async
from api.dependencies import VerifiedDomain
from api.quota import QUOTA_429_RESPONSE, enforce_create_quotas, register_dispatch
from api.resources.grids.utils import incomplete_inventory_columns
from api.resources.inventories.schema import Inventory
from api.resources.inventories.tree.allometry.crown_base_height.examples import (
    CREATE_CROWN_BASE_HEIGHT_OPENAPI_EXAMPLES,
)
from api.resources.inventories.tree.allometry.crown_base_height.schema import (
    CROWN_BASE_HEIGHT_GDAM_INPUTS,
    CROWN_BASE_HEIGHT_REQUIRED_SOURCE_COLUMNS,
    CreateCrownBaseHeightAllometryRequest,
    CrownBaseHeightAllometrySource,
    resolve_crown_base_height_columns,
)
from api.resources.inventories.tree.allometry.schema import (
    GDAM_CROWN_BASE_HEIGHT_MODEL,
)
from api.schema import JobStatus
from api.tasks import create_http_task_async
from lib.config import (
    INVENTORIES_COLLECTION,
    STANDGEN_QUEUE,
    STANDGEN_SERVICE,
)

router = APIRouter()

COLLECTION = INVENTORIES_COLLECTION


@router.post(
    "",
    response_model=Inventory,
    status_code=status.HTTP_201_CREATED,
    summary="Create an inventory by computing crown base heights",
    responses=QUOTA_429_RESPONSE,
)
async def create_crown_base_height_inventory(
    request: Request,
    response: Response,
    background_tasks: BackgroundTasks,
    domain: VerifiedDomain,
    body: Annotated[
        CreateCrownBaseHeightAllometryRequest,
        Body(openapi_examples=CREATE_CROWN_BASE_HEIGHT_OPENAPI_EXAMPLES),
    ],
):
    """
    # Create Crown Base Height Inventory

    Creates a new tree inventory with each tree's crown base height computed
    from its species, dbh, crown ratio and height by a crown base height
    equation, or predicted by GDAM with the ``gdam`` model. Use it for a tree
    inventory without a crown base height, such as an uploaded inventory,
    before voxelizing it or building canopy grids from it.

    With an equation, an inventory without dbh, crown ratio or species needs
    the GDAM route (``POST /domains/{domain_id}/inventories/tree/allometry/gdam``)
    instead, which fills those and the crown base height in one request. The
    ``gdam`` model needs only position and height, conditions on whichever of
    species, dbh, crown ratio and crown base height are present, and writes
    only the crown base height. GDAM covers the conterminous US; an inventory
    with a tree outside it fails.

    ## Request Body

    - **source_tree_inventory_id**: (required) ID of a completed tree inventory
      with the required columns below.
    - **model**: (optional) The crown base height model, selected by ``name``.
      Defaults to ``{"name": "fia_national_logit_2026"}``. Species an equation
      does not cover get no crown base height. ``{"name": "gdam"}`` predicts it
      with GDAM.
    - **overwrite**: (optional, default ``false``) ``false`` keeps existing
      crown base heights and fills only empty cells; a source where every tree
      already has one is rejected. ``true`` recomputes every tree's crown base
      height, and ``gdam`` is not given the old values.
    - **name**: (optional) Name for the inventory.
    - **description**: (optional) Description.
    - **tags**: (optional) Tags for organizing inventories.

    ## Columns

    **Required on the source inventory**, with a value for every tree:

    - **x**, **y**: tree position, in the domain CRS.
    - **height**: tree height, in metres (``m``); above zero for ``gdam``.
    - **dbh**: diameter at breast height, in centimetres (``cm``). Not
      required for ``gdam``.
    - **crown_ratio**: FIA compacted live crown ratio (0–1 fraction). Not
      required for ``gdam``.
    - **fia_species_code**: FIA species code. Not required for ``gdam``.

    **Written**:

    - **crown_base_height**: height of the lowest live foliage, in metres
      (``m``): the base of a continuous, gap-free crown, equal to ``height``
      for a tree with no live crown.

    Every other source column is copied unchanged.

    ## Source

    The created inventory's ``source`` records ``model`` and ``overwrite``.

    ## Response

    Returns the created Inventory resource with status ``"pending"``. The
    backend (Standgen) computes the crown base heights asynchronously and
    updates status to ``"completed"`` when ready.
    """
    owner_id = request.state.id
    domain_id = domain["id"]

    await enforce_create_quotas(COLLECTION, request)

    # owner/domain/status filters return 404 on mismatch (no existence leak).
    _, source_snapshot = await get_document_async(
        COLLECTION,
        body.source_tree_inventory_id,
        owner_id=owner_id,
        domain_id=domain_id,
        document_status="completed",
    )
    source_data = source_snapshot.to_dict()

    if source_data.get("type") != "tree":
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=(
                f"Source inventory '{body.source_tree_inventory_id}' is not a tree "
                f"inventory. Crown base heights can only be computed for a tree "
                f"inventory."
            ),
        )

    source_columns = source_data.get("columns", [])
    column_keys = [c["key"] if isinstance(c, dict) else c for c in source_columns]
    missing = sorted(
        incomplete_inventory_columns(
            source_data, set(CROWN_BASE_HEIGHT_REQUIRED_SOURCE_COLUMNS)
        )
    )
    if missing:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=(
                f"Source inventory is missing, or has empty values in, column(s) "
                f"{missing}. Every tree needs a value in "
                f"{list(CROWN_BASE_HEIGHT_REQUIRED_SOURCE_COLUMNS)}, and these "
                f"cannot be imputed. Available columns: {column_keys}."
            ),
        )

    # GDAM's crown base needs no equation inputs.
    incomplete_inputs = set()
    if body.model.name != GDAM_CROWN_BASE_HEIGHT_MODEL:
        incomplete_inputs = incomplete_inventory_columns(
            source_data, set(CROWN_BASE_HEIGHT_GDAM_INPUTS)
        )
    if incomplete_inputs:
        impute_columns = [
            c for c in CROWN_BASE_HEIGHT_GDAM_INPUTS if c in incomplete_inputs
        ]
        gdam_body = {
            "source_tree_inventory_id": body.source_tree_inventory_id,
            "impute_columns": [*impute_columns, "crown_base_height"],
            "crown_base_height_model": body.model.model_dump(),
        }
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=(
                f"Source inventory is missing, or has empty values in, column(s) "
                f"{impute_columns}, which crown base height model "
                f"{body.model.name} needs for every tree. Send POST "
                f"/domains/{domain_id}/inventories/tree/allometry/gdam with body "
                f"{json.dumps(gdam_body)}; it fills those columns and the crown "
                f"base height in one request."
            ),
        )

    if not body.overwrite:
        crown_base = next(
            (
                c
                for c in source_columns
                if isinstance(c, dict) and c.get("key") == "crown_base_height"
            ),
            None,
        )
        summary = (crown_base or {}).get("summary") or {}
        if summary.get("null_count") == 0:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=(
                    "Every tree already has a crown base height; send "
                    "overwrite: true to recompute it with this model."
                ),
            )

    inventory_id = uuid.uuid4().hex
    request_time = datetime.now(UTC)

    source = CrownBaseHeightAllometrySource(
        source_tree_inventory_id=body.source_tree_inventory_id,
        source_tree_inventory_checksum=source_data.get("checksum"),
        model=body.model,
        overwrite=body.overwrite,
    )

    inventory_data = {
        "id": inventory_id,
        "checksum": uuid.uuid4().hex,
        "domain_id": domain_id,
        "type": body.type.value,
        "name": body.name,
        "description": body.description,
        "status": JobStatus.pending.value,
        "progress": None,
        "created_on": request_time,
        "modified_on": request_time,
        "source": source.model_dump(),
        "modifications": [],
        "columns": resolve_crown_base_height_columns(source_columns),
        "georeference": None,  # Will be set by standgen
        "error": None,
        "tags": body.tags,
        "owner_id": owner_id,
    }

    await set_document_async(COLLECTION, inventory_id, inventory_data)

    await create_http_task_async(STANDGEN_QUEUE, STANDGEN_SERVICE, inventory_id)
    register_dispatch(request, response, background_tasks)

    return Inventory(**inventory_data)
