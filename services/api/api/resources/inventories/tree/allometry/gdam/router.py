"""
api/v2/resources/inventories/tree/allometry/gdam/router.py

Router for GDAM allometry imputation inventory creation.
"""

import uuid
from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Body, HTTPException, Request, status

from api.db.documents import get_document_async, set_document_async
from api.dependencies import VerifiedDomain
from api.resources.inventories.schema import BASE_INVENTORY_COLUMNS, Inventory
from api.resources.inventories.tree.allometry.gdam.examples import (
    CREATE_GDAM_OPENAPI_EXAMPLES,
)
from api.resources.inventories.tree.allometry.gdam.schema import (
    CreateGdamInventoryRequest,
    GdamInventorySource,
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

# Columns GDAM needs on the source inventory to make a prediction.
_REQUIRED_SOURCE_COLUMNS = ("x", "y", "height")


@router.post(
    "",
    response_model=Inventory,
    status_code=status.HTTP_201_CREATED,
    summary="Create an inventory by filling in another via GDAM",
)
async def create_gdam_inventory(
    request: Request,
    domain: VerifiedDomain,
    body: Annotated[
        CreateGdamInventoryRequest,
        Body(openapi_examples=CREATE_GDAM_OPENAPI_EXAMPLES),
    ],
):
    """
    # Create GDAM Allometry Inventory

    Creates a new tree inventory by calling the GDAM machine-learning API to fill
    in the missing morphology columns (diameter, crown ratio, species) of an
    existing tree inventory.

    The typical input is an uploaded **position + height** inventory (`x`, `y`,
    `height`). GDAM predicts the missing fields; any values already present are
    preserved and passed to GDAM as conditioning inputs.

    ## Request Body

    - **source_tree_inventory_id**: (required) ID of a completed tree inventory to
      fill in.
    - **impute_columns**: (optional) Which morphology columns to impute. Defaults
      to all of ``dbh``, ``crown_ratio``, ``fia_species_code``. Narrow it (e.g.
      ``["fia_species_code"]``) to impute fewer columns and write less to disk;
      columns left out are not imputed. Must be non-empty with no duplicates.
    - **name**: (optional) Name for the inventory.
    - **description**: (optional) Description.
    - **tags**: (optional) Tags for organizing inventories.

    ## Columns

    **Required on the source inventory** (the typical uploaded position+height set):

    - **x**, **y**: tree position, in the domain CRS.
    - **height**: tree height, in metres (``m``).

    **Imputable by GDAM** (select via ``impute_columns``) — filled only where
    missing; existing values are preserved:

    - **dbh**: diameter at breast height, in centimetres (``cm``).
    - **crown_ratio**: live crown ratio, as a 0–1 fraction.
    - **fia_species_code**: FIA species code.

    ## Response

    Returns the created Inventory resource with status ``"pending"``. The backend
    (Standgen) calls GDAM asynchronously and updates status to ``"completed"`` when
    ready.
    """
    owner_id = request.state.id
    domain_id = domain["id"]

    # Validate the source inventory exists, is owned, in this domain, and completed.
    # owner/domain/status filters return 404 on mismatch (no existence leak).
    _, source_snapshot = await get_document_async(
        COLLECTION,
        body.source_tree_inventory_id,
        owner_id=owner_id,
        domain_id=domain_id,
        document_status="completed",
    )
    source_data = source_snapshot.to_dict()

    # Must be a tree inventory.
    if source_data.get("type") != "tree":
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=(
                f"Source inventory '{body.source_tree_inventory_id}' is not a tree "
                f"inventory. GDAM allometry imputation requires a tree inventory."
            ),
        )

    # Must carry the columns GDAM needs (position + height).
    source_columns = source_data.get("columns", [])
    column_keys = [c["key"] if isinstance(c, dict) else c for c in source_columns]
    missing = [c for c in _REQUIRED_SOURCE_COLUMNS if c not in column_keys]
    if missing:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=(
                f"Source inventory is missing required column(s) {missing}. "
                f"GDAM allometry imputation requires {list(_REQUIRED_SOURCE_COLUMNS)}. "
                f"Available columns: {column_keys}."
            ),
        )

    inventory_id = uuid.uuid4().hex
    request_time = datetime.now(UTC)

    source = GdamInventorySource(
        source_tree_inventory_id=body.source_tree_inventory_id,
        source_tree_inventory_checksum=source_data.get("checksum"),
        impute_columns=body.impute_columns,
    )

    inventory_data = {
        "id": inventory_id,
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
        "columns": [c.model_dump() for c in BASE_INVENTORY_COLUMNS],
        "georeference": None,  # Will be set by standgen
        "error": None,
        "tags": body.tags,
        "owner_id": owner_id,
    }

    await set_document_async(COLLECTION, inventory_id, inventory_data)

    # Enqueue task to Standgen for processing
    await create_http_task_async(STANDGEN_QUEUE, STANDGEN_SERVICE, inventory_id)

    return Inventory(**inventory_data)
