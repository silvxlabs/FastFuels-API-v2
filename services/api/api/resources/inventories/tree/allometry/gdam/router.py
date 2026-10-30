"""
api/v2/resources/inventories/tree/allometry/gdam/router.py

Router for GDAM allometry imputation inventory creation.
"""

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
from api.resources.inventories.tree.allometry.gdam.examples import (
    CREATE_GDAM_OPENAPI_EXAMPLES,
)
from api.resources.inventories.tree.allometry.gdam.schema import (
    GDAM_REQUIRED_SOURCE_COLUMNS,
    CreateGdamInventoryRequest,
    GdamInventorySource,
    crown_base_height_missing_inputs,
    gdam_conditioned_columns,
    resolve_gdam_columns,
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
    summary="Create an inventory by filling in another via GDAM",
    responses=QUOTA_429_RESPONSE,
)
async def create_gdam_inventory(
    request: Request,
    response: Response,
    background_tasks: BackgroundTasks,
    domain: VerifiedDomain,
    body: Annotated[
        CreateGdamInventoryRequest,
        Body(openapi_examples=CREATE_GDAM_OPENAPI_EXAMPLES),
    ],
):
    """
    # Create GDAM Allometry Inventory

    **GDAM (Generalized Dendro Allometric Model)** is a machine-learning model
    that predicts tree morphology — diameter at breast height, live crown ratio,
    species and uncompacted crown ratio — from simple stem metrics (position and
    height). It replaces legacy region-specific allometric equations with a
    single national generative model: each predicted value is one draw from the
    predicted distribution given the values you already supply. Every tree is treated as visible from above, as trees
    detected on a canopy height model are.

    This endpoint creates a new tree inventory by calling the GDAM API to fill in
    the morphology columns (diameter, crown ratio, species, crown base height)
    of an existing tree inventory. With the default ``gdam``
    ``crown_base_height_model`` GDAM predicts the crown base too; any other
    model computes it from the final species, diameter, crown ratio and height.
    GDAM covers the conterminous US; an inventory with a tree outside it fails.

    The typical input is an uploaded **position + height** inventory (`x`, `y`,
    `height`). GDAM conditions on every dbh, crown ratio and species column the
    source has, and on its crown base height with the ``gdam`` model, except
    those being overwritten.

    ## Request Body

    - **source_tree_inventory_id**: (required) ID of a completed tree inventory to
      fill in.
    - **impute_columns**: (optional) Which morphology columns to impute. Defaults
      to all of ``dbh``, ``crown_ratio``, ``fia_species_code``,
      ``crown_base_height``. Narrow it (e.g. ``["fia_species_code"]``) to impute
      fewer columns and write less to disk; columns left out are not imputed.
      With a ``crown_base_height_model`` other than ``gdam``, requesting
      ``crown_base_height`` requires each of ``dbh``, ``crown_ratio`` and
      ``fia_species_code`` to be imputed too or to be a column of the source
      inventory. Must be non-empty with no duplicates.
    - **overwrite**: (optional, default ``false``) ``false`` keeps existing
      values and fills only empty cells. ``true`` replaces every tree's value in
      ``impute_columns``, and GDAM does not see the old values. With a model
      other than ``gdam``, overwriting ``crown_ratio`` without
      ``crown_base_height`` leaves crown bases that no longer match the new
      crown ratios.
    - **crown_base_height_model**: (optional) The model that gives
      ``crown_base_height``, selected by ``name``. Defaults to
      ``{"name": "gdam"}``: GDAM predicts it with the other columns. The other
      models compute it from species, dbh, crown ratio and height.
    - **name**: (optional) Name for the inventory.
    - **description**: (optional) Description.
    - **tags**: (optional) Tags for organizing inventories.

    ## Columns

    **Required on the source inventory** (the typical uploaded position+height set):

    - **x**, **y**: tree position, in the domain CRS.
    - **height**: tree height above zero, in metres (``m``).

    **Imputable** (select via ``impute_columns``):

    - **dbh**: diameter at breast height, in centimetres (``cm``). Predicted by
      GDAM.
    - **crown_ratio**: FIA compacted live crown ratio (0–1 fraction), as
      recorded by FIA field crews. Predicted by GDAM.
    - **fia_species_code**: FIA species code. Predicted by GDAM.
    - **crown_base_height**: height of the lowest live foliage, in metres
      (``m``): the base of a continuous, gap-free crown, equal to ``height`` for
      a tree with no live crown. Predicted by GDAM with the ``gdam`` model.
      Any other ``crown_base_height_model`` computes it from each tree's final
      species, crown ratio, diameter and height; species the model does not
      cover get no value, and with only ``crown_base_height`` requested no GDAM
      call is made.

    ## Source

    The created inventory's ``source`` records ``impute_columns``,
    ``overwrite``, ``crown_base_height_model`` and ``conditioned_columns`` (the
    source columns GDAM conditioned on, including ``crown_base_height`` when
    the ``gdam`` model was sent it).

    ## Response

    Returns the created Inventory resource with status ``"pending"``. The backend
    (Standgen) calls GDAM asynchronously and updates status to ``"completed"`` when
    ready.
    """
    owner_id = request.state.id
    domain_id = domain["id"]

    await enforce_create_quotas(COLLECTION, request)

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

    # GDAM predicts from position and height, so every tree needs both.
    source_columns = source_data.get("columns", [])
    missing = sorted(
        incomplete_inventory_columns(source_data, set(GDAM_REQUIRED_SOURCE_COLUMNS))
    )
    if missing:
        column_keys = [c["key"] if isinstance(c, dict) else c for c in source_columns]
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=(
                f"Source inventory is missing, or has empty values in, column(s) "
                f"{missing}. GDAM allometry imputation needs a value for every "
                f"tree in {list(GDAM_REQUIRED_SOURCE_COLUMNS)}. "
                f"Available columns: {column_keys}."
            ),
        )

    # A crown base height equation reads species, crown ratio and diameter.
    source_keys = {c["key"] if isinstance(c, dict) else c for c in source_columns}
    model_name = body.crown_base_height_model.name
    missing_inputs = crown_base_height_missing_inputs(
        source_keys, body.impute_columns, model_name
    )
    if missing_inputs:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=(
                f"crown_base_height_model {model_name} computes "
                f"crown_base_height from fia_species_code, crown_ratio, dbh and "
                f"height, but the source inventory has no column(s) "
                f"{missing_inputs}. Add {missing_inputs} to impute_columns, or "
                f"use crown_base_height_model gdam."
            ),
        )

    inventory_id = uuid.uuid4().hex
    request_time = datetime.now(UTC)

    source = GdamInventorySource(
        source_tree_inventory_id=body.source_tree_inventory_id,
        source_tree_inventory_checksum=source_data.get("checksum"),
        impute_columns=body.impute_columns,
        overwrite=body.overwrite,
        crown_base_height_model=body.crown_base_height_model,
        conditioned_columns=gdam_conditioned_columns(
            source_keys, body.impute_columns, body.overwrite, model_name
        ),
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
        "columns": resolve_gdam_columns(source_columns, body.impute_columns),
        "georeference": None,  # Will be set by standgen
        "error": None,
        "tags": body.tags,
        "owner_id": owner_id,
    }

    await set_document_async(COLLECTION, inventory_id, inventory_data)

    # Enqueue task to Standgen for processing
    await create_http_task_async(STANDGEN_QUEUE, STANDGEN_SERVICE, inventory_id)
    register_dispatch(request, response, background_tasks)

    return Inventory(**inventory_data)
