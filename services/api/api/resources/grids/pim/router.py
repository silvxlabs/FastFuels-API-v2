"""
api/v2/resources/grids/pim/router.py

Router for PIM grid product endpoints.
"""

import uuid
from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Body, Request, status

from api.db.documents import set_document_async
from api.dependencies import VerifiedDomain
from api.quota import QUOTA_429_RESPONSE, enforce_create_quotas
from api.resources.grids.pim.examples import CREATE_TREEMAP_OPENAPI_EXAMPLES
from api.resources.grids.pim.schema import (
    CreateTreeMapRequest,
    TreeMapSource,
    build_treemap_bands,
)
from api.resources.grids.schema import CHUNK_SHAPE, Grid
from api.resources.grids.utils import (
    dump_modifications_for_firestore,
    validate_feature_modifications,
    validate_target_grid_alignment,
)
from api.schema import JobStatus
from api.tasks import create_http_task_async
from lib.config import GRIDDLE_QUEUE, GRIDDLE_SERVICE, GRIDS_COLLECTION

router = APIRouter()

COLLECTION = GRIDS_COLLECTION


@router.post(
    "/treemap",
    response_model=Grid,
    status_code=status.HTTP_201_CREATED,
    summary="Create a grid from TreeMap",
    responses=QUOTA_429_RESPONSE,
)
async def create_treemap(
    request: Request,
    domain: VerifiedDomain,
    body: Annotated[
        CreateTreeMapRequest,
        Body(openapi_examples=CREATE_TREEMAP_OPENAPI_EXAMPLES),
    ],
):
    """
    # Create TreeMap Grid

    Creates a grid with plot imputation data from TreeMap at 30m resolution.

    Each pixel contains a plot ID that maps to FIA tree records. Available bands:
    - **tm_id**: TreeMap raster pixel values (small integers, 1-70K)
    - **plt_cn**: FIA plot condition number (large integers, derived from tree table)

    By default only `tm_id` is included. Use the `bands` field to also request
    `plt_cn`.

    ## Request Body

    - **bands**: (optional) Which bands to include. Default: ["tm_id"].
    - **name**: (optional) Name for the grid.
    - **description**: (optional) Description.
    - **tags**: (optional) Tags for organizing grids.
    - **version**: (optional) TreeMap version. Default: "2022".

    ## Response

    Returns the created Grid resource with status "pending". The backend will
    fetch the data and update status to "completed" when ready.
    """
    owner_id = request.state.id
    domain_id = domain["id"]

    await enforce_create_quotas(COLLECTION, request)

    await validate_target_grid_alignment(body.alignment, owner_id, domain_id)
    await validate_feature_modifications(body.modifications, owner_id, domain_id)

    grid_id = uuid.uuid4().hex
    request_time = datetime.now()
    source = TreeMapSource(
        version=body.version,
        bands=body.bands,
        extent_buffer_cells=body.extent_buffer_cells,
        alignment=body.alignment,
    )
    bands = build_treemap_bands(body.bands)

    grid_data = {
        "id": grid_id,
        "checksum": uuid.uuid4().hex,
        "domain_id": domain_id,
        "name": body.name,
        "description": body.description,
        "status": JobStatus.pending.value,
        "created_on": request_time,
        "modified_on": request_time,
        "source": source.model_dump(),
        "modifications": dump_modifications_for_firestore(body.modifications),
        "bands": [b.model_dump() for b in bands],
        "georeference": None,
        "tags": body.tags,
        "chunks": {"shape": CHUNK_SHAPE, "count": None, "count_by_axis": None},
        "owner_id": owner_id,
    }

    await set_document_async(COLLECTION, grid_id, grid_data)

    # Enqueue task to Griddle for processing
    await create_http_task_async(GRIDDLE_QUEUE, GRIDDLE_SERVICE, grid_id)

    return Grid(**grid_data)
