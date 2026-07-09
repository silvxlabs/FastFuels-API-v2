"""
api/v2/resources/grids/fbfm40/router.py

Router for FBFM40 grid product endpoints.
"""

import uuid
from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Body, Request, status

from api.db.documents import set_document_async
from api.dependencies import VerifiedDomain
from api.quota import QUOTA_429_RESPONSE, enforce_create_quotas
from api.resources.grids.fbfm40.examples import (
    CREATE_LANDFIRE_FBFM40_OPENAPI_EXAMPLES,
)
from api.resources.grids.fbfm40.schema import (
    FBFM40_BAND,
    CreateLandfireFbfm40Request,
    LandfireFbfm40Source,
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
    "/landfire",
    response_model=Grid,
    status_code=status.HTTP_201_CREATED,
    summary="Create a grid from LANDFIRE FBFM40",
    responses=QUOTA_429_RESPONSE,
)
async def create_landfire_fbfm40(
    request: Request,
    domain: VerifiedDomain,
    body: Annotated[
        CreateLandfireFbfm40Request,
        Body(openapi_examples=CREATE_LANDFIRE_FBFM40_OPENAPI_EXAMPLES),
    ],
):
    """
    # Create LANDFIRE FBFM40 Grid

    Creates a grid with FBFM40 fuel model codes from LANDFIRE at 30m resolution.

    The grid contains a single categorical band (`fbfm`) with Scott-Burgan 40
    fuel model codes (e.g., GR1, TL3, SH5).

    To convert fuel model codes to fuel parameters (fuel loads, SAV, depth),
    use the `/grids/lookup/fbfm40` endpoint.

    ## Request Body

    - **name**: (optional) Name for the grid.
    - **description**: (optional) Description.
    - **tags**: (optional) Tags for organizing grids.
    - **version**: (optional) LANDFIRE version. Default: "2024".

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
    source = LandfireFbfm40Source(
        version=body.version,
        remove_non_burnable=body.remove_non_burnable,
        extent_buffer_cells=body.extent_buffer_cells,
        alignment=body.alignment,
    )

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
        "bands": [FBFM40_BAND.model_dump()],
        "georeference": None,
        "tags": body.tags,
        "chunks": {"shape": CHUNK_SHAPE, "count": None, "count_by_axis": None},
        "owner_id": owner_id,
    }

    await set_document_async(COLLECTION, grid_id, grid_data)

    # Enqueue task to Griddle for processing
    await create_http_task_async(GRIDDLE_QUEUE, GRIDDLE_SERVICE, grid_id)

    return Grid(**grid_data)
