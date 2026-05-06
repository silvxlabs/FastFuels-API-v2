"""
api/v2/resources/grids/chm/router.py

Router for CHM grid product endpoints.
"""

import uuid
from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Body, Request, status

from api.db.documents import set_document_async
from api.dependencies import VerifiedDomain
from api.resources.grids.chm.examples import (
    CREATE_META_CHM_OPENAPI_EXAMPLES,
    CREATE_NAIP_CHM_OPENAPI_EXAMPLES,
)
from api.resources.grids.chm.schema import (
    CreateMetaChmRequest,
    CreateNaipChmRequest,
    MetaChmSource,
    NaipChmSource,
    build_chm_bands,
)
from api.resources.grids.schema import CHUNK_SHAPE, Grid
from api.schema import JobStatus
from api.tasks import create_http_task_async
from lib.config import GRIDDLE_QUEUE, GRIDDLE_SERVICE, GRIDS_COLLECTION

router = APIRouter()

COLLECTION = GRIDS_COLLECTION


@router.post(
    "/meta",
    response_model=Grid,
    status_code=status.HTTP_201_CREATED,
    summary="Create a grid from Meta CHM",
)
async def create_meta_chm(
    request: Request,
    domain: VerifiedDomain,
    body: Annotated[
        CreateMetaChmRequest,
        Body(openapi_examples=CREATE_META_CHM_OPENAPI_EXAMPLES),
    ],
):
    """
    # Create Meta CHM Grid

    Creates a grid with canopy height data from Meta's global canopy height
    model at ~1m resolution.

    ## Request Body

    - **name**: (optional) Name for the grid.
    - **description**: (optional) Description.
    - **tags**: (optional) Tags for organizing grids.
    - **version**: (optional) Meta CHM version. Default: "2".
        - **1**: Tolan, J. et al. (2024). Very high resolution canopy height maps from RGB imagery.
        - **2**: Brandt, J. et al. (2026). CHMv2: Improvements in Global Canopy Height Mapping using DINOv3.

    ## Response

    Returns the created Grid resource with status "pending". The backend will
    fetch the data and update status to "completed" when ready.
    """
    owner_id = request.state.id
    domain_id = domain["id"]

    grid_id = uuid.uuid4().hex
    request_time = datetime.now()
    source = MetaChmSource(
        version=body.version,
        extent_buffer_cells=body.resolved_extent_buffer_cells(0),
    )
    bands = build_chm_bands()

    grid_data = {
        "id": grid_id,
        "domain_id": domain_id,
        "name": body.name,
        "description": body.description,
        "status": JobStatus.pending.value,
        "created_on": request_time,
        "modified_on": request_time,
        "source": source.model_dump(),
        "modifications": [m.model_dump() for m in body.modifications],
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


@router.post(
    "/naip",
    response_model=Grid,
    status_code=status.HTTP_201_CREATED,
    summary="Create a grid from NAIP CHM",
)
async def create_naip_chm(
    request: Request,
    domain: VerifiedDomain,
    body: Annotated[
        CreateNaipChmRequest,
        Body(openapi_examples=CREATE_NAIP_CHM_OPENAPI_EXAMPLES),
    ],
):
    """
    # Create NAIP CHM Grid

    Creates a grid with canopy height data from the NAIP high-resolution
    canopy height model at ~0.6m resolution (CONUS).

    ## Request Body
    - **name**: (optional) Name for the grid.
    - **description**: (optional) Description.
    - **tags**: (optional) Tags for organizing grids.

    ## Response

    Returns the created Grid resource with status "pending". The backend will
    fetch the data and update status to "completed" when ready.
    """
    owner_id = request.state.id
    domain_id = domain["id"]

    grid_id = uuid.uuid4().hex
    request_time = datetime.now()
    source = NaipChmSource(extent_buffer_cells=body.resolved_extent_buffer_cells(0))
    bands = build_chm_bands()

    grid_data = {
        "id": grid_id,
        "domain_id": domain_id,
        "name": body.name,
        "description": body.description,
        "status": JobStatus.pending.value,
        "created_on": request_time,
        "modified_on": request_time,
        "source": source.model_dump(),
        "modifications": [m.model_dump() for m in body.modifications],
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
