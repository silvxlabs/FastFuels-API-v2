"""
api/v2/resources/grids/landfire/router.py

Router for LANDFIRE grid source endpoints.
"""

import uuid
from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Body, Request, status

from api.db.documents import set_document_async
from api.dependencies import VerifiedDomain
from api.resources.grids.landfire.examples import (
    CREATE_LANDFIRE_FBFM40_OPENAPI_EXAMPLES,
    CREATE_LANDFIRE_TOPOGRAPHY_OPENAPI_EXAMPLES,
)
from api.resources.grids.landfire.schema import (
    FBFM40_BAND,
    CreateLandfireFbfm40Request,
    CreateLandfireTopographyRequest,
    LandfireFbfm40Source,
    LandfireTopographySource,
    build_topography_bands,
)
from api.resources.grids.schema import Grid
from api.schema import JobStatus
from api.tasks import create_http_task_async
from lib.config import GRIDDLE_QUEUE, GRIDDLE_SERVICE, GRIDS_COLLECTION

router = APIRouter()

COLLECTION = GRIDS_COLLECTION


@router.post(
    "/fbfm40",
    response_model=Grid,
    status_code=status.HTTP_201_CREATED,
    summary="Create a grid from LANDFIRE FBFM40",
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
    - **version**: (optional) LANDFIRE version. Default: "2022".

    ## Response

    Returns the created Grid resource with status "pending". The backend will
    fetch the data and update status to "completed" when ready.
    """
    owner_id = request.state.id
    domain_id = domain["id"]

    grid_id = uuid.uuid4().hex
    request_time = datetime.now()
    source = LandfireFbfm40Source(version=body.version)

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
        "bands": [FBFM40_BAND.model_dump()],
        "georeference": None,
        "tags": body.tags,
        "owner_id": owner_id,
    }

    await set_document_async(COLLECTION, grid_id, grid_data)

    # Enqueue task to Griddle for processing
    await create_http_task_async(GRIDDLE_QUEUE, GRIDDLE_SERVICE, grid_id)

    return Grid(**grid_data)


@router.post(
    "/topography",
    response_model=Grid,
    status_code=status.HTTP_201_CREATED,
    summary="Create a grid from LANDFIRE topographic data",
)
async def create_landfire_topography(
    request: Request,
    domain: VerifiedDomain,
    body: Annotated[
        CreateLandfireTopographyRequest,
        Body(openapi_examples=CREATE_LANDFIRE_TOPOGRAPHY_OPENAPI_EXAMPLES),
    ],
):
    """
    # Create LANDFIRE Topography Grid

    Creates a grid with topographic data from LANDFIRE at 30m resolution.

    Available bands:
    - **elevation**: meters above sea level
    - **slope**: degrees (0-90)
    - **aspect**: degrees clockwise from north (0-360)

    By default all three bands are included. Use the `bands` field to select
    a subset.

    ## Request Body

    - **bands**: (optional) Which bands to include. Default: all three.
    - **name**: (optional) Name for the grid.
    - **description**: (optional) Description.
    - **tags**: (optional) Tags for organizing grids.
    - **version**: (optional) LANDFIRE version. Default: "2020".

    ## Response

    Returns the created Grid resource with status "pending". The backend will
    fetch the data and update status to "completed" when ready.
    """
    owner_id = request.state.id
    domain_id = domain["id"]

    grid_id = uuid.uuid4().hex
    request_time = datetime.now()
    source = LandfireTopographySource(version=body.version, bands=body.bands)
    bands = build_topography_bands(body.bands)

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
        "owner_id": owner_id,
    }

    await set_document_async(COLLECTION, grid_id, grid_data)

    # Enqueue task to Griddle for processing
    await create_http_task_async(GRIDDLE_QUEUE, GRIDDLE_SERVICE, grid_id)

    return Grid(**grid_data)
