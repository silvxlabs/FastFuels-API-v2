"""
api/v2/resources/grids/disturbance/annual/router.py

Router for the annual LANDFIRE Limited Annual Disturbance grid product.
"""

import asyncio
import uuid
from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Body, Request, Response, status

from api.db.documents import set_document_async
from api.dependencies import VerifiedDomain
from api.quota import QUOTA_429_RESPONSE, enforce_create_quotas, register_dispatch
from api.resources.grids.disturbance.annual.examples import (
    CREATE_LANDFIRE_DISTURBANCE_OPENAPI_EXAMPLES,
)
from api.resources.grids.disturbance.annual.schema import (
    DISTURBANCE_BAND,
    CreateLandfireDisturbanceRequest,
    LandfireDisturbanceSource,
)
from api.resources.grids.schema import CHUNK_SHAPE, Grid
from api.resources.grids.utils import (
    dump_modifications_for_firestore,
    validate_feature_modifications,
    validate_lfps_coverage,
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
    summary="Create a grid from LANDFIRE Limited Annual Disturbance",
    responses=QUOTA_429_RESPONSE,
)
async def create_landfire_disturbance(
    request: Request,
    response: Response,
    background_tasks: BackgroundTasks,
    domain: VerifiedDomain,
    body: Annotated[
        CreateLandfireDisturbanceRequest,
        Body(openapi_examples=CREATE_LANDFIRE_DISTURBANCE_OPENAPI_EXAMPLES),
    ],
):
    """
    # Create LANDFIRE Limited Annual Disturbance Grid

    Creates a grid with LANDFIRE Limited Annual Disturbance (LDist) codes,
    always fetched on demand from LANDFIRE Product Service at 30m resolution.

    The grid contains a single categorical band (`annual_disturbance`) with
    raw LDist codes.

    LDist is a single annual release with a "first look" at the disturbance and
    treatment events (fire, mechanical treatment, insects/disease), from that
    version's fiscal year, released the following January/February. See
    https://landfire.gov/disturbance/annualdisturbance for details.

    LANDFIRE's current-year fuels layers (FBFM40, FBFM13, FCCS) roll out
    region by region, with disturbance already incorporated wherever they've
    landed. LDist covers all of CONUS, so in regions where the rollout
    hasn't reached yet, you can pair LDist with the last complete  national
    fuels release to incorporate more recent disturbances. For example,
    if your region is still on LF2024 fuels because LF2025 hasn't reached it yet,
    pair LF2024 with LDist25 for a better estimate of current conditons.

    ## Request Body

    - **name**: (optional) Name for the grid.
    - **description**: (optional) Description.
    - **tags**: (optional) Tags for organizing grids.
    - **version**: (optional) LANDFIRE version. Default: "2025".

    ## Response

    Returns the created Grid resource with status "pending". The backend will
    fetch the data and update status to "completed" when ready.
    """
    owner_id = request.state.id
    domain_id = domain["id"]

    await enforce_create_quotas(COLLECTION, request)

    await validate_target_grid_alignment(body.alignment, owner_id, domain_id)
    await validate_feature_modifications(body.modifications, owner_id, domain_id)

    await asyncio.to_thread(
        validate_lfps_coverage,
        "annual_disturbance",
        body.version,
        domain,
    )

    grid_id = uuid.uuid4().hex
    request_time = datetime.now()
    source = LandfireDisturbanceSource(
        version=body.version,
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
        "bands": [DISTURBANCE_BAND.model_dump()],
        "georeference": None,
        "tags": body.tags,
        "chunks": {"shape": CHUNK_SHAPE, "count": None, "count_by_axis": None},
        "owner_id": owner_id,
    }

    await set_document_async(COLLECTION, grid_id, grid_data)

    # Enqueue task to Griddle for processing
    await create_http_task_async(GRIDDLE_QUEUE, GRIDDLE_SERVICE, grid_id)
    register_dispatch(request, response, background_tasks)

    return Grid(**grid_data)
