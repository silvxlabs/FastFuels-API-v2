"""
api/v2/resources/grids/fccs/router.py

Router for FCCS grid product endpoints.
"""

import asyncio
import uuid
from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Body, Request, Response, status

from api.db.documents import set_document_async
from api.dependencies import VerifiedDomain
from api.quota import QUOTA_429_RESPONSE, enforce_create_quotas, register_dispatch
from api.resources.grids.fccs.examples import (
    CREATE_LANDFIRE_FCCS_OPENAPI_EXAMPLES,
)
from api.resources.grids.fccs.schema import (
    FCCS_BAND,
    CreateLandfireFccsRequest,
    LandfireFccsSource,
)
from api.resources.grids.providers.landfire import (
    LandfireCoverageResponse,
    build_landfire_coverage_response,
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
from lib.domain_utils import parse_domain_gdf
from lib.landfire import LANDFIRE_VERSIONS, list_releases

router = APIRouter()

COLLECTION = GRIDS_COLLECTION


@router.post(
    "/landfire",
    response_model=Grid,
    status_code=status.HTTP_201_CREATED,
    summary="Create a grid from LANDFIRE FCCS",
    responses=QUOTA_429_RESPONSE,
)
async def create_landfire_fccs(
    request: Request,
    response: Response,
    background_tasks: BackgroundTasks,
    domain: VerifiedDomain,
    body: Annotated[
        CreateLandfireFccsRequest,
        Body(openapi_examples=CREATE_LANDFIRE_FCCS_OPENAPI_EXAMPLES),
    ],
):
    """
    # Create LANDFIRE FCCS Grid

    Creates a grid with FCCS fuelbed IDs from LANDFIRE at 30m resolution.

    The grid contains a single categorical band (`fccs`) with fuel
    classification system fuelbed IDs (e.g., 26, 598, 34721).

    To convert fuelbed IDs to fuel parameters (fuel loads, SAV, depth),
    use the `/grids/lookup/fccs` endpoint.

    ## Request Body

    - **name**: (optional) Name for the grid.
    - **description**: (optional) Description.
    - **tags**: (optional) Tags for organizing grids.
    - **version**: (optional) LANDFIRE version. Default: "2023".
    - **remove_bare_ground**: (optional) Remove bare ground cells (fuelbed ID 0),
                              replaced by neighboring majority. Default: False.

    ## Response

    Returns the created Grid resource with status "pending". The backend will
    fetch the data and update status to "completed" when ready.
    """
    owner_id = request.state.id
    domain_id = domain["id"]

    await enforce_create_quotas(COLLECTION, request)

    await validate_target_grid_alignment(body.alignment, owner_id, domain_id)
    await validate_feature_modifications(body.modifications, owner_id, domain_id)

    if body.version in LANDFIRE_VERSIONS["fccs"]["lfps_available"]:
        await asyncio.to_thread(
            validate_lfps_coverage,
            "fccs",
            body.version,
            domain,
        )

    grid_id = uuid.uuid4().hex
    request_time = datetime.now()
    source = LandfireFccsSource(
        version=body.version,
        remove_bare_ground=body.remove_bare_ground,
        boundary_scatter=body.boundary_scatter,
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
        "bands": [FCCS_BAND.model_dump()],
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


@router.get(
    "/landfire/coverage",
    response_model=LandfireCoverageResponse,
    summary="Check LANDFIRE FCCS release coverage for a domain",
)
async def check_landfire_fccs_coverage(request: Request, domain: VerifiedDomain):
    """
    # Check LANDFIRE FCCS Coverage

    Immediate pre-flight check reporting every FCCS release the API serves
    and how much of this domain each one covers. Staged annual releases are
    national; the current-year release is served by LANDFIRE Product Service
    region by region, so its coverage depends on where the domain is.

    ## Response

    `latest` is the release representing the most recent point in time that
    fully covers the domain. `releases` lists every release, newest first.
    Each release that covers the domain carries a `links.create` request:
    send its `body` to its `href`, a path relative to this API's base URL,
    to create the grid.
    """
    geometry = parse_domain_gdf(domain).to_crs(epsg=5070).geometry.union_all()
    releases = await asyncio.to_thread(list_releases, "fccs", geometry)
    create_href = str(
        request.app.url_path_for("create_landfire_fccs", domain_id=domain["id"])
    )
    return build_landfire_coverage_response("fccs", releases, create_href)
