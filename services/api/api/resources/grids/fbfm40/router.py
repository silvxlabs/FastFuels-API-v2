"""
api/v2/resources/grids/fbfm40/router.py

Router for FBFM40 grid product endpoints.
"""

import asyncio
import uuid
from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Body, Request, Response, status

from api.db.documents import set_document_async
from api.dependencies import VerifiedDomain
from api.quota import QUOTA_429_RESPONSE, enforce_create_quotas, register_dispatch
from api.resources.grids.fbfm40.examples import (
    CREATE_LANDFIRE_FBFM40_OPENAPI_EXAMPLES,
)
from api.resources.grids.fbfm40.schema import (
    FBFM40_BAND,
    CreateLandfireFbfm40Request,
    LandfireFbfm40Source,
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
from lib.landfire import (
    LANDFIRE_VERSIONS,
    SEASON_CODES,
    list_releases,
    resolve_lf_product,
)

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
    response: Response,
    background_tasks: BackgroundTasks,
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
    - **season**: (optional) LANDFIRE Seasonal Fuels release: "ES" (early
      spring), "SP" (spring), "SU" (summer), or "FA" (fall).

    ## Response

    Returns the created Grid resource with status "pending". The backend will
    fetch the data and update status to "completed" when ready.

    The response `source` reports `year`: the calendar year the fuel data
    represents. For an annual grid this is the landscape vintage (same as
    `version`); for a seasonal grid it is the projected season year (e.g.
    `version` 2025 + `season` "SP" is spring 2026).
    """
    owner_id = request.state.id
    domain_id = domain["id"]

    await enforce_create_quotas(COLLECTION, request)

    await validate_target_grid_alignment(body.alignment, owner_id, domain_id)
    await validate_feature_modifications(body.modifications, owner_id, domain_id)
    use_lfps = (
        body.season is not None
        or body.version in LANDFIRE_VERSIONS["fbfm40"]["lfps_available"]
    )
    if use_lfps:
        await asyncio.to_thread(
            validate_lfps_coverage,
            "fbfm40",
            body.version,
            domain,
            season=body.season,
        )
    if body.season is not None:
        # Read the represented year off the live LFPS catalog entry rather
        # than assuming it is `version + 1`. Coverage validation above already
        # confirmed the product is live, so the match is present (cached call).
        matched = await asyncio.to_thread(
            resolve_lf_product, "fbfm40", body.version, body.season
        )
        year = matched.season_year if matched else None
    else:
        # Annual FBFM40's version IS the landscape vintage year.
        year = int(body.version)

    grid_id = uuid.uuid4().hex
    request_time = datetime.now()
    source = LandfireFbfm40Source(
        version=body.version,
        remove_non_burnable=body.remove_non_burnable,
        extent_buffer_cells=body.extent_buffer_cells,
        alignment=body.alignment,
        season=body.season,
        year=year,
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
    register_dispatch(request, response, background_tasks)

    return Grid(**grid_data)


@router.get(
    "/landfire/coverage",
    response_model=LandfireCoverageResponse,
    summary="Check LANDFIRE FBFM40 release coverage for a domain",
)
async def check_landfire_fbfm40_coverage(request: Request, domain: VerifiedDomain):
    """
    # Check LANDFIRE FBFM40 Coverage

    Immediate pre-flight check reporting every FBFM40 release the API serves
    and how much of this domain each one covers. Staged annual releases are
    national; the current-year release and the Seasonal Fuels windows are
    served by LANDFIRE Product Service region by region, so coverage depends
    on where the domain is and on what LANDFIRE has published so far.

    ## Response

    `latest` is the release representing the most recent point in time that
    fully covers the domain. `releases` lists every release, newest first.
    Each release that covers the domain carries a `links.create` request:
    send its `body` to its `href`, a path relative to this API's base URL,
    to create the grid.
    """
    geometry = parse_domain_gdf(domain).to_crs(epsg=5070).geometry.union_all()
    releases = await asyncio.to_thread(list_releases, "fbfm40", geometry, SEASON_CODES)
    create_href = str(
        request.app.url_path_for("create_landfire_fbfm40", domain_id=domain["id"])
    )
    return build_landfire_coverage_response("fbfm40", releases, create_href)
