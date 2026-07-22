"""
api/v2/resources/grids/lookup/fbfm13/router.py

Router for the FBFM13 lookup endpoint.
"""

import uuid
from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Body, Request, Response, status

from api.db.documents import get_document_async, set_document_async
from api.dependencies import VerifiedDomain
from api.quota import QUOTA_429_RESPONSE, enforce_create_quotas, register_dispatch
from api.resources.grids.lookup.fbfm13.examples import (
    CREATE_FBFM13_LOOKUP_OPENAPI_EXAMPLES,
)
from api.resources.grids.lookup.fbfm13.schema import (
    CreateFbfm13LookupRequest,
    Fbfm13LookupSource,
    get_fbfm13_lookup_band,
)
from api.resources.grids.schema import CHUNK_SHAPE, Grid
from api.resources.grids.utils import (
    dump_modifications_for_firestore,
    validate_feature_modifications,
    validate_grid_has_band,
)
from api.schema import JobStatus
from api.tasks import create_http_task_async
from lib.config import GRIDDLE_QUEUE, GRIDDLE_SERVICE, GRIDS_COLLECTION

router = APIRouter()

COLLECTION = GRIDS_COLLECTION


@router.post(
    "",
    response_model=Grid,
    status_code=status.HTTP_201_CREATED,
    summary="Create a grid by looking up FBFM13 fuel parameters",
    responses=QUOTA_429_RESPONSE,
)
async def create_fbfm13_lookup(
    request: Request,
    response: Response,
    background_tasks: BackgroundTasks,
    domain: VerifiedDomain,
    body: Annotated[
        CreateFbfm13LookupRequest,
        Body(openapi_examples=CREATE_FBFM13_LOOKUP_OPENAPI_EXAMPLES),
    ],
):
    """
    # Create FBFM13 Lookup Grid

    Converts Anderson 13 fuel model codes to fuel parameters using the
    Anderson 13 lookup table.

    Takes a source grid containing categorical FBFM13 codes (from
    `/grids/fbfm13/landfire`) and produces a new grid with the requested
    continuous fuel parameters.

    ## Request Body

    - **source_grid_id**: (required) Grid containing FBFM13 codes.
    - **bands**: (required) Bands to look up. Valid values:
      - `fuel_load.1hr`, `fuel_load.10hr`, `fuel_load.100hr` - Dead fuel loads (kg/m**2)
      - `fuel_load.live_foliage` - Live foliage fuel loads (kg/m**2)
      - `savr.1hr`, `savr.10hr`, `savr.100hr` - Dead fuel SAV ratios (1/m)
      - `savr.live_foliage` - Live foliage fuel SAV ratios (1/m)
      - `fuel_depth` - Fuel bed depth (m)
    - **source_band**: (optional) Band in source grid containing FBFM13 codes. Defaults to `"fbfm13"`.
    - **name**: (optional) Name for the grid.
    - **description**: (optional) Description.
    - **tags**: (optional) Tags for organizing grids.

    ## Valid FBFM13 Codes

    The source grid must contain only valid Anderson 13 fuel model codes.
    The 18 valid codes are:

    - **NB** (non-burnable): 91, 92, 93, 98, 99
    - **Anderson 13 models**: 1–13

    If any cell in the source grid contains a code not in this set (including 0
    or nodata), the job will fail with an `INVALID_FBFM_CODES` error listing
    the invalid codes found.

    ## Response

    Returns the created Grid with status "pending". The backend applies the
    lookup transformation and updates status to "completed" when ready.

    ## Notes

    - Domain is propagated from the source grid (derived grids carry the
      same domain reference as their source).
    - The output grid inherits georeference from the source grid.
    - Non-burnable codes (91-99) produce zero values for all bands.
    - Fuel loads, fuel bed depth, and moisture of extinction are from
      Anderson, Hal E. 1982. *Aids to determining fuel models for estimating
      fire behavior.* USDA Forest Service General Technical Report INT-122.
    - All output values are in metric units (converted from Anderson 13 imperial values).
    """
    owner_id = request.state.id
    domain_id = domain["id"]

    await enforce_create_quotas(COLLECTION, request)

    await validate_feature_modifications(body.modifications, owner_id, domain_id)

    _, source_snapshot = await get_document_async(
        COLLECTION,
        body.source_grid_id,
        owner_id=owner_id,
        domain_id=domain_id,
        document_status="completed",
    )
    source_grid_data = source_snapshot.to_dict()

    validate_grid_has_band(source_grid_data, body.source_grid_id, body.source_band)

    bands = [
        get_fbfm13_lookup_band(band, index) for index, band in enumerate(body.bands)
    ]

    grid_id = uuid.uuid4().hex
    request_time = datetime.now()
    source = Fbfm13LookupSource(
        source_grid_id=body.source_grid_id,
        source_grid_checksum=source_grid_data.get("checksum"),
        source_band=body.source_band,
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
        "bands": [b.model_dump() for b in bands],
        "georeference": source_grid_data.get("georeference"),
        "tags": body.tags,
        "chunks": {"shape": CHUNK_SHAPE, "count": None, "count_by_axis": None},
        "owner_id": owner_id,
    }

    await set_document_async(COLLECTION, grid_id, grid_data)

    await create_http_task_async(GRIDDLE_QUEUE, GRIDDLE_SERVICE, grid_id)
    register_dispatch(request, response, background_tasks)

    return Grid(**grid_data)
