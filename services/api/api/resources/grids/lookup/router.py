"""
api/v2/resources/grids/lookup/router.py

Router for lookup grid source endpoints.
"""

import uuid
from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Body, Request, Response, status

from api.db.documents import get_document_async, set_document_async
from api.dependencies import VerifiedDomain
from api.quota import QUOTA_429_RESPONSE, enforce_create_quotas, register_dispatch
from api.resources.grids.lookup.examples import (
    CREATE_FBFM40_LOOKUP_OPENAPI_EXAMPLES,
)
from api.resources.grids.lookup.schema import (
    CreateFbfm40LookupRequest,
    Fbfm40LookupSource,
    get_fbfm40_lookup_band,
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
    "/fbfm40",
    response_model=Grid,
    status_code=status.HTTP_201_CREATED,
    summary="Create a grid by looking up FBFM40 fuel parameters",
    responses=QUOTA_429_RESPONSE,
)
async def create_fbfm40_lookup(
    request: Request,
    response: Response,
    background_tasks: BackgroundTasks,
    domain: VerifiedDomain,
    body: Annotated[
        CreateFbfm40LookupRequest,
        Body(openapi_examples=CREATE_FBFM40_LOOKUP_OPENAPI_EXAMPLES),
    ],
):
    """
    # Create FBFM40 Lookup Grid

    Converts FBFM40 fuel model codes to fuel parameters using Scott-Burgan 40
    lookup tables.

    Takes a source grid containing categorical FBFM codes (from
    `/grids/fbfm40/landfire`) and produces a new grid with the requested
    continuous fuel parameters.

    ## Request Body

    - **source_grid_id**: (required) Grid containing FBFM40 codes.
    - **bands**: (required) Bands to look up. Valid values:
      - `fuel_load.1hr`, `fuel_load.10hr`, `fuel_load.100hr` - Dead fuel loads (kg/m**2)
      - `fuel_load.live_herb`, `fuel_load.live_woody` - Live fuel loads (kg/m**2)
      - `savr.1hr`, `savr.10hr`, `savr.100hr` - Dead fuel SAV ratios (1/m)
      - `savr.live_herb`, `savr.live_woody` - Live fuel SAV ratios (1/m)
      - `fuel_depth` - Fuel bed depth (m)
    - **source_band**: (optional) Band in source grid containing FBFM codes. Defaults to `"fbfm"`.
    - **name**: (optional) Name for the grid.
    - **description**: (optional) Description.
    - **tags**: (optional) Tags for organizing grids.

    ## Valid FBFM40 Codes

    The source grid must contain only valid Scott-Burgan 40 fuel model codes.
    The 46 valid codes are:

    - **NB** (non-burnable): 91, 92, 93, 98, 99
    - **GR** (grass): 101–109
    - **GS** (grass-shrub): 121–124
    - **SH** (shrub): 141–149
    - **TU** (timber-understory): 161–165
    - **TL** (timber litter): 181–189
    - **SB** (slash-blowdown): 201–204

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
    - Non-burnable codes (NB1–NB9) produce zero values for all bands.
    - All output values are in metric units (converted from SB40 imperial values).
    """
    owner_id = request.state.id
    domain_id = domain["id"]

    await enforce_create_quotas(COLLECTION, request)

    await validate_feature_modifications(body.modifications, owner_id, domain_id)

    # Validate source grid: exists, owned, in this domain, and completed
    _, source_snapshot = await get_document_async(
        COLLECTION,
        body.source_grid_id,
        owner_id=owner_id,
        domain_id=domain_id,
        document_status="completed",
    )
    source_grid_data = source_snapshot.to_dict()

    # Validate source grid has the required band
    validate_grid_has_band(source_grid_data, body.source_grid_id, body.source_band)

    bands = [
        get_fbfm40_lookup_band(band, index) for index, band in enumerate(body.bands)
    ]

    grid_id = uuid.uuid4().hex
    request_time = datetime.now()
    source = Fbfm40LookupSource(
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
