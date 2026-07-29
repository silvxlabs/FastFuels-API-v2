"""
api/v2/resources/grids/lookup/fccs/router.py

Router for the FCCS lookup endpoint.
"""

import uuid
from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Body, Request, Response, status

from api.db.documents import get_document_async, set_document_async
from api.dependencies import VerifiedDomain
from api.quota import QUOTA_429_RESPONSE, enforce_create_quotas, register_dispatch
from api.resources.grids.lookup.fccs.examples import (
    CREATE_FCCS_LOOKUP_OPENAPI_EXAMPLES,
)
from api.resources.grids.lookup.fccs.schema import (
    CreateFccsLookupRequest,
    FccsLookupSource,
    get_fccs_lookup_band,
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
    summary="Create a grid by looking up FCCS fuel parameters",
    responses=QUOTA_429_RESPONSE,
)
async def create_fccs_lookup(
    request: Request,
    response: Response,
    background_tasks: BackgroundTasks,
    domain: VerifiedDomain,
    body: Annotated[
        CreateFccsLookupRequest,
        Body(openapi_examples=CREATE_FCCS_LOOKUP_OPENAPI_EXAMPLES),
    ],
):
    """
    # Create FCCS Lookup Grid

    Converts FCCS fuelbed codes to fuel parameters using the FOFEM FCCS
    fuelbed lookup table (see the
    [FOFEM/SpatialFOFEM FCCS lookup table](https://www.landfire.gov/sites/default/files/CSV/SpatialFOFEM_FCCS_Formatted_TS_06-27-24.csv),
    the USDA Forest Service data source this endpoint converts).

    Takes a source grid containing categorical FCCS codes (from
    `/grids/fccs/landfire`) and produces a new grid with the requested
    continuous fuel parameters.

    ## Request Body

    - **source_grid_id**: (required) Grid containing FCCS codes.
    - **bands**: (required) Bands to look up. Valid values:
      - `fuel_load.litter`, `fuel_load.duff` - Ground fuel loads (kg/m**2)
      - `duff_depth` - Duff layer depth (m)
      - `fuel_load.live_shrub`, `fuel_load.live_herb` - Live surface fuel loads (kg/m**2)
      - `fuel_load.1hr`, `fuel_load.10hr`, `fuel_load.100hr` - Dead fuel loads (kg/m**2)
      - `fuel_load.1000hr_sound`, `fuel_load.1000hr_rotten` - Dead fuel loads, >3in. diameter (kg/m**2)
      - `fuel_load.live_foliage`, `fuel_load.live_branch` - Live crown fuel loads (kg/m**2)
    - **source_band**: (optional) Band in source grid containing FCCS codes. Defaults to `"fccs"`.
    - **name**: (optional) Name for the grid.
    - **description**: (optional) Description.
    - **tags**: (optional) Tags for organizing grids.

    These 12 bands are a starting subset of what FOFEM provides, not the
    full table. `fuel_load.1000hr_sound` and `fuel_load.1000hr_rotten`
    are each calculated by summing three FOFEM size-class columns
    (3-9 in., 9-20 in., 20+ in.) rather than mapping to a single source
    column. FOFEM also provides finer sound/rotten size-class
    breakdowns, a cover-group code, and emission factors that aren't
    exposed as bands here — additional bands can be added on request.
    - **source_band**: (optional) Band in source grid containing FCCS codes. Defaults to `"fccs"`.
    - **name**: (optional) Name for the grid.
    - **description**: (optional) Description.
    - **tags**: (optional) Tags for organizing grids.

    ## Valid FCCS Codes

    Each `FCCS` code is a synthetic key: `base * 10_000 + suffix`, where
    `base` is the `FCCSID` fuelbed number and the 3-digit `suffix`
    encodes an FCCS Potential rating (Fire Behavior / Crown Fire /
    Available Fuel Potential, each a 0-9 digit) per the [Fuel Characteristic
    Classification System Version 3.0: Technical Documentation (PNW-GTR-887)](https://www.fs.usda.gov/pnw/pubs/pnw_gtr887.pdf).

    The source grid must contain FCCS codes whose base fuelbed number matches
    a recognized `FCCSID`. A code with a valid base but no matching row in the
    FOFEM lookup table is not an error. It's a fuelbed/fire-potential combination
    the table doesn't cover, so its output is `NaN` for every band, and a
    progress warning lists these codes.

    ## Response

    Returns the created Grid with status "pending". The backend applies the
    lookup transformation and updates status to "completed" when ready.

    ## Notes

    - Domain is propagated from the source grid (derived grids carry the
      same domain reference as their source).
    - The output grid inherits georeference from the source grid.
    - All output values are in metric units (converted from FOFEM imperial
      values).
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

    bands = [get_fccs_lookup_band(band, index) for index, band in enumerate(body.bands)]

    grid_id = uuid.uuid4().hex
    request_time = datetime.now()
    source = FccsLookupSource(
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
