"""
api/v2/resources/features/building/router.py

Router for Building feature creation via Overture Maps.
"""

import uuid
from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Body, Request, Response, status

from api.db.documents import set_document_async
from api.dependencies import VerifiedDomain
from api.quota import QUOTA_429_RESPONSE, enforce_create_quotas, register_dispatch
from api.resources.features.building.examples import (
    CREATE_BUILDING_OPENAPI_EXAMPLES,
)
from api.resources.features.building.schema import (
    CreateOvertureBuildingFeatureRequest,
    OvertureBuildingSource,
)
from api.resources.features.schema import Feature, FeatureType
from api.schema import JobStatus
from api.tasks import create_http_task_async
from lib.config import (
    FEATURES_COLLECTION,
    FEATURES_QUEUE,
    FEATURES_SERVICE,
)

router = APIRouter()

COLLECTION = FEATURES_COLLECTION


@router.post(
    "/overture",
    response_model=Feature,
    status_code=status.HTTP_201_CREATED,
    summary="Create a building feature from Overture Maps",
    responses=QUOTA_429_RESPONSE,
)
async def create_overture_building_feature(
    request: Request,
    response: Response,
    background_tasks: BackgroundTasks,
    domain: VerifiedDomain,
    body: Annotated[
        CreateOvertureBuildingFeatureRequest,
        Body(openapi_examples=CREATE_BUILDING_OPENAPI_EXAMPLES),
    ],
):
    """
    # Create Overture Building Feature

    Creates building footprint polygons for the domain from the
    [Overture Maps](https://overturemaps.org) buildings dataset, which combines
    OpenStreetMap and Microsoft ML Buildings footprints.

    Data comes from the latest Overture release. The release used is recorded
    in `source.release` once the feature completes.

    ## Request Body

    - **name**: (optional) Name for the building feature.
    - **description**: (optional) Description.
    - **tags**: (optional) Tags for organizing features.
    - **extent_buffer_m**: (optional) Distance in meters to expand the domain
      extent before clipping buildings. Range: 0–100. Default: 0.

    ## Response

    Returns the created Feature with status `"pending"`; status changes to
    `"completed"` when the buildings are ready.

    Each building has:

    - **height**: building height in meters, or null if unknown.
    - **height_source**: dataset that supplied the height, or null.
    - **source_dataset**: dataset that supplied the footprint, e.g.
      `OpenStreetMap` or `Microsoft ML Buildings`.

    The feature's `attributions` list Overture Maps and each upstream dataset
    its buildings came from. The data is licensed under ODbL; reproduce each
    `citation` wherever you use or share it.
    """
    owner_id = request.state.id
    domain_id = domain["id"]

    await enforce_create_quotas(COLLECTION, request)

    feature_id = uuid.uuid4().hex
    request_time = datetime.now()
    source = OvertureBuildingSource(extent_buffer_m=body.extent_buffer_m)

    feature_data = {
        "id": feature_id,
        "domain_id": domain_id,
        "type": FeatureType.building.value,
        "name": body.name,
        "description": body.description,
        "status": JobStatus.pending.value,
        "progress": None,
        "created_on": request_time,
        "modified_on": request_time,
        "source": source.model_dump(),
        "georeference": None,
        "error": None,
        "tags": body.tags,
        "owner_id": owner_id,
    }

    await set_document_async(COLLECTION, feature_id, feature_data)

    # Enqueue task to the Features worker for processing
    await create_http_task_async(FEATURES_QUEUE, FEATURES_SERVICE, feature_id)
    register_dispatch(request, response, background_tasks)

    return Feature(**feature_data)
