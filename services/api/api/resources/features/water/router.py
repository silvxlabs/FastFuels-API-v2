"""
api/v2/resources/features/water/router.py

Router for Water feature creation via OpenStreetMap.
"""

import uuid
from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Body, Request, status

from api.db.documents import set_document_async
from api.dependencies import VerifiedDomain
from api.resources.features.schema import Feature, FeatureType
from api.resources.features.water.examples import CREATE_WATER_OPENAPI_EXAMPLES
from api.resources.features.water.schema import (
    CreateOsmWaterFeatureRequest,
    OsmWaterSource,
)
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
    "",
    response_model=Feature,
    status_code=status.HTTP_201_CREATED,
    summary="Create a water feature from OpenStreetMap",
)
async def create_osm_water_feature(
    request: Request,
    domain: VerifiedDomain,
    body: Annotated[
        CreateOsmWaterFeatureRequest,
        Body(openapi_examples=CREATE_WATER_OPENAPI_EXAMPLES),
    ],
):
    """
    # Create OSM Water Feature

    Generates a polygon representation of water bodies and waterways within the
    specified domain using data from OpenStreetMap (OSM).

    The backend worker will:
    1. Fetch the bounding box for the target domain.
    2. Query OpenStreetMap for water features (e.g., `water=*`, `waterway=*`, `natural=water`).
    3. Extract existing polygon features (lakes, ponds, wide rivers).
    4. Dynamically buffer linear water features (streams, creeks, narrow rivers)
       into polygon areas based on their specific OSM classification.
    5. Merge the geometries and save the resulting GeoJSON to the features bucket.

    ## Request Body

    - **name**: (optional) Name for the water feature.
    - **description**: (optional) Description.
    - **tags**: (optional) Tags for organizing features.

    ## Response

    Returns the created Feature resource with status ``"pending"``. The
    backend worker will process the OSM extraction asynchronously and update
    status to ``"completed"`` when ready.
    """
    owner_id = request.state.id
    domain_id = domain["id"]

    feature_id = uuid.uuid4().hex
    request_time = datetime.now()
    source = OsmWaterSource()

    feature_data = {
        "id": feature_id,
        "domain_id": domain_id,
        "type": FeatureType.water.value,
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

    return Feature(**feature_data)
