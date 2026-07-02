"""
api/v2/resources/features/road/router.py

Router for Road feature creation via OpenStreetMap.
"""

import uuid
from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Body, Request, status

from api.db.documents import set_document_async
from api.dependencies import VerifiedDomain
from api.quota import QUOTA_429_RESPONSE, enforce_create_quotas
from api.resources.features.road.examples import CREATE_ROAD_OPENAPI_EXAMPLES
from api.resources.features.road.schema import (
    CreateOsmRoadFeatureRequest,
    OsmRoadSource,
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
    "/osm",
    response_model=Feature,
    status_code=status.HTTP_201_CREATED,
    summary="Create a road feature from OpenStreetMap",
    responses=QUOTA_429_RESPONSE,
)
async def create_osm_road_feature(
    request: Request,
    domain: VerifiedDomain,
    body: Annotated[
        CreateOsmRoadFeatureRequest,
        Body(openapi_examples=CREATE_ROAD_OPENAPI_EXAMPLES),
    ],
):
    """
    # Create OSM Road Feature

    Generates a polygon representation of the road network within the specified
    domain using data from OpenStreetMap (OSM).

    The backend worker will:
    1. Fetch the bounding box for the target domain.
    2. Query OpenStreetMap for linear road segments (`highway=*`).
    3. Dynamically buffer the line strings into realistic polygon areas
       based on their specific OSM classification (e.g., motorways receive
       a wider buffer than residential streets or trails).
    4. Save the resulting GeoJSON to the features bucket.

    ## Request Body

    - **name**: (optional) Name for the road feature.
    - **description**: (optional) Description.
    - **tags**: (optional) Tags for organizing features.
    - **extent_buffer_m**: (optional) Distance in meters to expand the domain
      extent outward before clipping fetched roads. Lets roads that exit the
      domain at the boundary extend slightly past the edge, providing context
      for visualization and downstream operations. Applied in the domain's
      projected CRS. If omitted, roads are clipped exactly to the domain
      boundary. Range: 0–100 meters.

    ## Response

    Returns the created Feature resource with status ``"pending"``. The
    backend worker will process the OSM extraction asynchronously and update
    status to ``"completed"`` when ready.
    """
    owner_id = request.state.id
    domain_id = domain["id"]

    await enforce_create_quotas(COLLECTION, request)

    feature_id = uuid.uuid4().hex
    request_time = datetime.now()
    source = OsmRoadSource(extent_buffer_m=body.extent_buffer_m)

    feature_data = {
        "id": feature_id,
        "domain_id": domain_id,
        "type": FeatureType.road.value,
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
