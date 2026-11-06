"""
api/v2/resources/features/road/router.py

Router for Road feature creation via OpenStreetMap.
"""

import uuid
from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Body, Request, Response, status

from api.db.documents import set_document_async
from api.dependencies import VerifiedDomain
from api.quota import QUOTA_429_RESPONSE, enforce_create_quotas, register_dispatch
from api.resources.features.concatenate import (
    ConcatenateFeaturesRequest,
    create_concatenated_feature,
)
from api.resources.features.geojson import (
    CreateGeojsonFeatureRequest,
    create_geojson_feature,
)
from api.resources.features.road.examples import (
    CONCATENATE_ROAD_OPENAPI_EXAMPLES,
    CREATE_ROAD_GEOJSON_OPENAPI_EXAMPLES,
    CREATE_ROAD_OPENAPI_EXAMPLES,
)
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
    response: Response,
    background_tasks: BackgroundTasks,
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
    register_dispatch(request, response, background_tasks)

    return Feature(**feature_data)


@router.post(
    "/geojson",
    response_model=Feature,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a road feature as GeoJSON",
    responses=QUOTA_429_RESPONSE,
)
async def create_geojson_road_feature(
    request: Request,
    domain: VerifiedDomain,
    body: Annotated[
        CreateGeojsonFeatureRequest,
        Body(openapi_examples=CREATE_ROAD_GEOJSON_OPENAPI_EXAMPLES),
    ],
) -> Feature:
    """
    # Upload Road Feature

    Creates a road feature from your own road footprint polygons, such as driveways missing from OpenStreetMap. The feature is used
    the same way as one from `/road/osm`.

    The polygons are reprojected to the domain CRS and clipped to the domain,
    plus `extent_buffer_m`.

    ## Request Body

    The body **is** a GeoJSON FeatureCollection, with resource metadata as
    top-level fields:

    - **features**: (array) At least one Feature with a `Polygon` or
      `MultiPolygon` geometry. `properties` are optional and are stored as
      columns; each property must have one type across all features.
    - **crs**: (object, optional) The GeoJSON `crs` block. Any CRS is
      accepted. When omitted, coordinates are read as `EPSG:4326`.
    - **name**: (string, optional) Name of the feature.
    - **description**: (string, optional) Description.
    - **tags**: (array of strings, optional) Searchable tags.
    - **extent_buffer_m**: (number, optional) Distance in meters to expand the
      domain extent before clipping. Range: 0–100. Default: 0.

    Returns 422 if a geometry is invalid or no polygon intersects the domain.

    ## Response

    Returns the created Feature with status `"completed"`.
    """
    return await create_geojson_feature(request, domain, body, FeatureType.road)


@router.post(
    "/concatenate",
    response_model=Feature,
    status_code=status.HTTP_201_CREATED,
    summary="Concatenate road features",
    responses=QUOTA_429_RESPONSE,
)
async def concatenate_road_features(
    request: Request,
    domain: VerifiedDomain,
    body: Annotated[
        ConcatenateFeaturesRequest,
        Body(openapi_examples=CONCATENATE_ROAD_OPENAPI_EXAMPLES),
    ],
) -> Feature:
    """
    # Concatenate Road Features

    Creates a new road feature whose rows are the rows of other road features
    in this domain, appended in order. Use it to combine OpenStreetMap roads with
    your own uploads.

    Geometries are not merged: overlapping polygons stay separate rows. The
    columns are the union of the inputs' columns, empty where an input lacks
    one, plus `source_feature_id`, the input each row came from.

    ## Request Body

    - **source_feature_ids**: (array of strings) 2–100 feature IDs, each
      listed once. Every feature must be a completed road feature in this
      domain.
    - **name**: (string, optional) Name of the feature.
    - **description**: (string, optional) Description.
    - **tags**: (array of strings, optional) Searchable tags.

    ## Response

    Returns the created Feature with status `"completed"`. Its
    `attributions` combine those of the input features.
    """
    return await create_concatenated_feature(request, domain, body, FeatureType.road)
