"""
api/v2/resources/features/water/router.py

Router for Water feature creation via OpenStreetMap.
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
from api.resources.features.schema import Feature, FeatureType
from api.resources.features.water.examples import (
    CONCATENATE_WATER_OPENAPI_EXAMPLES,
    CREATE_WATER_GEOJSON_OPENAPI_EXAMPLES,
    CREATE_WATER_OPENAPI_EXAMPLES,
)
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
    "/osm",
    response_model=Feature,
    status_code=status.HTTP_201_CREATED,
    summary="Create a water feature from OpenStreetMap",
    responses=QUOTA_429_RESPONSE,
)
async def create_osm_water_feature(
    request: Request,
    response: Response,
    background_tasks: BackgroundTasks,
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
    - **extent_buffer_m**: (optional) Distance in meters to expand the domain
      extent outward before clipping fetched water features. Lets streams and
      rivers that exit the domain at the boundary extend slightly past the
      edge, providing context for visualization and downstream operations.
      Applied in the domain's projected CRS. If omitted, water features are
      clipped exactly to the domain boundary. Range: 0–100 meters.

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
    source = OsmWaterSource(extent_buffer_m=body.extent_buffer_m)

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
    register_dispatch(request, response, background_tasks)

    return Feature(**feature_data)


@router.post(
    "/geojson",
    response_model=Feature,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a water feature as GeoJSON",
    responses=QUOTA_429_RESPONSE,
)
async def create_geojson_water_feature(
    request: Request,
    domain: VerifiedDomain,
    body: Annotated[
        CreateGeojsonFeatureRequest,
        Body(openapi_examples=CREATE_WATER_GEOJSON_OPENAPI_EXAMPLES),
    ],
) -> Feature:
    """
    # Upload Water Feature

    Creates a water feature from your own water polygons, such as ponds missing from OpenStreetMap. The feature is used
    the same way as one from `/water/osm`.

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
    return await create_geojson_feature(request, domain, body, FeatureType.water)


@router.post(
    "/concatenate",
    response_model=Feature,
    status_code=status.HTTP_201_CREATED,
    summary="Concatenate water features",
    responses=QUOTA_429_RESPONSE,
)
async def concatenate_water_features(
    request: Request,
    domain: VerifiedDomain,
    body: Annotated[
        ConcatenateFeaturesRequest,
        Body(openapi_examples=CONCATENATE_WATER_OPENAPI_EXAMPLES),
    ],
) -> Feature:
    """
    # Concatenate Water Features

    Creates a new water feature whose rows are the rows of other water features
    in this domain, appended in order. Use it to combine OpenStreetMap water with
    your own uploads.

    Geometries are not merged: overlapping polygons stay separate rows. The
    columns are the union of the inputs' columns, empty where an input lacks
    one, plus `source_feature_id`, the input each row came from.

    ## Request Body

    - **source_feature_ids**: (array of strings) 2–100 feature IDs, each
      listed once. Every feature must be a completed water feature in this
      domain.
    - **name**: (string, optional) Name of the feature.
    - **description**: (string, optional) Description.
    - **tags**: (array of strings, optional) Searchable tags.

    ## Response

    Returns the created Feature with status `"completed"`. Its
    `attributions` combine those of the input features.
    """
    return await create_concatenated_feature(request, domain, body, FeatureType.water)
