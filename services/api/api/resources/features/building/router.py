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
    CONCATENATE_BUILDING_OPENAPI_EXAMPLES,
    CREATE_BUILDING_GEOJSON_OPENAPI_EXAMPLES,
    CREATE_BUILDING_OPENAPI_EXAMPLES,
)
from api.resources.features.building.schema import (
    CreateOvertureBuildingFeatureRequest,
    OvertureBuildingSource,
)
from api.resources.features.concatenate import (
    ConcatenateFeaturesRequest,
    create_concatenated_feature,
)
from api.resources.features.geojson import (
    CreateGeojsonFeatureRequest,
    create_geojson_feature,
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


@router.post(
    "/geojson",
    response_model=Feature,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a building feature as GeoJSON",
    responses=QUOTA_429_RESPONSE,
)
async def create_geojson_building_feature(
    request: Request,
    domain: VerifiedDomain,
    body: Annotated[
        CreateGeojsonFeatureRequest,
        Body(openapi_examples=CREATE_BUILDING_GEOJSON_OPENAPI_EXAMPLES),
    ],
) -> Feature:
    """
    # Upload Building Feature

    Creates a building feature from your own building footprint polygons, such as surveyed buildings missing from Overture Maps. The feature is used
    the same way as one from `/building/overture`.

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
    return await create_geojson_feature(request, domain, body, FeatureType.building)


@router.post(
    "/concatenate",
    response_model=Feature,
    status_code=status.HTTP_201_CREATED,
    summary="Concatenate building features",
    responses=QUOTA_429_RESPONSE,
)
async def concatenate_building_features(
    request: Request,
    domain: VerifiedDomain,
    body: Annotated[
        ConcatenateFeaturesRequest,
        Body(openapi_examples=CONCATENATE_BUILDING_OPENAPI_EXAMPLES),
    ],
) -> Feature:
    """
    # Concatenate Building Features

    Creates a new building feature whose rows are the rows of other building features
    in this domain, appended in order. Use it to combine Overture buildings with
    your own uploads.

    Geometries are not merged: overlapping polygons stay separate rows. The
    columns are the union of the inputs' columns, empty where an input lacks
    one, plus `source_feature_id`, the input each row came from.

    ## Request Body

    - **source_feature_ids**: (array of strings) 2–100 feature IDs, each
      listed once. Every feature must be a completed building feature in this
      domain.
    - **name**: (string, optional) Name of the feature.
    - **description**: (string, optional) Description.
    - **tags**: (array of strings, optional) Searchable tags.

    ## Response

    Returns the created Feature with status `"completed"`. Its
    `attributions` combine those of the input features.
    """
    return await create_concatenated_feature(
        request, domain, body, FeatureType.building
    )
