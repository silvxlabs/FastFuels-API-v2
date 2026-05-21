"""
api/v2/resources/features/router.py

Router for the Feature resource with standard CRUD endpoints.
Algorithm-specific creation endpoints are in their respective subdirectories.
"""

from datetime import datetime

from fastapi import (
    APIRouter,
    BackgroundTasks,
    HTTPException,
    Path,
    Query,
    Request,
    Response,
    status,
)

from api.db.blobs import delete_file_safe
from api.db.documents import (
    delete_document_async,
    get_document_async,
    list_documents_async,
    update_document_async,
)
from api.dependencies import VerifiedDomain
from api.resources.features.cache import (
    GEOJSON_MEDIA_TYPE,
    InvalidFeatureParquet,
    PartitionOutOfRange,
    PartitionTooLarge,
    fetch_partition_geojson,
    get_feature_metadata,
)
from api.resources.features.layerset.router import router as layerset_router
from api.resources.features.road.router import router as road_router
from api.resources.features.schema import (
    Feature,
    FeatureDataMetadata,
    FeatureSortField,
    FeatureType,
    ListFeaturesResponse,
    UpdateFeatureRequestBody,
)
from api.resources.features.water.router import router as water_router
from api.schema import SortOrder
from lib.config import FEATURES_BUCKET, FEATURES_COLLECTION

router = APIRouter()
wildcard_router = APIRouter()


@wildcard_router.get(
    "",
    response_model=ListFeaturesResponse,
    status_code=status.HTTP_200_OK,
    summary="List features across all domains",
)
async def list_features_cross_domain(
    request: Request,
    page: int = Query(
        0,
        ge=0,
        description="The page number to retrieve (zero-indexed).",
    ),
    size: int = Query(
        100,
        ge=1,
        le=1000,
        description="The number of features to retrieve per page.",
    ),
    sort_by: FeatureSortField | None = Query(
        None,
        description="The field to sort results by.",
    ),
    sort_order: SortOrder | None = Query(
        None,
        description="The order to sort results (ascending or descending).",
    ),
    type: FeatureType | None = Query(
        None,
        description="Filter features by entity type (e.g., 'road', 'water').",
    ),
    product: str | None = Query(
        None,
        description="Filter features by source product (e.g., 'osm').",
    ),
    tag: str | None = Query(
        None,
        description="Filter features that contain this tag.",
    ),
) -> ListFeaturesResponse:
    """
    # List Features Endpoint

    Retrieves a paginated list of all features across all domains belonging to
    the authenticated user.

    ## Query Parameters

    - **page**: (integer, optional) Page number (zero-indexed). Default: 0.
    - **size**: (integer, optional) Items per page (1-1000). Default: 100.
    - **sort_by**: (string, optional) Field to sort by: `created_on`, `modified_on`, `name`.
    - **sort_order**: (string, optional) Sort direction: `ascending`, `descending`.
    - **type**: (string, optional) Filter by entity type (e.g., `road`).
    - **product**: (string, optional) Filter by source product (e.g., `osm`).
    - **tag**: (string, optional) Filter features that contain this tag.

    ## Response

    Returns a paginated list of features with metadata.
    """
    owner_id = request.state.id

    # Build equality filters
    filters = {}
    if type:
        filters["type"] = type.value
    if product:
        filters["source.product"] = product

    # Build array-contains filters
    array_contains_filters = {}
    if tag:
        array_contains_filters["tags"] = tag

    # Query Firestore
    documents, total_count = await list_documents_async(
        collection=FEATURES_COLLECTION,
        owner_id=owner_id,
        page=page,
        size=size,
        sort_by=sort_by.value if sort_by else None,
        sort_order=sort_order.value if sort_order else None,
        filters=filters if filters else None,
        array_contains_filters=(
            array_contains_filters if array_contains_filters else None
        ),
    )

    features = [Feature(**doc.to_dict()) for doc in documents]

    return ListFeaturesResponse(
        features=features,
        current_page=page,
        page_size=size,
        total_items=total_count,
    )


@router.get(
    "",
    response_model=ListFeaturesResponse,
    status_code=status.HTTP_200_OK,
    summary="List all features",
)
async def list_features(
    request: Request,
    domain: VerifiedDomain,
    page: int = Query(
        0,
        ge=0,
        description="The page number to retrieve (zero-indexed).",
    ),
    size: int = Query(
        100,
        ge=1,
        le=1000,
        description="The number of features to retrieve per page.",
    ),
    sort_by: FeatureSortField | None = Query(
        None,
        description="The field to sort results by.",
    ),
    sort_order: SortOrder | None = Query(
        None,
        description="The order to sort results (ascending or descending).",
    ),
    type: FeatureType | None = Query(
        None,
        description="Filter features by entity type (e.g., 'road', 'water').",
    ),
    product: str | None = Query(
        None,
        description="Filter features by source product (e.g., 'osm').",
    ),
    tag: str | None = Query(
        None,
        description="Filter features that contain this tag.",
    ),
) -> ListFeaturesResponse:
    """
    # List Features Endpoint

    Retrieves a paginated list of all features within a domain belonging to
    the authenticated user.

    ## Path Parameters

    - **domain_id**: (string) The domain to list features for.

    ## Query Parameters

    - **page**: (integer, optional) Page number (zero-indexed). Default: 0.
    - **size**: (integer, optional) Items per page (1-1000). Default: 100.
    - **sort_by**: (string, optional) Field to sort by: `created_on`, `modified_on`, `name`.
    - **sort_order**: (string, optional) Sort direction: `ascending`, `descending`.
    - **type**: (string, optional) Filter by entity type (e.g., `road`).
    - **product**: (string, optional) Filter by source product (e.g., `osm`).
    - **tag**: (string, optional) Filter features that contain this tag.

    ## Response

    Returns a paginated list of features with metadata.
    """
    owner_id = request.state.id
    domain_id = domain["id"]

    # Build equality filters
    filters = {"domain_id": domain_id}
    if type:
        filters["type"] = type.value
    if product:
        filters["source.product"] = product

    # Build array-contains filters
    array_contains_filters = {}
    if tag:
        array_contains_filters["tags"] = tag

    # Query Firestore
    documents, total_count = await list_documents_async(
        collection=FEATURES_COLLECTION,
        owner_id=owner_id,
        page=page,
        size=size,
        sort_by=sort_by.value if sort_by else None,
        sort_order=sort_order.value if sort_order else None,
        filters=filters if filters else None,
        array_contains_filters=(
            array_contains_filters if array_contains_filters else None
        ),
    )

    features = [Feature(**doc.to_dict()) for doc in documents]

    return ListFeaturesResponse(
        features=features,
        current_page=page,
        page_size=size,
        total_items=total_count,
    )


@router.get(
    "/{feature_id}",
    response_model=Feature,
    status_code=status.HTTP_200_OK,
    summary="Get a feature by ID",
)
async def get_feature(
    request: Request,
    domain: VerifiedDomain,
    feature_id: str,
):
    """
    # Get Feature Endpoint

    Retrieves a specific feature resource by its unique identifier.

    ## Path Parameters

    - **domain_id**: (string) The domain the feature belongs to.
    - **feature_id**: (string) The unique identifier of the feature.

    ## Response

    Returns the feature resource.

    ## Error Responses

    - **404 Not Found**: The feature does not exist or the user does not have access.
    """
    _, snapshot = await get_document_async(
        FEATURES_COLLECTION,
        feature_id,
        owner_id=request.state.id,
        domain_id=domain["id"],
    )
    return Feature(**snapshot.to_dict())


@router.patch(
    "/{feature_id}",
    response_model=Feature,
    status_code=status.HTTP_200_OK,
    summary="Update a feature",
)
async def update_feature(
    request: Request,
    domain: VerifiedDomain,
    feature_id: str,
    body: UpdateFeatureRequestBody,
):
    """
    # Update Feature Endpoint

    Updates the metadata of an existing feature resource. Only the fields
    provided in the request body will be modified.

    ## Path Parameters

    - **domain_id**: (string) The domain the feature belongs to.
    - **feature_id**: (string) The unique identifier of the feature.

    ## Request Body

    All fields are optional:

    - **name**: (string) New name for the feature.
    - **description**: (string) New description.
    - **tags**: (array of strings) New tags (replaces existing).

    ## What Cannot Be Updated

    The following fields are immutable:

    - **id**, **domain_id**, **type**, **source**, **georeference**
    - **created_on** (creation timestamp is permanent)

    The **modified_on** field is automatically updated.

    ## Response

    Returns the updated feature resource.
    """
    _, snapshot = await get_document_async(
        FEATURES_COLLECTION,
        feature_id,
        owner_id=request.state.id,
        domain_id=domain["id"],
    )
    feature_data = snapshot.to_dict()

    update_data = body.model_dump(exclude_none=True)
    update_data["modified_on"] = datetime.now()

    await update_document_async(
        collection=FEATURES_COLLECTION,
        document_id=feature_id,
        data=update_data,
    )

    feature_data.update(update_data)

    return Feature(**feature_data)


@router.delete(
    "/{feature_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a feature",
)
async def delete_feature(
    request: Request,
    domain: VerifiedDomain,
    feature_id: str,
    background_tasks: BackgroundTasks,
):
    """
    # Delete Feature Endpoint

    Permanently deletes a feature resource by its unique identifier.
    """
    domain_id = domain["id"]

    await get_document_async(
        FEATURES_COLLECTION, feature_id, owner_id=request.state.id, domain_id=domain_id
    )

    await delete_document_async(
        collection=FEATURES_COLLECTION,
        document_id=feature_id,
    )

    # Delete the target Parquet blob asynchronously
    file_path = f"{domain_id}/{feature_id}.parquet"
    background_tasks.add_task(delete_file_safe, FEATURES_BUCKET, file_path)


@router.get(
    "/{feature_id}/data/metadata",
    response_model=FeatureDataMetadata,
    status_code=status.HTTP_200_OK,
    summary="Get feature data partition layout",
)
async def get_feature_data_metadata(
    request: Request,
    domain: VerifiedDomain,
    feature_id: str,
) -> FeatureDataMetadata:
    """
    # Get Feature Data Metadata

    Returns the partition layout for a completed feature's data blob. Use
    this to discover how many partitions exist before streaming them via
    `GET /domains/{domain_id}/features/{feature_id}/data/{partition_index}`.

    ## Path Parameters

    - **domain_id**: The domain the feature belongs to.
    - **feature_id**: The unique identifier of the feature.

    ## Response

    JSON object describing the partition layout of the underlying
    GeoParquet blob:

    ```json
    {
      "total_features": 5400,
      "partition_count": 6,
      "partitions": [
        {"index": 0, "num_features": 1000},
        {"index": 1, "num_features": 1000},
        {"index": 2, "num_features": 1000},
        {"index": 3, "num_features": 1000},
        {"index": 4, "num_features": 1000},
        {"index": 5, "num_features": 400}
      ]
    }
    ```

    - **total_features**: Total number of features across all partitions.
    - **partition_count**: Number of valid `partition_index` values. Iterate
      from `0` to `partition_count - 1` to retrieve every feature exactly
      once, in source order.
    - **partitions**: Per-partition row counts read directly from the
      GeoParquet footer. Sum of `num_features` equals `total_features`.

    A feature with zero features has `partition_count = 0` and an empty
    `partitions` list — no `/data/{partition_index}` calls are valid in
    that case.

    ## Error Responses

    - **404 Not Found**: Feature does not exist or is not accessible to the
      caller.
    - **422 Unprocessable Entity**: Feature is not in `completed` status, or
      the underlying GeoParquet blob is missing / malformed.
    """
    await get_document_async(
        FEATURES_COLLECTION,
        feature_id,
        owner_id=request.state.id,
        domain_id=domain["id"],
        document_status="completed",
    )

    try:
        return await get_feature_metadata(domain["id"], feature_id)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=(
                "Feature data blob not found in storage. "
                "Re-create the feature to enable data streaming."
            ),
        ) from exc
    except InvalidFeatureParquet as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"Feature data blob is malformed: {exc}",
        ) from exc


@router.get(
    "/{feature_id}/data/{partition_index}",
    status_code=status.HTTP_200_OK,
    summary="Get one partition of feature data as GeoJSON",
    responses={200: {"content": {GEOJSON_MEDIA_TYPE: {}}}},
)
async def get_feature_data_partition(
    request: Request,
    domain: VerifiedDomain,
    feature_id: str,
    partition_index: int = Path(
        ...,
        ge=0,
        description=(
            "Zero-indexed partition number. Valid range is "
            "`0` ≤ `partition_index` < `partition_count` from "
            "`GET /data/metadata`."
        ),
    ),
):
    """
    # Get Feature Data Partition

    Returns one partition of a completed feature's data as a self-contained
    GeoJSON `FeatureCollection`. To stream the full collection, first call
    `GET /domains/{domain_id}/features/{feature_id}/data/metadata` to
    discover `partition_count`, then GET this endpoint for each
    `partition_index` from `0` to `partition_count - 1`. The concatenated
    `features` arrays reproduce the source feature list in source order.

    ## Path Parameters

    - **domain_id**: The domain the feature belongs to.
    - **feature_id**: The unique identifier of the feature.
    - **partition_index**: Zero-indexed partition number. Must be `< partition_count`
      from `/data/metadata`.

    ## Response

    `application/geo+json` body — a valid GeoJSON `FeatureCollection`
    containing up to `partition_size` features. Each feature's `properties`
    and `geometry` round-trip from the source GeoParquet via geopandas.

    ## Error Responses

    - **404 Not Found**: Feature does not exist or is not accessible to the
      caller.
    - **413 Content Too Large**: Serialized partition exceeds the 30 MB
      response cap. Partition size is fixed server-side and cannot be
      adjusted by re-creating the feature; contact
      `support.fastfuels@silvxlabs.com` so the partition size can be
      reduced for your feature.
    - **422 Unprocessable Entity**: `partition_index` is past the last
      partition, the feature is not in `completed` status, or the
      underlying GeoParquet blob is missing / malformed.
    """
    await get_document_async(
        FEATURES_COLLECTION,
        feature_id,
        owner_id=request.state.id,
        domain_id=domain["id"],
        document_status="completed",
    )

    try:
        payload, _num_features, _total = await fetch_partition_geojson(
            domain["id"], feature_id, partition_index
        )
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=(
                "Feature data blob not found in storage. "
                "Re-create the feature to enable data streaming."
            ),
        ) from exc
    except InvalidFeatureParquet as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"Feature data blob is malformed: {exc}",
        ) from exc
    except PartitionOutOfRange as exc:
        if exc.partition_count == 0:
            detail = (
                f"Partition {exc.partition_index} out of range. "
                "Feature has 0 partitions; no /data/{i} calls are valid."
            )
        else:
            detail = (
                f"Partition {exc.partition_index} out of range. "
                f"Feature has {exc.partition_count} partition(s); "
                f"valid indices are 0..{exc.partition_count - 1}."
            )
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=detail,
        ) from exc
    except PartitionTooLarge as exc:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=(
                f"Partition payload ({exc.payload_bytes} bytes) exceeds API "
                f"response limit ({exc.limit} bytes). Partition size is fixed "
                "server-side and cannot be adjusted by re-creating the "
                "feature; contact support.fastfuels@silvxlabs.com so the "
                "partition size can be reduced for your feature."
            ),
        ) from exc

    return Response(content=payload, media_type=GEOJSON_MEDIA_TYPE)


router.include_router(road_router, prefix="/road", tags=["Features - Road"])
router.include_router(water_router, prefix="/water", tags=["Features - Water"])
router.include_router(layerset_router, prefix="/layerset", tags=["Features - Layerset"])
