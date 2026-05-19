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
    DEFAULT_PAGE_SIZE,
    GEOJSON_MEDIA_TYPE,
    MAX_PAGE_SIZE,
    InvalidFeatureGeoJSON,
    PageOutOfRange,
    PageTooLarge,
    fetch_feature_page,
)
from api.resources.features.layerset.router import router as layerset_router
from api.resources.features.road.router import router as road_router
from api.resources.features.schema import (
    Feature,
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
    This action cannot be undone and will delete the associated GeoJSON file in GCS.
    """
    domain_id = domain["id"]

    await get_document_async(
        FEATURES_COLLECTION, feature_id, owner_id=request.state.id, domain_id=domain_id
    )

    await delete_document_async(
        collection=FEATURES_COLLECTION,
        document_id=feature_id,
    )

    # Delete the target GeoJSON blob asynchronously
    file_path = f"{domain_id}/{feature_id}.geojson"
    background_tasks.add_task(delete_file_safe, FEATURES_BUCKET, file_path)


@router.get(
    "/{feature_id}/data",
    status_code=status.HTTP_200_OK,
    summary="Get feature GeoJSON (paginated)",
    responses={200: {"content": {GEOJSON_MEDIA_TYPE: {}}}},
)
async def get_feature_data(
    request: Request,
    domain: VerifiedDomain,
    feature_id: str,
    page: int = Query(0, ge=0, description="Zero-indexed page number."),
    size: int = Query(
        DEFAULT_PAGE_SIZE,
        ge=1,
        le=MAX_PAGE_SIZE,
        description="Features per page.",
    ),
):
    """
    # Get Feature Data

    Returns a page of features from a completed feature resource's GeoJSON
    blob as a self-contained `FeatureCollection`. Defaults are sized so a
    typical OSM road/water blob comes back in a single request; clients with
    larger feature counts can paginate via `page` / `size`.

    ## Path Parameters

    - **domain_id**: The domain the feature belongs to.
    - **feature_id**: The unique identifier of the feature.

    ## Query Parameters

    - **page**: (integer, optional) Zero-indexed page number. Default `0`.
    - **size**: (integer, optional) Features per page (1-5000). Default `1000`.

    ## Response

    `application/geo+json` body — a valid GeoJSON `FeatureCollection` that
    can be saved straight to `.geojson` or piped into a renderer. The
    top-level `crs` block from the source file is preserved unchanged.

    Headers carry pagination metadata so the body stays a clean GeoJSON
    object:

    - `X-Page` — page index echoed back.
    - `X-Page-Size` — page size echoed back.
    - `X-Num-Features` — features in this page (≤ `size`).
    - `X-Total-Features` — total features in the source blob.
    - `X-Has-More` — `"true"` if more pages remain, else `"false"`.

    ## Error Responses

    - **404 Not Found**: Feature does not exist, is not accessible, or is
      not in `completed` status.
    - **413 Content Too Large**: Page payload exceeds the 30 MB cap. Lower
      `size` and retry.
    - **422 Unprocessable Entity**: `page` is out of range, or the
      underlying GeoJSON blob is missing / malformed.
    """
    await get_document_async(
        FEATURES_COLLECTION,
        feature_id,
        owner_id=request.state.id,
        domain_id=domain["id"],
        document_status="completed",
    )

    try:
        page_bytes, num_features, total = await fetch_feature_page(
            domain["id"], feature_id, page, size
        )
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=(
                "Feature GeoJSON not found in storage. "
                "Re-create the feature to enable data streaming."
            ),
        ) from exc
    except InvalidFeatureGeoJSON as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"Feature GeoJSON is malformed: {exc}",
        ) from exc
    except PageOutOfRange as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=(
                f"Page {exc.page} (size {exc.size}) out of range. "
                f"Feature has {exc.total} feature(s)."
            ),
        ) from exc
    except PageTooLarge as exc:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=(
                f"Page payload ({exc.page_bytes} bytes) exceeds API response "
                f"limit ({exc.limit} bytes). Request a smaller `size`."
            ),
        ) from exc

    has_more = (page + 1) * size < total
    headers = {
        "X-Page": str(page),
        "X-Page-Size": str(size),
        "X-Num-Features": str(num_features),
        "X-Total-Features": str(total),
        "X-Has-More": "true" if has_more else "false",
    }
    return Response(content=page_bytes, media_type=GEOJSON_MEDIA_TYPE, headers=headers)


router.include_router(road_router, prefix="/road", tags=["Features - Road"])
router.include_router(water_router, prefix="/water", tags=["Features - Water"])
router.include_router(layerset_router, prefix="/layerset", tags=["Features - Layerset"])
