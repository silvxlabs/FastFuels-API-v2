"""
api/v2/resources/grids/router.py

Router for the Grid resource with standard CRUD endpoints.
Product-specific endpoints (FBFM40, Topography, etc.) are in their respective subdirectories.
"""

from datetime import datetime

import numpy as np
from fastapi import (
    APIRouter,
    BackgroundTasks,
    HTTPException,
    Query,
    Request,
    Response,
    status,
)

from api.db.blobs import delete_directory_safe
from api.db.documents import (
    delete_document_async,
    get_document_async,
    list_documents_async,
    update_document_async,
)
from api.dependencies import VerifiedDomain
from api.resources.grids.cache import get_grid_array
from api.resources.grids.chm.router import router as chm_router
from api.resources.grids.exports.router import router as grid_exports_router
from api.resources.grids.fbfm40.router import router as fbfm40_router
from api.resources.grids.lookup.router import router as lookup_router
from api.resources.grids.pim.router import router as pim_router
from api.resources.grids.resample.router import router as resample_router
from api.resources.grids.schema import (
    DenseGridData,
    Grid,
    GridDataArrayFormat,
    GridDataChunkMetadata,
    GridDataOrder,
    GridDataResponse,
    GridDataResponseFormat,
    GridSortField,
    ListGridsResponse,
    SparseGridData,
    UpdateGridRequestBody,
)
from api.resources.grids.topography.router import router as topography_router
from api.resources.grids.tree.router import router as tree_router
from api.resources.grids.uniform.router import router as uniform_router
from api.resources.grids.utils import (
    compute_chunk_metadata,
    compute_chunk_slices,
    validate_grid_has_band,
)
from api.schema import SortOrder
from lib.config import GRIDS_BUCKET, GRIDS_COLLECTION

router = APIRouter()
wildcard_router = APIRouter()

COLLECTION = GRIDS_COLLECTION
MAX_BINARY_BYTES = 30 * 1024 * 1024
MAX_JSON_SCALARS = 1_000_000


def _check_size(actual: int, limit: int, what: str) -> None:
    if actual > limit:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(
                f"{what} ({actual}) exceeds API response limit ({limit}). "
                "Request array_format=sparse, format=binary, or a smaller chunk."
            ),
        )


def _sparse_components(
    data: np.ndarray,
    fill_value: float | int,
    order: GridDataOrder,
) -> tuple[np.ndarray, np.ndarray]:
    flat = data.ravel(order=order.value)
    if np.issubdtype(flat.dtype, np.floating) and np.isnan(fill_value):
        mask = ~np.isnan(flat)
    else:
        mask = flat != fill_value
    indices = np.flatnonzero(mask).astype(np.int32, copy=False)
    return indices, flat[indices]


@wildcard_router.get(
    "",
    response_model=ListGridsResponse,
    status_code=status.HTTP_200_OK,
    summary="List grids across all domains",
)
async def list_grids_cross_domain(
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
        description="The number of grids to retrieve per page.",
    ),
    sort_by: GridSortField | None = Query(
        None,
        description="The field to sort results by.",
    ),
    sort_order: SortOrder | None = Query(
        None,
        description="The order to sort results (ascending or descending).",
    ),
    source: str | None = Query(
        None,
        description="Filter grids by source name (e.g., 'landfire', '3dep').",
    ),
    product: str | None = Query(
        None,
        description="Filter grids by source product (e.g., 'fbfm40', 'topography'). Requires source filter.",
    ),
    tag: str | None = Query(
        None,
        description="Filter grids that contain this tag.",
    ),
) -> ListGridsResponse:
    """
    # List Grids Across All Domains Endpoint

    Retrieves a paginated list of all grids across all domains belonging to the
    authenticated user.

    ## Query Parameters

    - **page**: (integer, optional) Page number (zero-indexed). Default: 0.
    - **size**: (integer, optional) Items per page (1-1000). Default: 100.
    - **sort_by**: (string, optional) Field to sort by: `created_on`, `modified_on`, `name`.
    - **sort_order**: (string, optional) Sort direction: `ascending`, `descending`.
    - **source**: (string, optional) Filter grids by source name (e.g., `landfire`, `3dep`).
    - **product**: (string, optional) Filter grids by source product (e.g., `fbfm40`, `topography`).
    - **tag**: (string, optional) Filter grids that contain this tag.

    ## Response

    Returns a paginated list of grids with metadata.
    """
    owner_id = request.state.id

    filters = {}
    if source:
        filters["source.name"] = source
    if product:
        filters["source.product"] = product

    array_contains_filters = {}
    if tag:
        array_contains_filters["tags"] = tag

    documents, total_count = await list_documents_async(
        collection=COLLECTION,
        owner_id=owner_id,
        page=page,
        size=size,
        sort_by=sort_by.value if sort_by else None,
        sort_order=sort_order.value if sort_order else None,
        filters=filters if filters else None,
        array_contains_filters=array_contains_filters
        if array_contains_filters
        else None,
    )

    # Convert Firestore documents to Grid models
    grids = [Grid(**doc.to_dict()) for doc in documents]

    return ListGridsResponse(
        grids=grids,
        current_page=page,
        page_size=size,
        total_items=total_count,
    )


@router.get(
    "",
    response_model=ListGridsResponse,
    status_code=status.HTTP_200_OK,
    summary="List all grids",
)
async def list_grids(
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
        description="The number of grids to retrieve per page.",
    ),
    sort_by: GridSortField | None = Query(
        None,
        description="The field to sort results by.",
    ),
    sort_order: SortOrder | None = Query(
        None,
        description="The order to sort results (ascending or descending).",
    ),
    source: str | None = Query(
        None,
        description="Filter grids by source name (e.g., 'landfire', '3dep').",
    ),
    product: str | None = Query(
        None,
        description="Filter grids by source product (e.g., 'fbfm40', 'topography'). Requires source filter.",
    ),
    tag: str | None = Query(
        None,
        description="Filter grids that contain this tag.",
    ),
) -> ListGridsResponse:
    """
    # List Grids Endpoint

    Retrieves a paginated list of all grids within a domain belonging to the
    authenticated user.

    ## Path Parameters

    - **domain_id**: (string) The domain to list grids for.

    ## Query Parameters

    - **page**: (integer, optional) Page number (zero-indexed). Default: 0.
    - **size**: (integer, optional) Items per page (1-1000). Default: 100.
    - **sort_by**: (string, optional) Field to sort by: `created_on`, `modified_on`, `name`.
    - **sort_order**: (string, optional) Sort direction: `ascending`, `descending`.
    - **source**: (string, optional) Filter grids by source name (e.g., `landfire`, `3dep`).
    - **product**: (string, optional) Filter grids by source product (e.g., `fbfm40`, `topography`).
    - **tag**: (string, optional) Filter grids that contain this tag.

    ## Response

    Returns a paginated list of grids with metadata.
    """
    owner_id = request.state.id
    domain_id = domain["id"]

    # Build equality filters — domain_id is always present from URL
    filters = {"domain_id": domain_id}
    if source:
        filters["source.name"] = source
    if product:
        filters["source.product"] = product

    # Build array-contains filters
    array_contains_filters = {}
    if tag:
        array_contains_filters["tags"] = tag

    # Query Firestore
    documents, total_count = await list_documents_async(
        collection=COLLECTION,
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

    # Convert Firestore documents to Grid models
    grids = [Grid(**doc.to_dict()) for doc in documents]

    return ListGridsResponse(
        grids=grids,
        current_page=page,
        page_size=size,
        total_items=total_count,
    )


@router.get(
    "/{grid_id}",
    response_model=Grid,
    status_code=status.HTTP_200_OK,
    summary="Get a grid by ID",
)
async def get_grid(
    request: Request,
    domain: VerifiedDomain,
    grid_id: str,
):
    """
    # Get Grid Endpoint

    Retrieves a specific grid resource by its unique identifier.

    ## Path Parameters

    - **domain_id**: (string) The domain the grid belongs to.
    - **grid_id**: (string) The unique 32-character hex identifier of the grid.

    ## Response

    Returns the grid resource.

    ## Error Responses

    - **404 Not Found**: The grid does not exist or the user does not have access.
    """
    _, snapshot = await get_document_async(
        COLLECTION, grid_id, owner_id=request.state.id, domain_id=domain["id"]
    )
    return Grid(**snapshot.to_dict())


@router.patch(
    "/{grid_id}",
    response_model=Grid,
    status_code=status.HTTP_200_OK,
    summary="Update a grid",
)
async def update_grid(
    request: Request,
    domain: VerifiedDomain,
    grid_id: str,
    body: UpdateGridRequestBody,
):
    """
    # Update Grid Endpoint

    Updates the metadata of an existing grid resource. Only the fields provided
    in the request body will be modified.

    ## Path Parameters

    - **domain_id**: (string) The domain the grid belongs to.
    - **grid_id**: (string) The unique identifier of the grid.

    ## Request Body

    All fields are optional:

    - **name**: (string) New name for the grid.
    - **description**: (string) New description.
    - **tags**: (array of strings) New tags (replaces existing).

    ## What Cannot Be Updated

    The following fields are immutable:

    - **id**, **domain_id**, **source**, **modifications**, **bands**, **georeference**
    - **created_on** (creation timestamp is permanent)

    The **modified_on** field is automatically updated.

    ## Response

    Returns the updated grid resource.
    """
    _, snapshot = await get_document_async(
        COLLECTION, grid_id, owner_id=request.state.id, domain_id=domain["id"]
    )
    grid_data = snapshot.to_dict()

    # Build update data from provided fields only
    update_data = body.model_dump(exclude_none=True)

    # Always update modified_on timestamp
    update_data["modified_on"] = datetime.now()

    # Perform the partial update
    await update_document_async(
        collection=COLLECTION,
        document_id=grid_id,
        data=update_data,
    )

    # Merge updates with existing data to return the full grid
    grid_data.update(update_data)

    return Grid(**grid_data)


@router.delete(
    "/{grid_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a grid",
)
async def delete_grid(
    request: Request,
    domain: VerifiedDomain,
    grid_id: str,
    background_tasks: BackgroundTasks,
):
    """
    # Delete Grid Endpoint

    Permanently deletes a grid resource by its unique identifier.
    This action cannot be undone.

    ## Path Parameters

    - **domain_id**: (string) The domain the grid belongs to.
    - **grid_id**: (string) The unique identifier of the grid.

    ## Response

    Returns HTTP 204 No Content with an empty response body.

    ## Error Responses

    - **404 Not Found**: The grid does not exist or the user does not have access.
    """
    await get_document_async(
        COLLECTION, grid_id, owner_id=request.state.id, domain_id=domain["id"]
    )

    await delete_document_async(
        collection=COLLECTION,
        document_id=grid_id,
    )

    background_tasks.add_task(delete_directory_safe, GRIDS_BUCKET, grid_id)


@router.get(
    "/{grid_id}/chunks/{chunk_index}",
    response_model=GridDataChunkMetadata,
    response_model_exclude_none=True,
    status_code=status.HTTP_200_OK,
    summary="Get chunk metadata",
)
async def get_chunk_metadata(
    request: Request,
    domain: VerifiedDomain,
    grid_id: str,
    chunk_index: int,
):
    """
    # Get Chunk Metadata Endpoint

    Retrieves the shape, pixel offset, and affine transform for a single 2D or
    3D chunk of a completed grid. This is a lightweight call that performs pure
    arithmetic on the grid's stored georeference and chunk shape — no raster
    data is read.

    ## Path Parameters

    - **domain_id**: (string) The domain the grid belongs to.
    - **grid_id**: (string) The unique identifier of the grid.
    - **chunk_index**: (integer) Zero-based flat chunk index. 2D grids use
      y,x order. 3D grids use z,y,x order.

    ## Response

    Returns chunk metadata:

    - **index**: The chunk index.
    - **shape**: 2D `(height, width)` or 3D `(z, height, width)`. Edge chunks
      may be smaller than the grid's chunk shape.
    - **offset**: 2D `(row, column)` or 3D `(z, row, column)` pixel offset of
      the chunk within the full grid.
    - **transform**: Six-element affine transform for the chunk's spatial extent.
    - **z_origin**, **z_resolution**: Present only for 3D grids.

    ## Error Responses

    - **404 Not Found**: The grid does not exist, is not completed, or the user
      does not have access.
    - **422 Unprocessable Entity**: The chunk index is out of range.
    """
    _, snapshot = await get_document_async(
        COLLECTION,
        grid_id,
        owner_id=request.state.id,
        domain_id=domain["id"],
        document_status="completed",
    )
    grid_data = snapshot.to_dict()

    try:
        return compute_chunk_metadata(
            grid_data["georeference"], grid_data["chunk_shape"], chunk_index
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )


@router.get(
    "/{grid_id}/data/{band}/{chunk_index}",
    response_model=GridDataResponse,
    status_code=status.HTTP_200_OK,
    summary="Get band data for a chunk",
)
async def get_grid_data(
    request: Request,
    domain: VerifiedDomain,
    grid_id: str,
    chunk_index: int,
    band: str,
    response_format: GridDataResponseFormat = Query(
        GridDataResponseFormat.json, alias="format", description="Response format."
    ),
    array_format: GridDataArrayFormat = Query(
        GridDataArrayFormat.dense,
        description="Array format: dense values or sparse COO flat indices.",
    ),
    order: GridDataOrder = Query(
        GridDataOrder.C, description="Array memory order for flattening."
    ),
):
    """
    # Get Grid Data Endpoint

    Reads a single chunk of a single band from a completed grid's Zarr store
    on GCS. Returns dense raster values or sparse COO flat indices as either
    JSON or raw binary bytes.

    ## Path Parameters

    - **domain_id**: (string) The domain the grid belongs to.
    - **grid_id**: (string) The unique identifier of the grid.
    - **band**: (string) Band key to read (e.g., `elevation`, `fbfm`). Must match
      a band present in the grid.
    - **chunk_index**: (integer) Zero-based flat chunk index. 2D grids use
      y,x order. 3D grids use z,y,x order.

    ## Query Parameters

    - **format**: (string, optional) Response format: `json` (default) or `binary`.
    - **array_format**: (string, optional) Array format: `dense` (default) or
      `sparse`.
    - **order**: (string, optional) Array memory order for flattening: `C`
      (row-major, default) or `F` (column-major).

    ## Response

    **Dense JSON** (`format=json&array_format=dense`): Returns shape, order,
    and a flat list of values.

    **Dense binary** (`format=binary&array_format=dense`): Returns raw bytes
    with metadata in headers:

    - `X-Data-Shape`: Comma-separated dimensions (e.g., `47,61`).
    - `X-Data-Dtype`: NumPy dtype string (e.g., `float32`).
    - `X-Data-Format`: `dense`.
    - `X-Data-Order`: Memory order used for flattening (`C` or `F`).

    **Sparse JSON** (`format=json&array_format=sparse`): Returns shape, order,
    fill value, flat int32 indices, and values.

    **Sparse binary** (`format=binary&array_format=sparse`): Returns
    `indices.tobytes() + values.tobytes()` with sparse metadata in headers.

    ## Error Responses

    - **404 Not Found**: The grid does not exist, is not completed, or the user
      does not have access.
    - **422 Unprocessable Entity**: The requested band does not exist on this
      grid, or the chunk index is out of range.
    - **413 Payload Too Large**: The requested dense or JSON response is too
      large for the API response limit.
    """
    _, snapshot = await get_document_async(
        COLLECTION,
        grid_id,
        owner_id=request.state.id,
        domain_id=domain["id"],
        document_status="completed",
    )
    grid_data = snapshot.to_dict()

    validate_grid_has_band(grid_data, grid_id, band)

    try:
        meta = compute_chunk_metadata(
            grid_data["georeference"], grid_data["chunk_shape"], chunk_index
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )

    chunk_slices = compute_chunk_slices(meta)

    array = await get_grid_array(grid_id, band)
    data = await array.getitem(chunk_slices)

    shape_header = ",".join(str(s) for s in data.shape)

    if array_format == GridDataArrayFormat.sparse:
        raw_fill = getattr(getattr(array, "metadata", None), "fill_value", None)
        if raw_fill is None:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Sparse grid data requires a zarr fill value.",
            )
        fill_value = raw_fill.item() if hasattr(raw_fill, "item") else raw_fill

        indices, values = _sparse_components(data, fill_value, order)

        if response_format == GridDataResponseFormat.binary:
            raw = indices.tobytes() + values.tobytes()
            _check_size(len(raw), MAX_BINARY_BYTES, "Sparse binary grid data")
            return Response(
                content=raw,
                media_type="application/octet-stream",
                headers={
                    "X-Data-Shape": shape_header,
                    "X-Data-Order": order.value,
                    "X-Data-Format": "sparse",
                    "X-Data-Fill-Value": str(fill_value),
                    "X-Data-NNZ": str(len(indices)),
                    "X-Data-Index-Dtype": str(indices.dtype),
                    "X-Data-Value-Dtype": str(values.dtype),
                },
            )

        return GridDataResponse(
            shape=list(data.shape),
            order=order.value,
            data=SparseGridData(
                format="sparse",
                fill_value=fill_value,
                indices=indices.tolist(),
                values=values.tolist(),
            ),
        )

    if response_format == GridDataResponseFormat.binary:
        raw = data.ravel(order=order.value).tobytes()
        _check_size(len(raw), MAX_BINARY_BYTES, "Dense binary grid data")
        return Response(
            content=raw,
            media_type="application/octet-stream",
            headers={
                "X-Data-Shape": shape_header,
                "X-Data-Dtype": str(data.dtype),
                "X-Data-Order": order.value,
                "X-Data-Format": "dense",
            },
        )

    _check_size(data.size, MAX_JSON_SCALARS, "Dense JSON grid data")
    return GridDataResponse(
        shape=list(data.shape),
        order=order.value,
        data=DenseGridData(
            format="dense",
            values=data.ravel(order=order.value).tolist(),
        ),
    )


router.include_router(
    grid_exports_router, prefix="/{grid_id}/exports", tags=["Grids - Exports"]
)
router.include_router(fbfm40_router, prefix="/fbfm40", tags=["Grids - FBFM40"])
router.include_router(
    topography_router, prefix="/topography", tags=["Grids - Topography"]
)
router.include_router(pim_router, prefix="/pim", tags=["Grids - PIM"])
router.include_router(chm_router, prefix="/chm", tags=["Grids - CHM"])
router.include_router(lookup_router, prefix="/lookup", tags=["Grids - Lookup"])
router.include_router(resample_router, prefix="/resample", tags=["Grids - Resample"])
router.include_router(tree_router, prefix="/tree")
router.include_router(uniform_router, prefix="/uniform", tags=["Grids - Uniform"])
