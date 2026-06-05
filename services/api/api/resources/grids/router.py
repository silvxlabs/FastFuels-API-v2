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
from api.resources.grids.canopy.router import router as canopy_router
from api.resources.grids.exports.quicfire.router import (
    router as quicfire_export_router,
)
from api.resources.grids.exports.router import router as grid_exports_router
from api.resources.grids.fbfm40.router import router as fbfm40_router
from api.resources.grids.fccs.router import router as fccs_router
from api.resources.grids.lookup.router import router as lookup_router
from api.resources.grids.pim.router import router as pim_router
from api.resources.grids.rasterize.router import router as rasterize_router
from api.resources.grids.resample.router import router as resample_router
from api.resources.grids.schema import (
    DenseGridData,
    Grid,
    GridDataArrayFormat,
    GridDataChunkMetadata,
    GridDataOrder,
    GridDataResponse,
    GridSortField,
    ListGridsResponse,
    SparseGridData,
    UpdateGridRequestBody,
)
from api.resources.grids.topography.router import router as topography_router
from api.resources.grids.uniform.router import router as uniform_router
from api.resources.grids.upload.router import router as upload_router
from api.resources.grids.utils import (
    compute_chunk_metadata,
    compute_chunk_slices,
    validate_grid_has_band,
)
from api.resources.grids.voxelize.router import router as voxelize_router
from api.schema import SortOrder
from lib.config import GRIDS_BUCKET, GRIDS_COLLECTION

router = APIRouter()
wildcard_router = APIRouter()

COLLECTION = GRIDS_COLLECTION
MAX_BINARY_BYTES = 30 * 1024 * 1024
MAX_JSON_SCALARS = 1_000_000
MAX_SPARSE_INDEX = int(np.iinfo(np.int32).max)


def _check_size(actual: int, limit: int, what: str, hint: str) -> None:
    if actual > limit:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"{what} ({actual}) exceeds API response limit ({limit}). {hint}",
        )


def _sparse_components(
    data: np.ndarray,
    fill_value: float | int | None,
    order: GridDataOrder,
) -> tuple[np.ndarray, np.ndarray]:
    flat = data.ravel(order=order.value)
    if flat.size > MAX_SPARSE_INDEX:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=(
                f"Chunk size ({flat.size}) exceeds int32 sparse-index limit "
                f"({MAX_SPARSE_INDEX}). Request a smaller chunk."
            ),
        )
    if fill_value is None:
        return np.arange(flat.size, dtype=np.int32), flat
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
    - **checksum** (changes only when the grid's content is rebuilt, never via
      metadata updates)

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
    3D chunk of a completed grid. This is a lightweight call — no raster data
    is read.

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
    chunks = grid_data.get("chunks")
    if not chunks:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"Grid {grid_id} has no chunk layout.",
        )

    try:
        return compute_chunk_metadata(
            grid_data["georeference"], chunks["shape"], chunk_index
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        )


def _chunk_metadata_headers(meta: GridDataChunkMetadata) -> dict[str, str]:
    headers = {
        "X-Data-Offset": ",".join(str(v) for v in meta.offset),
        "X-Data-Transform": ",".join(str(v) for v in meta.transform),
    }
    if meta.z_origin is not None:
        headers["X-Data-Z-Origin"] = str(meta.z_origin)
    if meta.z_resolution is not None:
        headers["X-Data-Z-Resolution"] = str(meta.z_resolution)
    return headers


async def _read_grid_chunk(
    request: Request,
    domain: VerifiedDomain,
    grid_id: str,
    band: str,
    chunk_index: int,
):
    _, snapshot = await get_document_async(
        COLLECTION,
        grid_id,
        owner_id=request.state.id,
        domain_id=domain["id"],
        document_status="completed",
    )
    grid_data = snapshot.to_dict()

    validate_grid_has_band(grid_data, grid_id, band)
    chunks = grid_data.get("chunks")
    if not chunks:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"Grid {grid_id} has no chunk layout.",
        )

    try:
        meta = compute_chunk_metadata(
            grid_data["georeference"], chunks["shape"], chunk_index
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        )

    array = await get_grid_array(grid_id, band)
    data = await array.getitem(compute_chunk_slices(meta))
    return data, array, meta


@router.get(
    "/{grid_id}/data/{band}/{chunk_index}",
    response_model=GridDataResponse,
    response_model_exclude_none=True,
    status_code=status.HTTP_200_OK,
    summary="Get band data for a chunk (JSON)",
)
async def get_grid_data_json(
    request: Request,
    domain: VerifiedDomain,
    grid_id: str,
    chunk_index: int,
    band: str,
    array_format: GridDataArrayFormat = Query(
        GridDataArrayFormat.dense,
        description="Array format: dense values or sparse COO flat indices.",
    ),
    order: GridDataOrder = Query(
        GridDataOrder.C, description="Array memory order for flattening."
    ),
):
    """
    # Get Grid Data (JSON)

    Returns the values of a single band within a single chunk of a completed
    grid as a JSON payload — either dense values or a sparse representation,
    selected via `array_format`.

    For raw bytes (smaller, faster to parse), use the `/binary` variant of
    this endpoint.

    ## Path Parameters

    - **domain_id**: The domain the grid belongs to.
    - **grid_id**: The grid identifier.
    - **band**: Band key to read (must be present in the grid's `bands`).
    - **chunk_index**: Zero-based flat chunk index. 2D grids index in (y, x);
      3D grids index in (z, y, x). The chunk's shape, offset, and affine
      transform are returned alongside the data — no separate metadata call
      is required. (`GET …/chunks/{chunk_index}` is still available for
      clients that want to lay out chunks before fetching data.)

    ## Query Parameters

    - **array_format**: `dense` (default) or `sparse`. Sparse compresses out
      cells equal to the band's fill value, returning only the non-fill
      entries.
    - **order**: Flattening order — `C` (row-major, default) or `F`
      (column-major).

    ## Response

    Both variants share `shape` (chunk dimensions), `order` (flattening
    order used for the values), and `metadata` (the chunk's index, shape,
    offset, and affine transform — and `z_origin`/`z_resolution` for 3D
    grids). The CRS does not vary per chunk; read it from the grid's
    `georeference.crs`.

    **Dense** (`array_format=dense`): `data.format = "dense"`, `data.values`
    is a flat list of all cells.

    **Sparse** (`array_format=sparse`): `data.format = "sparse"`,
    `data.indices` are flat positions of non-fill cells, `data.values` are
    their values. `data.fill_value` is the band's fill value, or `null` if
    the band does not define one — in which case every cell is listed and no
    compression has been applied.

    ## Errors

    - **404**: Grid not found, not completed, or not accessible.
    - **422**: Band does not exist on this grid, or chunk index out of range.
    - **413**: Response would exceed the size limit. Try `array_format=sparse`,
      the `/binary` variant, or a smaller chunk.
    """
    data, array, meta = await _read_grid_chunk(
        request, domain, grid_id, band, chunk_index
    )

    if array_format == GridDataArrayFormat.sparse:
        raw_fill = getattr(getattr(array, "metadata", None), "fill_value", None)
        fill_value = raw_fill.item() if hasattr(raw_fill, "item") else raw_fill
        indices, values = _sparse_components(data, fill_value, order)
        _check_size(
            2 * len(indices),
            MAX_JSON_SCALARS,
            "Sparse JSON grid data",
            "Request /binary or a smaller chunk.",
        )
        return GridDataResponse(
            shape=list(data.shape),
            order=order.value,
            metadata=meta,
            data=SparseGridData(
                format="sparse",
                fill_value=fill_value,
                indices=indices.tolist(),
                values=values.tolist(),
            ),
        )

    _check_size(
        data.size,
        MAX_JSON_SCALARS,
        "Dense JSON grid data",
        "Request array_format=sparse, /binary, or a smaller chunk.",
    )
    return GridDataResponse(
        shape=list(data.shape),
        order=order.value,
        metadata=meta,
        data=DenseGridData(
            format="dense",
            values=data.ravel(order=order.value).tolist(),
        ),
    )


@router.get(
    "/{grid_id}/data/{band}/{chunk_index}/binary",
    status_code=status.HTTP_200_OK,
    summary="Get band data for a chunk (binary)",
    responses={200: {"content": {"application/octet-stream": {}}}},
)
async def get_grid_data_binary(
    request: Request,
    domain: VerifiedDomain,
    grid_id: str,
    chunk_index: int,
    band: str,
    array_format: GridDataArrayFormat = Query(
        GridDataArrayFormat.dense,
        description="Array format: dense values or sparse COO flat indices.",
    ),
    order: GridDataOrder = Query(
        GridDataOrder.C, description="Array memory order for flattening."
    ),
):
    """
    # Get Grid Data (binary)

    Returns the values of a single band within a single chunk of a completed
    grid as raw bytes, with shape and type metadata in `X-Data-*` response
    headers. Use this when you need the most compact wire format and intend
    to deserialize into a typed array on the client.

    For a structured JSON payload, use the JSON variant of this endpoint
    (drop the trailing `/binary`).

    ## Path Parameters

    - **domain_id**: The domain the grid belongs to.
    - **grid_id**: The grid identifier.
    - **band**: Band key to read (must be present in the grid's `bands`).
    - **chunk_index**: Zero-based flat chunk index. 2D grids index in (y, x);
      3D grids index in (z, y, x). The chunk's shape, offset, and affine
      transform are returned alongside the data — no separate metadata call
      is required. (`GET …/chunks/{chunk_index}` is still available for
      clients that want to lay out chunks before fetching data.)

    ## Query Parameters

    - **array_format**: `dense` (default) or `sparse`. Sparse compresses out
      cells equal to the band's fill value, returning only the non-fill
      entries.
    - **order**: Flattening order — `C` (row-major, default) or `F`
      (column-major).

    ## Response

    All variants return `application/octet-stream` with these common headers:

    - `X-Data-Shape`: comma-separated chunk dimensions (e.g., `47,61`).
    - `X-Data-Order`: flattening order used (`C` or `F`).
    - `X-Data-Format`: `dense` or `sparse`.
    - `X-Data-Offset`: comma-separated pixel offset of this chunk within
      the full grid (2D: `row,col`; 3D: `z,row,col`).
    - `X-Data-Transform`: comma-separated six-element affine transform for
      the chunk's spatial extent.
    - `X-Data-Z-Origin`, `X-Data-Z-Resolution`: present only for 3D grids.

    **Dense** (`array_format=dense`): body is the flattened cells as raw
    bytes.

    - `X-Data-Dtype`: numeric type of the cells (e.g., `float32`).

    **Sparse** (`array_format=sparse`): body is the index array bytes
    immediately followed by the value array bytes.

    - `X-Data-NNZ`: number of non-fill entries (length of both arrays).
    - `X-Data-Index-Dtype`: numeric type of the index array (`int32`).
    - `X-Data-Value-Dtype`: numeric type of the value array.
    - `X-Data-Fill-Value`: the band's fill value (stringified). Omitted when
      the band does not define a fill value, in which case every cell is
      listed and no compression has been applied.

    Slice the response body at `NNZ * sizeof(index_dtype)` to separate
    indices from values.

    ## Errors

    - **404**: Grid not found, not completed, or not accessible.
    - **422**: Band does not exist on this grid, or chunk index out of range.
    - **413**: Response would exceed the size limit. Try `array_format=sparse`
      or a smaller chunk.
    """
    data, array, meta = await _read_grid_chunk(
        request, domain, grid_id, band, chunk_index
    )
    shape_header = ",".join(str(s) for s in data.shape)
    chunk_headers = _chunk_metadata_headers(meta)

    if array_format == GridDataArrayFormat.sparse:
        raw_fill = getattr(getattr(array, "metadata", None), "fill_value", None)
        fill_value = raw_fill.item() if hasattr(raw_fill, "item") else raw_fill
        indices, values = _sparse_components(data, fill_value, order)
        raw = indices.tobytes() + values.tobytes()
        _check_size(
            len(raw),
            MAX_BINARY_BYTES,
            "Sparse binary grid data",
            "Request a smaller chunk.",
        )
        headers = {
            "X-Data-Shape": shape_header,
            "X-Data-Order": order.value,
            "X-Data-Format": "sparse",
            "X-Data-NNZ": str(len(indices)),
            "X-Data-Index-Dtype": str(indices.dtype),
            "X-Data-Value-Dtype": str(values.dtype),
            **chunk_headers,
        }
        if fill_value is not None:
            headers["X-Data-Fill-Value"] = str(fill_value)
        return Response(
            content=raw,
            media_type="application/octet-stream",
            headers=headers,
        )

    raw = data.ravel(order=order.value).tobytes()
    _check_size(
        len(raw),
        MAX_BINARY_BYTES,
        "Dense binary grid data",
        "Request array_format=sparse or a smaller chunk.",
    )
    return Response(
        content=raw,
        media_type="application/octet-stream",
        headers={
            "X-Data-Shape": shape_header,
            "X-Data-Dtype": str(data.dtype),
            "X-Data-Order": order.value,
            "X-Data-Format": "dense",
            **chunk_headers,
        },
    )


router.include_router(
    grid_exports_router, prefix="/{grid_id}/exports", tags=["Grids - Exports"]
)
router.include_router(
    quicfire_export_router,
    prefix="/exports/quicfire",
    tags=["Grids - Exports"],
)
router.include_router(fbfm40_router, prefix="/fbfm40", tags=["Grids - FBFM40"])
router.include_router(fccs_router, prefix="/fccs", tags=["Grids - FCCS"])
router.include_router(
    topography_router, prefix="/topography", tags=["Grids - Topography"]
)
router.include_router(pim_router, prefix="/pim", tags=["Grids - PIM"])
router.include_router(canopy_router, prefix="/canopy", tags=["Grids - Canopy"])
router.include_router(lookup_router, prefix="/lookup", tags=["Grids - Lookup"])
router.include_router(rasterize_router, prefix="/rasterize", tags=["Grids - Rasterize"])
router.include_router(resample_router, prefix="/resample", tags=["Grids - Resample"])
router.include_router(voxelize_router, prefix="/voxelize")
router.include_router(uniform_router, prefix="/uniform", tags=["Grids - Uniform"])
router.include_router(upload_router, prefix="/upload", tags=["Grids - Upload"])
