"""
api/v2/resources/point_clouds/router.py

Routes for managing point-cloud resources and reading their stored point data.

Source-specific routers create point clouds from uploads or USGS 3DEP. This
module provides the shared list / get / update / delete surface, plus indexed
tile reads for completed point clouds. Point parsing and reprojection remain in
worker services; the API read path uses only the GDAL-free Parquet reader.
"""

from datetime import datetime

from fastapi import APIRouter, Path, Query, Request, Response, status

from api.db.documents import (
    delete_document_async,
    get_document_async,
    list_documents_async,
    update_document_async,
)
from api.dependencies import VerifiedDomain
from api.resources.point_clouds import utils
from api.resources.point_clouds.schema import (
    ListPointCloudsResponse,
    PointCloud,
    PointCloudDataMetadata,
    PointCloudSortField,
    PointCloudTileDataResponse,
    PointCloudType,
    UpdatePointCloudRequestBody,
)
from api.resources.point_clouds.threedep.router import router as threedep_router
from api.resources.point_clouds.upload.router import router as upload_router
from api.schema import SortOrder
from lib.config import POINT_CLOUDS_COLLECTION

router = APIRouter()
wildcard_router = APIRouter()

COLLECTION = POINT_CLOUDS_COLLECTION

# Literal creation paths must be registered before /{point_cloud_id}.
router.include_router(upload_router, prefix="/upload", tags=["Point Clouds - Upload"])
router.include_router(threedep_router, prefix="/3dep", tags=["Point Clouds - 3DEP"])


@wildcard_router.get(
    "",
    response_model=ListPointCloudsResponse,
    status_code=status.HTTP_200_OK,
    summary="List point clouds across all domains",
)
async def list_point_clouds_cross_domain(
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
        description="The number of point clouds to retrieve per page.",
    ),
    sort_by: PointCloudSortField | None = Query(
        None,
        description="The field to sort results by.",
    ),
    sort_order: SortOrder | None = Query(
        None,
        description="The order to sort results (ascending or descending).",
    ),
    type: PointCloudType | None = Query(
        None,
        description="Filter point clouds by acquisition type (`als` or `tls`).",
    ),
    source: str | None = Query(
        None,
        description="Filter point clouds by source name (for example, `3dep`).",
    ),
    tag: str | None = Query(
        None,
        description="Filter point clouds that contain this tag.",
    ),
) -> ListPointCloudsResponse:
    """
    # List Point Clouds Across All Domains

    Returns a paginated list of every point cloud owned by the authenticated
    caller, regardless of which domain contains it. Use the domain-scoped list
    endpoint when the caller already knows the domain.

    ## Query Parameters

    - **page**: Zero-indexed page number. Defaults to `0`.
    - **size**: Results per page, from `1` through `1000`. Defaults to `100`.
    - **sort_by**: Sort by `created_on`, `modified_on`, or `name`.
    - **sort_order**: Sort in `ascending` or `descending` order.
    - **type**: Keep only airborne (`als`) or terrestrial (`tls`) clouds.
    - **source**: Keep only clouds from a source such as `3dep` or `upload`.
    - **tag**: Keep only clouds whose `tags` array contains this value.

    ## Response

    A standard paginated response. `point_clouds` contains the resources on the
    requested page; `current_page`, `page_size`, and `total_items` describe the
    page and the complete filtered result set.
    """
    filters = {}
    if type is not None:
        filters["type"] = type.value
    if source is not None:
        filters["source.name"] = source

    documents, total_count = await list_documents_async(
        collection=COLLECTION,
        owner_id=request.state.id,
        page=page,
        size=size,
        sort_by=sort_by.value if sort_by else None,
        sort_order=sort_order.value if sort_order else None,
        filters=filters or None,
        array_contains_filters={"tags": tag} if tag else None,
    )
    return ListPointCloudsResponse(
        point_clouds=[PointCloud(**doc.to_dict()) for doc in documents],
        current_page=page,
        page_size=size,
        total_items=total_count,
    )


@router.get(
    "",
    response_model=ListPointCloudsResponse,
    status_code=status.HTTP_200_OK,
    summary="List point clouds in a domain",
)
async def list_point_clouds(
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
        description="The number of point clouds to retrieve per page.",
    ),
    sort_by: PointCloudSortField | None = Query(
        None,
        description="The field to sort results by.",
    ),
    sort_order: SortOrder | None = Query(
        None,
        description="The order to sort results (ascending or descending).",
    ),
    type: PointCloudType | None = Query(
        None,
        description="Filter point clouds by acquisition type (`als` or `tls`).",
    ),
    source: str | None = Query(
        None,
        description="Filter point clouds by source name (for example, `3dep`).",
    ),
    tag: str | None = Query(
        None,
        description="Filter point clouds that contain this tag.",
    ),
) -> ListPointCloudsResponse:
    """
    # List Point Clouds in a Domain

    Returns a paginated list of point clouds owned by the authenticated caller
    within one domain.

    ## Path Parameters

    - **domain_id**: Domain whose point clouds should be listed.

    ## Query Parameters

    - **page**: Zero-indexed page number. Defaults to `0`.
    - **size**: Results per page, from `1` through `1000`. Defaults to `100`.
    - **sort_by**: Sort by `created_on`, `modified_on`, or `name`.
    - **sort_order**: Sort in `ascending` or `descending` order.
    - **type**: Keep only airborne (`als`) or terrestrial (`tls`) clouds.
    - **source**: Keep only clouds from a source such as `3dep` or `upload`.
    - **tag**: Keep only clouds whose `tags` array contains this value.

    ## Response

    A standard paginated response. `point_clouds` contains the resources on the
    requested page; `current_page`, `page_size`, and `total_items` describe the
    page and the complete filtered result set.

    ## Error Responses

    - **404 Not Found**: The domain does not exist or is not accessible to the
      caller.
    """
    filters = {"domain_id": domain["id"]}
    if type is not None:
        filters["type"] = type.value
    if source is not None:
        filters["source.name"] = source

    documents, total_count = await list_documents_async(
        collection=COLLECTION,
        owner_id=request.state.id,
        page=page,
        size=size,
        sort_by=sort_by.value if sort_by else None,
        sort_order=sort_order.value if sort_order else None,
        filters=filters,
        array_contains_filters={"tags": tag} if tag else None,
    )
    return ListPointCloudsResponse(
        point_clouds=[PointCloud(**doc.to_dict()) for doc in documents],
        current_page=page,
        page_size=size,
        total_items=total_count,
    )


@router.get(
    "/{point_cloud_id}/data/metadata",
    response_model=PointCloudDataMetadata,
    status_code=status.HTTP_200_OK,
    summary="Get point-cloud tile metadata",
)
async def get_point_cloud_data_metadata(
    request: Request,
    domain: VerifiedDomain,
    point_cloud_id: str,
) -> PointCloudDataMetadata:
    """
    # Get Point-Cloud Data Metadata

    Returns the public read index for a completed point cloud without returning
    any point values. Call this endpoint first to discover the occupied tiles,
    stored columns and dtypes, coordinate encoding, and the number of points
    each level of detail (LOD) would return.

    A typical client workflow is:

    1. Read this metadata once.
    2. Select one of the entries in `tiles`.
    3. Choose an LOD whose cumulative point count fits the client workload.
    4. Request that tile from the JSON or binary data endpoint.

    ## Path Parameters

    - **domain_id**: Domain the point cloud belongs to.
    - **point_cloud_id**: Unique point-cloud identifier.

    ## Response

    - **tile_m**: Width and height of each tile in the units of `crs`.
    - **lod_levels**: Number of available cumulative LOD selections. The
      current format has six levels, numbered `0` through `5`.
    - **crs**: Coordinate reference system for the decoded point coordinates
      and all reported bounds.
    - **bounds**: Horizontal point-cloud extent as
      `[min_x, min_y, max_x, max_y]`.
    - **scales** and **offsets**: Three values in X/Y/Z order used to decode
      stored integer coordinates:

      `coordinate = stored_integer * scale + offset`

    - **columns**: Stored public column names mapped to their NumPy-compatible
      dtypes. `X`, `Y`, and `Z` are encoded integers; `classification` contains
      ASPRS classification codes. Other columns, such as `intensity`, are
      source-dependent.
    - **tiles**: Occupied tiles only. Empty positions in the tiling are omitted.
      Each entry contains its integer `tile_x` and `tile_y`, horizontal
      `bounds`, and `points_by_lod`.

    `points_by_lod[k]` is the number of rows returned by `lod=k` before an
    optional classification filter. Counts are cumulative: LOD 0 is the
    coarsest sample, each higher value includes every preceding level, and the
    final value is the complete tile. A sparse boundary tile may legitimately
    repeat counts across several LODs when it contains too few points to
    populate every level.

    Internal GCS object names, Parquet part paths, row-group offsets, and byte
    ranges are deliberately not part of the API response. The server uses that
    storage index to satisfy tile requests; clients only need this stable tile
    catalogue.

    ## Error Responses

    - **404 Not Found**: The point cloud does not exist, belongs to another
      domain, or is not accessible to the caller.
    - **422 Unprocessable Entity**: The point cloud is not completed, its
      resource metadata does not match its stored data, or the stored Parquet
      index is missing or malformed. Re-create the point cloud before retrying.
    """
    resource, storage = await utils.load_point_cloud(
        request.state.id, domain["id"], point_cloud_id
    )
    return utils.metadata_response(resource, storage)


@router.get(
    "/{point_cloud_id}/data/{tile_x}/{tile_y}",
    response_model=PointCloudTileDataResponse,
    status_code=status.HTTP_200_OK,
    summary="Get point-cloud tile data (JSON)",
)
async def get_point_cloud_data_json(
    request: Request,
    domain: VerifiedDomain,
    point_cloud_id: str,
    tile_x: int = Path(
        ...,
        description="Horizontal tile index from `GET /data/metadata`.",
    ),
    tile_y: int = Path(
        ...,
        description="Vertical tile index from `GET /data/metadata`.",
    ),
    lod: int | None = Query(
        None,
        ge=0,
        description=(
            "Inclusive LOD ceiling. Omit to read the complete tile. Valid "
            "values are `0` through `lod_levels - 1` from `/data/metadata`."
        ),
    ),
    classes: str | None = Query(
        None,
        description=(
            "Comma-separated ASPRS classification codes to retain, such as "
            "`2,5`. Omit to retain every class."
        ),
    ),
    columns: str | None = Query(
        None,
        description=(
            "Comma-separated stored columns to return, such as `X,Y,Z`. "
            "Omit to return every public stored column."
        ),
    ),
) -> PointCloudTileDataResponse:
    """
    # Get Point-Cloud Tile Data as JSON

    Returns selected columns from one occupied point-cloud tile as columnar
    JSON. Use this representation for inspection, small previews, and clients
    that do not need the more compact binary response.

    Call `GET /domains/{domain_id}/pointclouds/{point_cloud_id}/data/metadata`
    first. Its `tiles` array supplies valid tile coordinates and exact
    cumulative point counts, while its `columns` object supplies the available
    names and dtypes.

    ## Path Parameters

    - **domain_id**: Domain the point cloud belongs to.
    - **point_cloud_id**: Unique point-cloud identifier.
    - **tile_x** and **tile_y**: Integer coordinates of an occupied tile from
      the metadata response. They are indices in the point cloud's own tiling,
      not projected map coordinates.

    ## Query Parameters

    - **lod**: Inclusive LOD ceiling. `lod=0` returns the coarsest sample;
      `lod=k` returns levels `0` through `k`; omitting it returns the complete
      tile. Valid values are `0` through `lod_levels - 1` from the metadata
      response.
    - **classes**: Optional comma-separated ASPRS classification filter, for
      example `?classes=2,5`. Duplicate values are ignored. Omit it to retain
      every classification.
    - **columns**: Optional comma-separated column projection in the desired
      response order, for example `?columns=X,Y,Z`. Omit it to return all
      public stored columns.

    ## Response

    The response is columnar: arrays in `data` have equal length and values at
    the same array index describe the same point.

    For example, this is the complete response for tile `(-1, 0)` from the
    `static-test-blackfoot-3dep` point cloud with
    `?lod=5&classes=1,2&columns=X,Y,Z,classification`:

    ```json
    {
      "tile_x": -1,
      "tile_y": 0,
      "bounds": [
        293711.08485993545,
        5198981.669894749,
        294094.99218481116,
        5199365.577219625
      ],
      "lod": 5,
      "classes": [1, 2],
      "scales": [0.001, 0.001, 0.001],
      "offsets": [294094.0, 5198981.0, 0.0],
      "columns": {
        "X": "int32",
        "Y": "int32",
        "Z": "int32",
        "classification": "uint8"
      },
      "data": {
        "X": [992, 992],
        "Y": [346497, 64586],
        "Z": [1077190, 1051360],
        "classification": [1, 2]
      }
    }
    ```

    `X`, `Y`, and `Z` remain stored integers so the response is exact and does
    not expand them to float64. Decode coordinate axis `i` with:

    `coordinate = stored_integer * scales[i] + offsets[i]`

    The echoed `lod`, `classes`, `columns`, and tile bounds make the response
    self-describing. When `classes` was omitted, the response field is `null`.

    JSON responses are capped at 1,000,000 numeric values, calculated as rows
    multiplied by selected columns. If a request is too large, lower `lod`,
    select fewer classes or columns, or use the `/binary` endpoint.

    ## Error Responses

    - **404 Not Found**: The point cloud does not exist, belongs to another
      domain, or is not accessible to the caller.
    - **413 Content Too Large**: The selected rows and columns exceed the JSON
      response limit.
    - **422 Unprocessable Entity**: The cloud is not completed; the tile, LOD,
      class, or column selection is invalid; or stored data is unreadable or
      inconsistent with its index.
    """
    selection = await utils.select_tile(
        request.state.id,
        domain["id"],
        point_cloud_id,
        tile_x,
        tile_y,
        lod,
        classes,
        columns,
    )
    utils.check_json_size(selection)
    data = await utils.read_tile(selection)
    utils.check_json_size(selection, data.num_rows)
    return utils.json_response(selection, data)


class PointCloudBinaryResponse(Response):
    """Response class carrying the binary media type, so OpenAPI documents
    ``application/octet-stream`` rather than the default ``application/json``.
    """

    media_type = "application/octet-stream"


@router.get(
    "/{point_cloud_id}/data/{tile_x}/{tile_y}/binary",
    status_code=status.HTTP_200_OK,
    summary="Get point-cloud tile data (binary)",
    response_class=PointCloudBinaryResponse,
)
async def get_point_cloud_data_binary(
    request: Request,
    domain: VerifiedDomain,
    point_cloud_id: str,
    tile_x: int = Path(
        ...,
        description="Horizontal tile index from `GET /data/metadata`.",
    ),
    tile_y: int = Path(
        ...,
        description="Vertical tile index from `GET /data/metadata`.",
    ),
    lod: int | None = Query(
        None,
        ge=0,
        description=(
            "Inclusive LOD ceiling. Omit to read the complete tile. Valid "
            "values are `0` through `lod_levels - 1` from `/data/metadata`."
        ),
    ),
    classes: str | None = Query(
        None,
        description=(
            "Comma-separated ASPRS classification codes to retain, such as "
            "`2,5`. Omit to retain every class."
        ),
    ),
    columns: str | None = Query(
        None,
        description=(
            "Comma-separated stored columns in the desired binary block order, "
            "such as `X,Y,Z`. Omit to return every public stored column."
        ),
    ),
) -> PointCloudBinaryResponse:
    """
    # Get Point-Cloud Tile Data as Binary

    Returns selected columns from one occupied point-cloud tile as raw
    little-endian typed arrays. This is the compact counterpart to the JSON
    endpoint and is intended for clients that can construct NumPy, JavaScript,
    Rust, or C/C++ typed arrays directly from response bytes.

    Call `GET /domains/{domain_id}/pointclouds/{point_cloud_id}/data/metadata`
    first to discover valid tiles, cumulative LOD costs, available columns, and
    coordinate scaling.

    ## Path Parameters

    - **domain_id**: Domain the point cloud belongs to.
    - **point_cloud_id**: Unique point-cloud identifier.
    - **tile_x** and **tile_y**: Integer coordinates of an occupied tile from
      the metadata response.

    ## Query Parameters

    - **lod**: Inclusive LOD ceiling. `lod=0` returns the coarsest sample;
      `lod=k` returns levels `0` through `k`; omitting it returns the complete
      tile.
    - **classes**: Optional comma-separated ASPRS classification filter, for
      example `?classes=2,5`. Omit it for all classes.
    - **columns**: Optional comma-separated column projection in the exact
      desired block order, for example `?columns=X,Y,Z`. Omit it for all public
      stored columns.

    ## Response Body

    The body is one contiguous column block after another in `X-Data-Columns`
    order. Every block contains `X-Data-Count` values and uses the corresponding
    dtype in `X-Data-Dtypes`. All multi-byte values are little-endian.

    For example, these headers:

    ```text
    X-Data-Columns: X,Z,classification
    X-Data-Dtypes: int32,int32,uint8
    X-Data-Count: 1000
    ```

    describe `4000` bytes of X values, followed by `4000` bytes of Z values,
    followed by `1000` classification bytes. In general, each block occupies:

    `X-Data-Count * sizeof(corresponding dtype)`

    Slice the body at the cumulative block sizes. Values with the same position
    within each block describe the same point.

    ## Response Headers

    - **X-Data-Columns**: Comma-separated column block order.
    - **X-Data-Dtypes**: Comma-separated NumPy dtype for each column block.
    - **X-Data-Count**: Number of values in every block.
    - **X-Data-Tile**: Requested tile as `tile_x,tile_y`.
    - **X-Data-Bounds**: Horizontal tile bounds as
      `min_x,min_y,max_x,max_y`.
    - **X-Data-LOD**: Inclusive LOD ceiling used for the response.
    - **X-Data-Classes**: Comma-separated selected ASPRS classes, or `all` when
      no class filter was supplied.
    - **X-Data-Scales** and **X-Data-Offsets**: X/Y/Z coordinate encoding.
      Decode coordinate axis `i` with
      `stored_integer * scale[i] + offset[i]`.

    These headers are exposed through CORS, so browser JavaScript can read them.

    Binary responses are capped at 30 MiB. If a request is too large, lower
    `lod` or select fewer classes or columns.

    ## Error Responses

    - **404 Not Found**: The point cloud does not exist, belongs to another
      domain, or is not accessible to the caller.
    - **413 Content Too Large**: The selected binary column blocks exceed the
      30 MiB response limit.
    - **422 Unprocessable Entity**: The cloud is not completed; the tile, LOD,
      class, or column selection is invalid; or stored data is unreadable or
      inconsistent with its index.
    """
    selection = await utils.select_tile(
        request.state.id,
        domain["id"],
        point_cloud_id,
        tile_x,
        tile_y,
        lod,
        classes,
        columns,
    )
    utils.check_binary_size(selection)
    data = await utils.read_tile(selection)
    utils.check_binary_size(selection, data.num_rows)
    return PointCloudBinaryResponse(
        content=utils.binary_payload(selection, data),
        headers=utils.binary_headers(selection, data.num_rows),
    )


@router.get(
    "/{point_cloud_id}",
    response_model=PointCloud,
    status_code=status.HTTP_200_OK,
    summary="Get a point cloud by ID",
)
async def get_point_cloud(
    request: Request,
    domain: VerifiedDomain,
    point_cloud_id: str,
) -> PointCloud:
    """
    # Get a Point Cloud

    Returns one point-cloud resource by ID. This endpoint returns resource
    metadata and processing state; it does not return the individual points.
    Use `/data/metadata` and the tile data endpoints for point values after the
    resource reaches `status="completed"`.

    ## Path Parameters

    - **domain_id**: Domain the point cloud belongs to.
    - **point_cloud_id**: Unique point-cloud identifier.

    ## Response

    The complete point-cloud resource, including its acquisition `type`,
    `source` provenance, processing `status`, georeference, content summary,
    checksum, and user-editable metadata. Derived fields such as `georeference`
    and `summary` are null until processing completes.

    ## Error Responses

    - **404 Not Found**: The point cloud does not exist, belongs to another
      domain, or is not accessible to the caller.
    """
    _, snapshot = await get_document_async(
        COLLECTION,
        point_cloud_id,
        owner_id=request.state.id,
        domain_id=domain["id"],
    )
    return PointCloud(**snapshot.to_dict())


@router.patch(
    "/{point_cloud_id}",
    response_model=PointCloud,
    status_code=status.HTTP_200_OK,
    summary="Update a point cloud",
)
async def update_point_cloud(
    request: Request,
    domain: VerifiedDomain,
    point_cloud_id: str,
    body: UpdatePointCloudRequestBody,
) -> PointCloud:
    """
    # Update a Point Cloud

    Updates the user-editable metadata of an existing point cloud. Only fields
    present in the request body are changed.

    ## Path Parameters

    - **domain_id**: Domain the point cloud belongs to.
    - **point_cloud_id**: Unique point-cloud identifier.

    ## Request Body

    Every field is optional:

    - **name**: New human-readable name.
    - **description**: New free-text description.
    - **tags**: Replacement tag list. Supplying an empty list removes all tags.

    Omitted fields retain their current values.

    ## Immutable Fields

    This endpoint cannot alter stored point data or derived/provenance fields,
    including `id`, `domain_id`, `type`, `source`, `georeference`, `summary`,
    `status`, `created_on`, or `checksum`. A metadata-only update therefore does
    not make resources derived from the point cloud stale.

    `modified_on` is updated automatically.

    ## Response

    The updated point-cloud resource.

    ## Error Responses

    - **404 Not Found**: The point cloud does not exist, belongs to another
      domain, or is not accessible to the caller.
    """
    _, snapshot = await get_document_async(
        COLLECTION,
        point_cloud_id,
        owner_id=request.state.id,
        domain_id=domain["id"],
    )
    point_cloud_data = snapshot.to_dict()

    update_data = body.model_dump(exclude_none=True)
    update_data["modified_on"] = datetime.now()
    await update_document_async(
        collection=COLLECTION,
        document_id=point_cloud_id,
        data=update_data,
    )

    point_cloud_data.update(update_data)
    return PointCloud(**point_cloud_data)


@router.delete(
    "/{point_cloud_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a point cloud",
)
async def delete_point_cloud(
    request: Request,
    domain: VerifiedDomain,
    point_cloud_id: str,
) -> None:
    """
    # Delete a Point Cloud

    Deletes a point-cloud resource. This action cannot be undone through the
    API. Its GCS artifact becomes orphaned and is reclaimed asynchronously by
    the storage cleanup service; callers should treat the point cloud as deleted
    as soon as this endpoint returns.

    Deleting a point cloud does not delete grids or inventories that were
    derived from it. Those resources retain their recorded provenance, although
    the source point cloud can no longer be queried.

    ## Path Parameters

    - **domain_id**: Domain the point cloud belongs to.
    - **point_cloud_id**: Unique point-cloud identifier.

    ## Response

    HTTP `204 No Content` with an empty response body.

    ## Error Responses

    - **404 Not Found**: The point cloud does not exist, belongs to another
      domain, or is not accessible to the caller.
    """
    await get_document_async(
        COLLECTION,
        point_cloud_id,
        owner_id=request.state.id,
        domain_id=domain["id"],
    )
    await delete_document_async(
        collection=COLLECTION,
        document_id=point_cloud_id,
    )
