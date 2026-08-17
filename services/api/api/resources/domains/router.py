"""
api/v2/resources/domains/router.py
"""

import math
import uuid
from datetime import datetime
from typing import Annotated

from fastapi import (
    APIRouter,
    Body,
    HTTPException,
    Query,
    Request,
    status,
)
from google.cloud.firestore import FieldFilter

from api.db.documents import (
    delete_document_async,
    firestore_client,
    get_document_async,
    list_documents_async,
    set_document_async,
    update_document_async,
)
from api.dependencies import VerifiedDomain, invalidate_domain_cache
from api.quota import QUOTA_429_RESPONSE, enforce_create_quotas
from api.resources.domains.examples import CREATE_DOMAIN_OPENAPI_EXAMPLES
from api.resources.domains.schema import (
    CreateDomainRequestBody,
    Domain,
    DomainLattice,
    DomainSortField,
    DomainSortOrder,
    ListDomainsResponse,
    UpdateDomainRequestBody,
)
from api.resources.domains.validate import (
    reproject_features,
    validate_crs,
    validate_domain,
)
from lib.config import (
    DOMAINS_COLLECTION,
    FEATURES_COLLECTION,
    GRIDS_COLLECTION,
    INVENTORIES_COLLECTION,
    POINT_CLOUDS_COLLECTION,
)
from lib.domain_utils import parse_domain_gdf

# Child resource types that cascade-delete when a domain is force-deleted.
# Each entry is a Firestore collection and the foreign key linking it to the
# domain. Child docs are deleted synchronously (keeping quota accurate); walle
# reclaims their GCS artifacts. Exports are deliberately excluded — they survive
# domain deletion as standalone provenance artifacts (walle likewise never
# orphan-reaps an export by a missing domain).
CHILD_RESOURCES = [
    {"collection": GRIDS_COLLECTION, "foreign_key": "domain_id"},
    {"collection": INVENTORIES_COLLECTION, "foreign_key": "domain_id"},
    {"collection": FEATURES_COLLECTION, "foreign_key": "domain_id"},
    {"collection": POINT_CLOUDS_COLLECTION, "foreign_key": "domain_id"},
]

router = APIRouter()


@router.post(
    "",
    response_model=Domain,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new domain",
    response_model_exclude_none=True,
    responses=QUOTA_429_RESPONSE,
)
async def create_domain(
    request: Request,
    body: Annotated[
        CreateDomainRequestBody,
        Body(openapi_examples=CREATE_DOMAIN_OPENAPI_EXAMPLES),
    ],
):
    """
    # Create Domain Endpoint

    This endpoint creates a new domain resource based on a spatial extent and
    additional details provided by the user. The domain resource acts as the
    spatial container for all other resources that create data within the system.

    ## What is a Domain Resource?

    A domain resource is a spatial container that represents a specific geographical
    area. It includes metadata such as the name, description, creation date, and the
    spatial extent defined by geographic coordinates. Domains are used to organize
    and manage spatial data and operations within a defined area.

    ## Request Body

    The request body must be a GeoJSON FeatureCollection as defined by the
    [GeoJSON specification (RFC 7946)](https://datatracker.ietf.org/doc/html/rfc7946).

    ### Required Fields

    - **type**: (string) Must be "FeatureCollection".
    - **features**: (array) An array of Feature objects. Each Feature must have:
      - **type**: (string) Must be "Feature".
      - **geometry**: (GeoJSON Geometry) A geometry object (typically Polygon).
        - **type**: (string) Must be a valid GeoJSON type, e.g., "Polygon".
        - **coordinates**: (array) An array of coordinates defining the geometry.

    ### Optional Fields

    - **name**: (string) The name of the domain. Default: empty string.
    - **description**: (string) A brief description of the domain. Default: empty string.
    - **tags**: (array of strings) Tags for organizing and filtering domains.
    - **crs**: (object) The coordinate reference system. Default: EPSG:4326 (WGS84).
      - **type**: (string) Must be "name".
      - **properties**: (object) Contains the CRS details.
        - **name**: (string) The CRS identifier, e.g., "EPSG:4326", "EPSG:5070",
          or URN format "urn:ogc:def:crs:EPSG::32611".
    - **pad_to_resolution**: (number) Optional resolution in meters to snap the
      domain bounding box to. When set, the bounding box (the "domain" feature)
      is snapped outward to the nearest multiple of this value. Grids whose
      resolutions divide evenly into this value will produce identical, aligned
      footprints on this domain. Useful for compositional workflows where
      multiple grids at different resolutions need to share an extent.
    - **style**: (object) Optional visual style for rendering the domain on a
      map. Sub-fields: `stroke_color`, `stroke_opacity` (0-1), `stroke_width`
      (>= 0), `fill_color`, `fill_opacity` (0-1). Color strings accept any
      format the renderer understands (hex, named, `rgb()`, ...) and are
      capped at 64 characters.

    ## Response

    On successful creation, returns the domain resource with:

    - **id**: (string) A unique 32-character hex identifier for the domain.
    - **type**: (string) Always "FeatureCollection".
    - **name**: (string) The name of the domain.
    - **description**: (string) The description of the domain.
    - **created_on**: (datetime) When the domain was created.
    - **modified_on**: (datetime) When the domain was last modified.
    - **tags**: (array) The tags associated with the domain.
    - **crs**: (object) The coordinate reference system (always projected).
    - **features**: (array) A single feature named `"domain"` — a polygon
      covering the working extent (bounding box of the input, possibly
      padded). This is what griddle, standgen, and exporter use as the
      authoritative spatial extent.
    - **bbox**: (array) Standard GeoJSON bbox `[minx, miny, maxx, maxy]` in the
      domain's projected CRS. Equals the bounds of the "domain" feature.
    - **pad_to_resolution**: (number, optional) The padding value, if set.

    ## CRS Handling

    The API handles coordinate reference systems as follows:

    1. **Geographic CRS (e.g., EPSG:4326)**: Automatically projected to the
       appropriate UTM zone based on the geometry's centroid. The response CRS
       will be the UTM zone (e.g., EPSG:32611 for UTM Zone 11N).

    2. **Projected CRS (e.g., EPSG:5070, EPSG:32611)**: Used as-is without
       reprojection. The response CRS will match the input CRS.

    ## Validation

    The following validations are performed:

    1. **CRS Validation**: Must be a valid EPSG code or URN format.
    2. **Area Validation**: Geometry must have non-zero area (no points or lines).
    3. **Location**: Geometry must be entirely within CONUS (Continental US).
       Validated against the original input polygon (not the padded bbox).
    4. **Size Limit**: The working extent (possibly padded bbox) must be less
       than 16 square kilometers.

    ## Important Notes

    1. **FeatureCollection Only**: Unlike v1, this endpoint only accepts
       FeatureCollection input, not individual Feature objects. Wrap single
       features in a FeatureCollection.

    2. **Working-Extent Output**: The created domain stores a single "domain"
       feature — the bounding box of the input geometry, which is the working
       extent used by all downstream services. The submitted geometry itself
       is not stored.

    3. **Projection**: Geographic coordinates are always projected to a suitable
       UTM zone for accurate area calculations and grid operations.

    4. **Maximum Area**: The 16 sq km limit ensures reasonable processing times.
       Contact support if you need larger domains.

    ## Error Responses

    - **422 Unprocessable Entity**:
      - "Invalid CRS '{crs}'. Must be a valid authority string (e.g., 'EPSG:4326')."
      - "Invalid geometry. The feature must have an area greater than zero."
      - "Invalid spatial extent. Area must be less than 16 square kilometers."
      - "Invalid spatial extent. The domain must be entirely within CONUS."
    """
    await enforce_create_quotas(DOMAINS_COLLECTION, request)

    # Validate domain geometry (parses GeoJSON, validates CRS, area, CONUS check)
    # Raises HTTPException if validation fails
    validation_result = validate_domain(body.model_dump())

    # Build domain data from validated result
    domain_id = uuid.uuid4().hex
    request_time = datetime.now()
    domain_data = {
        "type": "FeatureCollection",
        "id": domain_id,
        "name": body.name,
        "description": body.description,
        "created_on": request_time,
        "modified_on": request_time,
        "tags": body.tags,
        "crs": {
            "type": "name",
            "properties": {"name": str(validation_result.crs)},
        },
        "features": validation_result.features,
        "bbox": list(validation_result.bbox),
        "pad_to_resolution": body.pad_to_resolution,
        "style": body.style,
    }

    domain = Domain(**domain_data)

    # Serialize with Firestore context (stringifies nested coordinate arrays)
    firestore_data = domain.model_dump(context={"for_firestore": True})

    # Add owner_id to Firestore data (not part of the Domain model, but needed for access control)
    firestore_data["owner_id"] = request.state.id

    # Write to Firestore
    await set_document_async(DOMAINS_COLLECTION, domain.id, firestore_data)

    # Send data back to the client
    return domain


@router.post(
    "/preview",
    response_model=Domain,
    status_code=status.HTTP_200_OK,
    summary="Preview a domain without persisting it",
    response_model_exclude_none=True,
)
async def preview_domain(
    body: Annotated[
        CreateDomainRequestBody,
        Body(openapi_examples=CREATE_DOMAIN_OPENAPI_EXAMPLES),
    ],
):
    """
    # Preview Domain Endpoint

    Runs the same validation and projection pipeline as `POST /v2/domains` but
    returns the resulting `Domain` resource without writing to Firestore. Use
    this to let users inspect the projected, padded bounding box before committing
    to a create.

    ## Request Body

    Identical to `POST /v2/domains`. See that endpoint for full documentation.

    ## Response

    Returns the same `Domain` response model as create, with:

    - **id**: Always `"preview"` — not a real domain identifier.
    - **created_on** / **modified_on**: Set to the current request time (not persisted).
    - **features**: A single `"domain"` feature (the working extent),
      identical to what create would return.
    - **bbox**: Bounding box of the `"domain"` feature.
    - **crs**: Projected CRS, identical to what create would return.

    ## Error Responses

    Same 422 error responses as `POST /v2/domains`:

    - "Invalid CRS '{crs}'. Must be a valid authority string (e.g., 'EPSG:4326')."
    - "Invalid geometry. The feature must have an area greater than zero."
    - "Invalid spatial extent. Area must be less than 16 square kilometers."
    - "Invalid spatial extent. The domain must be entirely within CONUS."
    """
    validation_result = validate_domain(body.model_dump())

    request_time = datetime.now()
    domain_data = {
        "type": "FeatureCollection",
        "id": "preview",
        "name": body.name,
        "description": body.description,
        "created_on": request_time,
        "modified_on": request_time,
        "tags": body.tags,
        "crs": {
            "type": "name",
            "properties": {"name": str(validation_result.crs)},
        },
        "features": validation_result.features,
        "bbox": list(validation_result.bbox),
        "pad_to_resolution": body.pad_to_resolution,
        "style": body.style,
    }

    return Domain(**domain_data)


@router.post(
    "/reproject",
    response_model=CreateDomainRequestBody,
    status_code=status.HTTP_200_OK,
    summary="Reproject a FeatureCollection to a target CRS",
    response_model_exclude_none=True,
)
async def reproject_domain(
    body: CreateDomainRequestBody,
    target_epsg: int = Query(..., description="EPSG code of the target CRS."),
):
    """
    # Reproject Domain Endpoint

    Stateless utility that reprojects a GeoJSON `FeatureCollection` from one
    coordinate reference system to another. No resource is created; the
    reprojected `FeatureCollection` is returned immediately.

    ## Query Parameters

    - **target_epsg**: (integer, required) EPSG code of the target CRS
      (e.g., `4326` for WGS84, `32611` for UTM zone 11N).

    ## Request Body

    A GeoJSON `FeatureCollection`. The source CRS is read from the
    `crs.properties.name` field if present; otherwise EPSG:4326 is assumed.

    ## Response

    Returns the reprojected `FeatureCollection` with:

    - **features**: All input features reprojected to the target CRS, with
      original feature properties preserved.
    - **crs**: Set to the target EPSG code.

    ## Error Responses

    - **422**: Invalid source CRS, invalid target EPSG, or geometry that
      cannot be reprojected.
    """
    source_crs = validate_crs(body.crs.properties.name)
    target_crs = validate_crs(f"EPSG:{target_epsg}")

    features = [feat.model_dump() for feat in body.features]
    reprojected = reproject_features(features, source_crs, target_crs)

    return CreateDomainRequestBody(
        **{
            "type": "FeatureCollection",
            "features": reprojected,
            "name": body.name,
            "description": body.description,
            "tags": body.tags,
            "crs": {"type": "name", "properties": {"name": f"EPSG:{target_epsg}"}},
        }
    )


@router.get(
    "",
    response_model=ListDomainsResponse,
    status_code=status.HTTP_200_OK,
    summary="List all domains",
)
async def list_domains(
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
        description="The number of domains to retrieve per page.",
    ),
    sort_by: DomainSortField | None = Query(
        None,
        description="The field to sort results by.",
    ),
    sort_order: DomainSortOrder | None = Query(
        None,
        description="The order to sort results (ascending or descending).",
    ),
) -> ListDomainsResponse:
    """
    # List Domains Endpoint

    This endpoint retrieves a paginated list of all domains belonging to the
    authenticated user.

    ## Query Parameters

    - **page**: (integer, optional) The page number to retrieve. Zero-indexed,
      meaning the first page is `0`. Default: 0.
    - **size**: (integer, optional) The number of domains to retrieve per page.
      Must be between 1 and 1000. Default: 100.
    - **sort_by**: (string, optional) The field to sort results by. Valid values:
      - `created_on`: Sort by creation date.
      - `modified_on`: Sort by last modification date.
      - `name`: Sort alphabetically by name.
    - **sort_order**: (string, optional) The order to sort results. Valid values:
      - `ascending`: Sort in ascending order (A-Z, oldest first).
      - `descending`: Sort in descending order (Z-A, newest first).
      Default: descending when sort_by is specified.

    ## Response

    Returns a paginated list of domains with metadata:

    - **domains**: (array) List of domain resources for the current page.
      Each domain includes:
      - **id**: (string) The unique identifier for the domain.
      - **type**: (string) Always "FeatureCollection".
      - **name**: (string) The name of the domain.
      - **description**: (string) The description of the domain.
      - **created_on**: (datetime) When the domain was created.
      - **modified_on**: (datetime) When the domain was last modified.
      - **tags**: (array) The tags associated with the domain.
      - **crs**: (object) The coordinate reference system.
      - **features**: (array) The domain geometry features.
    - **current_page**: (integer) The current page number (zero-indexed).
    - **page_size**: (integer) The number of domains per page.
    - **total_items**: (integer) The total number of domains owned by the user.

    ## Pagination

    Use `page` and `size` parameters to navigate through large result sets:

    - First page: `?page=0&size=10`
    - Second page: `?page=1&size=10`
    - Calculate total pages: `ceil(total_items / page_size)`

    ## Sorting

    Combine `sort_by` and `sort_order` for custom ordering:

    - Newest first: `?sort_by=created_on&sort_order=descending`
    - Alphabetical: `?sort_by=name&sort_order=ascending`
    - Recently modified: `?sort_by=modified_on&sort_order=descending`

    ## Example Request

    ```http
    GET /v2/domains?page=0&size=10&sort_by=created_on&sort_order=descending
    ```

    ## Example Response

    ```json
    {
      "domains": [
        {
          "id": "abc123...",
          "type": "FeatureCollection",
          "name": "My Domain",
          "description": "A test domain",
          "created_on": "2024-01-15T10:30:00",
          "modified_on": "2024-01-15T10:30:00",
          "tags": ["test"],
          "crs": {"type": "name", "properties": {"name": "EPSG:32611"}},
          "features": [...]
        }
      ],
      "current_page": 0,
      "page_size": 10,
      "total_items": 42
    }
    ```

    ## Error Responses

    - **422 Unprocessable Entity**: Invalid query parameters.
      - Page must be a non-negative integer.
      - Size must be between 1 and 1000.
      - Invalid sort_by or sort_order values.
    """
    owner_id = request.state.id

    # Query Firestore for user's domains with pagination and sorting
    documents, total_count = await list_documents_async(
        collection=DOMAINS_COLLECTION,
        owner_id=owner_id,
        page=page,
        size=size,
        sort_by=sort_by.value if sort_by else None,
        sort_order=sort_order.value if sort_order else None,
    )

    # Convert Firestore documents to Domain models
    domains = []
    for doc in documents:
        domain_data = doc.to_dict()
        domain = Domain(**domain_data)
        domains.append(domain)

    return ListDomainsResponse(
        domains=domains,
        current_page=page,
        page_size=size,
        total_items=total_count,
    )


@router.get(
    "/{domain_id}",
    response_model=Domain,
    status_code=status.HTTP_200_OK,
    summary="Get a domain by ID",
)
async def get_domain(
    request: Request,
    domain_id: str,
):
    """
    # Get Domain Endpoint

    This endpoint retrieves a specific domain resource by its unique identifier.

    ## Path Parameters

    - **domain_id**: (string) The unique 32-character hex identifier of the domain.

    ## Response

    On success, returns the domain resource with:

    - **id**: (string) The unique identifier for the domain.
    - **type**: (string) Always "FeatureCollection".
    - **name**: (string) The name of the domain.
    - **description**: (string) The description of the domain.
    - **created_on**: (datetime) When the domain was created.
    - **modified_on**: (datetime) When the domain was last modified.
    - **tags**: (array) The tags associated with the domain.
    - **crs**: (object) The coordinate reference system (always projected).
    - **features**: (array) The domain geometry features.

    ## Error Responses

    - **404 Not Found**: The domain does not exist or the user does not have access.
      - Returns 404 for both missing documents and ownership mismatches to avoid
        leaking information about document existence.
    """
    owner_id = request.state.id

    # Fetch the domain document from Firestore with ownership validation
    _, document_snapshot = await get_document_async(
        collection=DOMAINS_COLLECTION,
        document_id=domain_id,
        owner_id=owner_id,
    )

    # Convert Firestore document to Domain model
    # The model validator automatically handles coordinate deserialization
    domain_data = document_snapshot.to_dict()
    domain = Domain(**domain_data)

    return domain


@router.get(
    "/{domain_id}/lattice",
    response_model=DomainLattice,
    status_code=status.HTTP_200_OK,
    summary="Get the pixel lattice for a domain at a given resolution",
)
async def get_domain_lattice(
    domain: VerifiedDomain,
    resolution: Annotated[
        float,
        Query(
            gt=0,
            description="Pixel size in meters (domain CRS units, always projected).",
        ),
    ],
    num_buffer_cells: Annotated[
        int,
        Query(
            ge=0,
            description=(
                "Expand the lattice by N cells on each side. Mirrors the "
                "buffer semantics of POST /domains/{domain_id}/grids/upload "
                "and the LANDFIRE/3DEP grid creation endpoints."
            ),
        ),
    ] = 0,
):
    """
    # Get Domain Lattice Endpoint

    Returns the pixel lattice (transform + shape) for the domain at the
    requested resolution. Use this to align a GeoTIFF before uploading it
    via `POST /domains/{domain_id}/grids/upload`.

    ## Query Parameters

    - **resolution** (required): Pixel size in meters.
    - **num_buffer_cells** (optional, default 0): Expand the lattice by
      `N * resolution` meters on each side.

    ## Response

    - **crs**: The domain CRS (always projected).
    - **resolution**: Echoes the input.
    - **num_buffer_cells**: Echoes the input.
    - **transform**: Affine coefficients `[a, b, c, d, e, f]` (rasterio
      convention).
    - **shape**: `[height, width]` in pixels.

    ## Error Responses

    - **404 Not Found**: The domain does not exist or the user does not
      have access.
    - **422 Unprocessable Entity**: `resolution` is missing or
      non-positive, or `num_buffer_cells` is negative.
    """
    domain_gdf = parse_domain_gdf(domain)
    minx, miny, maxx, maxy = domain_gdf.total_bounds
    pad = num_buffer_cells * resolution
    minx -= pad
    miny -= pad
    maxx += pad
    maxy += pad
    # North-up lattice anchored at (minx, miny), ceil-covering the padded bounds.
    width = max(1, math.ceil((maxx - minx) / resolution))
    height = max(1, math.ceil((maxy - miny) / resolution))
    return DomainLattice(
        crs=domain["crs"]["properties"]["name"],
        resolution=resolution,
        num_buffer_cells=num_buffer_cells,
        transform=(resolution, 0.0, minx, 0.0, -resolution, miny + height * resolution),
        shape=(height, width),
    )


@router.patch(
    "/{domain_id}",
    response_model=Domain,
    status_code=status.HTTP_200_OK,
    summary="Update a domain",
)
async def update_domain(
    request: Request,
    domain_id: str,
    body: UpdateDomainRequestBody,
):
    """
    # Update Domain Endpoint

    This endpoint updates the metadata of an existing domain resource. Only the
    fields provided in the request body will be modified; other fields remain
    unchanged.

    ## Path Parameters

    - **domain_id**: (string) The unique 32-character hex identifier of the domain.

    ## Request Body

    All fields are optional. Only provided fields will be updated.

    - **name**: (string, optional) The new name for the domain.
    - **description**: (string, optional) The new description for the domain.
    - **tags**: (array of strings, optional) The new tags for the domain.
      This replaces the existing tags array entirely.

    ## What Cannot Be Updated

    The following fields are immutable after domain creation:

    - **id**: The domain identifier is permanent.
    - **features**: Geometry cannot be modified. Create a new domain instead.
    - **crs**: Coordinate reference system is tied to the geometry.
    - **created_on**: Creation timestamp is permanent.

    The **modified_on** field is automatically updated to the current time.

    ## Response

    On success, returns the updated domain resource with all fields,
    including the new `modified_on` timestamp.

    ## Example Request

    ```http
    PATCH /v2/domains/abc123def456...
    Content-Type: application/json

    {
      "name": "Updated Domain Name",
      "tags": ["production", "verified"]
    }
    ```

    ## Error Responses

    - **404 Not Found**: The domain does not exist or the user does not have access.
    - **422 Unprocessable Entity**: Invalid request body.
    """
    owner_id = request.state.id

    # Validate ownership and get current document
    _, document_snapshot = await get_document_async(
        collection=DOMAINS_COLLECTION,
        document_id=domain_id,
        owner_id=owner_id,
    )

    existing_data = document_snapshot.to_dict()

    # Build update data from provided fields only
    update_data = body.model_dump(exclude_none=True)

    # Merge style sub-fields with the existing stored style; a top-level write
    # would replace the whole `style` object and clobber unspecified sub-fields.
    if "style" in update_data:
        existing_style = existing_data.get("style") or {}
        update_data["style"] = {**existing_style, **update_data["style"]}

    # Always update modified_on timestamp
    update_data["modified_on"] = datetime.now()

    # Perform the partial update
    await update_document_async(
        collection=DOMAINS_COLLECTION,
        document_id=domain_id,
        data=update_data,
    )

    await invalidate_domain_cache(domain_id, owner_id)

    # Merge updates with existing data to return the full domain
    existing_data.update(update_data)
    domain = Domain(**existing_data)

    return domain


@router.delete(
    "/{domain_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a domain",
)
async def delete_domain(
    request: Request,
    domain_id: str,
    force: bool = Query(
        False,
        description=(
            "Force cascade delete of all child resources (grids, etc.). "
            "Without this, returns 412 if child resources exist."
        ),
    ),
):
    """
    # Delete Domain Endpoint

    This endpoint permanently deletes a domain resource by its unique identifier.
    This action cannot be undone.

    ## Path Parameters

    - **domain_id**: (string) The unique 32-character hex identifier of the domain.

    ## Query Parameters

    - **force**: (boolean, optional) If true, cascade-deletes all child resources
      (grids, etc.) before deleting the domain. Default: false.

    ## Response

    On success, returns HTTP 204 No Content with an empty response body.

    ## Cascade Behavior (AIP-135)

    - **Without `force`**: If the domain has child grids, returns 412 Precondition
      Failed. Delete child resources first, or use `force=true`.
    - **With `force=true`**: Deletes the domain and all child grids in a single
      operation.

    ## Error Responses

    - **404 Not Found**: The domain does not exist or the user does not have access.
    - **412 Precondition Failed**: The domain has child resources and `force` was
      not set to true.
    """
    owner_id = request.state.id

    # Validate ownership (raises 404 if not found or not owned)
    await get_document_async(
        collection=DOMAINS_COLLECTION,
        document_id=domain_id,
        owner_id=owner_id,
    )

    # Check for child resources across all registered types
    has_children = False
    for resource in CHILD_RESOURCES:
        query = (
            firestore_client.collection(resource["collection"])
            .where(filter=FieldFilter(resource["foreign_key"], "==", domain_id))
            .where(filter=FieldFilter("owner_id", "==", owner_id))
            .limit(1)
        )
        if await query.get():
            has_children = True
            break

    if has_children and not force:
        raise HTTPException(
            status_code=status.HTTP_412_PRECONDITION_FAILED,
            detail=(
                "Domain has child resources. "
                "Use force=true to cascade delete all child resources."
            ),
        )

    if has_children and force:
        for resource in CHILD_RESOURCES:
            all_docs_query = (
                firestore_client.collection(resource["collection"])
                .where(filter=FieldFilter(resource["foreign_key"], "==", domain_id))
                .where(filter=FieldFilter("owner_id", "==", owner_id))
            )
            all_docs = await all_docs_query.get()
            if all_docs:
                batch = firestore_client.batch()
                for doc in all_docs:
                    batch.delete(doc.reference)
                await batch.commit()

    # Delete the domain document
    await delete_document_async(
        collection=DOMAINS_COLLECTION,
        document_id=domain_id,
    )

    await invalidate_domain_cache(domain_id, owner_id)
