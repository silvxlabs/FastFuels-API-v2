"""
api/v2/resources/features/layerset/router.py

Router for custom Layerset uploads.
"""

import asyncio
import logging
from datetime import UTC, datetime
from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, Body, Request, status

from api.db.documents import set_document_async
from api.dependencies import VerifiedDomain
from api.quota import QUOTA_429_RESPONSE, enforce_create_quotas
from api.resources.features.layerset.examples import CREATE_LAYERSET_OPENAPI_EXAMPLES
from api.resources.features.layerset.schema import (
    CreateLayersetRequestBody,
    LayersetSource,
)
from api.resources.features.layerset.validate import validate_layerset
from api.resources.features.schema import Feature, FeatureGeoreference, FeatureType
from api.schema import JobStatus
from lib.config import FEATURES_BUCKET, FEATURES_COLLECTION

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post(
    "/geojson",
    response_model=Feature,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a custom Layerset",
    responses=QUOTA_429_RESPONSE,
)
async def create_layerset(
    request: Request,
    domain: VerifiedDomain,
    body: Annotated[
        CreateLayersetRequestBody,
        Body(openapi_examples=CREATE_LAYERSET_OPENAPI_EXAMPLES),
    ],
) -> Feature:
    """
    # Create Layerset Endpoint

    Uploads a flat GeoJSON FeatureCollection of fuelbed polygons as a new
    feature resource. Each Feature's ``properties`` block carries one
    fuelbed's input columns for ``fastfuels_core.rasterize_layerset``. The
    payload is validated and saved directly to Cloud Storage.

    ## Path Parameters
    - **domain_id**: (string) The domain this layerset belongs to.

    ## Request Body

    The body **is** the GeoJSON FeatureCollection (mirroring `POST /domains`),
    with optional resource metadata as top-level fields:

    - **type**: (string) Must be `"FeatureCollection"`.
    - **features**: (array) At least one Feature, each carrying one fuelbed's
      `properties` and a `Polygon`/`MultiPolygon` `geometry`.
    - **crs**: (object) The GeoJSON `crs` block declaring a **projected** CRS
      (e.g. `EPSG:32612`). Geographic CRSes are rejected — rasterization
      requires cell sizes in meters.
    - **name**: (string, optional) Name of the layerset.
    - **description**: (string, optional) Description of the data.
    - **tags**: (array of strings, optional) Searchable tags.

    ## Response
    Returns the created Feature resource with a status of `completed`.
    """
    owner_id = request.state.id
    domain_id = domain["id"]

    # dispatch=False: layerset features are written synchronously — no worker
    # job is commissioned, so the weekly dispatch budget does not apply.
    await enforce_create_quotas(FEATURES_COLLECTION, request, dispatch=False)

    feature_id = f"{owner_id[:8]}-{uuid4().hex}"

    # Validate the upload: parses + projected-CRS-checks (422 on geographic /
    # unparseable CRS), builds the GeoDataFrame, and computes union bounds in
    # one pass. Mirrors create_domain's validate_domain call.
    geojson_dict = body.model_dump(exclude_none=True)
    result = validate_layerset(geojson_dict)

    # Write GeoParquet to GCS. Format choices match etcher.storage.save_features
    # so all feature blobs share one on-disk schema regardless of writer. The
    # pyarrow-backed to_parquet path keeps the API GDAL-free.
    gcs_blob_path = f"gs://{FEATURES_BUCKET}/{domain_id}/{feature_id}.parquet"
    await asyncio.to_thread(
        result.gdf.to_parquet,
        gcs_blob_path,
        compression="zstd",
        row_group_size=1000,
    )

    # Georeference: union bounds anchored to the layerset's declared CRS. None
    # when no feature carries a non-empty geometry.
    georef = (
        FeatureGeoreference(crs=result.crs_string, bounds=result.bounds)
        if result.bounds is not None
        else None
    )

    # Construct the Feature metadata as a Firestore dict.
    # `owner_id` is intentionally not part of the Feature schema (matches
    # the repo-wide pattern for resource models); it lives on the
    # Firestore document for access-control filtering only. See
    # services/api/api/resources/domains/router.py:239 for the same note.
    now = datetime.now(UTC)
    source = LayersetSource()

    feature_data = {
        "id": feature_id,
        "domain_id": domain_id,
        "type": FeatureType.layerset.value,
        "name": body.name,
        "description": body.description,
        "status": JobStatus.completed.value,
        "progress": None,
        "created_on": now,
        "modified_on": now,
        "source": source.model_dump(),
        "georeference": georef.model_dump() if georef is not None else None,
        "error": None,
        "tags": body.tags,
        "owner_id": owner_id,
    }

    # Save metadata to Firestore
    await set_document_async(FEATURES_COLLECTION, feature_id, feature_data)

    return Feature(**feature_data)
