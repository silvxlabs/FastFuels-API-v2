"""
api/v2/resources/features/disturbance/router.py

Router for disturbance polygon uploads.
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
from api.resources.features.disturbance.examples import (
    CREATE_DISTURBANCE_OPENAPI_EXAMPLES,
)
from api.resources.features.disturbance.schema import (
    CreateDisturbanceRequestBody,
    DisturbanceSource,
)
from api.resources.features.disturbance.validate import validate_disturbance
from api.resources.features.schema import Feature, FeatureGeoreference, FeatureType
from api.schema import JobStatus
from lib.config import FEATURES_BUCKET, FEATURES_COLLECTION

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post(
    "/geojson",
    response_model=Feature,
    status_code=status.HTTP_201_CREATED,
    summary="Upload disturbance polygons",
    responses=QUOTA_429_RESPONSE,
)
async def create_disturbance(
    request: Request,
    domain: VerifiedDomain,
    body: Annotated[
        CreateDisturbanceRequestBody,
        Body(openapi_examples=CREATE_DISTURBANCE_OPENAPI_EXAMPLES),
    ],
) -> Feature:
    """
    # Create Disturbance Endpoint

    Uploads a GeoJSON FeatureCollection of disturbance polygons as a new
    feature resource. Each Feature carries the LANDFIRE FDist code of one
    disturbance in `properties.fdist` and a `Polygon` or `MultiPolygon`
    geometry. The payload is validated and saved directly to Cloud Storage.

    ## Path Parameters
    - **domain_id**: (string) The domain these polygons belong to.

    ## Request Body

    The body **is** the GeoJSON FeatureCollection (mirroring `POST /domains`),
    with optional resource metadata as top-level fields:

    - **type**: (string) Must be `"FeatureCollection"`.
    - **features**: (array) At least one Feature with a `Polygon`/`MultiPolygon`
      `geometry` and an integer `properties.fdist`. Optional `properties.name`,
      `type`, `year` and `severity` are kept as given. Other properties are ignored.
    - **crs**: (object, optional) The GeoJSON `crs` block. Any CRS is
      accepted. When omitted, coordinates are read as `EPSG:4326`.
    - **name**: (string, optional) Name of the feature.
    - **description**: (string, optional) Description of the data.
    - **tags**: (array of strings, optional) Searchable tags.

    ## FDist codes

    The FDist codes are defined in the Attribute Data Dictionary at
    https://landfire.gov/fuel/fdist.

    `fdist = 100 * D_TYPE + 10 * D_SEVERITY + D_TIME`, with D_TYPE 1-7
    (fire, mechanical add, mechanical remove, weather, insects/disease,
    mechanical unknown, mastication) and D_SEVERITY 1-3 (low, moderate,
    high). D_TIME, the last digit, is how long ago the disturbance happened,
    counted back from `mapping_year`:
    - 1 for a disturbance in `mapping_year`
    - 2 for a disturbance in the previous four years
    - 3 for one five to nine years earlier

    For example, with a `mapping_year` of 2025 (to update a 2024 fuel model to
    2025), a 2025 disturbance ends in 1, a disturbance from 2021 to 2024 ends
    in 2, and a disturbance from 2016 to 2020 ends in 3.

    Where polygons overlap, the more recent disturbance (the lower last digit)
    wins when the feature is used; if they are equally recent, the later
    feature in the file wins.

    ## Response
    Returns the created Feature resource with a status of `completed`.
    """
    owner_id = request.state.id
    domain_id = domain["id"]

    # dispatch=False: disturbance features are written synchronously, with no
    # worker job, so the weekly dispatch budget does not apply.
    await enforce_create_quotas(FEATURES_COLLECTION, request, dispatch=False)

    feature_id = f"{owner_id[:8]}-{uuid4().hex}"

    geojson_dict = body.model_dump(exclude_none=True)
    result = validate_disturbance(geojson_dict, domain)

    # Same GeoParquet format as every other feature blob; the pyarrow-backed
    # to_parquet path keeps the API GDAL-free.
    gcs_blob_path = f"gs://{FEATURES_BUCKET}/{domain_id}/{feature_id}.parquet"
    await asyncio.to_thread(
        result.gdf.to_parquet,
        gcs_blob_path,
        compression="zstd",
        row_group_size=1000,
    )

    georef = FeatureGeoreference(crs=result.crs_string, bounds=result.bounds)

    # `owner_id` is intentionally not part of the Feature schema; it lives on
    # the Firestore document for access-control filtering only.
    now = datetime.now(UTC)
    feature_data = {
        "id": feature_id,
        "domain_id": domain_id,
        "type": FeatureType.disturbance.value,
        "name": body.name,
        "description": body.description,
        "status": JobStatus.completed.value,
        "progress": None,
        "created_on": now,
        "modified_on": now,
        "source": DisturbanceSource(mapping_year=body.mapping_year).model_dump(),
        "georeference": georef.model_dump(),
        "error": None,
        "tags": body.tags,
        "owner_id": owner_id,
    }

    await set_document_async(FEATURES_COLLECTION, feature_id, feature_data)

    return Feature(**feature_data)
