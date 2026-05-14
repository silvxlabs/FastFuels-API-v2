"""
api/v2/resources/features/layerset/router.py

Router for custom Layerset uploads.
"""

import logging
from datetime import UTC, datetime
from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, Body, Request, status

from api.db.documents import set_document_async
from api.dependencies import VerifiedDomain
from api.resources.features.layerset.examples import CREATE_LAYERSET_OPENAPI_EXAMPLES
from api.resources.features.layerset.schema import (
    CreateLayersetRequestBody,
    LayersetSource,
)
from api.resources.features.schema import Feature, FeatureGeoreference
from api.schema import JobStatus
from lib.config import FEATURES_BUCKET, FEATURES_COLLECTION
from lib.gcs.blobs import upload_json

logger = logging.getLogger(__name__)
router = APIRouter()


def _extract_bounds(geojson: dict) -> tuple[float, float, float, float] | None:
    """
    Crawls the hierarchical layerset GeoJSON to find the total bounding box
    across all strata and fuelbeds. Uses a recursive generator to handle
    arbitrary coordinate nesting depths.
    """
    min_x, min_y, max_x, max_y = (
        float("inf"),
        float("inf"),
        float("-inf"),
        float("-inf"),
    )
    has_coords = False

    def _get_points(coords_array):
        """Recursively yields (x, y) tuples from arbitrarily nested lists."""
        if not coords_array:
            return

        # If the first item is a number, we've hit the bottom [x, y] array
        if isinstance(coords_array[0], (int, float)):
            if len(coords_array) >= 2:
                yield coords_array[0], coords_array[1]
        else:
            # Otherwise, it's a nested list, so keep digging
            for item in coords_array:
                if isinstance(item, list):
                    yield from _get_points(item)

    features = geojson.get("features", [])
    for feature in features:
        fuelbeds = feature.get("properties", {}).get("fuelbeds", [])
        for fuelbed in fuelbeds:
            coords = fuelbed.get("polygons", {}).get("coordinates", [])

            # Safely extract all points regardless of nesting depth
            for x, y in _get_points(coords):
                min_x = min(min_x, x)
                min_y = min(min_y, y)
                max_x = max(max_x, x)
                max_y = max(max_y, y)
                has_coords = True

    if has_coords:
        return (min_x, min_y, max_x, max_y)

    return None


@router.post(
    "/geojson",
    response_model=Feature,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a custom Layerset",
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

    Uploads a custom hierarchical GeoJSON file as a new feature resource.
    The GeoJSON payload is validated and directly saved to Cloud Storage.

    ## Path Parameters
    - **domain_id**: (string) The domain this layerset belongs to.

    ## Request Body
    - **name**: (string) Name of the layerset.
    - **description**: (string) Description of the data.
    - **tags**: (array of strings) Searchable tags.
    - **geojson**: (object) A valid hierarchical GeoJSON FeatureCollection.

    ## Response
    Returns the created Feature resource with a status of `completed`.
    """
    owner_id = request.state.id
    domain_id = domain["id"]
    feature_id = f"{owner_id[:8]}-{uuid4().hex}"

    # 1. Convert the pydantic GeoJSON model back to a dictionary
    geojson_dict = body.geojson.model_dump(exclude_none=True)

    # 2. Upload directly to GCS using the shared library
    gcs_blob_path = f"gs://{FEATURES_BUCKET}/{domain_id}/{feature_id}.geojson"
    upload_json(gcs_blob_path, geojson_dict)

    # 3. Compute total bounds across all nested geometries
    georef = None
    try:
        bounds = _extract_bounds(geojson_dict)
        if bounds:
            # Note: We assume standard unprojected WGS84 for raw geojson uploads
            georef = FeatureGeoreference(crs="EPSG:4326", bounds=bounds)
    except Exception as e:
        logger.warning(f"Failed to extract bounds from custom GeoJSON: {e}")

    # 4. Construct the Feature metadata
    now = datetime.now(UTC)
    source = LayersetSource()

    feature = Feature(
        id=feature_id,
        domain_id=domain_id,
        type=body.type,
        name=body.name,
        description=body.description,
        tags=body.tags,
        status=JobStatus.completed,
        created_on=now,
        modified_on=now,
        source=source.model_dump(),
        georeference=georef,
        owner_id=owner_id,
    )

    # 5. Save metadata to Firestore
    await set_document_async(
        collection=FEATURES_COLLECTION,
        document_id=feature_id,
        data=feature.model_dump(exclude_none=True),
    )

    return feature
