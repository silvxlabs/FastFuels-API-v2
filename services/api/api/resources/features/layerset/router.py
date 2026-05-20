"""
api/v2/resources/features/layerset/router.py

Router for custom Layerset uploads.
"""

import asyncio
import logging
import re
from datetime import UTC, datetime
from typing import Annotated
from uuid import uuid4

import geopandas as gpd
import pyproj
from fastapi import APIRouter, Body, HTTPException, Request, status
from shapely.geometry import shape

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

logger = logging.getLogger(__name__)
router = APIRouter()

# GeoJSON spec default when no crs block is declared on the FeatureCollection.
_DEFAULT_CRS = "EPSG:4326"


def _parse_crs_name(geojson: dict) -> str:
    """Extract an ``EPSG:<code>`` string from the GeoJSON's optional crs block.

    Accepts both bare ``"EPSG:32612"`` and the URN form
    ``"urn:ogc:def:crs:EPSG::32612"``. Falls back to ``EPSG:4326`` when the
    crs block is missing — the GeoJSON spec default.
    """
    name = (geojson.get("crs") or {}).get("properties", {}).get("name", "")
    if not name:
        return _DEFAULT_CRS
    if name.startswith("EPSG:"):
        return name
    match = re.search(r"EPSG::?(\d+)", name)
    if match:
        return f"EPSG:{match.group(1)}"
    return name  # pass through unrecognized forms; downstream surfaces the issue


def _require_projected_crs(crs_string: str) -> None:
    """Raise 422 if ``crs_string`` is geographic or unparseable.

    ``fastfuels_core.rasterize_layerset`` rejects geographic CRSes at
    rasterize time (resolution would otherwise be in degrees, not meters).
    Validating at upload turns a deferred worker crash into an immediate,
    actionable error for the user.
    """
    try:
        crs = pyproj.CRS.from_user_input(crs_string)
    except pyproj.exceptions.CRSError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=(
                f"Could not parse CRS {crs_string!r}: {exc}. Declare a "
                "projected CRS on the GeoJSON's top-level `crs` block "
                "(e.g. `EPSG:32612` for UTM 12N)."
            ),
        ) from exc
    if crs.is_geographic:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=(
                f"Layerset CRS is geographic ({crs_string}). Rasterization "
                "requires a projected CRS so cell sizes are in meters. "
                "Reproject the GeoJSON to a UTM (or other projected) CRS "
                "and declare it on the FeatureCollection's `crs` block."
            ),
        )


def _extract_bounds(geojson: dict) -> tuple[float, float, float, float] | None:
    """Compute the union bounding box across every ``Feature.geometry``.

    The bounds are expressed in the GeoJSON's own coordinate units — i.e.
    whatever the top-level ``crs`` block declares. Returns ``None`` when no
    feature carries a non-empty geometry.
    """
    min_x = min_y = float("inf")
    max_x = max_y = float("-inf")
    found = False

    for feature in geojson.get("features", []):
        geom_dict = feature.get("geometry") or {}
        if not geom_dict.get("coordinates"):
            continue
        try:
            geom = shape(geom_dict)
        except Exception as e:  # malformed geometry — log and skip, not fatal
            logger.warning(f"Skipping malformed geometry in layerset upload: {e}")
            continue
        if geom.is_empty:
            continue
        bx_min, by_min, bx_max, by_max = geom.bounds
        min_x = min(min_x, bx_min)
        min_y = min(min_y, by_min)
        max_x = max(max_x, bx_max)
        max_y = max(max_y, by_max)
        found = True

    return (min_x, min_y, max_x, max_y) if found else None


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

    Uploads a flat GeoJSON FeatureCollection of fuelbed polygons as a new
    feature resource. Each Feature's ``properties`` block carries one
    fuelbed's input columns for ``fastfuels_core.rasterize_layerset``. The
    payload is validated and saved directly to Cloud Storage.

    ## Path Parameters
    - **domain_id**: (string) The domain this layerset belongs to.

    ## Request Body
    - **name**: (string) Name of the layerset.
    - **description**: (string) Description of the data.
    - **tags**: (array of strings) Searchable tags.
    - **geojson**: (object) A valid flat GeoJSON FeatureCollection of fuelbed polygons.

    ## Response
    Returns the created Feature resource with a status of `completed`.
    """
    owner_id = request.state.id
    domain_id = domain["id"]
    feature_id = f"{owner_id[:8]}-{uuid4().hex}"

    # 1. Convert the pydantic GeoJSON model back to a dictionary
    geojson_dict = body.geojson.model_dump(exclude_none=True)

    # 2. Validate CRS before uploading. Geographic / unparseable CRSes
    #    are rejected here so the failure surfaces to the caller as a 422
    #    instead of a deferred rasterize crash.
    crs = _parse_crs_name(geojson_dict)
    _require_projected_crs(crs)

    # 3. Convert to a GeoDataFrame and write GeoParquet to GCS.
    #    Format choices match etcher.storage.save_features so all feature
    #    blobs share a single on-disk schema regardless of writer. The
    #    pyarrow-backed to_parquet path keeps the API GDAL-free.
    gcs_blob_path = f"gs://{FEATURES_BUCKET}/{domain_id}/{feature_id}.parquet"
    gdf = gpd.GeoDataFrame.from_features(geojson_dict["features"], crs=crs)
    await asyncio.to_thread(
        gdf.to_parquet,
        gcs_blob_path,
        compression="zstd",
        row_group_size=1000,
    )

    # 4. Compute bounds across every Feature.geometry, anchored to whatever
    #    CRS the GeoJSON declares (or EPSG:4326 per the GeoJSON default).
    georef = None
    try:
        bounds = _extract_bounds(geojson_dict)
        if bounds:
            georef = FeatureGeoreference(crs=crs, bounds=bounds)
    except Exception as e:
        logger.warning(f"Failed to extract bounds from layerset GeoJSON: {e}")

    # 5. Construct the Feature metadata as a Firestore dict.
    #    `owner_id` is intentionally not part of the Feature schema (matches
    #    the repo-wide pattern for resource models); it lives on the
    #    Firestore document for access-control filtering only. See
    #    services/api/api/resources/domains/router.py:239 for the same note.
    now = datetime.now(UTC)
    source = LayersetSource()

    feature_data = {
        "id": feature_id,
        "domain_id": domain_id,
        "type": body.type.value,
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

    # 6. Save metadata to Firestore
    await set_document_async(FEATURES_COLLECTION, feature_id, feature_data)

    return Feature(**feature_data)
