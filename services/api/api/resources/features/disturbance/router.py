"""
api/v2/resources/features/disturbance/router.py

Router for disturbance features uploaded as GeoJSON.
"""

from typing import Annotated

from fastapi import APIRouter, Body, Request, status

from api.dependencies import VerifiedDomain
from api.quota import QUOTA_429_RESPONSE
from api.resources.features.disturbance.examples import (
    CREATE_DISTURBANCE_GEOJSON_OPENAPI_EXAMPLES,
)
from api.resources.features.disturbance.schema import CreateDisturbanceFeatureRequest
from api.resources.features.geojson import create_geojson_feature
from api.resources.features.schema import Feature, FeatureType

router = APIRouter()


@router.post(
    "/geojson",
    response_model=Feature,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a disturbance feature as GeoJSON",
    responses=QUOTA_429_RESPONSE,
)
async def create_geojson_disturbance_feature(
    request: Request,
    domain: VerifiedDomain,
    body: Annotated[
        CreateDisturbanceFeatureRequest,
        Body(openapi_examples=CREATE_DISTURBANCE_GEOJSON_OPENAPI_EXAMPLES),
    ],
) -> Feature:
    """
    # Upload Disturbance Feature

    Creates a disturbance feature from your own disturbance polygons, such as
    fire perimeters or treatment units. Each polygon carries the LANDFIRE FDist
    code of one disturbance.

    The polygons are reprojected to the domain CRS and clipped to the domain,
    plus `extent_buffer_m`.

    ## Request Body

    The body **is** a GeoJSON FeatureCollection, with resource metadata as
    top-level fields:

    - **features**: (array) At least one Feature with a `Polygon` or
      `MultiPolygon` geometry and an integer `properties.fdist`. Optional
      `properties.name`, `type`, `year` and `severity` describe the
      disturbance. All properties are stored as columns; each property must
      have one type across all features.
    - **crs**: (object, optional) The GeoJSON `crs` block. Any CRS is
      accepted. When omitted, coordinates are read as `EPSG:4326`.
    - **name**: (string, optional) Name of the feature.
    - **description**: (string, optional) Description.
    - **tags**: (array of strings, optional) Searchable tags.
    - **extent_buffer_m**: (number, optional) Distance in meters to expand the
      domain extent before clipping. Range: 0–100. Default: 0.

    Returns 422 if an `fdist` code is invalid, a geometry is invalid, or no
    polygon intersects the domain.

    ## FDist codes

    The FDist codes are defined in the Attribute Data Dictionary at
    https://landfire.gov/fuel/fdist.

    `fdist = 100 * D_TYPE + 10 * D_SEVERITY + D_TIME`, with D_TYPE 1-7
    (fire, mechanical add, mechanical remove, weather, insects/disease,
    mechanical unknown, mastication) and D_SEVERITY 1-3 (low, moderate,
    high). D_TIME, the last digit, is how long ago the disturbance happened,
    counted back from the year you are mapping:
    - 1 for a disturbance in that year
    - 2 for a disturbance in the previous four years
    - 3 for one five to nine years earlier

    For example, to update a 2024 fuel model to 2025, a 2025 disturbance ends
    in 1, a disturbance from 2021 to 2024 ends in 2, and a disturbance from
    2016 to 2020 ends in 3.

    Where polygons overlap, the more recent disturbance (the lower last digit)
    wins when the feature is used; if they are equally recent, the later
    feature in the file wins.

    ## Response

    Returns the created Feature with status `"completed"`.
    """
    return await create_geojson_feature(request, domain, body, FeatureType.disturbance)
