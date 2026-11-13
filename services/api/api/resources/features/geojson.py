"""
api/v2/resources/features/geojson.py

GeoJSON upload source for road, water, building and disturbance features.

The upload is reprojected to the domain CRS and clipped to the domain extent
plus ``extent_buffer_m``, as the OSM and Overture sources are, then stored as
GeoParquet in-request with status ``completed``. Reprojection uses pyproj
through geopandas, so the API stays GDAL-free.
"""

import asyncio
from datetime import UTC, datetime
from typing import Any, Literal
from uuid import uuid4

import geopandas as gpd
import pyproj
import shapely
from fastapi import HTTPException, Request, status
from geojson_pydantic import Feature as GeojsonFeature
from geojson_pydantic import FeatureCollection
from geojson_pydantic.geometries import MultiPolygon, Polygon
from pydantic import BaseModel, Field, field_validator, model_validator
from pyproj.exceptions import CRSError

from api.db.documents import set_document_async
from api.quota import enforce_create_quotas
from api.resources.features.schema import Feature, FeatureGeoreference, FeatureType
from api.resources.features.storage import (
    encode_feature_parquet,
    upload_feature_parquet,
)
from api.schema import JobStatus
from lib.config import FEATURES_COLLECTION
from lib.domain_utils import buffer_domain, parse_domain_gdf

# GeoJSON spec default when no crs block is declared on the FeatureCollection.
DEFAULT_CRS = "EPSG:4326"


class GeojsonSource(BaseModel):
    """Source metadata stored on a feature uploaded as GeoJSON."""

    product: Literal["geojson"] = "geojson"
    description: Literal["User-uploaded GeoJSON"] = "User-uploaded GeoJSON"
    extent_buffer_m: float = Field(0, ge=0, le=100)


class GeojsonCrs(BaseModel):
    """Optional GeoJSON ``crs`` block (deprecated by RFC 7946, but widely emitted)."""

    type: str = "name"
    properties: dict[str, Any] = Field(default_factory=dict)


class GeojsonPolygonFeature(
    GeojsonFeature[Polygon | MultiPolygon, dict[str, Any] | None]
):
    """One uploaded polygon. Its ``properties`` are stored as columns."""

    geometry: Polygon | MultiPolygon

    # Before mode, so it also checks the raw dict when a subclass types the
    # properties as a model.
    @field_validator("properties", mode="before")
    @classmethod
    def geometry_is_not_a_property(cls, value: Any) -> Any:
        if isinstance(value, dict) and "geometry" in value:
            raise ValueError(
                "`geometry` can't be used as a property name: it is the name of "
                "the stored geometry column. Rename the property."
            )
        return value


class CreateGeojsonFeatureRequest(FeatureCollection[GeojsonPolygonFeature]):
    """Request body for uploading a road, water, building or disturbance feature.

    The body **is** the GeoJSON FeatureCollection, as for ``POST /domains`` and
    layerset uploads, extended with the resource-metadata fields. ``name``
    overrides the optional GeoJSON ``name`` member and is the resource name.
    """

    crs: GeojsonCrs | None = None
    name: str = Field("", max_length=255)
    description: str = Field("", max_length=2000)
    tags: list[str] = Field(default_factory=list, max_length=50)
    extent_buffer_m: float = Field(
        0,
        ge=0,
        le=100,
        description=(
            "Distance in meters to expand the domain extent outward before "
            "clipping the uploaded polygons. Default 0 clips exactly to the "
            "domain boundary. Maximum: 100 meters."
        ),
    )

    @model_validator(mode="after")
    def at_least_one_feature(self) -> "CreateGeojsonFeatureRequest":
        if not self.features:
            raise ValueError("The FeatureCollection must contain at least one Feature.")
        return self


def parse_upload_crs(body: CreateGeojsonFeatureRequest) -> pyproj.CRS:
    """CRS declared on the upload's ``crs`` block, ``EPSG:4326`` when absent.

    Raises:
        HTTPException: 422 if the CRS can't be parsed.
    """
    crs_name = (body.crs.properties.get("name") if body.crs else None) or DEFAULT_CRS
    try:
        return pyproj.CRS.from_user_input(crs_name)
    except CRSError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=(
                f"Could not parse CRS {crs_name!r}: {exc}. Declare a valid CRS "
                "on the GeoJSON's top-level `crs` block (e.g. `EPSG:4326` or "
                "`EPSG:32612`)."
            ),
        ) from exc


def clip_upload_to_domain(
    body: CreateGeojsonFeatureRequest, domain: dict
) -> tuple[gpd.GeoDataFrame, FeatureGeoreference]:
    """Reproject the uploaded polygons to the domain CRS and clip them to the
    domain extent plus ``extent_buffer_m``.

    Returns:
        The clipped polygons, and the feature georeference: the buffered
        domain bounds in the domain CRS, as for the OSM and Overture sources.

    Raises:
        HTTPException: 422 if the CRS can't be parsed, a polygon is invalid, or
            no polygon intersects the buffered domain.
    """
    crs = parse_upload_crs(body)
    features = [feature.model_dump(mode="json") for feature in body.features]
    gdf = gpd.GeoDataFrame.from_features(features, crs=crs)

    invalid = ~gdf.geometry.is_valid
    if invalid.any():
        index = int(invalid.to_numpy().nonzero()[0][0])
        reason = shapely.is_valid_reason(gdf.geometry.iloc[index])
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=(
                f"Feature {index} has an invalid geometry ({reason}), "
                f"and {int(invalid.sum())} feature(s) are invalid in all. Fix "
                "them, e.g. with QGIS's Fix Geometries tool or shapely's "
                "make_valid, and upload again."
            ),
        )

    extent = buffer_domain(parse_domain_gdf(domain), body.extent_buffer_m)
    clipped = gdf.to_crs(extent.crs).clip(extent, keep_geom_type=True)
    if clipped.empty:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=(
                "None of the polygons intersect the domain. Check that they are "
                "in the declared CRS and cover the domain's area."
            ),
        )

    georeference = FeatureGeoreference(
        crs=str(extent.crs),
        bounds=tuple(float(b) for b in extent.total_bounds),
    )
    return clipped.sort_index().reset_index(drop=True), georeference


async def create_geojson_feature(
    request: Request,
    domain: dict,
    body: CreateGeojsonFeatureRequest,
    feature_type: FeatureType,
) -> Feature:
    """Validate, clip and store an uploaded GeoJSON feature of ``feature_type``."""
    owner_id = request.state.id
    domain_id = domain["id"]

    # dispatch=False: no worker job is commissioned.
    await enforce_create_quotas(FEATURES_COLLECTION, request, dispatch=False)

    gdf, georeference = await asyncio.to_thread(clip_upload_to_domain, body, domain)
    data = await encode_feature_parquet(gdf)

    feature_id = uuid4().hex
    await asyncio.to_thread(upload_feature_parquet, domain_id, feature_id, data)

    now = datetime.now(UTC)
    feature_data = {
        "id": feature_id,
        "domain_id": domain_id,
        "type": feature_type.value,
        "name": body.name,
        "description": body.description,
        "status": JobStatus.completed.value,
        "progress": None,
        "created_on": now,
        "modified_on": now,
        "source": GeojsonSource(extent_buffer_m=body.extent_buffer_m).model_dump(),
        "georeference": georeference.model_dump(),
        "error": None,
        "tags": body.tags,
        "attributions": [],
        "size_bytes": len(data),
        "owner_id": owner_id,
    }
    await set_document_async(FEATURES_COLLECTION, feature_id, feature_data)

    return Feature(**feature_data)
