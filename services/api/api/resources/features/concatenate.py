"""
api/v2/resources/features/concatenate.py

Concatenate source for road, water and building features.

Creates a new feature whose rows are the rows of other features of the same
type, appended in order. There is no geometry operation: overlapping polygons
stay separate rows. Columns are the union of the inputs' columns, null-filled,
plus a per-row ``source_feature_id``. Runs in-request with status
``completed``.
"""

import asyncio
from datetime import UTC, datetime
from typing import Literal
from uuid import uuid4

import geopandas as gpd
import pandas as pd
from fastapi import HTTPException, Request, status
from pydantic import BaseModel, Field, field_validator

from api.db.documents import firestore_client, set_document_async
from api.quota import enforce_create_quotas
from api.resources.attribution import merge_attributions
from api.resources.features.schema import Feature, FeatureGeoreference, FeatureType
from api.resources.features.storage import (
    encode_feature_parquet,
    feature_blob_path,
    upload_feature_parquet,
)
from api.schema import JobStatus
from lib.config import FEATURES_COLLECTION
from lib.domain_utils import parse_domain_gdf

SOURCE_FEATURE_ID_COLUMN = "source_feature_id"


class ConcatenateSource(BaseModel):
    """Source metadata stored on a concatenated feature."""

    product: Literal["concatenate"] = "concatenate"
    description: Literal["Rows of other features, appended"] = (
        "Rows of other features, appended"
    )
    source_feature_ids: list[str]


class ConcatenateFeaturesRequest(BaseModel):
    """Request body for concatenating features of one type."""

    source_feature_ids: list[str] = Field(
        ...,
        min_length=2,
        max_length=100,
        description=(
            "IDs of the features to combine, in the order their rows are "
            "appended. Each must be a completed feature of the path's type in "
            "this domain, and listed once."
        ),
    )
    name: str = Field("", max_length=255)
    description: str = Field("", max_length=2000)
    tags: list[str] = Field(default_factory=list, max_length=50)

    @field_validator("source_feature_ids")
    @classmethod
    def ids_are_unique(cls, value: list[str]) -> list[str]:
        duplicates = sorted({fid for fid in value if value.count(fid) > 1})
        if duplicates:
            raise ValueError(f"Feature IDs are listed more than once: {duplicates}.")
        return value


async def load_features(feature_ids: list[str]) -> dict[str, dict | None]:
    """Feature documents by ID in one batched read; ``None`` when missing."""
    refs = [
        firestore_client.collection(FEATURES_COLLECTION).document(fid)
        for fid in feature_ids
    ]
    snapshots = {snap.id: snap async for snap in firestore_client.get_all(refs)}
    docs = {}
    for fid in feature_ids:
        snap = snapshots.get(fid)
        docs[fid] = snap.to_dict() if snap is not None and snap.exists else None
    return docs


def validate_source_features(
    features: dict[str, dict | None],
    owner_id: str,
    domain_id: str,
    feature_type: FeatureType,
) -> None:
    """Check every input is a completed feature of ``feature_type`` in this domain.

    Raises:
        HTTPException: 422 naming the first input that isn't.
    """
    for fid, doc in features.items():
        if (
            doc is None
            or doc.get("owner_id") != owner_id
            or doc.get("domain_id") != domain_id
        ):
            detail = f"Feature {fid!r} does not exist in this domain."
        elif doc.get("type") != feature_type.value:
            detail = (
                f"Feature {fid!r} is a {doc.get('type')} feature, not a "
                f"{feature_type.value} feature. Only features of one type can be "
                "concatenated."
            )
        elif doc.get("status") != JobStatus.completed.value:
            detail = (
                f"Feature {fid!r} has status {doc.get('status')!r}, expected "
                "'completed'. Wait for it to finish before concatenating it."
            )
        else:
            continue
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=detail
        )


def append_feature_rows(
    domain: dict, feature_ids: list[str]
) -> tuple[gpd.GeoDataFrame, str]:
    """Read the features' rows and append them in order, in the domain CRS.

    Returns:
        The appended rows with a ``source_feature_id`` column, and the domain
        CRS as a string.
    """
    crs = parse_domain_gdf(domain).crs
    frames = []
    for fid in feature_ids:
        gdf = gpd.read_parquet(feature_blob_path(domain["id"], fid)).to_crs(crs)
        gdf[SOURCE_FEATURE_ID_COLUMN] = fid
        frames.append(gdf)
    rows = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=crs)
    return rows, str(crs)


def union_bounds(features: list[dict]) -> tuple[float, float, float, float]:
    """Bounding box covering every input feature's georeference bounds."""
    bounds = [doc["georeference"]["bounds"] for doc in features]
    return (
        min(b[0] for b in bounds),
        min(b[1] for b in bounds),
        max(b[2] for b in bounds),
        max(b[3] for b in bounds),
    )


async def create_concatenated_feature(
    request: Request,
    domain: dict,
    body: ConcatenateFeaturesRequest,
    feature_type: FeatureType,
) -> Feature:
    """Validate the inputs, append their rows, and store the new feature."""
    owner_id = request.state.id
    domain_id = domain["id"]

    # dispatch=False: no worker job is commissioned.
    await enforce_create_quotas(FEATURES_COLLECTION, request, dispatch=False)

    sources = await load_features(body.source_feature_ids)
    validate_source_features(sources, owner_id, domain_id, feature_type)
    source_docs = list(sources.values())

    rows, crs = await asyncio.to_thread(
        append_feature_rows, domain, body.source_feature_ids
    )
    data = await encode_feature_parquet(rows)

    feature_id = uuid4().hex
    await asyncio.to_thread(upload_feature_parquet, domain_id, feature_id, data)

    georeference = FeatureGeoreference(crs=crs, bounds=union_bounds(source_docs))
    source = ConcatenateSource(source_feature_ids=body.source_feature_ids)

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
        "source": source.model_dump(),
        "georeference": georeference.model_dump(),
        "error": None,
        "tags": body.tags,
        "attributions": merge_attributions(
            *(doc.get("attributions") for doc in source_docs)
        ),
        "size_bytes": len(data),
        "owner_id": owner_id,
    }
    await set_document_async(FEATURES_COLLECTION, feature_id, feature_data)

    return Feature(**feature_data)
