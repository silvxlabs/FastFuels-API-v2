"""
api/v2/resources/features/storage.py

Write feature GeoParquet blobs from the API.

Same on-disk format as ``etcher.storage.save_features``: zstd compression and
row groups of 1000, each row group being one ``partition_index`` of the data
endpoint. Uploads go through pyarrow's native GCS filesystem, not gcsfs (#265).
"""

import asyncio
import io

import geopandas as gpd
import pyarrow as pa
from fastapi import HTTPException, status
from pyarrow import fs as pafs

from lib.config import FEATURES_BUCKET


def feature_blob_path(domain_id: str, feature_id: str) -> str:
    """GCS path of a feature's GeoParquet blob."""
    return f"gs://{FEATURES_BUCKET}/{domain_id}/{feature_id}.parquet"


def _to_parquet_bytes(gdf: gpd.GeoDataFrame) -> bytes:
    buffer = io.BytesIO()
    gdf.to_parquet(buffer, compression="zstd", row_group_size=1000)
    return buffer.getvalue()


async def encode_feature_parquet(gdf: gpd.GeoDataFrame) -> bytes:
    """Serialize features to GeoParquet bytes, off the event loop.

    Raises:
        HTTPException: 422 if a column holds values of more than one type,
            e.g. numbers in some rows and text in others.
    """
    try:
        return await asyncio.to_thread(_to_parquet_bytes, gdf)
    except (pa.ArrowInvalid, pa.ArrowTypeError) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=(
                f"Feature properties can't be stored: {exc}. Each property "
                "must have one type across all features, e.g. always a number "
                "or always text."
            ),
        ) from exc


def upload_feature_parquet(domain_id: str, feature_id: str, data: bytes) -> None:
    """Write encoded GeoParquet bytes as a feature's blob."""
    filesystem, path = pafs.FileSystem.from_uri(
        feature_blob_path(domain_id, feature_id)
    )
    with filesystem.open_output_stream(path) as out:
        out.write(data)
