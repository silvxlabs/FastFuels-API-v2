"""
api/v2/resources/grids/data/router.py

Read-only endpoints for streaming grid raster data directly from Zarr on GCS.
"""

import numpy as np
from fastapi import APIRouter, HTTPException, Query, Request, Response, status

from api.db.documents import get_document_async
from api.dependencies import VerifiedDomain
from api.resources.grids.data.schema import (
    GridDataChunkMetadata,
    GridDataFormat,
    GridDataOrder,
    GridDataResponse,
    compute_chunk_metadata,
)
from api.resources.grids.data.zarr import get_grid_array
from api.resources.grids.utils import validate_grid_has_band
from lib.config import GRIDS_COLLECTION

router = APIRouter()


@router.get(
    "/chunks/{chunk_index}",
    response_model=GridDataChunkMetadata,
    status_code=status.HTTP_200_OK,
    summary="Get chunk metadata",
)
async def get_chunk_metadata(
    request: Request,
    domain: VerifiedDomain,
    grid_id: str,
    chunk_index: int,
):
    """Return shape, offset, and affine transform for a single chunk.

    Pure arithmetic on Firestore data — no GCS reads.
    """
    _, snapshot = await get_document_async(
        GRIDS_COLLECTION,
        grid_id,
        owner_id=request.state.id,
        domain_id=domain["id"],
        document_status="completed",
    )
    grid_data = snapshot.to_dict()

    georeference = grid_data.get("georeference")
    chunk_shape = grid_data.get("chunk_shape")
    if not georeference or not chunk_shape:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Grid {grid_id} is missing georeference or chunk_shape.",
        )

    try:
        return compute_chunk_metadata(georeference, chunk_shape, chunk_index)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )


@router.get(
    "/data",
    response_model=GridDataResponse,
    status_code=status.HTTP_200_OK,
    summary="Get band data for a chunk",
)
async def get_grid_data(
    request: Request,
    domain: VerifiedDomain,
    grid_id: str,
    band: str = Query(..., description="Band key to read (e.g., 'elevation', 'fbfm')."),
    chunk: int = Query(0, ge=0, description="Chunk index (default 0)."),
    data_format: GridDataFormat = Query(
        GridDataFormat.json, alias="format", description="Response format."
    ),
    order: GridDataOrder = Query(
        GridDataOrder.C, description="Array memory order for flattening."
    ),
):
    """Read a single chunk of a single band from the Zarr store on GCS.

    Supports JSON and binary (raw bytes) response formats.
    """
    _, snapshot = await get_document_async(
        GRIDS_COLLECTION,
        grid_id,
        owner_id=request.state.id,
        domain_id=domain["id"],
        document_status="completed",
    )
    grid_data = snapshot.to_dict()

    georeference = grid_data.get("georeference")
    chunk_shape = grid_data.get("chunk_shape")
    if not georeference or not chunk_shape:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Grid {grid_id} is missing georeference or chunk_shape.",
        )

    validate_grid_has_band(grid_data, grid_id, band)

    try:
        meta = compute_chunk_metadata(georeference, chunk_shape, chunk)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )

    row_start, col_start = meta.offset
    row_end = row_start + meta.shape[0]
    col_end = col_start + meta.shape[1]

    array = await get_grid_array(grid_id, band)
    data = await array.getitem((slice(row_start, row_end), slice(col_start, col_end)))

    if not isinstance(data, np.ndarray):
        data = np.asarray(data)

    if data_format == GridDataFormat.binary:
        raw = data.ravel(order=order.value).tobytes()
        return Response(
            content=raw,
            media_type="application/octet-stream",
            headers={
                "X-Data-Shape": ",".join(str(s) for s in data.shape),
                "X-Data-Dtype": str(data.dtype),
                "X-Data-Order": order.value,
            },
        )

    return GridDataResponse(
        shape=list(data.shape),
        order=order.value,
        data=data.ravel(order=order.value).tolist(),
    )
