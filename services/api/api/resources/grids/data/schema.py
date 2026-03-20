"""
api/v2/resources/grids/data/schema.py

Response models and chunk math for grid data streaming endpoints.
"""

from enum import StrEnum
from math import ceil
from typing import Literal

from pydantic import BaseModel


class GridDataFormat(StrEnum):
    json = "json"
    binary = "binary"


class GridDataOrder(StrEnum):
    C = "C"
    F = "F"


class GridDataChunkMetadata(BaseModel):
    index: int
    shape: tuple[int, int]
    offset: tuple[int, int]
    transform: tuple[float, float, float, float, float, float]


class GridDataResponse(BaseModel):
    shape: list[int]
    order: Literal["C", "F"]
    data: list


def compute_chunk_metadata(
    georeference: dict,
    chunk_shape: list[int] | tuple[int, int],
    chunk_index: int,
) -> GridDataChunkMetadata:
    """Compute metadata for a single chunk of a grid.

    Args:
        georeference: Grid georeference dict with 'shape' and 'transform'.
        chunk_shape: Zarr chunk shape (height, width).
        chunk_index: Zero-based flat chunk index (row-major order).

    Returns:
        GridDataChunkMetadata with shape, offset, and transform for the chunk.

    Raises:
        ValueError: If chunk_index is out of range.
    """
    grid_h, grid_w = georeference["shape"]
    chunk_h, chunk_w = chunk_shape

    num_chunks_y = ceil(grid_h / chunk_h)
    num_chunks_x = ceil(grid_w / chunk_w)
    total_chunks = num_chunks_y * num_chunks_x

    if chunk_index < 0 or chunk_index >= total_chunks:
        raise ValueError(
            f"chunk_index {chunk_index} out of range for grid with "
            f"{total_chunks} chunks (shape={georeference['shape']}, "
            f"chunk_shape={list(chunk_shape)})"
        )

    chunk_row = chunk_index // num_chunks_x
    chunk_col = chunk_index % num_chunks_x

    row_offset = chunk_row * chunk_h
    col_offset = chunk_col * chunk_w

    # Actual shape (smaller for edge chunks)
    actual_h = min(chunk_h, grid_h - row_offset)
    actual_w = min(chunk_w, grid_w - col_offset)

    # Compute per-chunk affine transform
    a, b, c, d, e, f = georeference["transform"]
    chunk_c = c + a * col_offset + b * row_offset
    chunk_f = f + d * col_offset + e * row_offset

    return GridDataChunkMetadata(
        index=chunk_index,
        shape=(actual_h, actual_w),
        offset=(row_offset, col_offset),
        transform=(a, b, chunk_c, d, e, chunk_f),
    )
