"""
api/v2/resources/grids/utils.py

Shared validation and computation utilities for grid endpoints.
"""

from math import ceil

from fastapi import HTTPException, status

from api.resources.grids.schema import GridDataChunkMetadata


def validate_grid_has_band(
    grid_data: dict, grid_id: str, required: str | list[str]
) -> None:
    """Validate that a grid has one or more required bands.

    Args:
        grid_data: Grid document data from Firestore.
        grid_id: Grid ID (for error messages).
        required: A band key or list of band keys that must be present.

    Raises:
        HTTPException(422): If any required band is missing.
    """
    band_keys = {b["key"] for b in grid_data.get("bands", [])}
    if isinstance(required, str):
        required = [required]
    missing = [k for k in required if k not in band_keys]
    if missing:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"Source grid {grid_id} is missing required bands: {missing}. "
                f"Available bands: {sorted(band_keys)}"
            ),
        )


def validate_grid_has_georeference(grid_data: dict, grid_id: str) -> None:
    """Validate that a grid has a georeference.

    Args:
        grid_data: Grid document data from Firestore.
        grid_id: Grid ID (for error messages).

    Raises:
        HTTPException(422): If the grid has no georeference.
    """
    if not grid_data.get("georeference"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"Source grid {grid_id} has no georeference. "
                f"The grid must be fully processed before it can be resampled."
            ),
        )


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


def compute_chunk_slices(
    meta: GridDataChunkMetadata,
) -> tuple[slice, slice]:
    """Compute row and column slices for reading a chunk from a zarr array.

    Args:
        meta: Chunk metadata from compute_chunk_metadata().

    Returns:
        A (row_slice, col_slice) tuple for indexing into the array.
    """
    row_start, col_start = meta.offset
    row_end = row_start + meta.shape[0]
    col_end = col_start + meta.shape[1]
    return slice(row_start, row_end), slice(col_start, col_end)
