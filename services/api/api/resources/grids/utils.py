"""
api/v2/resources/grids/utils.py

Shared validation utilities for grid endpoints.
"""

from fastapi import HTTPException, status


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


def validate_grid_has_chunk_shape(grid_data: dict, grid_id: str) -> None:
    """Validate that a grid has a chunk_shape.

    Args:
        grid_data: Grid document data from Firestore.
        grid_id: Grid ID (for error messages).

    Raises:
        HTTPException(422): If the grid has no chunk_shape.
    """
    if not grid_data.get("chunk_shape"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Grid {grid_id} is missing chunk_shape.",
        )
