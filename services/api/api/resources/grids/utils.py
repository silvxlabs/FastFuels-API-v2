"""
api/v2/resources/grids/utils.py

Shared validation and computation utilities for grid endpoints.
"""

from math import ceil, isclose

from fastapi import HTTPException, status

from api.resources.grids.schema import GridDataChunkMetadata

# Tolerance for comparing floating-point grid transform coefficients in meters.
# 1e-6 m = 1 micrometer; well below any realistic raster precision.
_TRANSFORM_ABS_TOL = 1e-6


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


def validate_band_unit(
    grid_data: dict, grid_id: str, band_key: str, expected_unit: str
) -> None:
    """Validate that a band on a grid has the expected unit.

    Assumes the band already exists on the grid; callers should run
    `validate_grid_has_band` first.

    Args:
        grid_data: Grid document data from Firestore.
        grid_id: Grid ID (for error messages).
        band_key: The band key to check.
        expected_unit: The unit string the band must carry (e.g. 'kg/m³').

    Raises:
        HTTPException(422): If the band's unit does not match.
    """
    band = next(b for b in grid_data["bands"] if b["key"] == band_key)
    actual = band.get("unit")
    if actual != expected_unit:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"Grid {grid_id} band {band_key!r} has unit {actual!r}, "
                f"expected {expected_unit!r}."
            ),
        )


def validate_grid_dimensionality(grid_data: dict, grid_id: str, expected: int) -> None:
    """Validate that a grid is 2D or 3D as expected.

    Dimensionality is read from the grid's georeference shape: 2 for (h, w),
    3 for (z, h, w).

    Args:
        grid_data: Grid document data from Firestore.
        grid_id: Grid ID (for error messages).
        expected: 2 or 3.

    Raises:
        HTTPException(422): If the grid has no georeference or the wrong rank.
    """
    if expected not in {2, 3}:
        raise ValueError("expected must be 2 or 3")
    georeference = grid_data.get("georeference")
    if not georeference:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"Grid {grid_id} has no georeference. The grid must be fully "
                f"processed before it can be used in a combined export."
            ),
        )
    rank = len(georeference.get("shape", []))
    if rank != expected:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(f"Grid {grid_id} is {rank}D, expected {expected}D for this role."),
        )


def validate_grid_resolution_matches(
    grid_data: dict,
    grid_id: str,
    reference_data: dict,
    reference_id: str,
) -> None:
    """Validate that a grid shares CRS, transform, and shape (XY plane) with
    a reference grid.

    Once the alignment field landed (#205), grids on a domain are not
    guaranteed to share an origin — different fetchers can opt into
    ``target="native"`` and end up offset. Composition therefore requires
    matching the full transform (cell size *and* origin) along with CRS.
    For 3D grids, only the XY transform/shape are compared (z is checked
    elsewhere).

    Args:
        grid_data: Grid document data from Firestore.
        grid_id: Grid ID (for error messages).
        reference_data: Reference grid document data (the canopy grid).
        reference_id: Reference grid ID (for error messages).

    Raises:
        HTTPException(422): On CRS, transform, or XY shape mismatch.
    """
    grid_georef = grid_data["georeference"]
    ref_georef = reference_data["georeference"]

    grid_crs = grid_georef.get("crs")
    ref_crs = ref_georef.get("crs")
    if grid_crs != ref_crs:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"Grid {grid_id} CRS ({grid_crs}) does not match "
                f"reference grid {reference_id} ({ref_crs}). Run "
                f"POST /v2/domains/{{domain_id}}/grids/resample "
                f'with alignment.target="grid" to align.'
            ),
        )

    grid_transform = grid_georef["transform"]
    ref_transform = ref_georef["transform"]
    transforms_match = all(
        isclose(g, r, abs_tol=_TRANSFORM_ABS_TOL)
        for g, r in zip(grid_transform, ref_transform)
    )
    if not transforms_match:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"Grid {grid_id} transform {grid_transform} does not match "
                f"reference grid {reference_id} {ref_transform}. The grids "
                f"have different cell sizes or origins. Run "
                f"POST /v2/domains/{{domain_id}}/grids/resample "
                f'with alignment.target="grid" to align.'
            ),
        )

    grid_xy_shape = tuple(grid_georef["shape"][-2:])
    ref_xy_shape = tuple(ref_georef["shape"][-2:])
    if grid_xy_shape != ref_xy_shape:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"Grid {grid_id} XY shape {grid_xy_shape} does not match "
                f"reference grid {reference_id} {ref_xy_shape}. Run "
                f"POST /v2/domains/{{domain_id}}/grids/resample "
                f'with alignment.target="grid" to align.'
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
    chunk_shape: list[int] | tuple[int, ...],
    chunk_index: int,
) -> GridDataChunkMetadata:
    """Compute metadata for a single chunk of a grid.

    Args:
        georeference: Grid georeference dict with 'shape' and 'transform'.
        chunk_shape: Zarr chunk shape. 2D: (height, width). 3D: (z, height, width).
        chunk_index: Zero-based flat chunk index. 2D order is y, x. 3D order is
            z, y, x.

    Returns:
        GridDataChunkMetadata with shape, offset, and transform for the chunk.

    Raises:
        ValueError: If chunk_index is out of range.
    """
    grid_shape = tuple(georeference["shape"])
    chunk_shape = tuple(chunk_shape)
    if len(grid_shape) != len(chunk_shape) or len(grid_shape) not in {2, 3}:
        raise ValueError(
            "georeference shape and chunk_shape must both be 2D or both be 3D "
            f"(shape={list(grid_shape)}, chunk_shape={list(chunk_shape)})"
        )

    if len(grid_shape) == 3:
        grid_z, grid_h, grid_w = grid_shape
        chunk_z, chunk_h, chunk_w = chunk_shape

        num_chunks_z = ceil(grid_z / chunk_z)
        num_chunks_y = ceil(grid_h / chunk_h)
        num_chunks_x = ceil(grid_w / chunk_w)
        chunks_per_z = num_chunks_y * num_chunks_x
        total_chunks = num_chunks_z * chunks_per_z

        if chunk_index < 0 or chunk_index >= total_chunks:
            raise ValueError(
                f"chunk_index {chunk_index} out of range for grid with "
                f"{total_chunks} chunks (shape={list(grid_shape)}, "
                f"chunk_shape={list(chunk_shape)})"
            )

        chunk_z_index = chunk_index // chunks_per_z
        chunk_yx_index = chunk_index % chunks_per_z
        chunk_row = chunk_yx_index // num_chunks_x
        chunk_col = chunk_yx_index % num_chunks_x

        z_offset = chunk_z_index * chunk_z
        row_offset = chunk_row * chunk_h
        col_offset = chunk_col * chunk_w

        actual_z = min(chunk_z, grid_z - z_offset)
        actual_h = min(chunk_h, grid_h - row_offset)
        actual_w = min(chunk_w, grid_w - col_offset)

        a, b, c, d, e, f = georeference["transform"]
        chunk_c = c + a * col_offset + b * row_offset
        chunk_f = f + d * col_offset + e * row_offset
        try:
            grid_z_origin = georeference["z_origin"]
            z_resolution = georeference["z_resolution"]
        except KeyError as exc:
            raise ValueError(
                "3D georeference requires z_origin and z_resolution."
            ) from exc
        z_origin = grid_z_origin + z_offset * z_resolution

        return GridDataChunkMetadata(
            index=chunk_index,
            shape=(actual_z, actual_h, actual_w),
            offset=(z_offset, row_offset, col_offset),
            transform=(a, b, chunk_c, d, e, chunk_f),
            z_origin=z_origin,
            z_resolution=z_resolution,
        )

    grid_h, grid_w = grid_shape
    chunk_h, chunk_w = chunk_shape

    num_chunks_y = ceil(grid_h / chunk_h)
    num_chunks_x = ceil(grid_w / chunk_w)
    total_chunks = num_chunks_y * num_chunks_x

    if chunk_index < 0 or chunk_index >= total_chunks:
        raise ValueError(
            f"chunk_index {chunk_index} out of range for grid with "
            f"{total_chunks} chunks (shape={list(grid_shape)}, "
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
) -> tuple[slice, slice] | tuple[slice, slice, slice]:
    """Compute slices for reading a chunk from a zarr array.

    Args:
        meta: Chunk metadata from compute_chunk_metadata().

    Returns:
        2D: (row_slice, col_slice). 3D: (z_slice, row_slice, col_slice).
    """
    if len(meta.shape) == 3:
        z_start, row_start, col_start = meta.offset
        z_end = z_start + meta.shape[0]
        row_end = row_start + meta.shape[1]
        col_end = col_start + meta.shape[2]
        return (
            slice(z_start, z_end),
            slice(row_start, row_end),
            slice(col_start, col_end),
        )

    row_start, col_start = meta.offset
    row_end = row_start + meta.shape[0]
    col_end = col_start + meta.shape[1]
    return slice(row_start, row_end), slice(col_start, col_end)
