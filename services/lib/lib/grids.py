"""Shared grid utilities used by api, griddle, and treevox services."""

from collections.abc import Sequence
from math import ceil

AXES_2D = ("y", "x")
AXES_3D = ("z", "y", "x")


def compute_chunks_doc(
    grid_shape: Sequence[int],
    chunk_shape: Sequence[int],
) -> dict:
    """Build the persisted ``chunks`` subdocument for a grid.

    Args:
        grid_shape: Grid pixel shape from the georeference. 2D: (y, x). 3D: (z, y, x).
        chunk_shape: Zarr chunk shape with the same rank as ``grid_shape``.

    Returns:
        ``{"shape": [...], "count": int, "count_by_axis": {"y": ..., "x": ...}}``.
        For 3D grids ``count_by_axis`` also includes a ``"z"`` key.
    """
    if len(grid_shape) != len(chunk_shape) or len(grid_shape) not in (2, 3):
        raise ValueError(
            "grid_shape and chunk_shape must both be 2D or both be 3D "
            f"(grid_shape={list(grid_shape)}, chunk_shape={list(chunk_shape)})"
        )
    axes = AXES_2D if len(grid_shape) == 2 else AXES_3D
    count_by_axis = {ax: ceil(g / c) for ax, g, c in zip(axes, grid_shape, chunk_shape)}
    count = 1
    for v in count_by_axis.values():
        count *= v
    return {
        "shape": list(chunk_shape),
        "count": count,
        "count_by_axis": count_by_axis,
    }
