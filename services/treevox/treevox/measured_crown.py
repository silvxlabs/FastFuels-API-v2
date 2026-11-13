"""Measured crowns for tree voxelization (#617).

Worker-safe: imports only numpy, pandas and fastfuels_core, like
`treevox.voxelize`. Loading a crown Feature into footprints is handler-side
work, in `treevox.crown_feature`.

A measured crown's outline is its CHM cells and its top is each cell's CHM
height; its bottom follows the crown profile up from the crown base.
"""

from __future__ import annotations

import math
import zlib
from dataclasses import dataclass

import numpy as np
import pandas as pd
from fastfuels_core.trees import Tree
from fastfuels_core.voxelization import (
    compute_crown_probability_field,
    sample_occupancy,
    voxelize_measured_crown,
)

# A footprint has volume when some CHM cell rises this many vertical voxels
# above the crown base. Core drops overlaps under 1e-9 of a voxel, so a column
# at least this thick always reaches the output grid.
MEASURED_VOLUME_TOLERANCE = 4e-9


@dataclass(frozen=True)
class Footprint:
    """One measured crown's CHM cells, as a box on the CHM lattice (#617).

    Fields
    ------
    mask
        `(rows, cols)` bool, the crown's cells.
    top
        `(rows, cols)` float, CHM height (m) of each cell; NaN outside `mask`.
    origin
        `(x_west, y_north)` of `mask[0, 0]` in the grid's CRS.
    resolution
        CHM cell size (m).
    """

    mask: np.ndarray
    top: np.ndarray
    origin: tuple[float, float]
    resolution: float


def crown_geometry_type(source_config: dict) -> str:
    """The grid's crown geometry source type; `parametric` when absent."""
    return (source_config.get("crown_geometry_source") or {}).get("type", "parametric")


def footprint_has_volume(
    footprint: Footprint, crown_base_height: float, vr: float
) -> bool:
    """Whether any footprint cell rises above the crown base.

    Mirrors the test in core's `voxelize_measured_crown`, so a footprint that
    passes always yields a crown there.
    """
    tops = footprint.top[footprint.mask]
    return bool(np.any(tops - crown_base_height > MEASURED_VOLUME_TOLERANCE * vr))


def stem_in_crown(x: float, y: float, footprint: Footprint) -> bool:
    """Whether the stem at (x, y) lies in one of the footprint's cells.

    A crown Feature grows each crown from its tree's stem cell, so a crown
    that does not hold its tree's stem was built from another inventory.
    """
    res = footprint.resolution
    col = math.floor((x - footprint.origin[0]) / res)
    row = math.floor((footprint.origin[1] - y) / res)
    nrows, ncols = footprint.mask.shape
    return 0 <= row < nrows and 0 <= col < ncols and bool(footprint.mask[row, col])


CROWN_GEOMETRY_REASONS = (
    "measured",
    "parametric_no_crown",
    "parametric_stem_outside_crown",
    "parametric_no_volume",
)


def plan_measured_crowns(
    trees_in_chunk: pd.DataFrame,
    footprints: dict[int, Footprint],
    vr: float,
) -> tuple[dict[int, Footprint], dict[int, str]]:
    """Choose which trees get a measured crown, and why the rest do not.

    A tree gets a measured crown when the crown Feature holds a footprint for
    its `tree_id` and some footprint cell rises above its crown base, the
    `crown_base_height` column. Every other tree is drawn parametrically.
    `footprints` holds only crowns that contain their tree's stem; a tree
    whose crown does not is flagged in the `_stem_outside_crown` column
    (see `crown_feature.load_footprints`).

    Returns the footprints of trees drawn measured, keyed by `tree_id`, and
    each tree's `CROWN_GEOMETRY_REASONS` entry, keyed by `tree_id`.
    `voxelize_chunk` counts the reasons of the trees it actually draws.
    """
    measured: dict[int, Footprint] = {}
    reasons: dict[int, str] = {}
    tree_ids = trees_in_chunk["tree_id"].to_numpy()
    crown_bases = trees_in_chunk["crown_base_height"].to_numpy(dtype=float)
    if "_stem_outside_crown" in trees_in_chunk:
        stem_outside = trees_in_chunk["_stem_outside_crown"].to_numpy(dtype=bool)
    else:
        stem_outside = np.zeros(len(trees_in_chunk), dtype=bool)
    for tree_id, crown_base_height, outside in zip(tree_ids, crown_bases, stem_outside):
        tree_id = int(tree_id)
        footprint = footprints.get(tree_id)
        if outside:
            reasons[tree_id] = "parametric_stem_outside_crown"
            continue
        if footprint is None:
            reasons[tree_id] = "parametric_no_crown"
            continue
        if footprint_has_volume(footprint, crown_base_height, vr):
            measured[tree_id] = footprint
            reasons[tree_id] = "measured"
        else:
            reasons[tree_id] = "parametric_no_volume"
    return measured, reasons


def measured_crown_seed(base_seed: int, tree_id: int) -> int:
    """Occupancy seed for one measured crown.

    Keyed by the tree rather than the chunk, so a measured crown is the same
    whichever chunk layout renders it.
    """
    return zlib.crc32(f"{base_seed}:tree:{tree_id}".encode())


def measured_halo_cells(
    df: pd.DataFrame, footprints: dict[int, Footprint], dims: dict
) -> int:
    """Halo width in cells that holds every measured crown.

    The farthest any measured crown reaches from its stem's cell, in cells.
    Each footprint's box is converted to output columns and rows with the same
    arithmetic core's `voxelize_measured_crown` uses to place the crown, so
    the crown never extends past it. Stem cells are found as in
    `assign_trees_to_chunks`. Returns 0 when no tree has a footprint.
    """
    hr, nx, ny = dims["hr"], dims["nx"], dims["ny"]
    x_origin, y_origin = dims["x_origin"], dims["y_origin"]
    stem_x = dict(zip(df["tree_id"].astype("int64"), df["x"].astype("float64")))
    stem_y = dict(zip(df["tree_id"].astype("int64"), df["y"].astype("float64")))
    reach = 0
    for tree_id, footprint in footprints.items():
        if tree_id not in stem_x or not footprint.mask.any():
            continue
        stem_col = min(max(math.floor((stem_x[tree_id] - x_origin) / hr), 0), nx - 1)
        stem_row = min(max(math.floor((y_origin - stem_y[tree_id]) / hr), 0), ny - 1)
        res = footprint.resolution
        rows = np.flatnonzero(footprint.mask.any(axis=1))
        cols = np.flatnonzero(footprint.mask.any(axis=0))
        west = footprint.origin[0] - x_origin + cols[0] * res
        east = footprint.origin[0] - x_origin + (cols[-1] + 1) * res
        north = y_origin - footprint.origin[1] + rows[0] * res
        south = y_origin - footprint.origin[1] + (rows[-1] + 1) * res
        reach = max(
            reach,
            stem_col - math.floor(west / hr),
            math.ceil(east / hr) - 1 - stem_col,
            stem_row - math.floor(north / hr),
            math.ceil(south / hr) - 1 - stem_row,
        )
    return reach


def measured_crown_occupancy(
    tree: Tree,
    footprint: Footprint,
    hr: float,
    vr: float,
    x_origin: float,
    y_origin: float,
    seed: int,
) -> tuple[np.ndarray, tuple[int, int, int]] | None:
    """Voxelize one measured crown and draw the voxels it occupies.

    Core's `voxelize_measured_crown` builds the crown's volume fraction on the
    output lattice: each footprint cell is a column from its profile bottom to
    its CHM top. Occupancy is then drawn from it as for a parametric crown, so
    the tree's mass can be spread through it the same way.

    Returns the sampled `(nz, ny, nx)` occupancy and its `(z, row, col)` offset
    on the output grid, or None when the crown has no volume.
    """
    crown = voxelize_measured_crown(
        footprint.mask,
        footprint.top,
        footprint.origin,
        footprint.resolution,
        (tree.x, tree.y),
        tree.crown_base_height,
        tree.crown_profile_model,
        (x_origin, y_origin),
        (hr, vr),
    )
    if crown is None:
        return None
    field, n_occupied = compute_crown_probability_field(
        crown.volume_fraction, alpha=0.5, beta=0.5
    )
    sampled = sample_occupancy(crown.volume_fraction, field, n_occupied, seed=seed)
    return sampled, crown.offset
