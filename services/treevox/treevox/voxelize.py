"""Pure compute for tree-inventory voxelization.

Worker-safe: imports only numpy, pandas, and fastfuels_core. Never imports
xarray, rioxarray, gcsfs, zarr, or treevox.storage.

Contains: grid dimension computation, chunk math, tree construction, per-chunk
biomass cache, and the per-chunk voxelization loop.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from fastfuels_core.trees import Tree
from fastfuels_core.voxelization import (
    VoxelizedTree,
    discretize_crown_profile,
    sample_occupied_cells,
)


@dataclass(frozen=True)
class CacheEntry:
    """Per-chunk cache entry for one group of morphologically equivalent trees.

    `biomass_arrays` are the pre-sampled realizations. The remaining fields
    carry the bin-representative tree's placement attrs — every tree sharing
    this `_cache_key` uses these values, so `build_tree` runs once per group
    instead of once per tree during chunk rendering.
    """

    biomass_arrays: list[np.ndarray]
    crown_base_height: float
    foliage_sav: float
    species_code: int


# Constants


CHUNK_LENGTH_METERS = 1000  # v1 default horizontal chunk size
OVERLAP_CELLS = 10  # halo cells for batch union reads (v1 line 525)
MAX_BIOMASS_ARRAY_CACHE = 100  # v1 line 42

# Tree-binning widths for cache grouping — see TREEVOX.md "Tree-binning cache key".
HEIGHT_BIN_M = 1.0
DBH_BIN_CM = 2.75
CR_BIN = 0.1

# Map API biomass_model strings to fastfuels-core model names.
BIOMASS_MODEL_MAP = {"nsvb": "NSVB", "jenkins": "jenkins", "inventory": "NSVB"}


# Errors


class InvalidResolutionError(ValueError):
    """Resolution / domain combination produces a degenerate grid."""


def compute_grid_dimensions(
    domain_gdf,
    df: pd.DataFrame,
    resolution: tuple[float, float, float],
) -> dict:
    """Compute grid dimensions from a domain GeoDataFrame, tree DataFrame, and resolution.

    Snaps domain bounds outward to the nearest multiple of the horizontal
    resolution so cell boundaries align cleanly. No extra padding is added —
    the domain resource already handles spatial padding via pad_to_resolution.

    Returns a dict with: nx, ny, nz, hr, vr, x_origin, y_origin, z_origin,
    crs, transform, x_coords, y_coords, z_coords.
    """
    hr_x, hr_y, vr = resolution
    if not math.isclose(hr_x, hr_y):
        raise InvalidResolutionError(
            f"Anisotropic resolution (hr_x={hr_x}, hr_y={hr_y}) is not supported; "
            f"fastfuels-core requires hr_x == hr_y."
        )
    if hr_x <= 0 or vr <= 0:
        raise InvalidResolutionError(
            f"Resolution must be positive (got hr={hr_x}, vr={vr})."
        )
    hr = hr_x

    domain_minx, domain_miny, domain_maxx, domain_maxy = domain_gdf.total_bounds

    # Snap bounds outward to the nearest multiple of hr.
    minx = math.floor(domain_minx / hr) * hr
    miny = math.floor(domain_miny / hr) * hr
    maxx = math.ceil(domain_maxx / hr) * hr
    maxy = math.ceil(domain_maxy / hr) * hr

    nx = max(1, int(round((maxx - minx) / hr)))
    ny = max(1, int(round((maxy - miny) / hr)))

    if df.empty:
        max_height = vr
    else:
        max_height = float(math.ceil(df["height"].max()))
    nz = max(1, int(math.ceil(max_height / vr)))

    if nx * ny * nz == 0:
        raise InvalidResolutionError(
            f"Resolution {resolution} with snapped bounds "
            f"[{minx}, {miny}, {maxx}, {maxy}] and max_height {max_height} "
            f"produces a degenerate grid."
        )

    # Cell-center coords.
    x_coords = minx + (np.arange(nx) + 0.5) * hr
    # y is decreasing from north (top) to south (bottom) in raster convention.
    y_coords = maxy - (np.arange(ny) + 0.5) * hr
    z_coords = (np.arange(nz) + 0.5) * vr

    # rasterio-style affine: (a, b, c, d, e, f) where
    # x = a*col + b*row + c, y = d*col + e*row + f.
    transform = (hr, 0.0, minx, 0.0, -hr, maxy)

    crs = str(domain_gdf.crs) if domain_gdf.crs is not None else ""

    return {
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "hr": hr,
        "vr": vr,
        "x_origin": minx,
        "y_origin": maxy,
        "z_origin": 0.0,
        "crs": crs,
        "transform": transform,
        "x_coords": x_coords,
        "y_coords": y_coords,
        "z_coords": z_coords,
    }


def build_tree(row, source_config: dict) -> Tree:
    """Construct a fastfuels_core.Tree from a v2 inventory row.

    V2 columns map directly to Tree kwargs (no renames):
      fia_species_code -> species_code, dbh -> diameter, etc.

    `crown_fuel_load` is only supplied when `biomass_model == "inventory"`;
    otherwise biomass is computed allometrically via NSVB or Jenkins.
    """
    biomass_model = source_config["biomass_model"]
    crown_fuel_load = None
    if biomass_model == "inventory":
        column = source_config["biomass_column"]
        crown_fuel_load = float(row[column])

    return Tree(
        species_code=int(row["fia_species_code"]),
        status_code=int(row["fia_status_code"]),
        diameter=float(row["dbh"]),
        height=float(row["height"]),
        crown_ratio=float(row["crown_ratio"]),
        x=float(row["x"]),
        y=float(row["y"]),
        crown_profile_model_type=source_config["crown_profile_model"],
        biomass_allometry_model_type=BIOMASS_MODEL_MAP[biomass_model],
        crown_fuel_load=crown_fuel_load,
    )


def compute_cache_keys(df: pd.DataFrame) -> pd.Series:
    """Group trees into cache-equivalence classes by binned characteristics.

    Trees with the same (species, binned dbh, binned height, binned crown_ratio)
    share voxelized biomass realizations — they're morphologically
    indistinguishable within the chosen bin widths. Returns integer codes via
    `groupby().ngroup()`.

    See TREEVOX.md for rationale and bin widths.
    """
    dbh_bin = (df["dbh"] / DBH_BIN_CM).astype("int64")
    height_bin = (df["height"] / HEIGHT_BIN_M).astype("int64")
    cr_bin = (df["crown_ratio"] / CR_BIN).round().astype("int64")
    return df.groupby(
        [df["fia_species_code"].astype("int64"), dbh_bin, height_bin, cr_bin],
        sort=False,
    ).ngroup()


def calculate_arrays_to_cache(
    nonzero_voxels: int,
    tree_frequency: int,
    max_cache: int = MAX_BIOMASS_ARRAY_CACHE,
) -> int:
    """V1's 4/5-power scaling: more unique biomass realizations for trees
    with larger / more frequent crowns; capped by frequency and max_cache.
    """
    from_volume = int(nonzero_voxels ** (4 / 5)) if nonzero_voxels > 0 else 1
    return max(1, min(from_volume, max(1, tree_frequency), max_cache))


def assign_trees_to_chunks(
    df: pd.DataFrame,
    x_origin: float,
    y_origin: float,
    hr: float,
    nx: int,
    ny: int,
    chunk_xy: int,
) -> pd.DataFrame:
    """Assign each tree to its (row_chunk, col_chunk) and sort by height ASC.

    Returns a new DataFrame with added `row_chunk` and `col_chunk` columns,
    sorted such that tallest trees within each chunk are iterated last —
    this is how overwrite-style bands (spcd, tree_id, savr, fuel_moisture)
    achieve the "tallest tree wins" overlap policy documented on the API.
    """
    col_cell = np.floor((df["x"].to_numpy() - x_origin) / hr).astype("int64")
    row_cell = np.floor((y_origin - df["y"].to_numpy()) / hr).astype("int64")
    col_cell = np.clip(col_cell, 0, nx - 1)
    row_cell = np.clip(row_cell, 0, ny - 1)

    # assign() produces a new frame that shares existing columns' data; the
    # only materializing copy happens inside sort_values below.
    out = df.assign(
        row_chunk=row_cell // chunk_xy,
        col_chunk=col_cell // chunk_xy,
    )
    out = out.sort_values(
        by=["row_chunk", "col_chunk", "height"],
        kind="stable",
        ignore_index=True,
    )
    return out


def batch_union_slices(
    chunk_batch: list[tuple[int, int]],
    ny: int,
    nx: int,
    chunk_xy: int,
    overlap_cells: int = OVERLAP_CELLS,
) -> tuple[slice, slice]:
    """Compute (y_slice, x_slice) covering a batch of chunks plus halo.

    Slices are clamped to [0, ny) / [0, nx). Used to pull one halo-extended
    region from zarr per batch so disjoint worker writes can later be merged.
    """
    if not chunk_batch:
        raise ValueError("chunk_batch must be non-empty")

    min_y = min_x = float("inf")
    max_y = max_x = -1
    for row, col in chunk_batch:
        y_start = max(0, row * chunk_xy - overlap_cells)
        y_end = min(ny, (row + 1) * chunk_xy + overlap_cells)
        x_start = max(0, col * chunk_xy - overlap_cells)
        x_end = min(nx, (col + 1) * chunk_xy + overlap_cells)
        min_y = min(min_y, y_start)
        min_x = min(min_x, x_start)
        max_y = max(max_y, y_end)
        max_x = max(max_x, x_end)

    return slice(int(min_y), int(max_y)), slice(int(min_x), int(max_x))


def chunk_slice(
    chunk_location: tuple[int, int],
    ny: int,
    nx: int,
    chunk_xy: int,
    overlap_cells: int = OVERLAP_CELLS,
) -> tuple[slice, slice]:
    """Halo-extended (y_slice, x_slice) for a single chunk."""
    row, col = chunk_location
    y_start = max(0, row * chunk_xy - overlap_cells)
    y_end = min(ny, (row + 1) * chunk_xy + overlap_cells)
    x_start = max(0, col * chunk_xy - overlap_cells)
    x_end = min(nx, (col + 1) * chunk_xy + overlap_cells)
    return slice(y_start, y_end), slice(x_start, x_end)


def build_chunk_cache(
    trees_in_chunk: pd.DataFrame,
    hr: float,
    vr: float,
    source_config: dict,
    rng: np.random.Generator,
) -> dict[int, CacheEntry]:
    """Build per-chunk cache indexed by `_cache_key`.

    For each unique cache_key in `trees_in_chunk`:
      1. Build a Tree from the group's first row.
      2. `discretize_crown_profile` -> canopy volume-fraction mask.
      3. Sample N biomass realizations, where N scales with nonzero voxel
         count and tree frequency (v1's `calculate_arrays_to_cache`).
      4. Stash the bin-representative tree's placement attrs alongside the
         biomass arrays so `voxelize_chunk` doesn't rebuild Tree objects
         per-row.

    `rng` seeds `sample_occupied_cells` for deterministic output.
    """
    cache: dict[int, CacheEntry] = {}
    if trees_in_chunk.empty:
        return cache

    for cache_key, group in trees_in_chunk.groupby("_cache_key", sort=False):
        first_row = group.iloc[0]
        try:
            tree = build_tree(first_row, source_config)
            canopy_mask = discretize_crown_profile(tree, hr, vr)
        except Exception:
            # Degenerate tree (e.g. allometric failure) — skip entry, like v1.
            continue
        nonzero = int(np.count_nonzero(canopy_mask))
        if nonzero == 0:
            # Empty crown — no voxels to distribute biomass into.
            continue
        num_to_cache = calculate_arrays_to_cache(nonzero, len(group))
        arrays: list[np.ndarray] = []
        for _ in range(num_to_cache):
            seed = int(rng.integers(1, 2**31 - 1))
            try:
                sampled = sample_occupied_cells(
                    canopy_mask, alpha=0.5, beta=0.5, seed=seed
                )
                vt = VoxelizedTree(tree, sampled, hr, vr)
                biomass = vt.distribute_biomass()
            except Exception:
                continue
            # Guard against divide-by-zero in VoxelizedTree.distribute_biomass
            # (foliage_biomass / 0 volume -> inf/nan). Replace non-finite with
            # 0 so downstream accumulation never contaminates the zarr store.
            if not np.all(np.isfinite(biomass)):
                biomass = np.nan_to_num(biomass, nan=0.0, posinf=0.0, neginf=0.0)
            arrays.append(biomass)
        if arrays:
            cache[int(cache_key)] = CacheEntry(
                biomass_arrays=arrays,
                crown_base_height=float(tree.crown_base_height),
                foliage_sav=float(tree.foliage_sav),
                species_code=int(tree.species_code),
            )
    return cache


# Per-chunk voxelization


def _tree_cell_indices(
    x: float, y: float, x_origin: float, y_origin: float, hr: float
) -> tuple[int, int]:
    """World coords → absolute (col, row) cell indices.

    `y_origin` is the NORTH edge of the grid; rows increase southward. Uses
    floor so a stem exactly on a cell boundary lands in the cell to the
    east/south (the half-open [start, start+hr) convention).
    """
    abs_col = int(math.floor((x - x_origin) / hr))
    abs_row = int(math.floor((y_origin - y) / hr))
    return abs_col, abs_row


def _clip_1d(start: int, span: int, dim: int) -> tuple[slice, slice] | None:
    """Clip a placement [start, start+span) against a buffer [0, dim).

    Returns (buffer_slice, source_slice) where source_slice indexes into the
    unclipped biomass array, or None if the placement is entirely outside.
    """
    end = start + span
    if end <= 0 or start >= dim:
        return None
    return (
        slice(max(0, start), min(dim, end)),
        slice(max(0, -start), span - max(0, end - dim)),
    )


def _place_biomass(
    abs_col: int,
    abs_row: int,
    chunk_x_start: int,
    chunk_y_start: int,
    crown_base_height: float,
    biomass_shape: tuple[int, int, int],
    buffer_shape: tuple[int, int, int],
    vr: float,
) -> tuple[tuple[slice, slice, slice], tuple[slice, slice, slice]] | None:
    """Compute (buffer_slices, source_slices) for placing a biomass array
    into a chunk buffer.

    The biomass array is placed with its horizontal center at the stem cell
    and its vertical bottom at crown_base_height, then clipped against the
    chunk buffer's (nz, ny, nx) shape. Returns None if fully outside.
    """
    b_nz, b_ny, b_nx = biomass_shape
    nz, ny_chunk, nx_chunk = buffer_shape

    col_cell = abs_col - chunk_x_start
    row_cell = abs_row - chunk_y_start

    z = _clip_1d(int(crown_base_height / vr), b_nz, nz)
    y = _clip_1d(row_cell - b_ny // 2, b_ny, ny_chunk)
    x = _clip_1d(col_cell - b_nx // 2, b_nx, nx_chunk)
    if z is None or y is None or x is None:
        return None
    return (z[0], y[0], x[0]), (z[1], y[1], x[1])


def _apply_bands(
    buffers: dict[str, np.ndarray],
    buf_slices: tuple[slice, slice, slice],
    biomass_clip: np.ndarray,
    species_code: int,
    foliage_sav: float,
    tree_id: int,
    moisture_value: float | None,
) -> None:
    """Accumulate or overwrite one tree's contribution into the chunk buffers.

    Accumulate: volume_fraction, bulk_density.foliage.
    Overwrite:  savr.foliage, fuel_moisture.live, spcd, tree_id — rely on the
    caller iterating trees height-ASC so the tallest tree's value wins.
    """
    mask = biomass_clip > 0
    for key, buf in buffers.items():
        region = buf[buf_slices]
        if key == "volume_fraction":
            region += mask.astype(buf.dtype)
        elif key == "bulk_density.foliage":
            region += biomass_clip.astype(buf.dtype)
        elif key == "savr.foliage":
            region[mask] = foliage_sav
        elif key == "fuel_moisture.live":
            region[mask] = moisture_value
        elif key == "spcd":
            region[mask] = species_code
        elif key == "tree_id":
            region[mask] = tree_id


def voxelize_chunk(
    trees_in_chunk: pd.DataFrame,
    buffers: dict[str, np.ndarray],
    cache: dict[int, CacheEntry],
    chunk_y_start: int,
    chunk_x_start: int,
    hr: float,
    vr: float,
    x_origin: float,
    y_origin: float,
    source_config: dict,
    rng: np.random.Generator,
) -> None:
    """Render `trees_in_chunk` into `buffers` (mutated in place).

    - `buffers` is `{band_key: np.ndarray}` per band, already the chunk-local
      shape `(nz, halo_y, halo_x)` and filled with each band's fill value.
    - `chunk_y_start` / `chunk_x_start` are the absolute grid indices of
      buffer cell (y=0, x=0) — the chunk's north-west corner including halo.
    - Trees arrive pre-sorted by height ASC so tallest writes last; overwrite
      bands (spcd, tree_id, savr.foliage, fuel_moisture.live) therefore take
      the tallest tree's value in overlap cells.
    - Placement attrs (crown_base_height, foliage_sav, species_code) are read
      from the shared `CacheEntry` — `build_tree` runs once per cache_key in
      `build_chunk_cache`, not once per row here.
    """
    if trees_in_chunk.empty:
        return

    moisture_value = None
    if "fuel_moisture.live" in buffers:
        moisture_value = float(source_config["moisture_model"]["live"])

    buffer_shape = next(iter(buffers.values())).shape

    for _, row in trees_in_chunk.iterrows():
        entry = cache.get(int(row["_cache_key"]))
        if entry is None or not entry.biomass_arrays:
            continue
        arrays = entry.biomass_arrays
        biomass_array = (
            arrays[0] if len(arrays) == 1 else arrays[int(rng.integers(len(arrays)))]
        )

        abs_col, abs_row = _tree_cell_indices(
            float(row["x"]), float(row["y"]), x_origin, y_origin, hr
        )
        placement = _place_biomass(
            abs_col,
            abs_row,
            chunk_x_start,
            chunk_y_start,
            entry.crown_base_height,
            biomass_array.shape,
            buffer_shape,
            vr,
        )
        if placement is None:
            continue

        buf_slices, src_slices = placement
        _apply_bands(
            buffers,
            buf_slices,
            biomass_array[src_slices],
            entry.species_code,
            entry.foliage_sav,
            int(row["tree_id"]),
            moisture_value,
        )
