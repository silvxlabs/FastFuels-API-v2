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
    """One equivalence class of morphologically-identical trees, pre-voxelized.

    Fields
    ------
    biomass_arrays
        List of 3D numpy arrays — each a `(nz, ny, nx)` canopy biomass
        realization in kg/m³ sampled from the bin-representative tree's
        crown profile. Shape is consistent across the list (determined by
        the tree's crown dimensions + voxel resolution). Length varies by
        bin: scales with canopy volume and tree frequency via
        `calculate_arrays_to_cache`, capped at `MAX_BIOMASS_ARRAY_CACHE`.
        When a tree in this bin is rendered, `voxelize_chunk` picks one of
        these arrays uniformly at random.
    crown_base_height
        Meters above ground where the bin-representative tree's crown
        starts. Used by `_place_biomass` to position the biomass array
        vertically in the chunk buffer.
    foliage_sav
        Surface-area-to-volume ratio for foliage (1/m). Derived by
        fastfuels-core from the bin-representative tree's species. Written
        verbatim into the `savr.foliage` band for every voxel the tree
        occupies.
    species_code
        FIA species code (int). Written verbatim into the `spcd` band.

    Scope
    -----
    One entry is created per unique `_cache_key` in `build_chunk_cache`,
    lives for the duration of one chunk's rendering, and is discarded when
    the worker returns. Never persisted, never crosses process boundaries
    via pickle (cache construction and consumption both run inside the
    worker).

    Why the shared attrs exist
    --------------------------
    All trees sharing a `_cache_key` fall in the same
    `(species, dbh_bin, height_bin, crown_ratio_bin)` bucket, so
    `crown_base_height`, `foliage_sav`, and `species_code` are identical
    (or close enough — see TREEVOX.md). Caching them here means the outer
    `voxelize_chunk` loop makes zero Tree-allometry calls per row — those
    ran once per bin inside `build_chunk_cache`.
    """

    biomass_arrays: list[np.ndarray]
    crown_base_height: float
    foliage_sav: float
    species_code: int


# Constants


CHUNK_SIZE_HORIZONTAL = 500  # cells per chunk in x and y; resolution-independent
OVERLAP_CELLS = 10  # halo cells for batch union reads (v1 line 525)
MAX_BIOMASS_ARRAY_CACHE = 100  # v1 line 42

# Tree-binning widths for cache grouping — see TREEVOX.md "Tree-binning cache key".
HEIGHT_BIN_M = 1.0
DBH_BIN_CM = 2.75
CR_BIN = 0.1

# Map API allometry equation names to fastfuels-core model names.
BIOMASS_EQUATION_MAP = {"nsvb": "NSVB", "jenkins": "jenkins"}
BIOMASS_DENSITY_BAND_COMPONENTS = {
    "bulk_density.foliage.live": "foliage",
    "bulk_density.foliage.dead": "foliage",
    "bulk_density.branchwood.live": "branchwood",
    "bulk_density.branchwood.dead": "branchwood",
    "bulk_density.fine.live": "fine",
    "bulk_density.fine.dead": "fine",
}


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


def foliage_inventory_column(source_config: dict) -> str | None:
    """Return the foliage inventory biomass column used by current compute."""
    biomass_source = source_config["biomass_source"]
    if biomass_source["type"] != "inventory_columns":
        return None
    foliage = biomass_source.get("columns", {}).get("foliage")
    if foliage is None:
        return None
    return foliage["column"]


def _biomass_allometry_model_type(source_config: dict) -> str:
    """Return the fastfuels-core foliage allometry model for this source."""
    biomass_source = source_config["biomass_source"]
    if biomass_source["type"] == "allometry":
        return BIOMASS_EQUATION_MAP[biomass_source["equations"]]
    return BIOMASS_EQUATION_MAP["nsvb"]


def biomass_component_to_distribute(source_config: dict) -> str:
    """Select the biomass component represented by cached density arrays."""
    source_components = set(source_config["biomass_source"].get("components", {}))
    band_components = {
        component
        for band in source_config.get("bands", [])
        if (component := BIOMASS_DENSITY_BAND_COMPONENTS.get(band))
    }
    requested = source_components | band_components
    for component in ("branchwood", "fine", "foliage"):
        if component in requested:
            return component
    return "foliage"


def biomass_component_state(source_config: dict, component: str) -> dict[str, float]:
    """Return live/dead partition fractions for one biomass component."""
    states = source_config["biomass_source"].get("component_states", {})
    state = states.get(component, {})
    return {
        "live": float(state.get("live", 1.0)),
        "dead": float(state.get("dead", 0.0)),
    }


def distribute_component_biomass(vt: VoxelizedTree, component: str) -> np.ndarray:
    """Populate a biomass array for one component from a VoxelizedTree."""
    if component == "foliage":
        return vt.distribute_biomass()
    if component in {"branchwood", "fine"}:
        raise NotImplementedError(
            f"Treevox does not yet support {component} biomass distribution."
        )
    raise ValueError(f"Unknown biomass component: {component!r}.")


def build_tree(row, source_config: dict) -> Tree:
    """Construct a fastfuels_core.Tree from a v2 inventory row.

    V2 columns map directly to Tree kwargs (no renames):
      fia_species_code -> species_code, dbh -> diameter, etc.

    `crown_fuel_load` is only supplied when `biomass_source.type` is
    `inventory_columns` and a foliage column is configured; otherwise foliage
    biomass is computed allometrically via NSVB or Jenkins.
    """
    crown_fuel_load = None
    column = foliage_inventory_column(source_config)
    if column is not None:
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
        biomass_allometry_model_type=_biomass_allometry_model_type(source_config),
        crown_fuel_load=crown_fuel_load,
    )


def compute_cache_keys(
    df: pd.DataFrame, source_config: dict | None = None
) -> pd.Series:
    """Group trees into cache-equivalence classes by binned characteristics.

    Trees with the same (species, binned dbh, binned height, binned crown_ratio)
    share voxelized biomass realizations — they're morphologically
    indistinguishable within the chosen bin widths. When foliage biomass comes
    from an inventory column, that per-row biomass value is also part of the
    key so rows with the same morphology but different supplied biomass do not
    reuse the first row's cached density arrays. Returns integer codes via
    `groupby().ngroup()`.

    See TREEVOX.md for rationale and bin widths.
    """
    dbh_bin = (df["dbh"] / DBH_BIN_CM).astype("int64")
    height_bin = (df["height"] / HEIGHT_BIN_M).astype("int64")
    cr_bin = (df["crown_ratio"] / CR_BIN).round().astype("int64")
    groupers = [
        df["fia_species_code"].astype("int64"),
        dbh_bin,
        height_bin,
        cr_bin,
    ]
    if source_config is not None and (
        column := foliage_inventory_column(source_config)
    ):
        groupers.append(df[column].astype("float64"))
    return df.groupby(groupers, sort=False).ngroup()


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
    """Pre-sample biomass realizations for every equivalence class in a chunk.

    Inputs
    ------
    trees_in_chunk
        DataFrame slice holding every tree assigned to one chunk, already
        height-sorted ASC and carrying a `_cache_key` column (int group id
        from `compute_cache_keys`) plus the raw tree columns the v2
        inventory schema guarantees: `x`, `y`, `fia_species_code`,
        `fia_status_code`, `dbh`, `height`, `crown_ratio`. Empty frame
        returns an empty cache.
    hr
        Horizontal voxel resolution in meters. Threaded into
        `discretize_crown_profile` and `VoxelizedTree` so the sampled
        arrays are sized for the output grid.
    vr
        Vertical voxel resolution in meters. Same role as `hr` but for
        the z axis.
    source_config
        The grid's `source` sub-dict. Must carry `crown_profile_model`
        ("purves" | "beta") and `biomass_source`, whose source is either
        allometry equations or inventory columns. Passed verbatim into
        `build_tree`.
    rng
        Seeded numpy Generator. Used twice per cache entry: once to draw
        a per-realization int seed for `sample_occupied_cells` (so each
        realization varies deterministically), and once downstream in
        `voxelize_chunk` to pick which realization a given tree row uses.
        Seeded from `source.seed + (row_chunk, col_chunk)` in the
        orchestrator so the whole job is reproducible.

    What it does
    ------------
    Groups `trees_in_chunk` by `_cache_key`. For each group:

    1. Builds a fastfuels-core `Tree` from the group's first row (the
       "bin-representative").
    2. Calls `discretize_crown_profile(tree, hr, vr)` to get a 3D mask of
       which voxels the tree's crown occupies.
    3. Computes how many independent biomass realizations to cache via
       `calculate_arrays_to_cache(nonzero_voxels, group_size)`.
    4. For each realization:
         a. Draws a seed from `rng`.
         b. `sample_occupied_cells(canopy_mask, alpha=0.5, beta=0.5, seed)`
            — stochastic sub-voxel occupancy.
         c. `VoxelizedTree(tree, sampled, hr, vr).distribute_biomass()` →
            per-voxel kg/m³ array.
         d. Sanitizes non-finite values (`foliage_biomass / 0 volume`
            edge case inside fastfuels-core) to zero so the downstream
            zarr store never receives NaN/Inf.
    5. Packs the realizations plus the bin-representative's
       `crown_base_height`, `foliage_sav`, and `species_code` into one
       `CacheEntry`.

    Skip rules (never raise, always log by omission so the job makes
    progress on partial failure):
      - Tree construction raises → whole bin skipped (no cache_key entry).
      - Canopy mask is all-zero (empty crown) → whole bin skipped.
      - An individual realization fails mid-sample → that one realization
        dropped; the bin keeps whichever succeeded. Bin skipped only if
        every realization failed.

    Output
    ------
    `dict[int, CacheEntry]` keyed by `_cache_key`. Keys absent from the
    result are ones `voxelize_chunk` will silently skip (the outer loop
    does `cache.get(key)` and `continue`s on None). Keys present are
    guaranteed to have `biomass_arrays` non-empty and all arrays finite.
    """
    cache: dict[int, CacheEntry] = {}
    if trees_in_chunk.empty:
        return cache
    biomass_component = biomass_component_to_distribute(source_config)

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
                biomass = distribute_component_biomass(vt, biomass_component)
            except NotImplementedError:
                raise
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
    """Convert a tree stem's world coordinates to absolute grid cell indices.

    Inputs
    ------
    x, y
        Projected coordinates of the tree stem, in the grid's CRS
        (typically UTM meters — same CRS as the domain GeoDataFrame). For
        a domain spanning 500000..501000 m E and 5200000..5201000 m N in
        EPSG:32611, a tree would have `x` somewhere in that first range,
        `y` in the second.
    x_origin, y_origin
        Grid origin in the same projected CRS. By raster convention
        `x_origin` is the WEST edge and `y_origin` is the NORTH edge —
        so columns increase eastward as `x` increases, but rows increase
        southward as `y` DECREASES. Both come from `compute_grid_dimensions`.
    hr
        Horizontal cell size in meters (square cells, so the same value
        applies to both axes).

    What it does
    ------------
    Computes the integer cell index of the cell containing the stem on
    each axis:
      - `abs_col = floor((x - x_origin) / hr)` — offset east of the
        grid's west edge, divided by cell size.
      - `abs_row = floor((y_origin - y) / hr)` — offset south of the
        grid's north edge, divided by cell size.
    Floor means stems exactly on a cell boundary land in the east/south
    cell (the half-open `[start, start+hr)` convention).

    Does no clamping — if `x < x_origin` the returned `abs_col` is
    negative. The caller is responsible for bounds checking (typically
    downstream slice clipping in `_place_biomass` handles it).

    Output
    ------
    `(abs_col, abs_row)` — zero-based ints into the full grid's x/y
    axes. Example: a stem 5.5 m east and 3.2 m south of a grid origin at
    1 m resolution returns `(5, 3)`.
    """
    abs_col = int(math.floor((x - x_origin) / hr))
    abs_row = int(math.floor((y_origin - y) / hr))
    return abs_col, abs_row


def _clip_1d(start: int, span: int, dim: int) -> tuple[slice, slice] | None:
    """Clip a 1D placement window against a buffer axis.

    Inputs
    ------
    start
        Buffer-local index (int) where the placement window begins. May be
        negative when the source array overhangs the buffer's low side
        (e.g. the tree's crown extends north of the chunk's origin).
    span
        Length of the placement window — equals the source array's extent
        on this axis. `(start, start+span)` is the half-open buffer range
        the source WOULD write to if the buffer were infinite.
    dim
        Size of the buffer on this axis. Valid buffer indices are
        `[0, dim)`.

    What it does
    ------------
    Intersects the requested placement range `[start, start+span)` with
    the buffer range `[0, dim)` and returns a matched pair of slices:
      - `buffer_slice` — where to WRITE in the buffer, clamped to
        `[0, dim)`.
      - `source_slice` — where to READ from the (unclipped) source array;
        the beginning is shifted right when `start < 0`, and the end is
        pulled back when the window runs past `dim`.
    Both slices have the same length so assigning
    `buf[buffer_slice] = src[source_slice]` is always shape-safe.

    Returns `None` when the placement is ENTIRELY outside the buffer
    (`end <= 0` or `start >= dim`) — nothing to write, so the caller
    skips this axis/tree.

    Output
    ------
    Either `(buffer_slice, source_slice)` — both `slice` objects with
    equal length — or `None`. Examples:

      - Fully inside: `_clip_1d(2, 3, 10)` → `(slice(2, 5), slice(0, 3))`.
      - Overhang low: `_clip_1d(-2, 5, 10)` → `(slice(0, 3), slice(2, 5))`.
      - Overhang high: `_clip_1d(8, 5, 10)` → `(slice(8, 10), slice(0, 2))`.
      - Fully outside: `_clip_1d(-5, 3, 10)` → `None`.
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
    """Compute where to write one biomass array into a chunk buffer.

    Inputs
    ------
    abs_col, abs_row
        Absolute grid cell indices of the tree stem (from
        `_tree_cell_indices`). `abs_col` ranges `[0, nx)`; `abs_row`
        ranges `[0, ny)` in the ideal case but can fall outside when the
        stem is actually outside the domain.
    chunk_x_start, chunk_y_start
        Absolute grid cell indices of the chunk buffer's origin cell —
        i.e. the y=0, x=0 cell of `buffer`. Includes the halo offset:
        for chunk (1, 0) at `chunk_xy=1000, overlap_cells=10` the chunk
        buffer spans absolute x=`[0, 1010)` and y=`[990, 2010)`, so
        `chunk_x_start=0, chunk_y_start=990`.
    crown_base_height
        Meters above ground where the tree's crown starts. From
        `CacheEntry.crown_base_height` — shared across trees in the bin.
    biomass_shape
        `(nz, ny, nx)` of the pre-sampled biomass array being placed.
        Each biomass realization in the cache has this shape; it's
        determined by the representative tree's crown dimensions.
    buffer_shape
        `(nz, ny, nx)` of the chunk buffer being written to. `nz` equals
        the full grid's z extent; `ny`/`nx` are the chunk's halo-extended
        spans (typically `chunk_xy + 2 * overlap_cells` away from grid
        edges, less near boundaries).
    vr
        Vertical voxel resolution in meters. Used to convert
        `crown_base_height` into a z cell index.

    What it does
    ------------
    Determines the placement anchor on each axis:
      - z: crown bottom at `floor(crown_base_height / vr)`, filling
        upward through `z + b_nz`.
      - y: biomass array centered on the stem's buffer-local row
        (`row_cell - b_ny // 2`), filling north-to-south.
      - x: biomass array centered on the stem's buffer-local column
        (`col_cell - b_nx // 2`), filling west-to-east.
    Each axis is then clipped via `_clip_1d` against the corresponding
    buffer dimension. If any axis collapses to zero overlap (fully
    outside the buffer), returns `None` — the tree is entirely outside
    this chunk's halo and the caller skips it.

    Output
    ------
    Either `None` (fully outside) or a pair of 3-tuples:
      - `buffer_slices`: `(slice, slice, slice)` — where to WRITE in the
        chunk buffer. `buf[buffer_slices]` is the in-bounds region this
        tree contributes to.
      - `source_slices`: `(slice, slice, slice)` — where to READ from
        the unclipped biomass array. `biomass_array[source_slices]` has
        the same shape as `buf[buffer_slices]`, so
        `buf[buffer_slices] += biomass_array[source_slices]` (or any
        shape-matched write) is safe regardless of which buffer edges
        the crown overhangs.
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
    moisture_values: dict[str, float],
    component_state: dict[str, float],
    biomass_component: str = "foliage",
) -> None:
    """Write one tree's contribution into every requested band buffer.

    Inputs
    ------
    buffers
        Mapping from band key to the chunk's numpy buffer for that band,
        e.g. `{"volume_fraction": np.ndarray((nz, ny, nx), float32),
        "spcd": np.ndarray((nz, ny, nx), uint16), ...}`. Only keys that
        the user requested are present; unknown keys are silently ignored
        by the dispatch loop. Buffers are mutated IN PLACE.
    buf_slices
        `(z_slice, y_slice, x_slice)` identifying the 3D sub-region of
        each buffer this tree writes to — from `_place_biomass`. All
        buffers share the same shape, so the same slice tuple works for
        every band.
    biomass_clip
        The source-slice view into the bin's biomass array, i.e.
        `biomass_array[source_slices]`. Same shape as `buf[buf_slices]`.
        Non-zero values mark voxels the tree occupies; values are kg/m³
        of foliage biomass.
    species_code
        FIA species code (int), from `CacheEntry.species_code`. Written
        into the `spcd` band.
    foliage_sav
        Foliage surface-area-to-volume ratio (1/m), from
        `CacheEntry.foliage_sav`. Written into the `savr.foliage` band.
    tree_id
        Per-row unique tree identifier (int), from the DataFrame row.
        Written into the `tree_id` band so downstream consumers can trace
        a voxel back to an inventory row.
    moisture_values
        Fuel moisture percent by state, read once per chunk from
        `source.moisture_model`. Contains only requested moisture states.
    component_state
        Live/dead biomass partition fractions for `biomass_component`.

    What it does
    ------------
    Masks `biomass_clip > 0` to find the voxels this tree occupies, then
    dispatches per band into the corresponding buffer's sub-region:

      | Band                  | Rule      | Semantics                    |
      |-----------------------|-----------|------------------------------|
      | volume_fraction       | accumulate| `region += mask`             |
      | bulk_density.<component>.<state> | accumulate| `region += biomass_clip * fraction` |
      | savr.foliage          | overwrite | `region[mask] = foliage_sav` |
      | fuel_moisture.<state> | overwrite | `region[mask] = moisture`    |
      | spcd                  | overwrite | `region[mask] = species_code`|
      | tree_id               | overwrite | `region[mask] = tree_id`     |

    Accumulate bands SUM across overlapping crowns. Overwrite bands
    take the LAST writer's value on overlap — which, because the caller
    iterates trees in height-ASC order, means the tallest tree wins (the
    policy documented on the API).

    Uses `buf[buf_slices]` views (not copies), so `region += ...` and
    `region[mask] = ...` mutate `buffers` directly.

    Output
    ------
    None. Side effect: `buffers` is mutated in place.
    """
    mask = biomass_clip > 0
    for key, buf in buffers.items():
        region = buf[buf_slices]
        if key == "volume_fraction":
            region += mask.astype(buf.dtype)
        elif key == f"bulk_density.{biomass_component}.live":
            region += (biomass_clip * component_state["live"]).astype(buf.dtype)
        elif key == f"bulk_density.{biomass_component}.dead":
            region += (biomass_clip * component_state["dead"]).astype(buf.dtype)
        elif key == "savr.foliage":
            region[mask] = foliage_sav
        elif key == "fuel_moisture.live":
            region[mask] = moisture_values["live"]
        elif key == "fuel_moisture.dead":
            region[mask] = moisture_values["dead"]
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
    """Render every tree in a chunk into the per-band buffers.

    Inputs
    ------
    trees_in_chunk
        DataFrame of trees belonging to this chunk. Must carry `x`, `y`,
        `tree_id`, and `_cache_key` columns. Must be sorted by height
        ASC (the "tallest last" invariant drives the overlap policy);
        `assign_trees_to_chunks` guarantees this. Empty frame returns
        immediately with no buffer mutations.
    buffers
        Mapping from band key to the chunk's pre-allocated numpy buffer,
        e.g. `{"volume_fraction": np.zeros((nz, halo_y, halo_x), float32),
        ...}`. Pre-filled with each band's fill value (`BAND_SPECS`). All
        buffers share the same `(nz, halo_y, halo_x)` shape. Mutated IN
        PLACE; the orchestrator reads them back after this returns.
    cache
        Pre-built cache from `build_chunk_cache` mapping each
        `_cache_key` present in `trees_in_chunk` to a `CacheEntry`.
        Trees whose key is absent from `cache` (degenerate crown, build
        failure) are silently skipped.
    chunk_y_start, chunk_x_start
        Absolute grid cell indices of the chunk buffer's `(y=0, x=0)`
        corner, INCLUDING the halo. For chunk `(1, 0)` with
        `chunk_xy=1000, overlap=10` on a 2000-cell grid:
        `chunk_y_start=990, chunk_x_start=0`.
    hr, vr
        Horizontal and vertical voxel resolution in meters.
    x_origin, y_origin
        Full grid origin — west edge and north edge respectively in the
        grid's CRS. Threaded into `_tree_cell_indices` to turn each
        tree's world coords into absolute cell indices.
    source_config
        The grid's `source` sub-dict. Only used here to read
        `moisture_model` when fuel moisture bands are requested
        (per-chunk constants, resolved once before the loop).
    rng
        Seeded numpy Generator. Used to pick one biomass realization
        per tree when a `CacheEntry` carries multiple. Seed comes from
        `_chunk_rng_seed(base_seed, row_chunk, col_chunk)` in the
        orchestrator, so the same grid produces the same output on
        re-run.

    What it does
    ------------
    For each row in `trees_in_chunk`:

    1. Looks up the tree's `CacheEntry`. If missing or empty, skips.
    2. Picks one biomass realization from `entry.biomass_arrays` — the
       sole array when there's one, else a uniformly-sampled choice via
       `rng.integers`.
    3. `_tree_cell_indices` — stem world coords → absolute cell indices.
    4. `_place_biomass` — computes matched `(buffer_slices,
       source_slices)` with vertical anchor at
       `entry.crown_base_height`; returns `None` if the crown is fully
       outside this chunk buffer.
    5. `_apply_bands` — writes the tree's contribution into every
       requested band buffer with the appropriate accumulate/overwrite
       rule.

    Because trees are iterated height-ASC, the tallest writer wins on
    overwrite bands (spcd, tree_id, savr.foliage, fuel_moisture.*)
    at overlap cells — matches the policy documented on the API.

    Output
    ------
    None. Side effect: every buffer in `buffers` is mutated in place
    with one tree's contribution per iteration.
    """
    if trees_in_chunk.empty:
        return

    moisture_values: dict[str, float] = {}
    moisture_model = source_config.get("moisture_model") or {}
    if "fuel_moisture.live" in buffers:
        moisture_values["live"] = float(moisture_model["live"]["value"])
    if "fuel_moisture.dead" in buffers:
        moisture_values["dead"] = float(moisture_model["dead"]["value"])
    biomass_component = biomass_component_to_distribute(source_config)
    component_state = biomass_component_state(source_config, biomass_component)

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
            moisture_values,
            component_state,
            biomass_component,
        )
