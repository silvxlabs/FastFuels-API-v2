# Treevox Design Notes

See the architecture summary in the API repo's `docs/` folder for the full picture. This file captures the design decisions specific to this service.

## Layout

Mirrors griddle/standgen: a thin `dispatch.py` routes on the source triple, delegating the work to a `handlers/` package.

- `main.py` — Cloud Function entry (HTTP trigger, retry/error arms). Thin.
- `dispatch.py` — routes on the source `(operation, input, entity)` triple to a handler.
- `handlers/voxelize.py` — the tree-inventory voxelization job (`voxelize_inventory`) and its private stages, plus the `VoxelizationResult`/`GridLayout`/`BatchStats` dataclasses.
- `voxelize.py` — pure compute (GridSpec, chunk binning, cache, voxelize_chunk). Worker-safe imports only.
- `storage.py` — xarray-backed zarr I/O (init_store, read/write union, masked_merge). Not imported by workers.
- `_worker.py` — spawned worker entry. Strict imports: numpy, pandas, treevox.voxelize only.
- `measured_crowns.py` — handler-side: crown Feature + CHM → per-tree `Footprint`s for measured crowns (#617).
- `errors.py`, `firestore_io.py`, `inventory_io.py` — error types, Firestore helpers, parquet IO.

The `handlers/` package was added for the leaflux solar-irradiance source (#424); a second source module (e.g. `handlers/leaflux.py`) slots in alongside `handlers/voxelize.py` with a new `case` in `dispatch.py`.

## Hierarchy & Processing Flow

Two dimensions matter — **spatial** (how the output grid is partitioned) and **temporal** (how the work is scheduled through the multiprocessing Pool). Keep them distinct:

```
Grid                   Full voxel array in zarr: (nz, ny, nx).
 └─ Chunks             Spatial partition: every (chunk_xy × chunk_xy × nz)
                       tile indexed by (row_chunk, col_chunk). Data lives
                       here; zarr's on-disk chunking matches exactly so
                       region writes are chunk-aligned.
```

**Batches are NOT spatial containers** — they're temporal groupings of chunks that go through the mp Pool together:

```
Job
 ├─ Batch 1: [chunk(0,0), chunk(0,1), chunk(1,0), chunk(1,1)]
 ├─ Batch 2: [chunk(0,2), chunk(0,3), chunk(1,2), chunk(1,3)]
 └─ ...
```

`batch_size = num_workers` — one chunk per worker per batch. Chunks are sorted in 2×2 block order (`key=(row//2, col//2, row, col)`) so spatially adjacent chunks share a batch; their halos overlap into a single contiguous read region.

### Halo (per-chunk, clamped at grid edges)

Each chunk carries a halo on every side, clamped to `[0, nx)` / `[0, ny)` at the outer grid boundary. Trees are assigned only to the chunk holding their stem, so the halo must be as wide as the widest crown: `crown_halo_cells` sizes it per job as `ceil(max crown radius / hr)`, using the `max_crown_radius_source` inventory column where set and the allometric radius otherwise, widened to hold every measured crown (see "Measured crowns"), with `OVERLAP_CELLS = 10` as the minimum (#666). `_plan_grid_layout` stores it on `GridLayout.halo_cells`; `chunk_slice` and `batch_union_slices` both take it from there, so chunk buffers always sit inside the batch union. Halos let crowns whose stem is near a chunk boundary render into the neighbor's cells correctly.

A wider halo grows every chunk buffer to `nz × (chunk_xy + 2·halo)²` and the batch union to `nz × (2·chunk_xy + 2·halo)²` for a 2×2 batch. At `chunk_xy = 500` a 40-cell halo (20 m crown at 0.5 m) makes buffers 1.24× the 10-cell size.

The halo is capped at `MAX_HALO_CELLS = 125` (a quarter of a chunk; buffers ≤ 2.1× the 10-cell size). A job whose widest crown needs more fails with `CROWN_TOO_WIDE` instead of risking an OOM: one outlier radius (a bad inventory value, or Purves extrapolated to a very large dbh) would otherwise widen every chunk.

Overwrite bands (`spcd`, `tree_id`, `savr.*`, `fuel_moisture.*`) take the tallest tree's value only among trees in the same chunk. A chunk's worker starts from the union's current values and `masked_merge` keeps the last chunk's change, so where crowns from different chunks overlap, the chunk processed last wins whatever the heights. See #668.

### Per-batch flow

1. **Plan** — `batch_union_slices` computes the bounding slice covering every chunk's halo in the batch.
2. **Read once** — `storage.read_union` loads that union from zarr as an in-memory xarray Dataset (`.load()`, never lazy dask — workers must not re-hit GCS).
3. **Split** — `_build_payloads` carves one chunk-halo-sized numpy buffer per chunk out of the union, plus the trees, grid params, and per-chunk RNG seed.
4. **Render in parallel** — `pool.map(worker_run, payloads)`. Each worker `_worker.run`s independently: builds its own per-chunk biomass cache, voxelizes its trees, mutates its buffer in place.
5. **Merge** — `masked_merge` stitches the workers' buffers back into the union using `data != fill_value` per band so halo-overlap cells combine cleanly.
6. **Write once** — `storage.write_union` region-writes the merged union back to zarr with `align_chunks=True`.

One persistent `Pool` spans all batches (Constraint #2). The spatial chunk grid is fixed by `_plan_grid_layout`; batching is purely a scheduling decision.

## Why per-chunk caching (not v1's global cache)

V1 pre-computes every tree's biomass realizations before touching zarr. Memory grows with domain size and fails on large inventories. V2 builds the biomass cache *inside each chunk worker*, so peak memory is bounded by the largest chunk's tree diversity regardless of total domain size.

## Tree-binning cache key

V1 used the inventory's `TREE_ID` column as cache key — it wasn't guaranteed unique, so distinct trees accidentally shared biomass realizations. V2 inventories carry a stable, unique `tree_id` column (#611), which treevox writes into the `tree_id` band; the cache key is computed separately by binning `(fia_species_code, dbh_bin, height_bin, cr_bin)` at `DBH_BIN_CM = 2.75`, `HEIGHT_BIN_M = 1.0`, `CR_BIN = 0.1`. When `max_crown_radius_source` reads an inventory column, that radius joins the key binned at `CROWN_RADIUS_BIN_M = 0.25` (provisional, #613); each bin uses its first tree's radius.

The geometric crown profiles (`cone`, `cylinder`, `single_/dual_ellipsoid`, `single_/dual_paraboloid`) need no extra key field: with the allometric radius source their maximum radius is the Purves radius, a function of the already-binned species, DBH, height, and crown ratio.

## Measured crowns (#617)

With `source.crown_geometry_source.type == "crown_feature"`, trees with a crown in the crown Feature get a measured crown instead of a solid of revolution.

- **Parent** (`measured_crowns.py`, handler-side): reads the Feature document for its CHM grid id, the Feature's GeoParquet (`tree_id`, `geometry`) and the CHM window covering the matched crowns. Crowns match inventory trees by `tree_id`; crowns of trees not in the inventory are ignored. The polygons follow CHM cell edges, so `rasterio.features.rasterize` onto the CHM lattice recovers each crown's cells exactly. Each becomes a `voxelize.Footprint` (a bool mask and the CHM tops in its bounding box). No-data CHM cells are dropped from the footprint. The grid's `nz` reaches the highest footprint top as well as the tallest tree.
- **Payload**: a `crown_feature` payload also carries `footprints` (the chunk's trees only) and `base_seed`. Parametric payloads are unchanged.
- **Worker**: `plan_measured_crowns` sends a tree measured when it has a footprint with a cell above its crown base, and gives every other tree its reason, `parametric_no_crown` / `parametric_no_volume`. `voxelize_chunk` counts trees by reason as it draws them, so a tree that cannot be built (e.g. an unknown species) is skipped on either path and counted nowhere. Measured trees skip the binned cache (every crown differs); the others build it as before. In the height-ASC loop, a measured tree is built by core's `voxelize_measured_crown` (outline = footprint, top = CHM, bottom = `crown_profile_model`), sampled with the same occupancy field and uniform density as a parametric crown, and placed at the offset core returns. Its mass is the parametric tree's mass.
- **Seeds**: a measured crown's occupancy seed is `crc32(base_seed, tree_id)`, not the chunk RNG, so it does not depend on the chunk layout.
- **Counts**: workers return `crown_geometry_counts`; the handler sums them and `main.py` writes `source.crown_geometry_counts` on completion.

**Halo.** A measured crown is rendered by the chunk that owns its stem, like a parametric crown. `measured_halo_cells` measures how far each footprint reaches from its stem's cell, on the output lattice with core's placement arithmetic, and `_plan_grid_layout` widens the halo to the farthest reach, so no measured crown is clipped. The `MAX_HALO_CELLS` cap and `CROWN_TOO_WIDE` apply to the combined halo.

## Bulk density cap (`max_bulk_density`)

A voxel's bulk density is final only once every chunk whose halo covers it has been merged, so the cap cannot be applied in the workers or per batch. When `source.max_bulk_density` is set, `_plan_grid_layout` adds the scratch band `_bulk_density_total` (`BULK_DENSITY_TOTAL_KEY`), which workers accumulate with the distributed component's live + dead bulk density. After the batch loop, `storage.cap_bulk_density` multiplies every bulk density band and `leaf_area_density` by `cap / max(total, cap)` on the store's own chunks, then deletes the scratch band and reconsolidates metadata. Treevox distributes one biomass component per grid, so this caps each component; `volume_fraction` is not scaled. If the job fails during the cap pass, the store is deleted like any failed job.

## Chunk padding model

Two primitives together ensure crowns near chunk boundaries render correctly:
1. **Union reads with halo** — each batch reads a region covering all chunks in the batch plus the job's halo (see "Halo" above). Trees placed near a chunk interior boundary render into the halo.
2. **Masked merge on write** — after workers return, the voxelize handler merges per-chunk buffers into the union by `mask = data != fill_value`. Two chunks touching the same halo cell combine without clobbering.

V2 does **not** pad the outer grid itself (v1's `horizontal_padding_m = 10`). `compute_grid_dimensions` just snaps the domain's total bounds outward to the nearest multiple of `hr`. Domain-edge padding, where needed, is the domain resource's job via `pad_to_resolution` upstream.

V1's `ChunkedGrid3D`/`Chunk`/`SerializableChunk` classes are not ported — xarray's `to_zarr(region=...)` replaces them with ~50 LOC of `init_store`/`read_union`/`write_union`/`masked_merge`.

## Concurrency & Runtime Constraints

Ten constraints must be respected. See the implementation plan for full details; short summary:

1. `to_zarr(region=...)` needs `align_chunks=True` for non-aligned halo unions.
2. One persistent `Pool` across all batches — never per-batch.
3. Workers import only numpy, pandas, treevox.voxelize. Never xarray, rioxarray, gcsfs, zarr, treevox.storage.
4. Pickle overhead bounds `chunk_xy`; shared-memory zero-copy deferred.
5. Worker count capped by available memory, not just CPU count. 4 GB Cloud Run minimum.
6. `read_union` always `.load()`s; workers never see dask arrays.
7. `write_union` drops coord variables before region writes.
8. Stochastic steps take an injected RNG seeded from `(source.seed, row, col)` via `_resolve_base_seed` → `_chunk_rng_seed` (CRC32, stable across Python processes). The base seed is supplied by the API on grid creation — explicit when the user pins it, autogenerated otherwise — so re-running a grid always produces bit-identical output.
9. `ProcessingError` triggers `delete_zarr` before `update_status("failed")` — no stale consolidated metadata.
10. Test `_worker.run` as a pure function; one isolated Pool-roundtrip test covers pickle + spawn context.

## Correctness fix vs v1

V1's `write_combined_chunks` uses `mask = chunk.data > 0`. This breaks for:
- `spcd` with fill=0 (legitimate species code 0 would be indistinguishable from fill).

V2's `masked_merge` uses `mask = data != BAND_SPECS[key].fill_value` per band, so each band's merge respects its actual fill value.

## Out of scope (this PR)

- Cloud Run / Cloud Tasks provisioning — infra PR.
- Moisture methods beyond `uniform` — schema permits them; add when needed.
- Per-tree SAVR / fuel-moisture overrides — current `fuel_moisture.live` is uniform per moisture_model contract.
- Shared-memory worker payloads (Constraint #4 optimization) — defer until profiling shows IPC dominates wall time.
