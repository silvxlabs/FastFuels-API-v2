# Treevox Design Notes

See the architecture summary in the API repo's `docs/` folder for the full picture. This file captures the design decisions specific to this service.

## Layout

4 files:
- `main.py` — Cloud Function entry, dispatch, errors, handler orchestration, parquet IO.
- `voxelize.py` — pure compute (GridSpec, chunk binning, cache, voxelize_chunk). Worker-safe imports only.
- `storage.py` — xarray-backed zarr I/O (init_store, read/write union, masked_merge). Not imported by workers.
- `_worker.py` — spawned worker entry. Strict imports: numpy, pandas, treevox.voxelize only.

Griddle's `handlers/` subpackage and `dispatch.py`/`errors.py` splits exist because griddle has many source handlers. Treevox has exactly one (`inventory`) — premature scaffolding. Add the subpackage when a second source (lidar, treelist) lands.

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

Each chunk carries an `OVERLAP_CELLS = 10` halo on every side, clamped to `[0, nx)` / `[0, ny)` at the outer grid boundary. At `hr = 1 m` that's 10 m of physical overlap; at `hr = 2 m`, 20 m. Halos let crowns whose stem is near a chunk boundary render into the neighbor's cells correctly.

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

V1 used the inventory's `TREE_ID` column as cache key — it wasn't guaranteed unique, so distinct trees accidentally shared biomass realizations. V2 inventories have no `tree_id` column; treevox assigns `tree_id = np.arange(len(df))` (unique per row) and computes a separate cache key by binning `(fia_species_code, dbh_bin, height_bin, cr_bin)` at `DBH_BIN_CM = 2.75`, `HEIGHT_BIN_M = 1.0`, `CR_BIN = 0.1`.

## Chunk padding model

Two primitives together ensure crowns near chunk boundaries render correctly:
1. **Union reads with halo** — each batch reads a region covering all chunks in the batch plus a 10-cell halo. Trees placed near a chunk interior boundary render into the halo.
2. **Masked merge on write** — after workers return, the orchestrator merges per-chunk buffers into the union by `mask = data != fill_value`. Two chunks touching the same halo cell combine without clobbering.

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
- `tree_id` with fill=-1 (any value > -1 should win, including -1 → 0 transitions).
- `spcd` with fill=0 (legitimate species code 0 would be indistinguishable from fill).

V2's `masked_merge` uses `mask = data != BAND_SPECS[key].fill_value` per band, so each band's merge respects its actual fill value.

## Out of scope (this PR)

- Cloud Run / Cloud Tasks provisioning — infra PR.
- Moisture methods beyond `uniform` — schema permits them; add when needed.
- Per-tree SAVR / fuel-moisture overrides — current `fuel_moisture.live` is uniform per moisture_model contract.
- Shared-memory worker payloads (Constraint #4 optimization) — defer until profiling shows IPC dominates wall time.
- Tree-crown boundary overflow correctness beyond the halo+merge approach — at the canonical 1 m / 2 m horizontal resolutions, the 10-cell halo exceeds typical crown radii while staying tiny relative to the 500-cell chunk size, so boundary artifacts are sub-1%.
