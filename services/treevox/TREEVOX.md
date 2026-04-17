# Treevox Design Notes

See the architecture summary in the API repo's `docs/` folder for the full picture. This file captures the design decisions specific to this service.

## Layout

4 files:
- `main.py` — Cloud Function entry, dispatch, errors, handler orchestration, parquet IO.
- `voxelize.py` — pure compute (GridSpec, chunk binning, cache, voxelize_chunk). Worker-safe imports only.
- `storage.py` — xarray-backed zarr I/O (init_store, read/write union, masked_merge). Not imported by workers.
- `_worker.py` — spawned worker entry. Strict imports: numpy, pandas, treevox.voxelize only.

Griddle's `handlers/` subpackage and `dispatch.py`/`errors.py` splits exist because griddle has many source handlers. Treevox has exactly one (`inventory`) — premature scaffolding. Add the subpackage when a second source (lidar, treelist) lands.

## Why per-chunk caching (not v1's global cache)

V1 pre-computes every tree's biomass realizations before touching zarr. Memory grows with domain size and fails on large inventories. V2 builds the biomass cache *inside each chunk worker*, so peak memory is bounded by the largest chunk's tree diversity regardless of total domain size.

## Tree-binning cache key

V1 used the inventory's `TREE_ID` column as cache key — it wasn't guaranteed unique, so distinct trees accidentally shared biomass realizations. V2 inventories have no `tree_id` column; treevox assigns `tree_id = np.arange(len(df))` (unique per row) and computes a separate cache key by binning `(fia_species_code, dbh_bin, height_bin, cr_bin)` at `DBH_BIN_CM = 1.0`, `HEIGHT_BIN_M = 1.0`, `CR_BIN = 0.1`.

## Chunk padding model (ported from v1)

Three primitives together ensure crowns near chunk/domain boundaries render correctly:
1. **Grid padding** (`horizontal_padding_m = 10`) — the zarr grid extends 10 m past the domain on each side. Crowns near the domain edge render into padding rather than getting truncated.
2. **Union reads with halo** — each batch reads a region covering all chunks in the batch plus a 10-cell halo. Trees placed near a chunk interior boundary render into the halo.
3. **Masked merge on write** — after workers return, the orchestrator merges per-chunk buffers into the union by `mask = data != fill_value`. Two chunks touching the same halo cell combine without clobbering.

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
8. Stochastic steps take an injected RNG seeded from `(grid_id, row, col)`.
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
- Tree-crown boundary overflow correctness beyond the halo+merge approach — the 10 m halo is larger than typical crown radii at 1000 m chunks, so boundary artifacts are sub-1%.
