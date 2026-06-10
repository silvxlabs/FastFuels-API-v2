## Summary

Inventories today expose `columns[]` with `key`, `type`, `unit`, but no information about the values inside each column. The webapp summary card needs a small fixed set of per-column scalars (count, null count, basic spread) to render without re-fetching the partitioned parquet on the client.

Add a `summary` field on each `Column`, populated once at standgen creation time. Scope is deliberately minimal: scalar stats only, no histograms, no percentiles, no top-K — those can be added later as additive fields on the same discriminated union without breaking existing consumers.

This is the inventory-side mirror of #257 (per-band summary stats on grids). Same schema shape, same compute philosophy, different service. One material implementation difference: inventories are dask-lazy, so naive code can double-scan the source — see the compute section.

## Schema change

Add a new optional field `summary` to `Column` in `services/api/api/resources/inventories/schema.py`. The field is a discriminated union keyed on the column's `type`:

```python
class ContinuousColumnSummary(BaseModel):
    type: Literal["continuous"]
    count: int              # non-null rows
    null_count: int
    min: float | None       # None when count == 0
    max: float | None
    mean: float | None
    std: float | None

class CategoricalColumnSummary(BaseModel):
    type: Literal["categorical"]
    count: int
    null_count: int
    unique_count: int

ColumnSummary = Annotated[
    ContinuousColumnSummary | CategoricalColumnSummary,
    Field(discriminator="type"),
]

class Column(BaseModel):
    key: str
    type: ColumnType
    unit: str | None
    summary: ColumnSummary | None = None  # populated on completion
```

`summary` is `null` while the inventory is `pending` / `running`, and populated when it transitions to `completed`. Mirrors how `georeference` is already populated.

Stats are reported in the column's canonical `unit`.

## Where the compute lives — fused single-pass write

This is the load-bearing design decision. **Computing stats must happen as part of the same dask graph that writes the parquet, not as a separate step.** Inventories are returned by handlers as lazy `dd.DataFrame`s, not materialized numpy arrays. The naive sequence —

```python
stats = summarize_columns(ddf, columns)   # triggers full compute
save_parquet(inventory_id, ddf)           # triggers full compute again
```

— scans the source data **twice**, doubling GCS egress and CPU. The fix is to defer the write, build the stats reductions as a parallel dask graph, and fuse both into a single `dask.compute(...)` call so each partition is read once, contributes to both outputs, then is released.

To make this hard to misuse, fold both responsibilities into one function in `services/standgen/standgen/storage.py`, replacing the current `save_parquet`:

```python
def save_parquet_with_summary(
    inventory_id: str,
    ddf: dd.DataFrame,
    columns: list[Column],
) -> tuple[str, dict[str, ColumnSummary]]:
    """Write partitions to GCS and compute per-column summaries in one pass.

    Builds a deferred write (compute=False) and a per-column reduction graph,
    then fuses them with a single dask.compute call so each partition is
    materialized exactly once.
    """
    path = f"gs://{INVENTORIES_BUCKET}/{inventory_id}"
    write_delayed = ddf.to_parquet(path, write_metadata_file=True, compute=False)
    stats_delayed = _build_column_stats_graph(ddf, columns)
    _, stats = dask.compute(write_delayed, stats_delayed)
    return path, stats
```

`_build_column_stats_graph(ddf, columns)` returns a `dask.delayed` dict-of-`ColumnSummary` keyed by column key — see the algorithm section below.

`standgen/dispatch.py` swaps its `save_parquet(...)` call for `save_parquet_with_summary(...)` and merges the returned stats into the Firestore `columns` entries by key prior to the final document write.

This is the single chokepoint shared by every inventory handler (pim, chm) and by modifications, so one implementation covers every inventory source and ensures modifications produce recomputed summaries automatically.

## Compute algorithm

For each column, build a small per-column reduction graph and bundle them into a `dask.delayed` dict. The graph fuses with `to_parquet`'s graph during `dask.compute`, so each partition flows to both the writer and the reductions in one pass.

**Continuous columns:**
- mask is `~ddf[col].isna()` — null values per pandas convention
- accumulate: `count`, `null_count`, `min`, `max`, `mean`, `std` via dask's lazy reductions (`.count()`, `.isna().sum()`, `.min()`, `.max()`, `.mean()`, `.std()`)
- if `count == 0`, emit `min/max/mean/std = None`

**Categorical columns:**
- accumulate: `count`, `null_count`, `unique_count` via `.nunique()` (approximate for very large dask frames, exact for smaller ones; sufficient for a summary card)

No new dependencies. No histogram, no percentiles, no top-K.

## Tests

In `services/standgen/tests/`:

- Unit: `save_parquet_with_summary` on small pandas/dask fixtures
  - continuous column with mixed valid + null cells → exact `count` / `null_count` / `min` / `max` / `mean` / `std` match `numpy.nanmean` etc. ground truth
  - all-null continuous column → `count == 0`, scalars `None`
  - categorical column with nulls → null excluded from `unique_count`
  - **single-pass guarantee:** count source partition reads via a dask diagnostics hook (or a custom partition-counting wrapper) and assert it equals `npartitions`, not `2 * npartitions`. This is the regression test for the double-scan bug.
- Integration: one handler end-to-end (cheapest is a small PIM or hand-rolled CHM fixture) asserting the Firestore inventory doc carries `columns[*].summary` after completion, and that a modification produces an updated summary.

In `services/api/tests/`:

- Schema round-trip: `Column` with each variant of `ColumnSummary` serializes/deserializes and validates the discriminator.

## Out of scope (deliberate, deferable)

- Histograms / percentiles / top-K categorical values — additive fields on the same union, safe to add later.
- Forestry / stand-level metrics (basal area, TPA, QMD, SDI, species composition) — separate issue, since those are inventory-type-specific aggregates rather than per-column reductions.

## Files touched

- `services/api/api/resources/inventories/schema.py` — schema additions
- `services/standgen/standgen/storage.py` — replace `save_parquet` with `save_parquet_with_summary` (single chokepoint enforces single-pass)
- `services/standgen/standgen/dispatch.py` — call-site swap
- `services/standgen/tests/test_storage.py` (or `test_summarize.py`) — new, includes the single-pass regression test
- `services/api/tests/resources/inventories/test_schema.py` — schema tests
- `docs/inventories.md` — document the new field

## Related

- #257 — per-band summary statistics on grids (same schema shape; gridded data was already materialized, so the dask-fusion concern didn't apply there)

## Plan:

- `schema.py` — add the two summary models, ColumnSummary union, summary field on Column
- `storage.py` — add `save_parquet_with_summary` and `save_parquet_replace_with_summary`, with a shared `_build_column_stats_graph` internal helper. Keep existing `save_parquet` / `save_parquet_replace` untouched since they're the building blocks.
- `pim.py` — swap call site, return columns in result dict (I'll assume `chm.py` mirrors this)
- `modifications.py` / `treatments.py` — same swap, pass `inventory["columns"]`, return columns in result
- `main.py` — add `columns` to the `update_status` call when present in result
