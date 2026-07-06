# walle

walle is the FastFuels API v2 cleanup job — a nightly Cloud Run *job* (triggered
by Cloud Scheduler) that is the single owner of GCS artifact deletion. The API
deletes Firestore docs synchronously (so quota frees instantly); walle reclaims
the physical bytes and enforces retention.

It runs **one reconciliation pass** — a single projected scan per collection —
and deletes for three reasons ("deletion categories"), each with its own
dry-run switch:

| Category | Detects | Deletes |
|----------|---------|---------|
| Orphaned GCS blobs | an artifact whose owning doc is gone | the blob |
| Orphaned child docs | a child whose `domain_id` no longer exists | doc + artifact |
| TTL-expired docs | a doc past its owner's resolved retention | doc + artifact |

Deletion order is GCS-first, then the Firestore doc, so a crash between the two
leaves the doc behind and the next run re-reaps it — both idempotent. Doc
deletes are batched/throttled through a single `BulkWriter`.

## Configuration (env)

- `WALLE_ORPHAN_BLOBS_DRY_RUN`, `WALLE_ORPHAN_DOCS_DRY_RUN`, `WALLE_TTL_DRY_RUN` —
  default `false` (enforce). Set `true` to log a category's candidates without
  deleting; used to validate a category locally before shipping.
- `WALLE_TTL_FLOOR_DAYS` (default 7) — resolved TTLs are clamped to at least this.
- `WALLE_ORPHAN_MIN_AGE_HOURS` (default 24) — orphaned docs younger than this are
  left alone.

Plus the standard `lib.config` infrastructure vars (`GCP_PROJECT`, the bucket and
collection names). See the repo `.env.example`.

## Run locally (dry-run first)

```bash
cd services/walle
WALLE_ORPHAN_BLOBS_DRY_RUN=true \
WALLE_ORPHAN_DOCS_DRY_RUN=true \
WALLE_TTL_DRY_RUN=true \
uv run python -m walle
```

Inspect the `DRY-RUN` log lines against reality before running enforce.

## Tests

```bash
cd services/walle
uv run pytest tests/ -v                      # unit
uv run pytest tests/integration/ -v          # live Firestore + GCS
```

## Deployment

`.github/workflows/walle.yml` builds the image and rolls the Cloud Run **job**
`walle-v2-<env>` (`gcloud run jobs deploy`). Deploying the image does **not** run
it — a nightly Cloud Scheduler trigger (HTTP `POST .../jobs/walle-v2-<env>:run`,
mirroring v1 walle) runs it, and that trigger is provisioned **out-of-band**.

The job's service account needs Firestore access and GCS object-delete on the
five artifact buckets.

**Gate on the scheduler, not the code.** All categories default to enforce, so
before creating the nightly trigger:

1. Run locally with all categories dry-run and validate the candidates (above).
2. Enable blob + doc reconciliation — safe, they only touch already-orphaned
   records.
3. TTL deletes live user data. Per design §10.2 (retention-on-by-default), the
   expiry-warning emails must ship before it enforces on a schedule; until then,
   run the scheduled job with `WALLE_TTL_DRY_RUN=true` so blob/doc reconciliation
   runs while TTL only logs.
