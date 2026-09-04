# walle

walle is the FastFuels API v2 cleanup job — a nightly Cloud Run *job* (triggered
by Cloud Scheduler) that is the single owner of GCS artifact deletion. The API
deletes Firestore docs synchronously (so quota frees instantly); walle reclaims
the physical bytes and enforces retention.

It runs **one reconciliation pass** — a single projected scan per collection —
and deletes for five reasons ("deletion categories"), each with its own
dry-run switch:

| Category | Detects | Deletes |
|----------|---------|---------|
| Orphaned GCS blobs | an artifact whose owning doc is gone | the blob |
| Orphaned child docs | a child whose `domain_id` no longer exists | doc + artifact |
| TTL-expired docs | a doc past its owner's resolved retention | doc + artifact |
| Guest resources | a doc owned by an anonymous (guest) uid past a short window | doc + artifact (guest domains: doc only) |
| Stale test resources | a `test-` doc past a short retention window | doc + artifact (test domains: doc only) |

Deletion order is GCS-first, then the Firestore doc, so a crash between the two
leaves the doc behind and the next run re-reaps it — both idempotent. Reads
(re-checks, owner lookups) are batched through `get_all`, and all deletes are
executed in bulk at the end of the run — every GCS artifact first (chunked
`rm`), then every doc (one `BulkWriter`).

The **stale-test** category sweeps the ephemeral integration-test junk CI leaves
in the shared project. Real resource ids are server-generated `uuid4` hex (never
`test-`), so it only ever reaps test artifacts; the retention window is far
longer than any test run, so an in-flight test is never raced. The persistent
`static-test-` fixtures are excluded (they don't start with `test-`).

The **guest** category reaps no-account trial data. Nothing on the doc marks a
guest, so walle asks Firebase Auth (`get_users`, batched) which owners are
anonymous users — a Firebase user with no sign-in provider. Owners Auth does not
know (application ids, test owners) are not guests, and a guest who links a
credential stops being one.

## Configuration (env)

- `WALLE_ORPHAN_BLOBS_DRY_RUN`, `WALLE_ORPHAN_DOCS_DRY_RUN`, `WALLE_TTL_DRY_RUN`,
  `WALLE_TEST_PURGE_DRY_RUN` — default `false` (enforce). Set `true` to log a
  category's candidates without deleting; used to validate a category locally
  before shipping.
- `WALLE_GUEST_REAP_DRY_RUN` — default `true` (dry-run). The guest category ships
  logging-only; set `false` to enforce once the candidates look right in prod.
- `WALLE_TTL_FLOOR_DAYS` (default 7) — resolved TTLs are clamped to at least this.
- `WALLE_ORPHAN_MIN_AGE_HOURS` (default 24) — orphaned docs younger than this are
  left alone.
- `WALLE_GUEST_TTL_HOURS` (default 24) — anonymous-owned resources older than this
  are reaped (ignores the TTL floor).
- `WALLE_TEST_TTL_DAYS` (default 7) — `test-` resources older than this are purged.

Plus the standard `lib.config` infrastructure vars (`GCP_PROJECT`, the bucket and
collection names). See the repo `.env.example`.

## Run locally (dry-run first)

```bash
cd services/walle
WALLE_ORPHAN_BLOBS_DRY_RUN=true \
WALLE_ORPHAN_DOCS_DRY_RUN=true \
WALLE_TTL_DRY_RUN=true \
WALLE_TEST_PURGE_DRY_RUN=true \
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

`.github/workflows/walle.yml` (on push to `main`) builds the image and rolls the
single Cloud Run **job** `walle-v2` (`gcloud run jobs deploy`). walle reconciles
the shared Firestore + GCS state, so there is one job — not a per-environment
pair; a second instance would only race the first over identical data. Deploying
the image does **not** run it — a nightly Cloud Scheduler trigger (HTTP
`POST .../jobs/walle-v2:run`, mirroring v1 walle) runs it, and that trigger is
provisioned **out-of-band**.

The job's service account needs Firestore access, GCS object-delete on the five
artifact buckets, and Firebase Auth read (`get_users`) for the guest-owner probe
(the Firebase Authentication Viewer role, or `identitytoolkit` accounts lookup).

All categories except **guest** default to enforce (guest ships dry-run until
its candidates have been checked in prod). Before creating the nightly trigger, run
walle locally with the categories in dry-run and check the candidates against
reality (above); once they look right, schedule it. Retention (180 days
standard / never for applications) is a documented contract from a resource's
first day and is visible on `GET /users/me`, so TTL expiry needs no separate
gate.
