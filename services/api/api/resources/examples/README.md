# Examples Resource (v2)

The shared, prebuilt example (#582): one canonical Blue Mountain domain and one
completed NAIP canopy grid, built **once** and readable by **any authenticated
caller** — guests included — without owning it or spending quota. It replaces
the per-guest recompute that had every guest recreate the domain and re-dispatch
a NAIP grid, each spending from the shared weekly dispatch ceiling.

## Access model

Every other resource is owner-scoped: a read passes `owner_id ==
request.state.id` to `get_document_async`, which 404s on any mismatch. The
example is the single deliberate exception, implemented in `api/access.py`:

```
readable by a non-owner  ⟺  owner_id == EXAMPLE_OWNER_ID  AND  is_example is True
```

Both conditions are required, so opening read access can never leak an arbitrary
owner's resource:

- a normal owner's document fails the owner check, and
- any document the example owner has not explicitly flagged fails the flag
  check (a self-set `is_example` on someone else's document is worthless — the
  owner must be the example owner).

`get_readable_document_async` runs this authorization **before** the optional
status check, so a non-owned, non-example resource always 404s and can never
leak its existence through a 422 status mismatch.

### Read-only

Only the read paths use the relaxed access:

- `GET /domains/{id}` (via `get_readable_document_async`)
- `GET /domains/{id}/grids/{grid_id}` and the grid **data** endpoints
  (`.../chunks/...`, `.../data/{band}/{chunk}`, `.../data/{band}/{chunk}/binary`)
  — via the `ReadableDomain` dependency plus `get_readable_document_async`.

Writes, exports, and grid creation keep the owner-strict `VerifiedDomain` +
`get_document_async` path, so a guest can neither modify the example nor create
resources under the example domain (both 404 because they do not own it).

### Authentication

The read relaxation admits any authenticated caller, guest or account. It does
**not** admit unauthenticated requests: every route stays behind
`authenticate_user`. Minting an anonymous guest token is the cheap public entry
point, so there is no need to also open an unauthenticated door.

## Discovery

### GET /examples

Returns the example domain id and its completed grid id(s) so the web client can
deep-link into the 3D viewer without hardcoded ids:

```json
{
  "examples": [
    {
      "domain_id": "…",
      "name": "Blue Mountain Example",
      "grids": [{ "id": "…", "name": "NAIP Canopy Height", "domain_id": "…" }]
    }
  ]
}
```

Ids are resolved from Firestore (the same `owner_id == EXAMPLE_OWNER_ID AND
is_example` predicate the read path authorizes against), so a re-seed under new
ids is picked up automatically and the endpoint can never advertise something a
caller would then be denied. Only `completed` grids are listed. An empty
`examples` array means nothing has been seeded yet.

## Permanence (walle)

The example resources are permanent. `EXAMPLE_OWNER_ID` is exempt from every
doc-reap category in `walle/cleanup.py` (`_is_example_owned`), most importantly
the TTL sweep — the example owner has no owner document and would otherwise
resolve to the standard 180-day retention. The guard is applied in the finders
rather than by filtering the scan, so an example document stays in the live-id
set and its GCS artifact is never mistaken for an orphan blob.

## Seeding (operational)

There is no credential that authenticates *as* the example owner — it is only an
ownership tag. `scripts/seed_example.py` builds the example once through the
normal API using a real seeding key, waits for the NAIP grid to finish, then
promotes the two Firestore documents to `EXAMPLE_OWNER_ID` with
`is_example = True`. Re-tagging the owner is safe for the grid's GCS artifact
because artifacts are keyed by grid id, not owner. The script is idempotent (a
completed example short-circuits it) and supports resuming a slow build. See its
module docstring for usage.
