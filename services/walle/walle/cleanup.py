"""The nightly reconciliation pass.

One projected scan per collection produces light records; from those shared
reads walle derives four deletion categories and reaps them through one
idempotent GCS-then-doc routine:

1. Orphaned GCS blobs — an artifact whose owning doc is gone (blob only).
2. Orphaned child docs — a child whose ``domain_id`` is gone (doc + artifact).
3. TTL-expired docs — a doc past its owner's resolved retention (doc + artifact).
4. Stale test resources — an ephemeral ``test-`` doc past a short retention
   window (doc + artifact; test domains are purged doc-only).

Docs are deleted synchronously by the API, so orphaned blobs (category 1) are
the common case; 2 and 3 are the durable backstop and the retention policy; 4
sweeps the ephemeral integration-test junk that CI leaves in the shared project.
All Firestore re-checks and owner lookups are batched through ``get_all``; the
run accumulates the deletions and executes them in bulk at the end — every GCS
artifact first, then every Firestore doc — so a crash between the two leaves the
docs behind and the next run re-reaps them (both idempotent).
"""

import logging
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from lib.config import (
    APPLICATIONS_COLLECTION,
    DOMAINS_COLLECTION,
    USERS_COLLECTION,
)
from lib.firestore.documents import firestore_client
from walle.config import (
    ORPHAN_BLOBS_DRY_RUN,
    ORPHAN_DOCS_DRY_RUN,
    ORPHAN_MIN_AGE_HOURS,
    TEST_PURGE_DRY_RUN,
    TEST_TTL_DAYS,
    TTL_DRY_RUN,
    TTL_FLOOR_DAYS,
)
from walle.layouts import (
    RESOURCE_LAYOUTS,
    ResourceLayout,
    artifact_path,
    delete_artifacts,
    list_artifact_ids,
)

logger = logging.getLogger(__name__)

# Lifecycle defaults mirror api/quota.py Quotas / TIER_PRESETS. The ONLY tier
# rule that changes a TTL is that the application tier never expires; everything
# else uses the standard defaults plus any per-owner quota_overrides. Kept as a
# small local copy rather than importing the api package into walle's image
# (neither service depends on the other) — keep these values in sync with
# api/quota.py by hand.
FAILED_STATUS = "failed"
DEFAULT_RESOURCE_TTL_DAYS = 180
DEFAULT_FAILED_RESOURCE_TTL_DAYS = 14
_DEFAULT_TTLS = (DEFAULT_RESOURCE_TTL_DAYS, DEFAULT_FAILED_RESOURCE_TTL_DAYS)
_TIER_TTL_OVERRIDES: dict[str, dict] = {"application": {"resource_ttl_days": None}}

# Firestore fields the single scan projects — everything the four categories need.
_SCAN_FIELDS = ["domain_id", "owner_id", "status", "modified_on", "size_bytes"]

# Firestore batched reads (get_all) are chunked at this size. A fresh,
# mostly-orphaned bucket can have thousands of candidates; chunking bounds each
# BatchGetDocuments request (reads have no 500-doc write-batch limit — the cap is
# the 10 MiB request size, and doc paths are tiny) while avoiding per-candidate
# round-trips.
_GET_ALL_CHUNK = 1000


@dataclass(frozen=True)
class Record:
    """A projected view of one resource doc — all categories read from this."""

    collection: str
    doc_id: str
    domain_id: str | None
    owner_id: str | None
    status: str | None
    modified_on: datetime | None
    size_bytes: int | None


@dataclass
class Summary:
    """Per-run tallies, keyed by category, for the closing summary log."""

    deleted: Counter = field(default_factory=Counter)
    dry_run: Counter = field(default_factory=Counter)
    freed_bytes: int = 0

    def as_dict(self) -> dict:
        return {
            "deleted": dict(self.deleted),
            "dry_run": dict(self.dry_run),
            "freed_bytes": self.freed_bytes,
        }


# --- batched Firestore reads ----------------------------------------------


def _existing_ids(collection: str, ids) -> set[str]:
    """The subset of ``ids`` whose documents exist, via batched get_all."""
    coll = firestore_client.collection(collection)
    ids = list(ids)
    live: set[str] = set()
    for start in range(0, len(ids), _GET_ALL_CHUNK):
        refs = [coll.document(i) for i in ids[start : start + _GET_ALL_CHUNK]]
        live.update(snap.id for snap in firestore_client.get_all(refs) if snap.exists)
    return live


def _get_all_docs(collection: str, ids: list[str]) -> dict[str, dict]:
    """``{doc_id: data}`` for the documents that exist, via batched get_all."""
    coll = firestore_client.collection(collection)
    out: dict[str, dict] = {}
    for start in range(0, len(ids), _GET_ALL_CHUNK):
        refs = [coll.document(i) for i in ids[start : start + _GET_ALL_CHUNK]]
        for snap in firestore_client.get_all(refs):
            if snap.exists:
                out[snap.id] = snap.to_dict() or {}
    return out


# --- owner TTL resolution -------------------------------------------------


def _ttls_from_doc(data: dict) -> tuple[int | None, int | None]:
    """(resource_ttl_days, failed_resource_ttl_days) from an owner document's data."""
    tier = data.get("tier") or "standard"
    base = {
        "resource_ttl_days": DEFAULT_RESOURCE_TTL_DAYS,
        "failed_resource_ttl_days": DEFAULT_FAILED_RESOURCE_TTL_DAYS,
        **_TIER_TTL_OVERRIDES.get(tier, {}),
    }
    overrides = data.get("quota_overrides") or {}
    return (
        overrides.get("resource_ttl_days", base["resource_ttl_days"]),
        overrides.get("failed_resource_ttl_days", base["failed_resource_ttl_days"]),
    )


def resolve_owner_ttls_bulk(
    owner_ids,
) -> dict[str, tuple[int | None, int | None]]:
    """Resolve TTLs for many owners with batched reads.

    Owner-kind probe: check ``applications-v2`` first (an application always has
    a document), then ``users-v2`` for the misses; anything still missing gets
    the standard defaults. Two batched ``get_all`` passes instead of a serial
    one-or-two ``get()`` per owner.
    """
    remaining = [o for o in dict.fromkeys(owner_ids) if o]
    result: dict[str, tuple[int | None, int | None]] = {}
    for collection in (APPLICATIONS_COLLECTION, USERS_COLLECTION):
        if not remaining:
            break
        docs = _get_all_docs(collection, remaining)
        for owner_id, data in docs.items():
            result[owner_id] = _ttls_from_doc(data)
        remaining = [o for o in remaining if o not in docs]
    for owner_id in remaining:
        result[owner_id] = _DEFAULT_TTLS
    return result


def _effective_ttl_days(
    rec: Record, owner_ttls: dict[str, tuple[int | None, int | None]]
) -> int | None:
    """The retention window that applies to ``rec``, or None if it never expires."""
    resource_ttl, failed_ttl = owner_ttls.get(rec.owner_id, _DEFAULT_TTLS)
    ttl = failed_ttl if rec.status == FAILED_STATUS else resource_ttl
    if ttl is None:
        return None
    # Clamp to the floor so no override deletes below the safety minimum (§6).
    return max(ttl, TTL_FLOOR_DAYS)


# --- scanning -------------------------------------------------------------


def scan_collection(layout: ResourceLayout) -> list[Record]:
    """Stream one projected pass over a collection into light records."""
    stream = (
        firestore_client.collection(layout.collection).select(_SCAN_FIELDS).stream()
    )
    records = []
    for snap in stream:
        d = snap.to_dict() or {}
        records.append(
            Record(
                collection=layout.collection,
                doc_id=snap.id,
                domain_id=d.get("domain_id"),
                owner_id=d.get("owner_id"),
                status=d.get("status"),
                modified_on=d.get("modified_on"),
                size_bytes=d.get("size_bytes"),
            )
        )
    return records


def scan_domains() -> list[Record]:
    """Stream domain docs (id + modified_on + owner_id).

    Serves both the live-domain set (for orphaned-child detection) and the
    test-resource purge. ``owner_id`` is projected so the purge can match domains
    created through the API as a test owner (bare-hex ids). Domains have no GCS
    artifact, so the remaining fields are left empty.
    """
    stream = (
        firestore_client.collection(DOMAINS_COLLECTION)
        .select(["modified_on", "owner_id"])
        .stream()
    )
    records = []
    for snap in stream:
        d = snap.to_dict() or {}
        records.append(
            Record(
                collection=DOMAINS_COLLECTION,
                doc_id=snap.id,
                domain_id=None,
                owner_id=d.get("owner_id"),
                status=None,
                modified_on=d.get("modified_on"),
                size_bytes=None,
            )
        )
    return records


# --- category finders -----------------------------------------------------

# Persistent integration-test fixtures live in the shared project under this id
# prefix — often as GCS artifacts with no live doc. walle must never reap them,
# or a run would delete the fixtures the test suite depends on.
_PROTECTED_ID_PREFIXES = ("static-test-",)


def _is_protected(doc_id: str) -> bool:
    return doc_id.startswith(_PROTECTED_ID_PREFIXES)


def _is_test_record(rec: Record) -> bool:
    """Whether ``rec`` is an ephemeral integration-test artifact.

    Two independent markers: an id minted by the suite (``test-`` prefix), or
    ownership by a test owner. Test owner_ids are themselves ``test-``-prefixed
    (``test-owner`` and the per-test ``test-<uuid4hex>`` owners), so resources
    created through the API as a test owner — which get server-generated bare-hex
    ids the id check alone misses — are still caught. ``static-test-`` fixtures
    are excluded by the caller via ``_is_protected``.
    """
    return rec.doc_id.startswith("test-") or (rec.owner_id or "").startswith("test-")


def _older_than(ts, cutoff: datetime) -> bool:
    """Whether ``ts`` is a real datetime strictly before ``cutoff``.

    Some legacy domain docs store ``modified_on`` as a string; a non-datetime
    value is treated as "age unknown" and is never old enough to reap.
    """
    return isinstance(ts, datetime) and ts < cutoff


def find_orphan_docs(
    records: list[Record], domain_ids: set[str], now: datetime
) -> list[Record]:
    """Docs whose containing domain no longer exists (age-guarded)."""
    cutoff = now - timedelta(hours=ORPHAN_MIN_AGE_HOURS)
    out = []
    for rec in records:
        if _is_protected(rec.doc_id):
            continue
        if rec.domain_id is None or rec.domain_id in domain_ids:
            continue
        # Skip provably-recent docs so a resource mid-creation is never mistaken
        # for an orphan (unknown age is not skipped — the missing domain already
        # marks it an orphan).
        if isinstance(rec.modified_on, datetime) and rec.modified_on > cutoff:
            continue
        out.append(rec)
    return out


def confirm_orphan_docs(candidates: list[Record]) -> list[Record]:
    """Drop candidates whose domain actually still exists.

    ``find_orphan_docs`` already excluded domains present in the streamed
    domain-id set; this batched re-check catches a domain missed in that stream
    (Firestore eventual consistency) before anything is reaped.
    """
    domains = _existing_ids(DOMAINS_COLLECTION, {r.domain_id for r in candidates})
    return [r for r in candidates if r.domain_id not in domains]


def find_expired(
    records: list[Record],
    now: datetime,
    owner_ttls: dict[str, tuple[int | None, int | None]],
) -> list[Record]:
    """Docs older than their owner's resolved TTL."""
    out = []
    for rec in records:
        if _is_protected(rec.doc_id):
            continue
        ttl_days = _effective_ttl_days(rec, owner_ttls)
        if ttl_days is None:
            continue
        if _older_than(rec.modified_on, now - timedelta(days=ttl_days)):
            out.append(rec)
    return out


def find_stale_test(records: list[Record], now: datetime) -> list[Record]:
    """Ephemeral test docs older than the short test-retention window.

    Matches by suite-minted id or by test-owner ownership (see
    ``_is_test_record``), excluding protected ``static-test-`` fixtures. The
    window is far longer than any test run, so it never races an in-flight test.
    """
    cutoff = now - timedelta(days=TEST_TTL_DAYS)
    return [
        rec
        for rec in records
        if not _is_protected(rec.doc_id)
        and _is_test_record(rec)
        and _older_than(rec.modified_on, cutoff)
    ]


def find_orphan_blobs(
    layout: ResourceLayout, artifacts: dict[str, str], live_ids: set[str]
) -> dict[str, str]:
    """Artifact id -> path for artifacts whose owning doc is gone.

    Candidates (artifact id not in the live-id set) are re-checked directly
    before being returned: a doc missed in the collection stream (Firestore
    eventual consistency) must never have its live artifact reaped. Docs are
    always written before their GCS, so "artifact, no doc" is otherwise a
    reliable orphan signal. The re-check is batched — a mostly-orphaned bucket
    can have thousands of candidates.
    """
    candidate_ids = [i for i in set(artifacts) - live_ids if not _is_protected(i)]
    still_live = _existing_ids(layout.collection, candidate_ids)
    return {i: artifacts[i] for i in candidate_ids if i not in still_live}


# --- reaping --------------------------------------------------------------


def _age_days(ts: datetime | None, now: datetime) -> float | None:
    return None if ts is None else (now - ts).total_seconds() / 86400.0


def _reap_blob(
    layout: ResourceLayout,
    doc_id: str,
    path: str,
    *,
    dry_run: bool,
    summary: Summary,
    gcs_deletes: list[str],
) -> None:
    if dry_run:
        logger.info("DRY-RUN orphan_blob %s/%s -> %s", layout.name, doc_id, path)
        summary.dry_run["orphan_blob"] += 1
        return
    logger.info("delete orphan_blob %s/%s -> %s", layout.name, doc_id, path)
    gcs_deletes.append(path)
    summary.deleted["orphan_blob"] += 1


def _reap_doc(
    resource_name: str,
    rec: Record,
    path: str | None,
    now: datetime,
    *,
    category: str,
    dry_run: bool,
    summary: Summary,
    gcs_deletes: list[str],
    doc_deletes: list,
) -> None:
    """Reap one doc (+ its artifact if ``path`` is given). Domains pass ``path=None``."""
    age = _age_days(rec.modified_on, now)
    age_str = f"{age:.1f}d" if age is not None else "unknown"

    if dry_run:
        logger.info(
            "DRY-RUN %s %s/%s owner=%s age=%s",
            category,
            resource_name,
            rec.doc_id,
            rec.owner_id,
            age_str,
        )
        summary.dry_run[category] += 1
        return

    logger.info(
        "delete %s %s/%s owner=%s age=%s bytes=%s",
        category,
        resource_name,
        rec.doc_id,
        rec.owner_id,
        age_str,
        rec.size_bytes,
    )
    if path is not None:
        gcs_deletes.append(path)
    doc_deletes.append(firestore_client.collection(rec.collection).document(rec.doc_id))
    summary.deleted[category] += 1
    summary.freed_bytes += rec.size_bytes or 0


def _bulk_delete_docs(refs: list) -> None:
    """Delete many Firestore docs through one throttled BulkWriter."""
    if not refs:
        return
    bulk_writer = firestore_client.bulk_writer()
    for ref in refs:
        bulk_writer.delete(ref)
    bulk_writer.close()


def run() -> dict:
    """Run one reconciliation pass over every resource type. Returns a summary."""
    now = datetime.now(UTC)
    domain_records = scan_domains()
    domain_ids = {r.doc_id for r in domain_records}
    summary = Summary()
    owner_ttls: dict[str, tuple[int | None, int | None]] = {}
    gcs_deletes: list[str] = []
    doc_deletes: list = []
    logger.info("walle start: %d domains", len(domain_ids))

    for layout in RESOURCE_LAYOUTS:
        records = scan_collection(layout)
        live_ids = {r.doc_id for r in records}

        # Batch-resolve any owners not seen in an earlier collection.
        new_owners = [
            r.owner_id for r in records if r.owner_id and r.owner_id not in owner_ttls
        ]
        owner_ttls.update(resolve_owner_ttls_bulk(new_owners))

        # Categories on live docs, deduped so a doc is reaped under one reason.
        orphan_docs = (
            confirm_orphan_docs(find_orphan_docs(records, domain_ids, now))
            if layout.orphan_on_missing_domain
            else []
        )
        reaped_ids = {r.doc_id for r in orphan_docs}
        expired = [
            r
            for r in find_expired(records, now, owner_ttls)
            if r.doc_id not in reaped_ids
        ]
        reaped_ids |= {r.doc_id for r in expired}
        stale_test = [
            r for r in find_stale_test(records, now) if r.doc_id not in reaped_ids
        ]

        artifacts = list_artifact_ids(layout)
        orphan_blobs = find_orphan_blobs(layout, artifacts, live_ids)

        logger.info(
            "%s: %d docs, %d artifacts | orphan_blobs=%d orphan_docs=%d expired=%d test=%d",
            layout.name,
            len(records),
            len(artifacts),
            len(orphan_blobs),
            len(orphan_docs),
            len(expired),
            len(stale_test),
        )

        for doc_id, path in orphan_blobs.items():
            _reap_blob(
                layout,
                doc_id,
                path,
                dry_run=ORPHAN_BLOBS_DRY_RUN,
                summary=summary,
                gcs_deletes=gcs_deletes,
            )
        for category, dry_run, batch in (
            ("orphan_doc", ORPHAN_DOCS_DRY_RUN, orphan_docs),
            ("ttl", TTL_DRY_RUN, expired),
            ("test_stale", TEST_PURGE_DRY_RUN, stale_test),
        ):
            for rec in batch:
                _reap_doc(
                    layout.name,
                    rec,
                    artifact_path(layout, rec.doc_id, rec.domain_id),
                    now,
                    category=category,
                    dry_run=dry_run,
                    summary=summary,
                    gcs_deletes=gcs_deletes,
                    doc_deletes=doc_deletes,
                )

    # Ephemeral test domains carry no artifact — purge them doc-only.
    stale_test_domains = find_stale_test(domain_records, now)
    logger.info(
        "domains: %d docs | test=%d", len(domain_records), len(stale_test_domains)
    )
    for rec in stale_test_domains:
        _reap_doc(
            "domains",
            rec,
            None,
            now,
            category="test_stale",
            dry_run=TEST_PURGE_DRY_RUN,
            summary=summary,
            gcs_deletes=gcs_deletes,
            doc_deletes=doc_deletes,
        )

    # Execute the accumulated deletes in bulk: every GCS artifact first, then the
    # docs, so a crash between leaves the docs behind for the next run to re-reap.
    delete_artifacts(gcs_deletes)
    _bulk_delete_docs(doc_deletes)

    logger.info("walle done: %s", summary.as_dict())
    return summary.as_dict()
