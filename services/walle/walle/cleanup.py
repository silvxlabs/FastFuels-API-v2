"""The nightly reconciliation pass.

One projected scan per collection produces light records; from those shared
reads walle derives three deletion categories and reaps them through one
idempotent GCS-then-doc routine:

1. Orphaned GCS blobs — an artifact whose owning doc is gone (blob only).
2. Orphaned child docs — a child whose ``domain_id`` is gone (doc + artifact).
3. TTL-expired docs — a doc past its owner's resolved retention (doc + artifact).

Docs are deleted synchronously by the API, so orphaned blobs (category 1) are
the common case; categories 2 and 3 are the durable backstop and the retention
policy. Deletion order is GCS first, then the Firestore doc, so a crash between
the two leaves the doc behind and the next run re-reaps it — both idempotent.
"""

import functools
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
    TTL_DRY_RUN,
    TTL_FLOOR_DAYS,
)
from walle.layouts import (
    RESOURCE_LAYOUTS,
    ResourceLayout,
    artifact_path,
    delete_artifact,
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
_TIER_TTL_OVERRIDES: dict[str, dict] = {"application": {"resource_ttl_days": None}}

# Firestore fields the single scan projects — everything the three categories need.
_SCAN_FIELDS = ["domain_id", "owner_id", "status", "modified_on", "size_bytes"]

# The orphan-blob re-check is batched through get_all in chunks of this size. A
# fresh, mostly-orphaned bucket can have thousands of candidates; chunking bounds
# each BatchGetDocuments request (reads have no 500-doc write-batch limit — the
# cap is the 10 MiB request size, and doc paths are tiny) while avoiding
# per-candidate round-trips.
_GET_ALL_CHUNK = 1000


@dataclass(frozen=True)
class Record:
    """A projected view of one resource doc — all three categories read from this."""

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
    skipped: Counter = field(default_factory=Counter)
    freed_bytes: int = 0

    def as_dict(self) -> dict:
        return {
            "deleted": dict(self.deleted),
            "dry_run": dict(self.dry_run),
            "skipped": dict(self.skipped),
            "freed_bytes": self.freed_bytes,
        }


# --- owner TTL resolution -------------------------------------------------


@functools.cache
def resolve_owner_ttls(owner_id: str | None) -> tuple[int | None, int | None]:
    """(resource_ttl_days, failed_resource_ttl_days) for an owner.

    Owner-kind probe: an application always has an ``applications-v2`` document,
    so check there first; a miss means a user (whose ``users-v2`` doc may be
    absent → standard defaults). Cached for the process — one nightly run.
    """
    data = _owner_doc(owner_id) if owner_id else {}
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


def _owner_doc(owner_id: str) -> dict:
    for collection in (APPLICATIONS_COLLECTION, USERS_COLLECTION):
        snap = firestore_client.collection(collection).document(owner_id).get()
        if snap.exists:
            return snap.to_dict() or {}
    return {}


def _effective_ttl_days(rec: Record) -> int | None:
    """The retention window that applies to ``rec``, or None if it never expires."""
    resource_ttl, failed_ttl = resolve_owner_ttls(rec.owner_id)
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


def live_domain_ids() -> set[str]:
    """Every existing domain id (projected to ids only)."""
    stream = (
        firestore_client.collection(DOMAINS_COLLECTION).select(["owner_id"]).stream()
    )
    return {snap.id for snap in stream}


# --- category finders -----------------------------------------------------

# Persistent integration-test fixtures live in the shared project under this id
# prefix — often as GCS artifacts with no live doc. walle must never reap them,
# or a run would delete the fixtures the test suite depends on.
_PROTECTED_ID_PREFIXES = ("static-test-",)


def _is_protected(doc_id: str) -> bool:
    return doc_id.startswith(_PROTECTED_ID_PREFIXES)


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
        # Skip very recently modified docs so a resource mid-creation is never
        # mistaken for an orphan.
        if rec.modified_on is not None and rec.modified_on > cutoff:
            continue
        out.append(rec)
    return out


def find_expired(records: list[Record], now: datetime) -> list[Record]:
    """Docs older than their owner's resolved TTL."""
    out = []
    for rec in records:
        if _is_protected(rec.doc_id):
            continue
        if rec.modified_on is None:
            continue
        ttl_days = _effective_ttl_days(rec)
        if ttl_days is None:
            continue
        if rec.modified_on < now - timedelta(days=ttl_days):
            out.append(rec)
    return out


def find_orphan_blobs(
    layout: ResourceLayout, artifacts: dict[str, str], live_ids: set[str]
) -> dict[str, str]:
    """Artifact id -> path for artifacts whose owning doc is gone.

    Candidates (artifact id not in the live-id set) are re-checked directly
    before being returned: a doc missed in the collection stream (Firestore
    eventual consistency) must never have its live artifact reaped. Docs are
    always written before their GCS, so "artifact, no doc" is otherwise a
    reliable orphan signal.

    The re-check is batched through ``get_all`` — a fresh, mostly-orphaned
    bucket can have thousands of candidates, and a per-candidate ``get()`` would
    be thousands of serial round-trips.
    """
    candidate_ids = [i for i in set(artifacts) - live_ids if not _is_protected(i)]
    coll = firestore_client.collection(layout.collection)
    live: set[str] = set()
    for start in range(0, len(candidate_ids), _GET_ALL_CHUNK):
        refs = [
            coll.document(doc_id)
            for doc_id in candidate_ids[start : start + _GET_ALL_CHUNK]
        ]
        live.update(snap.id for snap in firestore_client.get_all(refs) if snap.exists)
    return {doc_id: artifacts[doc_id] for doc_id in candidate_ids if doc_id not in live}


# --- reaping --------------------------------------------------------------


def _age_days(ts: datetime | None, now: datetime) -> float | None:
    return None if ts is None else (now - ts).total_seconds() / 86400.0


def _reap_blob(
    layout: ResourceLayout, doc_id: str, path: str, *, dry_run: bool, summary: Summary
) -> None:
    if dry_run:
        logger.info("DRY-RUN orphan_blob %s/%s -> %s", layout.name, doc_id, path)
        summary.dry_run["orphan_blob"] += 1
        return
    logger.info("delete orphan_blob %s/%s -> %s", layout.name, doc_id, path)
    delete_artifact(path)
    summary.deleted["orphan_blob"] += 1


def _reap_doc(
    layout: ResourceLayout,
    rec: Record,
    now: datetime,
    bulk_writer,
    *,
    category: str,
    dry_run: bool,
    summary: Summary,
) -> None:
    # Re-check the containment edge for orphaned-doc reaps: a domain missed in
    # the domain listing must not orphan its live children.
    if category == "orphan_doc":
        dom = (
            firestore_client.collection(DOMAINS_COLLECTION)
            .document(rec.domain_id)
            .get()
        )
        if dom.exists:
            summary.skipped[category] += 1
            return

    path = artifact_path(layout, rec.doc_id, rec.domain_id)
    age = _age_days(rec.modified_on, now)
    age_str = f"{age:.1f}d" if age is not None else "unknown"

    if dry_run:
        logger.info(
            "DRY-RUN %s %s/%s owner=%s age=%s",
            category,
            layout.name,
            rec.doc_id,
            rec.owner_id,
            age_str,
        )
        summary.dry_run[category] += 1
        return

    logger.info(
        "delete %s %s/%s owner=%s age=%s bytes=%s",
        category,
        layout.name,
        rec.doc_id,
        rec.owner_id,
        age_str,
        rec.size_bytes,
    )
    delete_artifact(path)  # GCS first
    bulk_writer.delete(
        firestore_client.collection(layout.collection).document(rec.doc_id)
    )
    summary.deleted[category] += 1
    summary.freed_bytes += rec.size_bytes or 0


def run() -> dict:
    """Run one reconciliation pass over every resource type. Returns a summary."""
    now = datetime.now(UTC)
    resolve_owner_ttls.cache_clear()
    domain_ids = live_domain_ids()
    summary = Summary()
    logger.info("walle start: %d domains", len(domain_ids))

    # One BulkWriter batches/throttles every doc delete across all collections;
    # GCS deletes happen inline (before their doc is queued), so a crash before
    # close() leaves docs behind for the next run.
    bulk_writer = firestore_client.bulk_writer()

    for layout in RESOURCE_LAYOUTS:
        records = scan_collection(layout)
        live_ids = {r.doc_id for r in records}

        orphan_docs = (
            find_orphan_docs(records, domain_ids, now)
            if layout.orphan_on_missing_domain
            else []
        )
        orphan_ids = {r.doc_id for r in orphan_docs}
        expired = [r for r in find_expired(records, now) if r.doc_id not in orphan_ids]
        artifacts = list_artifact_ids(layout)
        orphan_blobs = find_orphan_blobs(layout, artifacts, live_ids)

        logger.info(
            "%s: %d docs, %d artifacts | orphan_blobs=%d orphan_docs=%d expired=%d",
            layout.name,
            len(records),
            len(artifacts),
            len(orphan_blobs),
            len(orphan_docs),
            len(expired),
        )

        for doc_id, path in orphan_blobs.items():
            _reap_blob(
                layout, doc_id, path, dry_run=ORPHAN_BLOBS_DRY_RUN, summary=summary
            )
        for rec in orphan_docs:
            _reap_doc(
                layout,
                rec,
                now,
                bulk_writer,
                category="orphan_doc",
                dry_run=ORPHAN_DOCS_DRY_RUN,
                summary=summary,
            )
        for rec in expired:
            _reap_doc(
                layout,
                rec,
                now,
                bulk_writer,
                category="ttl",
                dry_run=TTL_DRY_RUN,
                summary=summary,
            )

    bulk_writer.close()  # flush all queued doc deletes
    logger.info("walle done: %s", summary.as_dict())
    return summary.as_dict()
