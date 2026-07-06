"""Unit tests for walle's cleanup logic (finders, TTL resolution, blob re-check).

The pure finders take in-memory records, so they run without GCP; the Firestore
touching helpers are exercised with a monkeypatched client. The full ``run()``
pass is deliberately not exercised here — it reconciles whole buckets and would
act on real project data — its behaviour is covered by these piece-wise tests.
"""

from datetime import UTC, datetime, timedelta

from walle import cleanup
from walle.cleanup import (
    Record,
    find_expired,
    find_orphan_blobs,
    find_orphan_docs,
)
from walle.config import TTL_FLOOR_DAYS
from walle.layouts import RESOURCE_LAYOUTS

NOW = datetime(2026, 7, 6, tzinfo=UTC)


def rec(**kw) -> Record:
    base = dict(
        collection="grids-v2",
        doc_id="g1",
        domain_id="d1",
        owner_id="o1",
        status="completed",
        modified_on=NOW,
        size_bytes=100,
    )
    base.update(kw)
    return Record(**base)


# --- orphaned child docs --------------------------------------------------


def test_orphan_when_domain_missing():
    r = rec(domain_id="gone", modified_on=NOW - timedelta(days=2))
    assert find_orphan_docs([r], {"d1"}, NOW) == [r]


def test_not_orphan_when_domain_present():
    r = rec(domain_id="d1", modified_on=NOW - timedelta(days=2))
    assert find_orphan_docs([r], {"d1"}, NOW) == []


def test_orphan_age_guard_skips_recent():
    # Domain is gone, but the doc was modified an hour ago — mid-creation safety.
    r = rec(domain_id="gone", modified_on=NOW - timedelta(hours=1))
    assert find_orphan_docs([r], {"d1"}, NOW) == []


def test_orphan_skips_null_domain():
    r = rec(domain_id=None, modified_on=NOW - timedelta(days=5))
    assert find_orphan_docs([r], {"d1"}, NOW) == []


# --- TTL expiry -----------------------------------------------------------


def test_expired_past_standard_ttl(monkeypatch):
    monkeypatch.setattr(cleanup, "resolve_owner_ttls", lambda _o: (180, 14))
    old = rec(modified_on=NOW - timedelta(days=200))
    fresh = rec(modified_on=NOW - timedelta(days=10))
    assert find_expired([old, fresh], NOW) == [old]


def test_failed_uses_failed_clock(monkeypatch):
    monkeypatch.setattr(cleanup, "resolve_owner_ttls", lambda _o: (180, 14))
    expired = rec(status="failed", modified_on=NOW - timedelta(days=20))
    fresh = rec(status="failed", modified_on=NOW - timedelta(days=10))
    assert find_expired([expired, fresh], NOW) == [expired]


def test_application_tier_never_expires(monkeypatch):
    monkeypatch.setattr(cleanup, "resolve_owner_ttls", lambda _o: (None, 14))
    ancient = rec(modified_on=NOW - timedelta(days=9999))
    assert find_expired([ancient], NOW) == []


def test_null_modified_on_never_expires(monkeypatch):
    monkeypatch.setattr(cleanup, "resolve_owner_ttls", lambda _o: (180, 14))
    assert find_expired([rec(modified_on=None)], NOW) == []


def test_ttl_clamped_to_floor(monkeypatch):
    # An override below the floor is clamped up, so it can't sweep too eagerly.
    monkeypatch.setattr(cleanup, "resolve_owner_ttls", lambda _o: (3, 3))
    assert cleanup._effective_ttl_days(rec()) == TTL_FLOOR_DAYS
    just_inside = rec(modified_on=NOW - timedelta(days=TTL_FLOOR_DAYS - 1))
    assert find_expired([just_inside], NOW) == []


# --- owner TTL resolution -------------------------------------------------


def test_resolve_ttls_standard(monkeypatch):
    cleanup.resolve_owner_ttls.cache_clear()
    monkeypatch.setattr(cleanup, "_owner_doc", lambda _o: {})
    assert cleanup.resolve_owner_ttls("u-std") == (180, 14)


def test_resolve_ttls_application(monkeypatch):
    cleanup.resolve_owner_ttls.cache_clear()
    monkeypatch.setattr(cleanup, "_owner_doc", lambda _o: {"tier": "application"})
    assert cleanup.resolve_owner_ttls("a-app") == (None, 14)


def test_resolve_ttls_honours_overrides(monkeypatch):
    cleanup.resolve_owner_ttls.cache_clear()
    monkeypatch.setattr(
        cleanup, "_owner_doc", lambda _o: {"quota_overrides": {"resource_ttl_days": 30}}
    )
    assert cleanup.resolve_owner_ttls("u-ovr") == (30, 14)


def test_ttl_defaults_match_api_contract():
    # Pinned to api/quota.py Quotas defaults; changing either requires updating
    # both (the two services can't import each other — see cleanup.py).
    assert cleanup.DEFAULT_RESOURCE_TTL_DAYS == 180
    assert cleanup.DEFAULT_FAILED_RESOURCE_TTL_DAYS == 14
    assert cleanup._TIER_TTL_OVERRIDES["application"]["resource_ttl_days"] is None


# --- orphaned blobs (with the pre-delete re-check) ------------------------


def test_orphan_blob_diff_and_recheck(monkeypatch):
    layout = RESOURCE_LAYOUTS[0]  # grids, PREFIX
    artifacts = {"a": "b/a", "b": "b/b", "live": "b/live"}

    class _Ref:
        def __init__(self, doc_id):
            self.id = doc_id

    class _Coll:
        def document(self, doc_id):
            return _Ref(doc_id)

    class _Snap:
        def __init__(self, doc_id, exists):
            self.id = doc_id
            self.exists = exists

    monkeypatch.setattr(cleanup.firestore_client, "collection", lambda _c: _Coll())
    # Batched re-check: "b" still has a live doc (missed in the stream) -> spared.
    monkeypatch.setattr(
        cleanup.firestore_client,
        "get_all",
        lambda refs: [_Snap(r.id, r.id == "b") for r in refs],
    )

    # "live" is filtered by the id set; "b" is spared by the re-check; only "a".
    assert find_orphan_blobs(layout, artifacts, {"live"}) == {"a": "b/a"}


def test_exports_layout_exempt_from_orphan_docs():
    exports = next(x for x in RESOURCE_LAYOUTS if x.name == "exports")
    assert exports.orphan_on_missing_domain is False
    others = [x for x in RESOURCE_LAYOUTS if x.name != "exports"]
    assert all(x.orphan_on_missing_domain for x in others)


def test_static_test_fixtures_protected_both_directions(monkeypatch):
    layout = RESOURCE_LAYOUTS[0]

    # A static-test GCS blob with no live doc must not be reaped. get_all returns
    # [] (no live docs), so if it weren't filtered it would come back as an
    # orphan — the empty result proves it was excluded before the re-check.
    monkeypatch.setattr(cleanup.firestore_client, "get_all", lambda refs: [])
    artifacts = {"static-test-blue-mtn": "b/static-test-blue-mtn"}
    assert find_orphan_blobs(layout, artifacts, set()) == {}

    # A static-test doc must not be reaped as an orphaned child (domain gone)...
    orphan = rec(
        doc_id="static-test-fixture",
        domain_id="gone",
        modified_on=NOW - timedelta(days=5),
    )
    assert find_orphan_docs([orphan], {"d1"}, NOW) == []

    # ...nor as TTL-expired, however old.
    monkeypatch.setattr(cleanup, "resolve_owner_ttls", lambda _o: (180, 14))
    ancient = rec(doc_id="static-test-fixture", modified_on=NOW - timedelta(days=999))
    assert find_expired([ancient], NOW) == []
