"""Unit tests for walle's cleanup logic (finders, TTL resolution, batched reaping).

The pure finders take in-memory records, so they run without GCP; the Firestore
and GCS touching helpers are exercised with monkeypatched clients. The full
``run()`` pass is deliberately not exercised here — it reconciles whole buckets
and would act on real project data — its behaviour is covered piece-wise.
"""

from datetime import UTC, datetime, timedelta

from walle import cleanup, layouts
from walle.cleanup import (
    Record,
    find_expired,
    find_orphan_blobs,
    find_orphan_docs,
)
from walle.config import TEST_TTL_DAYS, TTL_FLOOR_DAYS
from walle.layouts import RESOURCE_LAYOUTS

NOW = datetime(2026, 7, 6, tzinfo=UTC)
STD = {"o1": (180, 14)}  # standard TTLs for rec()'s default owner "o1"


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


def test_confirm_orphan_docs_drops_live_domains(monkeypatch):
    # Batched re-check: "d-live" still exists (missed in the stream) -> its child
    # is spared; "d-gone" is genuinely absent -> its child is confirmed orphan.
    monkeypatch.setattr(cleanup, "_existing_ids", lambda _c, ids: {"d-live"} & set(ids))
    kept = rec(doc_id="c1", domain_id="d-gone")
    dropped = rec(doc_id="c2", domain_id="d-live")
    assert cleanup.confirm_orphan_docs([kept, dropped]) == [kept]


# --- TTL expiry -----------------------------------------------------------


def test_expired_past_standard_ttl():
    old = rec(modified_on=NOW - timedelta(days=200))
    fresh = rec(modified_on=NOW - timedelta(days=10))
    assert find_expired([old, fresh], NOW, STD) == [old]


def test_failed_uses_failed_clock():
    expired = rec(status="failed", modified_on=NOW - timedelta(days=20))
    fresh = rec(status="failed", modified_on=NOW - timedelta(days=10))
    assert find_expired([expired, fresh], NOW, STD) == [expired]


def test_application_tier_never_expires():
    ancient = rec(modified_on=NOW - timedelta(days=9999))
    assert find_expired([ancient], NOW, {"o1": (None, 14)}) == []


def test_null_modified_on_never_expires():
    assert find_expired([rec(modified_on=None)], NOW, STD) == []


def test_unknown_owner_falls_back_to_defaults():
    # An owner absent from the resolved map uses the standard defaults.
    old = rec(owner_id="not-resolved", modified_on=NOW - timedelta(days=200))
    assert find_expired([old], NOW, {}) == [old]


def test_ttl_clamped_to_floor():
    # An override below the floor is clamped up, so it can't sweep too eagerly.
    ttls = {"o1": (3, 3)}
    assert cleanup._effective_ttl_days(rec(), ttls) == TTL_FLOOR_DAYS
    just_inside = rec(modified_on=NOW - timedelta(days=TTL_FLOOR_DAYS - 1))
    assert find_expired([just_inside], NOW, ttls) == []


# --- stale test resources -------------------------------------------------


def test_find_stale_test_matches_old_test_ids():
    old = rec(doc_id="test-abc", modified_on=NOW - timedelta(days=TEST_TTL_DAYS + 1))
    assert cleanup.find_stale_test([old], NOW) == [old]


def test_find_stale_test_age_guard_spares_recent():
    # A test in flight (created minutes/hours ago) must never be raced.
    recent = rec(doc_id="test-abc", modified_on=NOW - timedelta(days=1))
    assert cleanup.find_stale_test([recent], NOW) == []


def test_find_stale_test_ignores_real_and_static_ids():
    # Real ids are server uuid4 (no prefix); static-test- fixtures are protected.
    real = rec(doc_id="abc123def", modified_on=NOW - timedelta(days=999))
    static = rec(doc_id="static-test-blue-mtn", modified_on=NOW - timedelta(days=999))
    assert cleanup.find_stale_test([real, static], NOW) == []


def test_find_stale_test_null_modified_on_spared():
    r = rec(doc_id="test-abc", modified_on=None)
    assert cleanup.find_stale_test([r], NOW) == []


def test_non_datetime_modified_on_spared_everywhere():
    # Some legacy domain docs store modified_on as a string; age comparisons must
    # treat that as unknown age (never reaped) rather than crash.
    s = "2020-01-01T00:00:00Z"
    assert cleanup.find_stale_test([rec(doc_id="test-x", modified_on=s)], NOW) == []
    assert find_expired([rec(modified_on=s)], NOW, STD) == []


# --- owner TTL resolution -------------------------------------------------


def test_ttls_from_doc_standard():
    assert cleanup._ttls_from_doc({}) == (180, 14)


def test_ttls_from_doc_application():
    assert cleanup._ttls_from_doc({"tier": "application"}) == (None, 14)


def test_ttls_from_doc_honours_overrides():
    data = {"quota_overrides": {"resource_ttl_days": 30}}
    assert cleanup._ttls_from_doc(data) == (30, 14)


def test_resolve_owner_ttls_bulk(monkeypatch):
    def fake_get_all_docs(collection, ids):
        if collection == cleanup.APPLICATIONS_COLLECTION:
            return {"a1": {"tier": "application"}} if "a1" in ids else {}
        if collection == cleanup.USERS_COLLECTION:
            return (
                {"u1": {"quota_overrides": {"resource_ttl_days": 30}}}
                if "u1" in ids
                else {}
            )
        return {}

    monkeypatch.setattr(cleanup, "_get_all_docs", fake_get_all_docs)
    # None is dropped, "a1" deduped; "x1" is in neither collection -> defaults.
    result = cleanup.resolve_owner_ttls_bulk(["a1", "u1", "x1", None, "a1"])
    assert result == {"a1": (None, 14), "u1": (30, 14), "x1": (180, 14)}


def test_ttl_defaults_match_api_contract():
    # Pinned to api/quota.py Quotas defaults; changing either requires updating
    # both (the two services can't import each other — see cleanup.py).
    assert cleanup.DEFAULT_RESOURCE_TTL_DAYS == 180
    assert cleanup.DEFAULT_FAILED_RESOURCE_TTL_DAYS == 14
    assert cleanup._TIER_TTL_OVERRIDES["application"]["resource_ttl_days"] is None


# --- orphaned blobs (with the batched re-check) ---------------------------


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
    ancient = rec(doc_id="static-test-fixture", modified_on=NOW - timedelta(days=999))
    assert find_expired([ancient], NOW, STD) == []


# --- batched reaping (accumulate, then bulk-delete) -----------------------


def test_reap_blob_accumulates_only_in_enforce():
    layout = RESOURCE_LAYOUTS[0]
    summary = cleanup.Summary()
    gcs: list[str] = []

    cleanup._reap_blob(
        layout, "b1", "grids/b1", dry_run=False, summary=summary, gcs_deletes=gcs
    )
    assert gcs == ["grids/b1"]
    assert summary.deleted["orphan_blob"] == 1

    # Dry-run logs but never accumulates a delete.
    cleanup._reap_blob(
        layout, "b2", "grids/b2", dry_run=True, summary=summary, gcs_deletes=gcs
    )
    assert gcs == ["grids/b1"]
    assert summary.dry_run["orphan_blob"] == 1


def test_delete_artifacts_batches_and_tolerates_missing(monkeypatch):
    calls = []

    class _FS:
        def rm(self, paths, recursive):
            calls.append(paths)
            names = paths if isinstance(paths, list) else [paths]
            if "missing" in names:  # simulate an already-gone path
                raise FileNotFoundError

    monkeypatch.setattr(layouts, "get_gcsfs_client", lambda: _FS())
    monkeypatch.setattr(layouts, "_RM_CHUNK", 2)  # force chunking

    layouts.delete_artifacts(["a", "b", "missing", "c"])

    assert ["a", "b"] in calls  # first chunk deleted as one batch
    # second chunk raised on "missing" -> retried per path so "c" still deletes
    assert "missing" in calls and "c" in calls


def test_delete_artifacts_empty_is_noop(monkeypatch):
    monkeypatch.setattr(
        layouts, "get_gcsfs_client", lambda: (_ for _ in ()).throw(AssertionError)
    )
    layouts.delete_artifacts([])  # must not touch GCS
