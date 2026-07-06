"""walle tunables and per-category dry-run switches.

Infrastructure values (buckets, collections) come from ``lib.config``. These are
walle's own operational knobs, kept local per the per-service-tuning convention.
"""

import os


def _flag(name: str) -> bool:
    return os.getenv(name, "false").strip().lower() in ("1", "true", "yes", "on")


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


# A resolved TTL is clamped to at least this many days, so no owner override can
# drop a resource's lifetime below the sweep's safety floor (design §6). Chosen
# below the smallest legitimate clock (14-day failed) so it never shortens a
# real TTL — it only caps how aggressive an override may be.
TTL_FLOOR_DAYS = _int("WALLE_TTL_FLOOR_DAYS", 7)

# Orphaned child docs modified more recently than this are left alone, so a
# resource mid-creation is never mistaken for garbage. Orphaned blobs use a
# per-candidate doc re-check instead (see cleanup.py).
ORPHAN_MIN_AGE_HOURS = _int("WALLE_ORPHAN_MIN_AGE_HOURS", 24)

# Ephemeral integration-test resources (id prefix "test-", but NOT the persistent
# "static-test-" fixtures) get a short retention. Real ids are server-generated
# uuid4 hex (never "test-"), so this only ever reaps test artifacts; the window
# is far longer than any test run, so an in-flight test is never raced.
TEST_TTL_DAYS = _int("WALLE_TEST_TTL_DAYS", 7)

# Per-category dry-run switches. Default enforce (delete); set true to log
# candidates without deleting — used to validate a category locally before
# shipping (deployed walle runs enforce).
ORPHAN_BLOBS_DRY_RUN = _flag("WALLE_ORPHAN_BLOBS_DRY_RUN")
ORPHAN_DOCS_DRY_RUN = _flag("WALLE_ORPHAN_DOCS_DRY_RUN")
TTL_DRY_RUN = _flag("WALLE_TTL_DRY_RUN")
TEST_PURGE_DRY_RUN = _flag("WALLE_TEST_PURGE_DRY_RUN")
