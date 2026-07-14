"""Shared test data paths and fixture loading used across all services."""

import json
from datetime import datetime
from pathlib import Path

SHARED_TEST_DATA_DIR = Path(__file__).resolve().parent.parent / "tests" / "shared_data"
SHARED_TEST_DOMAINS_DIR = SHARED_TEST_DATA_DIR / "domains"
SHARED_TEST_GRIDS_DIR = SHARED_TEST_DATA_DIR / "grids"
SHARED_TEST_INVENTORIES_DIR = SHARED_TEST_DATA_DIR / "inventories"
SHARED_TEST_EXPORTS_DIR = SHARED_TEST_DATA_DIR / "exports"
SHARED_TEST_FEATURES_DIR = SHARED_TEST_DATA_DIR / "features"

# Lifecycle fields the API stamps on every document it writes.
_TIMESTAMP_FIELDS = ("created_on", "modified_on")


def load_json(path: Path) -> dict:
    """Load a Firestore document fixture from the shared test data directory.

    JSON has no datetime type, so a fixture can only carry ``created_on`` /
    ``modified_on`` as ISO-8601 strings. Seeding those verbatim stores a string
    where every API-written doc holds a timestamp, and walle's stale-test purge
    skips a non-datetime ``modified_on`` as "age unknown" — so such docs are
    never reaped. Both fields are stamped fresh here (any value on disk is
    ignored), which keeps a seeded doc's age honest: recent enough that the
    purge can't race an in-flight test, old enough to reap once it finishes.
    """
    with open(path) as f:
        doc = json.load(f)
    now = datetime.now()
    for field in _TIMESTAMP_FIELDS:
        doc[field] = now
    return doc
