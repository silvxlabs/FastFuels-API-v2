# /// script
# requires-python = ">=3.11"
# dependencies = ["google-cloud-firestore>=2.16", "python-dotenv>=1.0"]
# ///
"""Backfill ``attributions`` on completed OSM road and water feature documents.

Features completed before #770 have no ``attributions`` field. Every road and
water feature is built from OpenStreetMap, so each gets the one OSM entry, with
``accessed_on`` set to the document's ``created_on`` date (today if missing).
Documents that already have a non-empty ``attributions`` list are left alone,
so re-running changes nothing.

Dry run by default: prints counts and example IDs. Pass ``--apply`` to write.

Reads GCP_PROJECT and FEATURES_COLLECTION (default ``features-v2``) from the
environment or the repository ``.env``; credentials come from ADC or
GOOGLE_APPLICATION_CREDENTIALS.

    uv run scripts/backfill_feature_attributions.py
    uv run scripts/backfill_feature_attributions.py --apply
"""

import argparse
import os
import sys
from datetime import date, datetime

from dotenv import load_dotenv
from google.cloud import firestore

FEATURE_TYPES = ("road", "water")
BATCH_SIZE = 400
EXAMPLES = 5

# Mirrors lib.attributions.DATASET_ATTRIBUTIONS["OpenStreetMap"]
# (services/lib/lib/attributions.py). Keep the two in sync.
OPENSTREETMAP_ATTRIBUTION = {
    "dataset": "OpenStreetMap",
    "license_name": "ODbL-1.0",
    "license_url": "https://opendatacommons.org/licenses/odbl/1-0/",
    "citation": "© OpenStreetMap contributors",
    "access_url": "https://www.openstreetmap.org/copyright",
}


def accessed_on(doc: dict) -> str:
    """ISO date of the document's ``created_on``, or today if it has none."""
    created_on = doc.get("created_on")
    if isinstance(created_on, datetime):
        return created_on.date().isoformat()
    if isinstance(created_on, str) and created_on:
        return datetime.fromisoformat(created_on).date().isoformat()
    return date.today().isoformat()


def find_targets(client: firestore.Client, collection: str):
    """Return (completed, targets): completed road/water docs and those to update."""
    completed = 0
    targets = []
    for feature_type in FEATURE_TYPES:
        query = (
            client.collection(collection)
            .where(filter=firestore.FieldFilter("type", "==", feature_type))
            .where(filter=firestore.FieldFilter("status", "==", "completed"))
        )
        for snapshot in query.stream():
            completed += 1
            doc = snapshot.to_dict()
            if not doc.get("attributions"):
                attributions = [
                    {**OPENSTREETMAP_ATTRIBUTION, "accessed_on": accessed_on(doc)}
                ]
                targets.append((snapshot.reference, attributions))
    return completed, targets


def apply_updates(client: firestore.Client, targets: list) -> int:
    """Write ``attributions`` to each target in batches; return the count written."""
    written = 0
    for start in range(0, len(targets), BATCH_SIZE):
        batch = client.batch()
        for ref, attributions in targets[start : start + BATCH_SIZE]:
            batch.update(ref, {"attributions": attributions})
        batch.commit()
        written += len(targets[start : start + BATCH_SIZE])
        print(f"  committed {written}/{len(targets)}")
    return written


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--project",
        default=os.getenv("GCP_PROJECT"),
        help="GCP project (default: $GCP_PROJECT)",
    )
    parser.add_argument(
        "--collection",
        default=os.getenv("FEATURES_COLLECTION", "features-v2"),
        help="Features collection (default: $FEATURES_COLLECTION or features-v2)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write the updates. Without it, only report what would change.",
    )
    args = parser.parse_args()
    if not args.project:
        parser.error("set --project or GCP_PROJECT")

    client = firestore.Client(project=args.project)
    completed, targets = find_targets(client, args.collection)

    mode = "APPLY" if args.apply else "DRY RUN"
    print(f"[{mode}] project={args.project} collection={args.collection}")
    print(f"  matched (completed {'/'.join(FEATURE_TYPES)}): {completed}")
    print(f"  would update (no attributions): {len(targets)}")
    print(f"  skipped (already attributed): {completed - len(targets)}")
    for ref, attributions in targets[:EXAMPLES]:
        print(f"    {ref.id} accessed_on={attributions[0]['accessed_on']}")

    if args.apply and targets:
        written = apply_updates(client, targets)
        print(f"  updated: {written}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
