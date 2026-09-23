"""
Add a `tree_id` column to tree inventories created before tree IDs existed (#611).

For each completed tree inventory whose document has no `tree_id` column:

  1. Number the trees 0 … N-1 in stored row order (`generate_tree_ids`).
  2. Rewrite the Parquet in place through the staged swap used by in-place
     modifications (`save_parquet_replace_with_summary`).
  3. Prepend the `tree_id` column, with its summary, to the document's
     `columns`, and refresh `size_bytes`.

`checksum` and `modified_on` are left unchanged: no existing value changes, so
derived resources are not made stale.

Dry run by default; pass --apply to write. Safe to re-run: inventories that
already have a `tree_id` column are skipped. Inventories that are not
completed, or have pending in-place work, are skipped and listed.

Usage:
    cd services/standgen
    PYTHONPATH=. uv run --env-file ../../.env python ../../scripts/migrate_inventory_tree_ids.py
    PYTHONPATH=. uv run --env-file ../../.env python ../../scripts/migrate_inventory_tree_ids.py --apply
    PYTHONPATH=. uv run --env-file ../../.env python ../../scripts/migrate_inventory_tree_ids.py --apply --ids ID1 ID2
"""

import argparse
import logging
import sys

import dask.dataframe as dd
from google.cloud.firestore import FieldFilter
from standgen.columns import TREE_ID_COLUMN, generate_tree_ids
from standgen.storage import inventory_size, save_parquet_replace_with_summary

from lib.config import INVENTORIES_BUCKET, INVENTORIES_COLLECTION
from lib.firestore.documents import firestore_client, update_document

TREE_ID_COLUMN_META = {"key": TREE_ID_COLUMN, "type": "categorical", "unit": None}

logger = logging.getLogger("migrate_inventory_tree_ids")


def _has_tree_id(doc: dict) -> bool:
    return any(c.get("key") == TREE_ID_COLUMN for c in doc.get("columns") or [])


def _skip_reason(doc: dict) -> str | None:
    if doc.get("status") != "completed":
        return f"status {doc.get('status')}"
    if doc.get("pending_modifications") or doc.get("pending_treatments"):
        return "pending in-place work"
    return None


def find_candidates(ids: list[str] | None) -> tuple[list, list, int]:
    """Return (to migrate, skipped with reason, already migrated count)."""
    collection = firestore_client.collection(INVENTORIES_COLLECTION)
    if ids:
        snapshots = [collection.document(i).get() for i in ids]
        missing = [s.id for s in snapshots if not s.exists]
        if missing:
            sys.exit(f"Inventories not found: {missing}")
    else:
        snapshots = collection.where(filter=FieldFilter("type", "==", "tree")).stream()

    todo, skipped, done = [], [], 0
    for snap in snapshots:
        doc = snap.to_dict()
        if _has_tree_id(doc):
            done += 1
        elif reason := _skip_reason(doc):
            skipped.append((snap.id, reason))
        else:
            todo.append(snap)
    return todo, skipped, done


def migrate(snap) -> int:
    """Add tree_id to one inventory; return its tree count."""
    inventory_id = snap.id
    # Re-read so a job started since the scan is not overwritten.
    doc = snap.reference.get().to_dict()
    if _has_tree_id(doc) or _skip_reason(doc):
        raise RuntimeError("changed since the scan; re-run to retry")

    ddf = dd.read_parquet(f"gs://{INVENTORIES_BUCKET}/{inventory_id}")
    if TREE_ID_COLUMN in ddf.columns:
        raise RuntimeError("data already has tree_id but the document does not")

    _, stats, _ = save_parquet_replace_with_summary(
        inventory_id, generate_tree_ids(ddf), [TREE_ID_COLUMN_META]
    )
    summary = stats[TREE_ID_COLUMN]
    update_document(
        INVENTORIES_COLLECTION,
        inventory_id,
        {
            "columns": [{**TREE_ID_COLUMN_META, "summary": summary}, *doc["columns"]],
            "size_bytes": inventory_size(inventory_id),
        },
    )
    return summary["count"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--apply", action="store_true", help="write changes")
    parser.add_argument("--ids", nargs="+", help="only these inventory IDs")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    logger.info(f"Collection {INVENTORIES_COLLECTION}, bucket {INVENTORIES_BUCKET}")
    todo, skipped, done = find_candidates(args.ids)
    logger.info(
        f"{len(todo)} to migrate, {done} already have tree_id, {len(skipped)} skipped"
    )
    for inventory_id, reason in skipped:
        logger.info(f"  skip {inventory_id}: {reason}")
    if not args.apply:
        for snap in todo:
            logger.info(f"  would migrate {snap.id}")
        logger.info("Dry run; pass --apply to write.")
        return

    failed = []
    for n, snap in enumerate(todo, 1):
        try:
            trees = migrate(snap)
            logger.info(f"[{n}/{len(todo)}] {snap.id}: {trees} trees")
        except Exception as e:
            failed.append(snap.id)
            logger.error(f"[{n}/{len(todo)}] {snap.id}: FAILED: {e}")
    logger.info(f"Done: {len(todo) - len(failed)} migrated, {len(failed)} failed")
    if failed:
        logger.info(f"Failed: {failed}")
        sys.exit(1)


if __name__ == "__main__":
    main()
