"""
Scrub dask's __null_dask_index__ column from existing inventory Parquet.

Inventories written before standgen and the uploader passed
``write_index=False`` (#335) carry a synthetic ``__null_dask_index__``
column in their Parquet file schema, which surfaced in the API's
data/metadata ``columns``. New writes are clean; this one-time sweep
rewrites the existing datasets.

For each inventory prefix in INVENTORIES_BUCKET whose ``_metadata`` footer
schema contains the artifact, the data is read back with dask and rewritten
in place via standgen's staged-swap path (``save_parquet_replace``, which
now writes clean). The inventory's Firestore ``checksum`` is then
reassigned so the API's per-content metadata cache drops the stale entry.

Usage (run from services/standgen, which has dask + standgen.storage;
PYTHONPATH=. because standgen is a virtual uv project, not an installed
package):
    cd services/standgen
    PYTHONPATH=. uv run --env-file ../../.env \
        python ../../scripts/scrub_inventory_dask_index.py --dry-run
    PYTHONPATH=. uv run --env-file ../../.env \
        python ../../scripts/scrub_inventory_dask_index.py --all
    PYTHONPATH=. uv run --env-file ../../.env \
        python ../../scripts/scrub_inventory_dask_index.py <inventory_id> [...]
"""

import argparse
import sys
from uuid import uuid4

import dask.dataframe as dd
import pyarrow.parquet as pq
from standgen.storage import save_parquet_replace

from lib.config import INVENTORIES_BUCKET, INVENTORIES_COLLECTION
from lib.firestore.documents import (
    DocumentNotFoundError,
    get_document,
    update_document,
)
from lib.gcs import get_gcsfs_client

ARTIFACT = "__null_dask_index__"


def has_artifact(inventory_id: str) -> bool | None:
    """Whether the inventory's ``_metadata`` footer schema carries the artifact.

    Returns None when the inventory has no ``_metadata`` file (nothing for the
    data/metadata endpoint to surface, so nothing to scrub).
    """
    fs = get_gcsfs_client()
    path = f"{INVENTORIES_BUCKET}/{inventory_id}/_metadata"
    if not fs.exists(path):
        return None
    with fs.open(path, "rb") as f:
        names = pq.read_metadata(f).schema.to_arrow_schema().names
    return ARTIFACT in names


def find_affected() -> list[str]:
    """Inventory IDs in INVENTORIES_BUCKET whose file schema has the artifact."""
    fs = get_gcsfs_client()
    fs.invalidate_cache()
    affected = []
    for prefix in sorted(fs.ls(INVENTORIES_BUCKET)):
        inventory_id = prefix.rstrip("/").rsplit("/", 1)[-1]
        # Staging leftovers from save_parquet_replace are not live inventories.
        if not inventory_id or inventory_id.endswith("__rev"):
            continue
        if has_artifact(inventory_id):
            affected.append(inventory_id)
    return affected


def scrub(inventory_id: str) -> None:
    """Rewrite one inventory in place and reassign its Firestore checksum."""
    ddf = dd.read_parquet(f"gs://{INVENTORIES_BUCKET}/{inventory_id}")
    save_parquet_replace(inventory_id, ddf)

    remaining = has_artifact(inventory_id)
    if remaining or remaining is None:
        raise RuntimeError(
            f"Rewrite of {inventory_id} did not produce a clean _metadata footer"
        )

    try:
        get_document(INVENTORIES_COLLECTION, inventory_id)
    except DocumentNotFoundError:
        print(f"  {inventory_id}: rewritten (no Firestore doc; checksum skipped)")
        return
    update_document(INVENTORIES_COLLECTION, inventory_id, {"checksum": uuid4().hex})
    print(f"  {inventory_id}: rewritten, checksum reassigned")


def main():
    parser = argparse.ArgumentParser(
        description="Rewrite inventory Parquet carrying __null_dask_index__ (#335)."
    )
    parser.add_argument(
        "inventory_ids", nargs="*", help="Specific inventory IDs to scrub"
    )
    parser.add_argument(
        "--all", action="store_true", help="Sweep every inventory in the bucket"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only list affected inventories; rewrite nothing",
    )
    args = parser.parse_args()

    if args.inventory_ids:
        targets = []
        for inventory_id in args.inventory_ids:
            artifact = has_artifact(inventory_id)
            if artifact is None:
                print(f"  {inventory_id}: no _metadata file; skipping")
            elif not artifact:
                print(f"  {inventory_id}: already clean; skipping")
            else:
                targets.append(inventory_id)
    elif args.all or args.dry_run:
        print(f"Scanning gs://{INVENTORIES_BUCKET} ...")
        targets = find_affected()
    else:
        parser.print_help()
        sys.exit(1)

    print(f"{len(targets)} inventories carry {ARTIFACT}")
    if args.dry_run:
        for inventory_id in targets:
            print(f"  {inventory_id}")
        return

    for inventory_id in targets:
        scrub(inventory_id)


if __name__ == "__main__":
    main()
