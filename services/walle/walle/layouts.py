"""How each resource type stores its GCS artifact, and the primitives to list
and delete those artifacts.

Every job-producing resource keeps its artifact under a bucket keyed by the
resource id — *except features*, which store a single ``.parquet`` under the
domain prefix. That one exception is why walle needs a per-resource layout
descriptor rather than a blanket ``<bucket>/<id>/`` rule.
"""

import logging
from dataclasses import dataclass
from enum import Enum

from lib.config import (
    EXPORTS_BUCKET,
    EXPORTS_COLLECTION,
    FEATURES_BUCKET,
    FEATURES_COLLECTION,
    GRIDS_BUCKET,
    GRIDS_COLLECTION,
    INVENTORIES_BUCKET,
    INVENTORIES_COLLECTION,
    POINT_CLOUDS_BUCKET,
    POINT_CLOUDS_COLLECTION,
)
from lib.gcs.blobs import get_gcsfs_client

logger = logging.getLogger(__name__)

_PARQUET = ".parquet"


class ArtifactKind(Enum):
    # Directory prefix keyed by resource id: <bucket>/<id>/...
    # (zarr grid, partitioned-parquet inventory, export bundle, .laz point cloud).
    PREFIX = "prefix"
    # Single file under the domain prefix: <bucket>/<domain_id>/<id>.parquet.
    FEATURE_FILE = "feature_file"


@dataclass(frozen=True)
class ResourceLayout:
    """A resource type's Firestore collection and GCS artifact layout.

    ``foreign_key`` is the containment edge used for orphaned-doc detection — a
    doc whose parent (``domain_id``) no longer exists is orphaned. Derivation
    edges (``source_grid_id`` etc.) are deliberately NOT used: dangling
    derivation references are tolerated by design.

    ``orphan_on_missing_domain`` is False for resources that intentionally
    outlive their domain (exports are standalone provenance artifacts — the API
    excludes them from domain cascade, so walle must not reap them by a missing
    domain either). Such resources are still subject to orphan-blob and TTL
    reaping.
    """

    name: str
    collection: str
    bucket: str
    kind: ArtifactKind
    foreign_key: str = "domain_id"
    orphan_on_missing_domain: bool = True


# Exports are grandchildren (under grids) but denormalize ``domain_id``, so the
# same containment edge works for every type.
RESOURCE_LAYOUTS: list[ResourceLayout] = [
    ResourceLayout("grids", GRIDS_COLLECTION, GRIDS_BUCKET, ArtifactKind.PREFIX),
    ResourceLayout(
        "exports",
        EXPORTS_COLLECTION,
        EXPORTS_BUCKET,
        ArtifactKind.PREFIX,
        orphan_on_missing_domain=False,
    ),
    ResourceLayout(
        "inventories",
        INVENTORIES_COLLECTION,
        INVENTORIES_BUCKET,
        ArtifactKind.PREFIX,
    ),
    ResourceLayout(
        "pointclouds",
        POINT_CLOUDS_COLLECTION,
        POINT_CLOUDS_BUCKET,
        ArtifactKind.PREFIX,
    ),
    ResourceLayout(
        "features",
        FEATURES_COLLECTION,
        FEATURES_BUCKET,
        ArtifactKind.FEATURE_FILE,
    ),
]


def artifact_path(layout: ResourceLayout, doc_id: str, domain_id: str | None) -> str:
    """The GCS path of a resource's artifact (no ``gs://`` prefix)."""
    if layout.kind is ArtifactKind.FEATURE_FILE:
        return f"{layout.bucket}/{domain_id}/{doc_id}{_PARQUET}"
    return f"{layout.bucket}/{doc_id}"


def list_artifact_ids(layout: ResourceLayout) -> dict[str, str]:
    """Map resource-id -> artifact GCS path for every artifact in the bucket.

    PREFIX: each top-level "directory" under the bucket is one resource id.
    FEATURE_FILE: each ``<domain_id>/<id>.parquet`` object is one resource id.
    An empty/absent bucket yields an empty map.
    """
    fs = get_gcsfs_client()
    fs.invalidate_cache(layout.bucket)
    result: dict[str, str] = {}

    if layout.kind is ArtifactKind.PREFIX:
        try:
            entries = fs.ls(layout.bucket, detail=False)
        except FileNotFoundError:
            return result
        for path in entries:
            doc_id = path.rstrip("/").rsplit("/", 1)[-1]
            if doc_id:
                result[doc_id] = path
        return result

    # FEATURE_FILE: flat-list every object and key by the .parquet basename.
    try:
        paths = fs.find(layout.bucket)
    except FileNotFoundError:
        return result
    for path in paths:
        base = path.rsplit("/", 1)[-1]
        if base.endswith(_PARQUET):
            result[base[: -len(_PARQUET)]] = path
    return result


def delete_artifact(path: str) -> None:
    """Delete a GCS artifact — prefix or single object. Idempotent."""
    fs = get_gcsfs_client()
    try:
        # recursive=True deletes a whole prefix and is a harmless no-op flag on a
        # single object.
        fs.rm(path, recursive=True)
    except FileNotFoundError:
        pass
