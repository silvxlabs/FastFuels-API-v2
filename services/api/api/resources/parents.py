"""
api/resources/parents.py

Load the parent resources a create request references, in one Firestore read.

Every resource a request references by ID is a parent. A create path loads all
of its parents once with ``load_parents``, validates the loaded documents, and
passes the same documents to ``inherit_attributions``.
"""

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field

from api.db.documents import firestore_client, validate_document
from lib.config import (
    FEATURES_COLLECTION,
    GRIDS_COLLECTION,
    INVENTORIES_COLLECTION,
    POINT_CLOUDS_COLLECTION,
)


@dataclass(frozen=True)
class Parents:
    """A request's parent documents by collection, keyed by document ID.

    Every requested ID has an entry; a document that does not exist is
    ``None``. Iterating yields every document, and raises if one is ``None``:
    a missing parent must be rejected by ``validate_parent`` before the
    documents are used, so reaching one here means validation was skipped.
    """

    grids: dict[str, dict | None] = field(default_factory=dict)
    features: dict[str, dict | None] = field(default_factory=dict)
    inventories: dict[str, dict | None] = field(default_factory=dict)
    point_clouds: dict[str, dict | None] = field(default_factory=dict)

    def documents(self, collection: str) -> dict[str, dict | None]:
        """The loaded documents of one collection."""
        return {
            GRIDS_COLLECTION: self.grids,
            FEATURES_COLLECTION: self.features,
            INVENTORIES_COLLECTION: self.inventories,
            POINT_CLOUDS_COLLECTION: self.point_clouds,
        }[collection]

    def __iter__(self) -> Iterator[dict]:
        for collection in (
            GRIDS_COLLECTION,
            FEATURES_COLLECTION,
            INVENTORIES_COLLECTION,
            POINT_CLOUDS_COLLECTION,
        ):
            for document_id, doc in self.documents(collection).items():
                if doc is None:
                    raise RuntimeError(
                        f"{collection}/{document_id} does not exist but was not "
                        "rejected by validate_parent"
                    )
                yield doc


async def load_parents(
    *,
    grids: Iterable[str | None] = (),
    features: Iterable[str | None] = (),
    inventories: Iterable[str | None] = (),
    point_clouds: Iterable[str | None] = (),
) -> Parents:
    """Read every referenced parent document in one batched Firestore read.

    Returns one entry per distinct requested ID, ``None`` when the document
    does not exist. ``None`` IDs (an optional reference that was not given)
    are skipped, and an ID repeated within a collection is read once.
    """
    parents = Parents(
        grids=dict.fromkeys(i for i in grids if i is not None),
        features=dict.fromkeys(i for i in features if i is not None),
        inventories=dict.fromkeys(i for i in inventories if i is not None),
        point_clouds=dict.fromkeys(i for i in point_clouds if i is not None),
    )
    refs = [
        firestore_client.collection(collection).document(document_id)
        for collection in (
            GRIDS_COLLECTION,
            FEATURES_COLLECTION,
            INVENTORIES_COLLECTION,
            POINT_CLOUDS_COLLECTION,
        )
        for document_id in parents.documents(collection)
    ]
    if not refs:
        return parents
    async for snapshot in firestore_client.get_all(refs):
        if snapshot.exists:
            collection = snapshot.reference.parent.id
            parents.documents(collection)[snapshot.id] = snapshot.to_dict() or {}
    return parents


def validate_parent(
    parents: Parents,
    collection: str,
    document_id: str,
    owner_id: str,
    domain_id: str,
) -> None:
    """Validate that a loaded parent exists, is owned by ``owner_id``, lives in
    ``domain_id``, and is completed.

    Raises the same errors as ``get_document_async``: 404 when missing, owned
    by another user, or in another domain; 422 when not completed.
    """
    validate_document(
        parents.documents(collection).get(document_id),
        collection,
        document_id,
        owner_id=owner_id,
        domain_id=domain_id,
        document_status="completed",
    )
