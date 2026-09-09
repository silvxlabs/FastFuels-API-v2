"""
api/access.py

Read access for the shared, prebuilt example (#582).

Every other resource is owner-scoped: a read passes ``owner_id ==
request.state.id`` to :func:`get_document_async`, which 404s on any mismatch.
The example is the single deliberate exception — one canonical Blue Mountain
domain and its NAIP canopy grid, owned by :data:`EXAMPLE_OWNER_ID`, that any
authenticated caller (guest included) may READ without owning it.

The relaxation is intentionally narrow. A document is readable by a non-owner
ONLY when BOTH hold:

1. it is owned by :data:`EXAMPLE_OWNER_ID`, and
2. it carries ``is_example is True``.

Both conditions are required so that opening read access can never leak an
arbitrary owner's resource: a normal owner's doc fails (1), and any doc the
example owner has not explicitly flagged fails (2). Writes are unaffected —
they keep using the owner-strict path, so the example stays read-only for
everyone but its seeding owner.
"""

from fastapi import HTTPException, status

from api.db.documents import get_document_async
from lib.config import EXAMPLE_OWNER_ID


def is_example_resource(data: dict) -> bool:
    """Whether ``data`` is a public example resource (owner + flag both required)."""
    return data.get("owner_id") == EXAMPLE_OWNER_ID and data.get("is_example") is True


async def get_readable_document_async(
    collection: str,
    document_id: str,
    viewer_id: str,
    *,
    domain_id: str | None = None,
    document_status: str | None = None,
):
    """Fetch a document the ``viewer_id`` may READ: their own, or the example.

    Mirrors :func:`get_document_async` but authorizes reads against
    ``owner_id == viewer_id OR is_example_resource(doc)`` instead of ownership
    alone. Existence and ``domain_id`` scoping are delegated to
    :func:`get_document_async` (both 404 without revealing ownership). The
    authorization check runs BEFORE the optional ``document_status`` check so a
    non-example resource owned by someone else always 404s — it can never leak
    its existence through a 422 status mismatch.

    Raises:
        HTTPException(404): document missing, wrong domain, or not readable by
            this viewer (owner mismatch and not a flagged example).
        HTTPException(422): document exists and is readable but its ``status``
            does not match ``document_status``.
    """
    # No owner filter here: authorize below so an example is reachable while a
    # non-owned, non-example doc still 404s. domain_id scoping still applies.
    ref, snapshot = await get_document_async(
        collection, document_id, domain_id=domain_id
    )
    data = snapshot.to_dict()

    collection_name = collection.split("-")[0]
    if not (data.get("owner_id") == viewer_id or is_example_resource(data)):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document not found: {collection_name}/{document_id}",
        )

    if document_status is not None:
        actual_status = data.get("status")
        if actual_status != document_status:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=(
                    f"{collection_name}/{document_id} status is "
                    f"'{actual_status}', expected '{document_status}'."
                ),
            )

    return ref, snapshot
