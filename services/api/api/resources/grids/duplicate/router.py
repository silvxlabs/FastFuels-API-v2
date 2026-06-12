"""
api/v2/resources/grids/duplicate/router.py

Router for duplicating a grid: POST /domains/{domain_id}/grids/{grid_id}/duplicate.

Creates an independent clone of a completed grid under a new ID. The zarr
artifact is server-side copied in GCS by a background task; griddle is never
involved.
"""

import logging
import traceback
import uuid
from datetime import datetime

from fastapi import APIRouter, BackgroundTasks, Request, status
from google.api_core.exceptions import NotFound

from api.db.blobs import copy_directory_verified
from api.db.documents import (
    get_document_async,
    set_document_async,
    update_document_async,
)
from api.dependencies import VerifiedDomain
from api.resources.grids.duplicate.schema import DuplicateGridRequest
from api.resources.grids.schema import Grid
from api.schema import JobStatus
from lib.config import GRIDS_BUCKET, GRIDS_COLLECTION

logger = logging.getLogger(__name__)

router = APIRouter()

COLLECTION = GRIDS_COLLECTION


async def _copy_grid_data(source_id: str, new_id: str) -> None:
    """Background task: server-side copy the zarr artifact from the source
    grid to the new one, then flip the new grid to ``completed``.

    The copy is verified against a pre-copy snapshot of the source listing,
    so a source deleted or rewritten mid-copy fails the duplicate instead of
    completing with a silently incomplete clone. On any failure the new grid
    is marked ``failed`` with a structured error so the dangling ``pending``
    document never lingers.
    """
    try:
        await copy_directory_verified(GRIDS_BUCKET, source_id, new_id)
        await update_document_async(
            COLLECTION,
            new_id,
            {"status": JobStatus.completed.value, "modified_on": datetime.now()},
        )
    except Exception:
        logger.exception("Failed to copy grid data %s -> %s", source_id, new_id)
        try:
            await update_document_async(
                COLLECTION,
                new_id,
                {
                    "status": JobStatus.failed.value,
                    "modified_on": datetime.now(),
                    "error": {
                        "code": "GRID_DUPLICATE_COPY_FAILED",
                        "message": "Failed to copy grid data during duplication.",
                        "suggestion": "Retry the duplicate request.",
                        "traceback": traceback.format_exc(),
                    },
                },
            )
        except NotFound:
            # The new grid was deleted (e.g. cancelled) before the copy
            # finished — there is no document left to mark failed.
            logger.info(
                "Duplicate target %s no longer exists; skipping failure update",
                new_id,
            )


@router.post(
    "",
    response_model=Grid,
    status_code=status.HTTP_201_CREATED,
    summary="Duplicate a grid",
)
async def duplicate_grid(
    request: Request,
    domain: VerifiedDomain,
    grid_id: str,
    background_tasks: BackgroundTasks,
    body: DuplicateGridRequest | None = None,
):
    """
    # Duplicate a Grid

    Creates an independent **copy** of a completed grid under a new ID. Use
    this to branch a scenario: duplicate, then edit the copy while the
    original stays untouched.

    This is a true clone, not a re-derivation. The finished data is
    byte-copied; no regeneration is performed and the upstream source is never
    re-fetched, so the copy is exact even if the upstream product has been
    updated since the original was built. The copy carries over the source's
    `source`, `modifications`, `bands`, `georeference`, `chunks`, and
    `checksum` verbatim — only its `id` and timestamps differ.

    ## Request Body (optional)

    All fields are optional. Any field omitted is carried over from the source.

    - **name**: Name for the copy.
    - **description**: Description for the copy.
    - **tags**: Tags for the copy.

    Send no body at all to copy the metadata unchanged.

    ## Response

    Returns the new Grid with status `"pending"`. The data is copied in the
    background; the status transitions to `"completed"` once the copy finishes
    (or `"failed"` if it does not). Data endpoints (`/chunks`, `/data`) become
    available only after the copy completes. The source grid is unchanged.

    ## Error Responses

    - **404 Not Found**: The source grid does not exist, is not owned by the
      caller, or is not in this domain.
    - **422 Unprocessable Content**: The source grid exists but is not yet
      `completed`, so there is no finished artifact to copy.
    """
    owner_id = request.state.id
    domain_id = domain["id"]

    # Source must exist, be owned, in this domain, and completed.
    _, source_snapshot = await get_document_async(
        COLLECTION,
        grid_id,
        owner_id=owner_id,
        domain_id=domain_id,
        document_status="completed",
    )
    source_data = source_snapshot.to_dict()

    overrides = body or DuplicateGridRequest()
    new_grid_id = uuid.uuid4().hex
    request_time = datetime.now()

    grid_data = {
        # Carry over source, checksum, modifications, bands, georeference,
        # and chunks verbatim; override identity, timestamps, transient
        # status fields, and any supplied metadata.
        **source_data,
        "id": new_grid_id,
        "owner_id": owner_id,
        "created_on": request_time,
        "modified_on": request_time,
        "status": JobStatus.pending.value,
        "progress": None,
        "error": None,
        "name": (
            overrides.name
            if overrides.name is not None
            else source_data.get("name", "")
        ),
        "description": (
            overrides.description
            if overrides.description is not None
            else source_data.get("description", "")
        ),
        "tags": (
            overrides.tags
            if overrides.tags is not None
            else source_data.get("tags", [])
        ),
    }

    # Write the document before constructing the response model: the Grid
    # before-validator decodes stringified modification coordinates in place,
    # so building it first would corrupt the values written to Firestore.
    await set_document_async(COLLECTION, new_grid_id, grid_data)
    background_tasks.add_task(_copy_grid_data, grid_id, new_grid_id)

    return Grid(**grid_data)
