"""Firestore read/write helpers for treevox.

All Firestore-facing I/O lives here so the HTTP entry (main.py) and the
voxelization handler (handlers/voxelize.py) stay independent of persistence
details.

`update_progress` and `update_status` treat `DocumentNotFoundError` as user
cancellation — the document was deleted mid-run, so we surface
`CancelledException` and let callers clean up.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import UTC, datetime

from lib.config import DOMAINS_COLLECTION, GRIDS_COLLECTION
from lib.domain_utils import EmptyDomainError, InvalidGeometryError, parse_domain_gdf
from lib.firestore import DocumentNotFoundError, get_document, update_document
from treevox.errors import CancelledException, ProcessingError

logger = logging.getLogger(__name__)


def load_grid(grid_id: str) -> dict:
    """Load a grid document from Firestore."""
    _, snapshot = get_document(GRIDS_COLLECTION, grid_id)
    return snapshot.to_dict()


def update_progress(grid_id: str, message: str, percent: int | None = None) -> None:
    progress: dict = {"message": message}
    if percent is not None:
        progress["percent"] = percent
    try:
        update_document(
            GRIDS_COLLECTION,
            grid_id,
            {"progress": progress, "modified_on": datetime.now(UTC)},
        )
    except DocumentNotFoundError:
        raise CancelledException(f"Grid {grid_id} was cancelled")


def update_status(
    grid_id: str,
    status: str,
    georeference: dict | None = None,
    error: dict | None = None,
) -> None:
    data: dict = {"status": status, "modified_on": datetime.now(UTC)}
    if status == "completed":
        data["progress"] = {"message": "Complete", "percent": 100}
    elif status == "failed":
        data["progress"] = {"message": "Failed", "percent": 100}
    if georeference is not None:
        data["georeference"] = georeference
    if error is not None:
        data["error"] = error
    try:
        update_document(GRIDS_COLLECTION, grid_id, data)
    except DocumentNotFoundError:
        raise CancelledException(f"Grid {grid_id} was cancelled")


def make_progress_callback(grid_id: str) -> Callable[[str, int | None], None]:
    def callback(message: str, percent: int | None = None):
        update_progress(grid_id, message, percent)

    return callback


def load_domain(domain_id: str):
    """Load a domain's GeoDataFrame, mapping Firestore errors to ProcessingError."""
    try:
        _, snapshot = get_document(DOMAINS_COLLECTION, domain_id)
        return parse_domain_gdf(snapshot.to_dict())
    except DocumentNotFoundError:
        raise ProcessingError(
            code="DOMAIN_NOT_FOUND",
            message=f"Domain {domain_id} not found.",
            suggestion="Ensure the domain exists before creating a grid.",
        )
    except EmptyDomainError:
        raise ProcessingError(
            code="EMPTY_DOMAIN",
            message="Domain has no geometry.",
            suggestion="Create a domain with at least one polygon feature.",
        )
    except InvalidGeometryError as e:
        raise ProcessingError(
            code="INVALID_GEOMETRY",
            message=str(e),
            suggestion="Ensure the domain has valid GeoJSON geometry.",
        )
