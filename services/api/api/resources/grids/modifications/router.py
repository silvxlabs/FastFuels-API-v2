"""
api/v2/resources/grids/modifications/router.py

Router for the in-place grid modifications endpoint.

POST /domains/{domain_id}/grids/{grid_id}/modifications
Applies modifications to the existing grid in place (same ID), asynchronously.
"""

import uuid
from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Body, HTTPException, Request, status
from google.cloud import firestore

from api.db.documents import firestore_client
from api.dependencies import VerifiedDomain
from api.resources.grids.modifications.examples import (
    APPLY_GRID_MODIFICATIONS_OPENAPI_EXAMPLES,
)
from api.resources.grids.modifications.schema import ApplyGridModificationsRequest
from api.resources.grids.schema import Grid
from api.resources.grids.utils import (
    dump_modifications_for_firestore,
    resolve_modification_fuel_model_labels,
    validate_feature_modifications,
    validate_grid_has_band,
)
from api.schema import JobStatus
from api.tasks import create_http_task_async
from lib.config import GRIDDLE_QUEUE, GRIDDLE_SERVICE, GRIDS_COLLECTION

router = APIRouter()

COLLECTION = GRIDS_COLLECTION


def _referenced_band_keys(body: ApplyGridModificationsRequest) -> list[str]:
    """Collect every band key referenced by the submitted rules."""
    keys: list[str] = []
    seen: set[str] = set()
    for modification in body.modifications:
        for condition in modification.conditions:
            band = getattr(condition, "band", None)
            if band is not None and band not in seen:
                seen.add(band)
                keys.append(band)
        for action in modification.actions:
            if action.band not in seen:
                seen.add(action.band)
                keys.append(action.band)
    return keys


@router.post(
    "",
    response_model=Grid,
    status_code=status.HTTP_200_OK,
    summary="Apply modifications to a grid in place",
)
async def apply_grid_modifications(
    request: Request,
    domain: VerifiedDomain,
    grid_id: str,
    body: Annotated[
        ApplyGridModificationsRequest,
        Body(openapi_examples=APPLY_GRID_MODIFICATIONS_OPENAPI_EXAMPLES),
    ],
):
    """
    # Apply Modifications to a Grid (in place)

    Applies modification rules to **this** grid in place — the grid keeps its
    ID and the submitted rules are applied on top of its current data
    asynchronously. To keep the original data instead, duplicate the grid
    first (`POST .../{grid_id}/duplicate`) and modify the copy.

    The grid's stored data is updated directly; the upstream source (LANDFIRE,
    3DEP, ...) is **not** re-fetched, so cells your rules don't touch are
    byte-for-byte unchanged — even if the upstream product has been updated
    since the grid was built.

    Modifications select cells by conditions and apply actions to the matching
    cells.

    ## Combining conditions: AND within a rule, OR across rules

    Each rule's `conditions` are **ANDed** — a cell is selected only when it
    satisfies *every* condition in that rule. Adding a condition to a rule
    therefore **narrows** the selection (the intersection). Example: a feature
    condition plus an attribute condition matches cells inside the feature
    **and** above a value threshold.

    There is **no OR within a rule**. To act on a **union** — "roads *or*
    water bodies", "GR1 *or* GR2 cells" — use **multiple rules**. Rules are
    applied independently and in order, so a cell matched by *any* rule is
    affected. Adding a rule therefore **widens** the overall selection.

    Putting two mutually exclusive conditions in one rule (e.g. a road feature
    AND a water feature) is the classic mistake: it selects cells that are
    both at once — usually none. Split them into one rule per feature instead.

    ## Conditions

    **Attribute conditions** compare a band's cell values against a value:
    - `band`: dot-notation band key (e.g., `fbfm`, `fuel_load.1hr`)
    - `operator`: `eq`, `ne`, `gt`, `lt`, `ge`, `le`
      (`eq`/`ne` also accept a list of values)
    - `value`: number or list for `eq`/`ne`. For `fbfm` bands you may use the
      human-readable Scott-Burgan labels (`"GR1"`) or the numeric codes (`101`)
      interchangeably — labels are resolved to codes when the rule is stored.

    **Spatial conditions** test each cell's location against a geometry. Two
    variants discriminated by the required `source` field:

    - `source: "geometry"` — supply GeoJSON directly via `geometry` (plus
      optional `crs`; defaults to the domain CRS).
    - `source: "feature"` — reference a persisted Feature resource by
      `feature_id` (road, water, layerset). The Feature must belong to the
      same domain as this grid and be in `completed` status; cross-domain,
      missing, or unfinished references are rejected with 422.

    Both spatial variants accept:
    - `operator`: `within`, `outside`, or `intersects`
    - `buffer_m`: (optional, meters) expands the geometry outward in the
      domain's projected CRS before testing.
    - `target`: `centroid` (default) tests the cell center; `cell` tests the
      cell's full footprint — use it with linestring features (e.g. roads)
      so every crossed cell matches.

    ## Actions

    - `{"band": "...", "modifier": "replace|multiply|divide|add|subtract", "value": ...}`
    - Non-`replace` results are clamped at zero (grid bands are physical
      quantities).

    ## Response

    Returns this grid (same ID) with status `"pending"`. Its `checksum`
    changes immediately, so any resource derived from it (resample, lookup,
    exports) can detect that the source has changed. The submitted rules
    appear in the grid's `modifications` list once processing completes —
    poll the grid until status returns to `"completed"`.

    If processing fails, the grid's status becomes `"failed"` with error
    details, the stored data is unchanged, and the queued rules are retained —
    submit another POST to retry (the new rules are applied together with the
    retained ones).

    ## Error Responses

    - **404 Not Found**: The grid does not exist, is not owned by the caller,
      or is not in this domain.
    - **422 Unprocessable Content**: The grid is not in `completed` status
      (and is not a retryable failed modification); the grid is a 3D voxel
      grid (apply modifications to the source tree inventory and re-voxelize
      instead); a referenced `feature_id` is missing, cross-domain, or not
      completed; or a referenced band does not exist on this grid.
    """
    owner_id = request.state.id
    domain_id = domain["id"]

    await validate_feature_modifications(body.modifications, owner_id, domain_id)

    new_checksum = uuid.uuid4().hex
    ref = firestore_client.collection(COLLECTION).document(grid_id)

    @firestore.async_transactional
    async def _append_pending(transaction) -> dict:
        """Read-validate-append atomically so concurrent POSTs can't drop a
        rule: the loser's transaction retries, re-reads the now-`pending`
        status, and is rejected instead of overwriting the winner's delta."""
        snapshot = await ref.get(transaction=transaction)
        grid_data = snapshot.to_dict() if snapshot.exists else None
        if (
            grid_data is None
            or grid_data.get("owner_id") != owner_id
            or grid_data.get("domain_id") != domain_id
        ):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document not found: grids/{grid_id}",
            )

        shape = (grid_data.get("georeference") or {}).get("shape", [])
        if len(shape) == 3:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=(
                    "Modifications are not supported for 3D voxel grids. "
                    "Apply modifications to the source tree inventory and "
                    "re-voxelize instead."
                ),
            )

        validate_grid_has_band(grid_data, grid_id, _referenced_band_keys(body))

        # Resolve FBFM labels against the grid's own band types (the
        # authoritative read), then snapshot the delta. Re-running on a
        # transaction retry is a no-op: resolved codes pass through unchanged.
        band_types = {b["key"]: b["type"] for b in grid_data.get("bands", [])}
        resolve_modification_fuel_model_labels(body.modifications, band_types)
        new_modifications = dump_modifications_for_firestore(body.modifications)

        pending = grid_data.get("pending_modifications") or []
        grid_status = grid_data.get("status")
        retryable_failed = grid_status == JobStatus.failed.value and pending
        if grid_status != JobStatus.completed.value and not retryable_failed:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=(
                    f"grids/{grid_id} status is '{grid_status}', expected 'completed'."
                ),
            )

        # Queue only the delta; griddle merges pending_modifications into the
        # cumulative `modifications` ledger atomically with status=completed,
        # so the ledger always equals the applied data (#319). The checksum
        # rotates now so derivatives become detectably stale (#304).
        update = {
            "pending_modifications": pending + new_modifications,
            "checksum": new_checksum,
            "status": JobStatus.pending.value,
            "progress": None,
            "error": None,
            "modified_on": datetime.now(),
        }
        transaction.update(ref, update)
        return {**grid_data, **update}

    grid_data = await _append_pending(firestore_client.transaction())

    # The task name embeds the fresh checksum: Cloud Tasks tombstones reused
    # task names, so re-using the bare grid_id (the create task's name) would
    # silently drop this task.
    await create_http_task_async(
        GRIDDLE_QUEUE,
        GRIDDLE_SERVICE,
        grid_id,
        task_name=f"{grid_id}-{new_checksum}",
    )

    # pending_modifications is an internal work-queue field, not part of the
    # Grid schema; Pydantic ignores it (and owner_id) on construction.
    return Grid(**grid_data)
