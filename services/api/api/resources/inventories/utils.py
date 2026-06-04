"""
api/v2/resources/inventories/utils.py

Shared validation utilities for inventory endpoints.
"""

from fastapi import HTTPException, status

from api.db.documents import firestore_client
from api.resources.inventories.modification_models import (
    InventoryFeatureSpatialCondition,
)
from lib.config import FEATURES_COLLECTION


async def validate_feature_conditions(
    items: list,
    owner_id: str,
    domain_id: str,
) -> None:
    """Verify every ``InventoryFeatureSpatialCondition.feature_id`` references a
    Feature that is owned by ``owner_id``, lives in ``domain_id``, and is in
    ``completed`` status.

    ``items`` is any list of objects exposing a ``.conditions`` list — both
    inventory modifications and treatments qualify. Looks up all distinct
    feature_ids in a single Firestore ``get_all`` call. The inventory-side
    mirror of ``grids.utils.validate_feature_modifications`` (issue #282 / #279).

    Raises:
        HTTPException(422): If any referenced feature is missing, owned by
            another user, in another domain, or not in ``completed`` status.
            All four cases use 422 — feature_id is a value the client supplied
            in the request body, so a bad reference is a validation error on
            the condition, not a path-level 404.
    """
    feature_ids: list[str] = []
    seen: set[str] = set()
    for item in items:
        for condition in item.conditions:
            if isinstance(condition, InventoryFeatureSpatialCondition):
                if condition.feature_id not in seen:
                    seen.add(condition.feature_id)
                    feature_ids.append(condition.feature_id)

    if not feature_ids:
        return

    refs = [
        firestore_client.collection(FEATURES_COLLECTION).document(fid)
        for fid in feature_ids
    ]
    snapshots = {snap.id: snap async for snap in firestore_client.get_all(refs)}

    for fid in feature_ids:
        snap = snapshots.get(fid)
        if snap is None or not snap.exists:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=(
                    f"Modification references feature_id {fid!r}, which does "
                    f"not exist in this domain."
                ),
            )
        data = snap.to_dict() or {}
        if data.get("owner_id") != owner_id or data.get("domain_id") != domain_id:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=(
                    f"Modification references feature_id {fid!r}, which does "
                    f"not exist in this domain."
                ),
            )
        feature_status = data.get("status")
        if feature_status != "completed":
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=(
                    f"Condition references feature_id {fid!r} whose status "
                    f"is {feature_status!r}, expected 'completed'. Wait for "
                    f"the feature to finish before referencing it."
                ),
            )
