"""
api/v2/resources/inventories/utils.py

Shared validation utilities for inventory endpoints.
"""

from fastapi import HTTPException, status

from api.resources.inventories.modification_models import (
    InventoryFeatureSpatialCondition,
)
from api.resources.parents import Parents
from lib.config import SUPPORT_EMAIL

# Max area (km²) an inventory-wide basal-area treatment may cover. A basal-area
# thin holds its whole treated population in memory at once; this bounds that to
# standgen's worker memory. standgen enforces the same value as its backstop.
MAX_TREATMENT_AREA_SQ_KM = 16.0


def validate_inventory_wide_treatment_area(domain: dict, treatments: list) -> None:
    """Reject an inventory-wide basal-area treatment over a domain too large to
    thin in memory.

    A basal-area treatment holds its entire treated population in memory at once,
    so the treated area is capped at ``MAX_TREATMENT_AREA_SQ_KM``. Only the
    inventory-wide case (no spatial conditions) is checked here — its region is
    the whole domain, whose area is the ``bbox`` already stored at domain create.
    Spatially scoped treatments are bounded by their (sub-domain) region and are
    checked by standgen, which resolves their geometry.

    Raises:
        HTTPException(422): If any basal-area treatment without conditions would
            cover a domain larger than the limit.
    """
    if not any(t.metric == "basal_area" and not t.conditions for t in treatments):
        return

    minx, miny, maxx, maxy = domain["bbox"]
    area_sq_km = (maxx - minx) * (maxy - miny) / 1e6
    if area_sq_km <= MAX_TREATMENT_AREA_SQ_KM:
        return

    raise HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        detail=(
            f"A basal-area treatment thins the whole stand at once, so its entire "
            f"treated population is held in memory. An inventory-wide treatment "
            f"would cover this domain's {area_sq_km:.1f} km², above the "
            f"{MAX_TREATMENT_AREA_SQ_KM:.0f} km² limit. Scope the treatment "
            f"to a smaller area with a spatial condition, or contact "
            f"{SUPPORT_EMAIL} to process a larger area."
        ),
    )


def condition_feature_ids(items: list) -> list[str]:
    """Distinct ``feature_id``s the items' conditions reference, in
    first-reference order.

    ``items`` is any list of objects exposing a ``.conditions`` list — both
    inventory modifications and treatments qualify.
    """
    feature_ids: dict[str, None] = {}
    for item in items:
        for condition in item.conditions:
            if isinstance(condition, InventoryFeatureSpatialCondition):
                feature_ids[condition.feature_id] = None
    return list(feature_ids)


def validate_feature_conditions(
    items: list,
    parents: Parents,
    owner_id: str,
    domain_id: str,
) -> None:
    """Verify every ``InventoryFeatureSpatialCondition.feature_id`` references a
    Feature that is owned by ``owner_id``, lives in ``domain_id``, and is in
    ``completed`` status.

    ``items`` is any list of objects exposing a ``.conditions`` list — both
    inventory modifications and treatments qualify. The referenced features
    must be among the loaded ``parents``. The inventory-side mirror of
    ``grids.utils.validate_feature_modifications`` (issue #282 / #279).

    Raises:
        HTTPException(422): If any referenced feature is missing, owned by
            another user, in another domain, or not in ``completed`` status.
            All four cases use 422 — feature_id is a value the client supplied
            in the request body, so a bad reference is a validation error on
            the condition, not a path-level 404.
    """
    for fid in condition_feature_ids(items):
        data = parents.features.get(fid)
        if (
            data is None
            or data.get("owner_id") != owner_id
            or data.get("domain_id") != domain_id
        ):
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
