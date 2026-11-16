"""
api/v2/resources/inventories/utils.py

Shared validation utilities for inventory endpoints.
"""

from fastapi import HTTPException, status

from api.db.documents import firestore_client
from api.resources.inventories.modification_models import (
    InventoryFeatureSpatialCondition,
)
from lib.config import FEATURES_COLLECTION, SUPPORT_EMAIL

# Max area (km²) an inventory-wide basal-area or crown-spacing treatment may
# cover. Both thin the whole treated population at once, in memory; this bounds
# that to standgen's worker memory. standgen enforces the same value as its
# backstop.
MAX_TREATMENT_AREA_SQ_KM = 16.0

# Metrics that thin the whole treated population at once, in memory.
IN_MEMORY_TREATMENT_METRICS = {"basal_area", "crown_spacing"}


def validate_inventory_wide_treatment_area(domain: dict, treatments: list) -> None:
    """Reject an inventory-wide basal-area or crown-spacing treatment over a
    domain too large to thin in memory.

    These treatments hold their entire treated population in memory at once, so
    the treated area is capped at ``MAX_TREATMENT_AREA_SQ_KM``. Only the
    inventory-wide case (no spatial conditions) is checked here — its region is
    the whole domain, whose area is the ``bbox`` already stored at domain create.
    Spatially scoped treatments are bounded by their (sub-domain) region and are
    checked by standgen, which resolves their geometry.

    Raises:
        HTTPException(422): If any such treatment without conditions would cover
            a domain larger than the limit.
    """
    if not any(
        t.metric in IN_MEMORY_TREATMENT_METRICS and not t.conditions for t in treatments
    ):
        return

    minx, miny, maxx, maxy = domain["bbox"]
    area_sq_km = (maxx - minx) * (maxy - miny) / 1e6
    if area_sq_km <= MAX_TREATMENT_AREA_SQ_KM:
        return

    raise HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        detail=(
            f"Basal-area and crown-spacing treatments thin the whole stand at "
            f"once, so the entire treated population is held in memory. An "
            f"inventory-wide treatment would cover this domain's "
            f"{area_sq_km:.1f} km², above the {MAX_TREATMENT_AREA_SQ_KM:.0f} km² "
            f"limit. Scope the treatment to a smaller area with a spatial "
            f"condition, or contact {SUPPORT_EMAIL} to process a larger area."
        ),
    )


def validate_treatment_columns(treatments: list, column_keys: set[str]) -> None:
    """Reject treatments that read a column the inventory does not have.

    Diameter and basal-area treatments need `dbh`; a crown-spacing treatment
    needs its `rank_by` column and its crown-radius column or allometry inputs
    (see ``lib.inventory_io.treatment_required_columns``).

    Raises:
        HTTPException(422): If any treatment needs a missing column.
    """
    for index, treatment in enumerate(treatments):
        missing = sorted(treatment.required_columns() - column_keys)
        if not missing:
            continue
        missing_names = ", ".join(f"`{c}`" for c in missing)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=(
                f"treatments[{index}] (`{treatment.metric}`) reads "
                f"{missing_names}, which this inventory does not have."
                f"{_missing_treatment_columns_hint(treatment, missing)}"
            ),
        )


def _missing_treatment_columns_hint(treatment, missing: list[str]) -> str:
    """A next step for the common missing-column cases."""
    if treatment.metric != "crown_spacing":
        return (
            " Inventories without a diameter, such as CHM-derived ones, can be "
            "thinned with a `crown_spacing` treatment instead."
        )
    if "dbh" in missing or "fia_species_code" in missing:
        return (
            " For an inventory without diameter or species, such as a "
            "CHM-derived one with crown segmentation, set `rank_by` to "
            '`height` and `max_crown_radius_source` to `{"type": '
            '"inventory_column", "column": "crown_radius"}`.'
        )
    return ""


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
