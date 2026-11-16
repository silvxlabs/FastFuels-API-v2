"""
Unit tests for the request-time treatment checks: the inventory-wide area limit
(``validate_inventory_wide_treatment_area``) and the per-treatment column check
(``validate_treatment_columns``).
"""

import pytest
from api.resources.inventories.schema import (
    BASE_INVENTORY_COLUMNS,
    CHM_INVENTORY_COLUMNS,
    CROWN_RADIUS_COLUMN,
)
from api.resources.inventories.treatment_models import (
    InventoryBasalAreaTreatment,
    InventoryCrownSpacingTreatment,
    InventoryDiameterTreatment,
)
from api.resources.inventories.utils import (
    validate_inventory_wide_treatment_area,
    validate_treatment_columns,
)
from fastapi import HTTPException

SMALL_DOMAIN = {"bbox": [0, 0, 1000, 1000]}  # 1 km²
LARGE_DOMAIN = {"bbox": [0, 0, 5000, 5000]}  # 25 km², over the 16 km² limit

GEOMETRY_CONDITION = {
    "source": "geometry",
    "operator": "within",
    "geometry": {
        "type": "Polygon",
        "coordinates": [
            [[-120.0, 38.0], [-119.5, 38.0], [-119.5, 38.5], [-120.0, 38.0]]
        ],
    },
}


def _basal(conditions=None):
    return InventoryBasalAreaTreatment(
        method="from_below", value=20.0, conditions=conditions or []
    )


def _diameter():
    return InventoryDiameterTreatment(method="from_below", value=30.0)


class TestValidateInventoryWideTreatmentArea:
    def test_no_treatments_ok(self):
        validate_inventory_wide_treatment_area(LARGE_DOMAIN, [])

    def test_small_domain_inventory_wide_basal_ok(self):
        validate_inventory_wide_treatment_area(SMALL_DOMAIN, [_basal()])

    def test_large_domain_inventory_wide_basal_raises(self):
        with pytest.raises(HTTPException) as exc:
            validate_inventory_wide_treatment_area(LARGE_DOMAIN, [_basal()])
        assert exc.value.status_code == 422

    def test_large_domain_diameter_ok(self):
        # Diameter never materializes the stand, so the area limit does not apply.
        validate_inventory_wide_treatment_area(LARGE_DOMAIN, [_diameter()])

    def test_large_domain_scoped_basal_ok(self):
        # Spatially scoped basal-area is bounded by its region; standgen checks it.
        validate_inventory_wide_treatment_area(
            LARGE_DOMAIN, [_basal([GEOMETRY_CONDITION])]
        )


def _crown_spacing(conditions=None, **kwargs):
    return InventoryCrownSpacingTreatment(
        value=3.0, conditions=conditions or [], **kwargs
    )


class TestCrownSpacingArea:
    def test_large_domain_inventory_wide_crown_spacing_raises(self):
        with pytest.raises(HTTPException) as exc:
            validate_inventory_wide_treatment_area(LARGE_DOMAIN, [_crown_spacing()])
        assert exc.value.status_code == 422

    def test_large_domain_scoped_crown_spacing_ok(self):
        validate_inventory_wide_treatment_area(
            LARGE_DOMAIN, [_crown_spacing([GEOMETRY_CONDITION])]
        )


PIM_COLUMNS = {c.key for c in BASE_INVENTORY_COLUMNS}
CHM_COLUMNS = {c.key for c in CHM_INVENTORY_COLUMNS}
SEGMENTED_CHM_COLUMNS = CHM_COLUMNS | {CROWN_RADIUS_COLUMN.key}
MEASURED_RADIUS = {"type": "inventory_column", "column": "crown_radius"}


class TestValidateTreatmentColumns:
    def test_no_treatments_ok(self):
        validate_treatment_columns([], CHM_COLUMNS)

    def test_pim_columns_satisfy_diameter_and_default_crown_spacing(self):
        validate_treatment_columns(
            [_diameter(), _basal(), _crown_spacing()], PIM_COLUMNS
        )

    def test_diameter_on_chm_rejected(self):
        with pytest.raises(HTTPException) as exc:
            validate_treatment_columns([_diameter()], SEGMENTED_CHM_COLUMNS)
        assert exc.value.status_code == 422
        assert "`dbh`" in exc.value.detail
        assert "crown_spacing" in exc.value.detail

    def test_default_crown_spacing_on_chm_rejected_with_hint(self):
        with pytest.raises(HTTPException) as exc:
            validate_treatment_columns([_crown_spacing()], SEGMENTED_CHM_COLUMNS)
        assert exc.value.status_code == 422
        assert "`rank_by`" in exc.value.detail

    def test_height_ranked_measured_crown_spacing_on_segmented_chm_ok(self):
        validate_treatment_columns(
            [_crown_spacing(rank_by="height", max_crown_radius_source=MEASURED_RADIUS)],
            SEGMENTED_CHM_COLUMNS,
        )

    def test_measured_radius_without_column_rejected(self):
        """A CHM inventory without segmentation has no crown_radius."""
        with pytest.raises(HTTPException) as exc:
            validate_treatment_columns(
                [
                    _crown_spacing(
                        rank_by="height", max_crown_radius_source=MEASURED_RADIUS
                    )
                ],
                CHM_COLUMNS,
            )
        assert "`crown_radius`" in exc.value.detail

    def test_names_the_failing_treatment(self):
        with pytest.raises(HTTPException) as exc:
            validate_treatment_columns(
                [
                    _crown_spacing(
                        rank_by="height", max_crown_radius_source=MEASURED_RADIUS
                    ),
                    _diameter(),
                ],
                SEGMENTED_CHM_COLUMNS,
            )
        assert "treatments[1]" in exc.value.detail
