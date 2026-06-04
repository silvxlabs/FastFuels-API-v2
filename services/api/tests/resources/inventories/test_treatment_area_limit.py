"""
Unit tests for the create-time inventory-wide basal-area area check
(``validate_inventory_wide_treatment_area``).
"""

import pytest
from api.resources.inventories.treatment_models import (
    InventoryBasalAreaTreatment,
    InventoryDiameterTreatment,
)
from api.resources.inventories.utils import validate_inventory_wide_treatment_area
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
