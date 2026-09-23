"""
Unit tests for validate_tree_id_conditions (#611).

A tree_id condition on an inventory without a tree_id column (created before
tree IDs existed) is rejected with 422. Pure function; no server needed.
"""

import pytest
from api.resources.inventories.modification_models import InventoryModification
from api.resources.inventories.utils import validate_tree_id_conditions
from fastapi import HTTPException

LEGACY_COLUMNS = [
    {"key": "x", "type": "continuous", "unit": "m"},
    {"key": "y", "type": "continuous", "unit": "m"},
    {"key": "height", "type": "continuous", "unit": "m"},
]
COLUMNS = [{"key": "tree_id", "type": "categorical", "unit": None}, *LEGACY_COLUMNS]


def _mods(*conditions):
    return [
        InventoryModification(
            conditions=list(conditions), actions={"modifier": "remove"}
        )
    ]


TREE_ID_CONDITION = {"attribute": "tree_id", "operator": "eq", "value": [1, 2]}
HEIGHT_CONDITION = {"attribute": "height", "operator": "lt", "value": 2.0}


def test_tree_id_condition_without_column_rejected():
    with pytest.raises(HTTPException) as exc_info:
        validate_tree_id_conditions(_mods(TREE_ID_CONDITION), LEGACY_COLUMNS)
    assert exc_info.value.status_code == 422
    assert "tree_id" in exc_info.value.detail


def test_tree_id_condition_among_others_rejected():
    mods = _mods(HEIGHT_CONDITION) + _mods(HEIGHT_CONDITION, TREE_ID_CONDITION)
    with pytest.raises(HTTPException) as exc_info:
        validate_tree_id_conditions(mods, LEGACY_COLUMNS)
    assert exc_info.value.status_code == 422


def test_tree_id_condition_with_column_accepted():
    validate_tree_id_conditions(_mods(TREE_ID_CONDITION), COLUMNS)


def test_other_conditions_on_legacy_inventory_accepted():
    validate_tree_id_conditions(_mods(HEIGHT_CONDITION), LEGACY_COLUMNS)
    validate_tree_id_conditions(_mods(), LEGACY_COLUMNS)
