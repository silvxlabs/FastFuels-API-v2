"""
Unit tests for api/v2/resources/inventories/tree/allometry/schema.py

Tests the shared crown base height model type.
"""

import pytest
from api.resources.inventories.tree.allometry.schema import (
    CROWN_BASE_HEIGHT_EQUATIONS,
    GDAM_CROWN_BASE_HEIGHT_MODEL,
    CrownBaseHeightModel,
    FiaNationalLogit2026Model,
    GdamModel,
    default_crown_base_height_model,
)
from pydantic import TypeAdapter, ValidationError

ADAPTER = TypeAdapter(CrownBaseHeightModel)

NAMES = [
    "fia_national_logit_2026",
    "monleon_2004_eq1",
    "monleon_2004_eq2",
    "toney_reeves_2009",
    "randolph_2010",
    "westfall_2019",
    "gdam",
]


@pytest.mark.parametrize("name", NAMES)
def test_each_name_validates(name):
    model = ADAPTER.validate_python({"name": name})
    assert model.name == name
    assert model.model_dump() == {"name": name}


def test_unknown_name_rejected():
    with pytest.raises(ValidationError):
        ADAPTER.validate_python({"name": "bogus"})


def test_gdam_member():
    model = ADAPTER.validate_python({"name": "gdam"})
    assert isinstance(model, GdamModel)
    assert GDAM_CROWN_BASE_HEIGHT_MODEL == "gdam"


def test_gdam_description_is_user_facing():
    schema = ADAPTER.json_schema()
    description = schema["$defs"]["GdamModel"]["description"]
    assert "flow-matching" in description
    assert "uncompacted crown" in description


def test_equations_exclude_gdam():
    names = [equation().name for equation in CROWN_BASE_HEIGHT_EQUATIONS]
    assert names == NAMES[:-1]


def test_missing_name_rejected():
    with pytest.raises(ValidationError):
        ADAPTER.validate_python({})


def test_default_is_national_refit():
    model = default_crown_base_height_model()
    assert isinstance(model, FiaNationalLogit2026Model)
    assert model.name == "fia_national_logit_2026"


def test_json_schema_has_discriminator():
    schema = ADAPTER.json_schema()
    assert schema["discriminator"]["propertyName"] == "name"
    assert set(schema["discriminator"]["mapping"]) == set(NAMES)
