"""
Null semantics shared by the metadata PATCH bodies (#712).

Omitted fields are untouched, an explicit null description clears it to "",
and name/tags reject null.
"""

import pytest
from api.resources.domains.schema import UpdateDomainRequestBody
from api.resources.exports.schema import UpdateExportRequestBody
from api.resources.features.schema import UpdateFeatureRequestBody
from api.resources.grids.schema import UpdateGridRequestBody
from api.resources.inventories.schema import UpdateInventoryRequestBody
from api.resources.point_clouds.schema import UpdatePointCloudRequestBody
from pydantic import ValidationError

BODIES = [
    UpdateDomainRequestBody,
    UpdateExportRequestBody,
    UpdateFeatureRequestBody,
    UpdateGridRequestBody,
    UpdateInventoryRequestBody,
    UpdatePointCloudRequestBody,
]


@pytest.mark.parametrize("body_cls", BODIES)
class TestUpdateBodyNulls:
    def test_omitted_fields_not_dumped(self, body_cls):
        assert body_cls().model_dump(exclude_unset=True) == {}

    def test_null_description_clears_to_empty(self, body_cls):
        body = body_cls.model_validate({"description": None})
        assert body.model_dump(exclude_unset=True) == {"description": ""}

    @pytest.mark.parametrize("field", ["name", "tags"])
    def test_null_rejected(self, body_cls, field):
        with pytest.raises(ValidationError):
            body_cls.model_validate({field: None})

    def test_empty_tags_clears(self, body_cls):
        body = body_cls.model_validate({"tags": []})
        assert body.model_dump(exclude_unset=True) == {"tags": []}

    def test_openapi_name_and_tags_not_nullable(self, body_cls):
        props = body_cls.model_json_schema()["properties"]
        for field in ("name", "tags"):
            assert "anyOf" not in props[field]
            assert "default" not in props[field]
        assert {"type": "null"} in props["description"]["anyOf"]
