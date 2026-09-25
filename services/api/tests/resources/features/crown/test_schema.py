"""
Unit tests for api/v2/resources/features/crown/schema.py
"""

import pytest
from api.resources.crown_segmentation import CrownSegmentationBase
from api.resources.features.crown.examples import CROWN_EXAMPLE_VALUES
from api.resources.features.crown.schema import (
    ChmCrownSegmentation,
    ChmCrownSource,
    CreateChmCrownFeatureRequest,
)
from api.resources.features.schema import FeatureType
from pydantic import ValidationError

IDS = {"source_inventory_id": "inv", "source_chm_grid_id": "grid"}


class TestChmCrownSegmentation:
    def test_shares_base_with_chm_detection(self):
        assert issubclass(ChmCrownSegmentation, CrownSegmentationBase)

    def test_defaults(self):
        assert ChmCrownSegmentation().model_dump(mode="json") == {
            "method": "dalponte2016",
            "min_relative_height": 0.45,
            "min_relative_crown_height": 0.55,
            "max_crown_radius": 10.0,
            "min_height": 2.0,
            "max_height": 120.0,
        }

    def test_max_height_null_means_no_ceiling(self):
        assert ChmCrownSegmentation(max_height=None).max_height is None

    @pytest.mark.parametrize(
        "field,value",
        [
            ("min_height", -1.0),
            ("min_relative_height", 1.0),
            ("min_relative_height", -0.1),
            ("min_relative_crown_height", 1.0),
            ("max_crown_radius", 0.0),
            ("max_crown_radius", 31.0),
            ("method", "watershed"),
        ],
    )
    def test_out_of_range_rejected(self, field, value):
        with pytest.raises(ValidationError):
            ChmCrownSegmentation(**{field: value})

    @pytest.mark.parametrize("max_height", [2.0, 1.0])
    def test_max_height_must_exceed_min_height(self, max_height):
        with pytest.raises(ValidationError, match="max_height"):
            ChmCrownSegmentation(min_height=2.0, max_height=max_height)


class TestChmCrownSource:
    def test_product_is_chm(self):
        source = ChmCrownSource(
            source_inventory_id="inv",
            source_chm_grid_id="grid",
            crown_segmentation=ChmCrownSegmentation(),
        )
        assert source.product == "chm"

    def test_dump_records_checksums_and_settings(self):
        source = ChmCrownSource(
            source_inventory_id="inv",
            source_inventory_checksum="a" * 32,
            source_chm_grid_id="grid",
            source_chm_grid_checksum="b" * 32,
            crown_segmentation=ChmCrownSegmentation(max_crown_radius=5.0),
        )
        data = source.model_dump(mode="json")
        assert data["source_inventory_checksum"] == "a" * 32
        assert data["source_chm_grid_checksum"] == "b" * 32
        assert data["crown_segmentation"]["max_crown_radius"] == 5.0
        assert data["crown_segmentation"]["method"] == "dalponte2016"


class TestCreateChmCrownFeatureRequest:
    def test_minimal(self):
        request = CreateChmCrownFeatureRequest(**IDS)
        assert request.type == FeatureType.crown
        assert request.crown_segmentation == ChmCrownSegmentation()

    @pytest.mark.parametrize("missing", ["source_inventory_id", "source_chm_grid_id"])
    def test_source_ids_required(self, missing):
        body = {k: v for k, v in IDS.items() if k != missing}
        with pytest.raises(ValidationError):
            CreateChmCrownFeatureRequest(**body)

    def test_other_type_rejected(self):
        with pytest.raises(ValidationError):
            CreateChmCrownFeatureRequest(type="road", **IDS)

    @pytest.mark.parametrize("name,example", CROWN_EXAMPLE_VALUES)
    def test_examples_validate(self, name, example):
        CreateChmCrownFeatureRequest(**example)
