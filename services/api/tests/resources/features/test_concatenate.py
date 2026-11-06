"""
Unit tests for api/v2/resources/features/concatenate.py and the GeoParquet
encoding in api/v2/resources/features/storage.py.

Request validation, input-feature validation, and bounds/attribution merging.
Pure unit tests with no external dependencies.
"""

import asyncio

import geopandas as gpd
import pytest
from api.resources.attribution import merge_attributions
from api.resources.features.concatenate import (
    ConcatenateFeaturesRequest,
    union_bounds,
    validate_source_features,
)
from api.resources.features.schema import FeatureType
from api.resources.features.storage import encode_feature_parquet
from fastapi import HTTPException
from pydantic import ValidationError
from shapely.geometry import box

OWNER = "owner-a"
DOMAIN = "domain-a"


def feature_doc(**overrides) -> dict:
    return {
        "owner_id": OWNER,
        "domain_id": DOMAIN,
        "type": "road",
        "status": "completed",
        **overrides,
    }


def osm_attribution(accessed_on: str) -> dict:
    return {
        "dataset": "OpenStreetMap",
        "license_name": "ODbL-1.0",
        "license_url": "https://opendatacommons.org/licenses/odbl/1-0/",
        "citation": "© OpenStreetMap contributors",
        "access_url": "https://www.openstreetmap.org/copyright",
        "accessed_on": accessed_on,
    }


class TestConcatenateFeaturesRequest:
    def test_valid_request(self):
        body = ConcatenateFeaturesRequest(source_feature_ids=["a", "b"])
        assert body.source_feature_ids == ["a", "b"]
        assert body.name == ""

    def test_one_feature_rejected(self):
        with pytest.raises(ValidationError):
            ConcatenateFeaturesRequest(source_feature_ids=["a"])

    def test_too_many_features_rejected(self):
        with pytest.raises(ValidationError):
            ConcatenateFeaturesRequest(source_feature_ids=[str(i) for i in range(101)])

    def test_duplicate_ids_rejected(self):
        with pytest.raises(ValidationError, match="more than once"):
            ConcatenateFeaturesRequest(source_feature_ids=["a", "b", "a"])


class TestValidateSourceFeatures:
    def validate(self, docs, feature_type=FeatureType.road):
        validate_source_features(docs, OWNER, DOMAIN, feature_type)

    def test_valid_inputs_pass(self):
        self.validate({"a": feature_doc(), "b": feature_doc()})

    @pytest.mark.parametrize(
        "doc",
        [
            None,
            feature_doc(owner_id="owner-b"),
            feature_doc(domain_id="domain-b"),
        ],
        ids=["missing", "other-owner", "other-domain"],
    )
    def test_unreachable_input_returns_422(self, doc):
        with pytest.raises(HTTPException) as exc:
            self.validate({"a": feature_doc(), "b": doc})
        assert exc.value.status_code == 422
        assert "'b' does not exist in this domain" in exc.value.detail

    def test_wrong_type_returns_422(self):
        with pytest.raises(HTTPException) as exc:
            self.validate({"a": feature_doc(), "b": feature_doc(type="building")})
        assert exc.value.status_code == 422
        assert "is a building feature, not a road feature" in exc.value.detail

    @pytest.mark.parametrize("status", ["pending", "running", "failed"])
    def test_incomplete_input_returns_422(self, status):
        with pytest.raises(HTTPException) as exc:
            self.validate({"a": feature_doc(), "b": feature_doc(status=status)})
        assert exc.value.status_code == 422
        assert "expected 'completed'" in exc.value.detail


class TestUnionBounds:
    def test_covers_every_input(self):
        docs = [
            {"georeference": {"bounds": [0, 0, 10, 10]}},
            {"georeference": {"bounds": [-5, 2, 8, 20]}},
        ]
        assert union_bounds(docs) == (-5, 0, 10, 20)


class TestMergeAttributions:
    def test_deduplicates_keeping_earliest(self):
        merged = merge_attributions(
            [osm_attribution("2026-11-05")], [osm_attribution("2026-10-01")], []
        )
        assert merged == [osm_attribution("2026-10-01")]

    def test_none_and_empty_lists_merge_to_empty(self):
        assert merge_attributions(None, []) == []


class TestEncodeFeatureParquet:
    def test_mixed_type_column_returns_422(self):
        gdf = gpd.GeoDataFrame(
            {"width": [5, "5 m"]}, geometry=[box(0, 0, 1, 1)] * 2, crs="EPSG:32612"
        )
        with pytest.raises(HTTPException) as exc:
            asyncio.run(encode_feature_parquet(gdf))
        assert exc.value.status_code == 422
        assert "width" in exc.value.detail

    def test_round_trips(self):
        gdf = gpd.GeoDataFrame(
            {"name": ["a", None]}, geometry=[box(0, 0, 1, 1)] * 2, crs="EPSG:32612"
        )
        data = asyncio.run(encode_feature_parquet(gdf))
        assert data[:4] == b"PAR1"
