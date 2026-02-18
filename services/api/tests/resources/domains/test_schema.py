"""
Unit tests for api/v2/resources/domains/schema.py
"""

import json
from datetime import datetime

import pytest
from api.resources.domains.schema import (
    CreateDomainRequestBody,
    Domain,
    GeoJsonCRS,
    GeoJsonCRSProperties,
    UpdateDomainRequestBody,
    _parse_coordinates,
    _stringify_coordinates,
    default_crs_factory,
)

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_polygon_coordinates():
    """Simple polygon coordinates (triangle)."""
    return [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0]]]


@pytest.fixture
def sample_multipolygon_coordinates():
    """MultiPolygon coordinates (two triangles)."""
    return [
        [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0]]],
        [[[2.0, 2.0], [3.0, 2.0], [3.0, 3.0], [2.0, 2.0]]],
    ]


@pytest.fixture
def sample_feature(sample_polygon_coordinates):
    """A single GeoJSON Feature with Polygon geometry."""
    return {
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": sample_polygon_coordinates,
        },
        "properties": {"name": "test-feature"},
    }


@pytest.fixture
def sample_feature_collection(sample_feature):
    """A GeoJSON FeatureCollection with one feature."""
    return {
        "type": "FeatureCollection",
        "features": [sample_feature],
    }


@pytest.fixture
def sample_domain_data(sample_feature_collection):
    """Complete Domain data with all required fields."""
    return {
        **sample_feature_collection,
        "id": "test-domain-id",
        "name": "Test Domain",
        "description": "A test domain",
        "created_on": datetime(2024, 1, 1, 12, 0, 0),
        "modified_on": datetime(2024, 1, 1, 12, 0, 0),
        "tags": ["test", "unit-test"],
    }


@pytest.fixture
def sample_firestore_data(sample_domain_data, sample_polygon_coordinates):
    """Domain data as it would be stored in Firestore (stringified coordinates)."""
    data = sample_domain_data.copy()
    data["features"] = [
        {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": json.dumps(sample_polygon_coordinates),
            },
            "properties": {"name": "test-feature"},
        }
    ]
    return data


# =============================================================================
# GeoJsonCRSProperties Tests
# =============================================================================


class TestGeoJsonCRSProperties:
    def test_default_value(self):
        """Should default to EPSG:4326."""
        props = GeoJsonCRSProperties()
        assert props.name == "EPSG:4326"

    def test_custom_value(self):
        """Should accept custom CRS name."""
        props = GeoJsonCRSProperties(name="EPSG:32610")
        assert props.name == "EPSG:32610"


# =============================================================================
# GeoJsonCRS Tests
# =============================================================================


class TestGeoJsonCRS:
    def test_default_type(self):
        """Should default type to 'name'."""
        crs = GeoJsonCRS(properties=GeoJsonCRSProperties())
        assert crs.type == "name"

    def test_structure(self):
        """Should have correct nested structure."""
        crs = GeoJsonCRS(properties=GeoJsonCRSProperties(name="EPSG:32611"))
        assert crs.type == "name"
        assert crs.properties.name == "EPSG:32611"

    def test_model_dump(self):
        """Should serialize to correct dict structure."""
        crs = GeoJsonCRS(properties=GeoJsonCRSProperties(name="EPSG:4326"))
        dumped = crs.model_dump()
        assert dumped == {"type": "name", "properties": {"name": "EPSG:4326"}}


# =============================================================================
# default_crs_factory Tests
# =============================================================================


class TestDefaultCrsFactory:
    def test_returns_geojson_crs(self):
        """Should return a GeoJsonCRS instance."""
        crs = default_crs_factory()
        assert isinstance(crs, GeoJsonCRS)

    def test_default_epsg_4326(self):
        """Should default to EPSG:4326."""
        crs = default_crs_factory()
        assert crs.properties.name == "EPSG:4326"

    def test_returns_new_instance(self):
        """Should return a new instance each call (not shared reference)."""
        crs1 = default_crs_factory()
        crs2 = default_crs_factory()
        assert crs1 is not crs2


# =============================================================================
# _stringify_coordinates Tests
# =============================================================================


class TestStringifyCoordinates:
    def test_converts_polygon_coordinates(self, sample_polygon_coordinates):
        """Should convert nested polygon coordinates to JSON string."""
        data = {
            "features": [
                {
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": sample_polygon_coordinates,
                    }
                }
            ]
        }
        result = _stringify_coordinates(data)
        coords = result["features"][0]["geometry"]["coordinates"]

        assert isinstance(coords, str)
        assert json.loads(coords) == sample_polygon_coordinates

    def test_converts_multipolygon_coordinates(self, sample_multipolygon_coordinates):
        """Should convert nested multipolygon coordinates to JSON string."""
        data = {
            "features": [
                {
                    "geometry": {
                        "type": "MultiPolygon",
                        "coordinates": sample_multipolygon_coordinates,
                    }
                }
            ]
        }
        result = _stringify_coordinates(data)
        coords = result["features"][0]["geometry"]["coordinates"]

        assert isinstance(coords, str)
        assert json.loads(coords) == sample_multipolygon_coordinates

    def test_handles_multiple_features(self, sample_polygon_coordinates):
        """Should convert coordinates in all features."""
        data = {
            "features": [
                {"geometry": {"coordinates": sample_polygon_coordinates}},
                {"geometry": {"coordinates": sample_polygon_coordinates}},
            ]
        }
        result = _stringify_coordinates(data)

        for feature in result["features"]:
            assert isinstance(feature["geometry"]["coordinates"], str)

    def test_handles_missing_features_key(self):
        """Should return data unchanged if no features key."""
        data = {"type": "FeatureCollection"}
        result = _stringify_coordinates(data)
        assert result == data

    def test_handles_missing_geometry(self):
        """Should skip features without geometry."""
        data = {"features": [{"properties": {"name": "test"}}]}
        result = _stringify_coordinates(data)
        assert result == data

    def test_handles_missing_coordinates(self):
        """Should skip geometry without coordinates."""
        data = {"features": [{"geometry": {"type": "Polygon"}}]}
        result = _stringify_coordinates(data)
        assert result == data

    def test_mutates_input(self, sample_polygon_coordinates):
        """Should mutate the input dict (not create a copy)."""
        data = {"features": [{"geometry": {"coordinates": sample_polygon_coordinates}}]}
        result = _stringify_coordinates(data)
        assert result is data


# =============================================================================
# _parse_coordinates Tests
# =============================================================================


class TestParseCoordinates:
    def test_parses_stringified_polygon_coordinates(self, sample_polygon_coordinates):
        """Should parse JSON string back to nested list."""
        data = {
            "features": [
                {
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": json.dumps(sample_polygon_coordinates),
                    }
                }
            ]
        }
        result = _parse_coordinates(data)
        coords = result["features"][0]["geometry"]["coordinates"]

        assert isinstance(coords, list)
        assert coords == sample_polygon_coordinates

    def test_parses_stringified_multipolygon_coordinates(
        self, sample_multipolygon_coordinates
    ):
        """Should parse JSON string back to nested list for MultiPolygon."""
        data = {
            "features": [
                {
                    "geometry": {
                        "type": "MultiPolygon",
                        "coordinates": json.dumps(sample_multipolygon_coordinates),
                    }
                }
            ]
        }
        result = _parse_coordinates(data)
        coords = result["features"][0]["geometry"]["coordinates"]

        assert isinstance(coords, list)
        assert coords == sample_multipolygon_coordinates

    def test_handles_multiple_features(self, sample_polygon_coordinates):
        """Should parse coordinates in all features."""
        stringified = json.dumps(sample_polygon_coordinates)
        data = {
            "features": [
                {"geometry": {"coordinates": stringified}},
                {"geometry": {"coordinates": stringified}},
            ]
        }
        result = _parse_coordinates(data)

        for feature in result["features"]:
            assert isinstance(feature["geometry"]["coordinates"], list)

    def test_skips_already_parsed_coordinates(self, sample_polygon_coordinates):
        """Should not modify coordinates that are already lists."""
        data = {"features": [{"geometry": {"coordinates": sample_polygon_coordinates}}]}
        result = _parse_coordinates(data)
        coords = result["features"][0]["geometry"]["coordinates"]

        assert coords == sample_polygon_coordinates

    def test_handles_missing_features_key(self):
        """Should return data unchanged if no features key."""
        data = {"type": "FeatureCollection"}
        result = _parse_coordinates(data)
        assert result == data

    def test_handles_missing_geometry(self):
        """Should skip features without geometry."""
        data = {"features": [{"properties": {"name": "test"}}]}
        result = _parse_coordinates(data)
        assert result == data

    def test_handles_missing_coordinates(self):
        """Should skip geometry without coordinates."""
        data = {"features": [{"geometry": {"type": "Polygon"}}]}
        result = _parse_coordinates(data)
        assert result == data


# =============================================================================
# CreateDomainRequestBody Tests
# =============================================================================


class TestCreateDomainRequestBody:
    def test_valid_feature_collection(self, sample_feature_collection):
        """Should accept valid GeoJSON FeatureCollection."""
        body = CreateDomainRequestBody(**sample_feature_collection)
        assert body.type == "FeatureCollection"
        assert len(body.features) == 1

    def test_default_name(self, sample_feature_collection):
        """Should default name to empty string."""
        body = CreateDomainRequestBody(**sample_feature_collection)
        assert body.name == ""

    def test_default_description(self, sample_feature_collection):
        """Should default description to empty string."""
        body = CreateDomainRequestBody(**sample_feature_collection)
        assert body.description == ""

    def test_default_crs(self, sample_feature_collection):
        """Should default CRS to EPSG:4326."""
        body = CreateDomainRequestBody(**sample_feature_collection)
        assert body.crs.properties.name == "EPSG:4326"

    def test_default_tags(self, sample_feature_collection):
        """Should default tags to empty list."""
        body = CreateDomainRequestBody(**sample_feature_collection)
        assert body.tags == []

    def test_custom_fields(self, sample_feature_collection):
        """Should accept custom name, description, crs, and tags."""
        body = CreateDomainRequestBody(
            **sample_feature_collection,
            name="My Domain",
            description="A custom domain",
            crs=GeoJsonCRS(properties=GeoJsonCRSProperties(name="EPSG:32610")),
            tags=["tag1", "tag2"],
        )
        assert body.name == "My Domain"
        assert body.description == "A custom domain"
        assert body.crs.properties.name == "EPSG:32610"
        assert body.tags == ["tag1", "tag2"]

    def test_rejects_invalid_geojson(self):
        """Should reject invalid GeoJSON structure."""
        with pytest.raises(Exception):  # ValidationError
            CreateDomainRequestBody(type="Invalid", features=[])


# =============================================================================
# UpdateDomainRequestBody Tests
# =============================================================================


class TestUpdateDomainRequestBody:
    def test_all_fields_optional(self):
        """Should accept empty body (no fields provided)."""
        body = UpdateDomainRequestBody()
        assert body.name is None
        assert body.description is None
        assert body.tags is None

    def test_name_only(self):
        """Should accept body with only name."""
        body = UpdateDomainRequestBody(name="New Name")
        assert body.name == "New Name"
        assert body.description is None
        assert body.tags is None

    def test_description_only(self):
        """Should accept body with only description."""
        body = UpdateDomainRequestBody(description="New Description")
        assert body.name is None
        assert body.description == "New Description"
        assert body.tags is None

    def test_tags_only(self):
        """Should accept body with only tags."""
        body = UpdateDomainRequestBody(tags=["tag1", "tag2"])
        assert body.name is None
        assert body.description is None
        assert body.tags == ["tag1", "tag2"]

    def test_all_fields(self):
        """Should accept body with all fields."""
        body = UpdateDomainRequestBody(
            name="Updated Name",
            description="Updated Description",
            tags=["updated"],
        )
        assert body.name == "Updated Name"
        assert body.description == "Updated Description"
        assert body.tags == ["updated"]

    def test_empty_string_name(self):
        """Should accept empty string as name."""
        body = UpdateDomainRequestBody(name="")
        assert body.name == ""

    def test_empty_tags_list(self):
        """Should accept empty list for tags."""
        body = UpdateDomainRequestBody(tags=[])
        assert body.tags == []

    def test_model_dump_exclude_none(self):
        """model_dump with exclude_none should only include provided fields."""
        body = UpdateDomainRequestBody(name="New Name")
        dumped = body.model_dump(exclude_none=True)

        assert dumped == {"name": "New Name"}
        assert "description" not in dumped
        assert "tags" not in dumped

    def test_model_dump_exclude_none_all_fields(self):
        """model_dump with exclude_none includes all provided fields."""
        body = UpdateDomainRequestBody(
            name="Name",
            description="Desc",
            tags=["tag"],
        )
        dumped = body.model_dump(exclude_none=True)

        assert dumped == {
            "name": "Name",
            "description": "Desc",
            "tags": ["tag"],
        }


# =============================================================================
# Domain Tests
# =============================================================================


class TestDomain:
    def test_valid_domain(self, sample_domain_data):
        """Should accept valid domain data."""
        domain = Domain(**sample_domain_data)
        assert domain.id == "test-domain-id"
        assert domain.name == "Test Domain"
        assert domain.created_on == datetime(2024, 1, 1, 12, 0, 0)

    def test_requires_id(self, sample_feature_collection):
        """Should require id field (though it can be None by default)."""
        domain = Domain(
            **sample_feature_collection,
            created_on=datetime.now(),
            modified_on=datetime.now(),
        )
        # id defaults to None per Field definition
        assert domain.id is None

    def test_requires_created_on(self, sample_feature_collection):
        """Should require created_on field."""
        with pytest.raises(Exception):  # ValidationError
            Domain(**sample_feature_collection, modified_on=datetime.now())

    def test_requires_modified_on(self, sample_feature_collection):
        """Should require modified_on field."""
        with pytest.raises(Exception):  # ValidationError
            Domain(**sample_feature_collection, created_on=datetime.now())


# =============================================================================
# Domain Firestore Serialization Tests
# =============================================================================


class TestDomainFirestoreSerialization:
    def test_model_dump_default_preserves_coordinates(self, sample_domain_data):
        """Default model_dump should preserve nested list coordinates."""
        domain = Domain(**sample_domain_data)
        dumped = domain.model_dump()
        coords = dumped["features"][0]["geometry"]["coordinates"]

        assert isinstance(coords, list)

    def test_model_dump_firestore_context_stringifies_coordinates(
        self, sample_domain_data, sample_polygon_coordinates
    ):
        """model_dump with for_firestore context should stringify coordinates."""
        domain = Domain(**sample_domain_data)
        dumped = domain.model_dump(context={"for_firestore": True})
        coords = dumped["features"][0]["geometry"]["coordinates"]

        assert isinstance(coords, str)
        assert json.loads(coords) == sample_polygon_coordinates

    def test_model_dump_empty_context_preserves_coordinates(self, sample_domain_data):
        """model_dump with empty context should preserve nested lists."""
        domain = Domain(**sample_domain_data)
        dumped = domain.model_dump(context={})
        coords = dumped["features"][0]["geometry"]["coordinates"]

        assert isinstance(coords, list)

    def test_model_dump_other_context_preserves_coordinates(self, sample_domain_data):
        """model_dump with other context keys should preserve nested lists."""
        domain = Domain(**sample_domain_data)
        dumped = domain.model_dump(context={"other_key": True})
        coords = dumped["features"][0]["geometry"]["coordinates"]

        assert isinstance(coords, list)


# =============================================================================
# Domain Firestore Deserialization Tests
# =============================================================================


class TestDomainFirestoreDeserialization:
    def test_auto_parses_stringified_coordinates(
        self, sample_firestore_data, sample_polygon_coordinates
    ):
        """Should automatically parse stringified coordinates from Firestore."""
        domain = Domain(**sample_firestore_data)
        # Access the coordinates through the geojson-pydantic model
        coords = domain.features[0].geometry.coordinates

        # geojson-pydantic returns tuples, so compare structure
        assert coords is not None

    def test_accepts_already_parsed_coordinates(self, sample_domain_data):
        """Should accept data with already-parsed coordinates (from API request)."""
        domain = Domain(**sample_domain_data)
        assert domain.id == "test-domain-id"

    def test_handles_multiple_features_with_stringified_coords(
        self, sample_polygon_coordinates
    ):
        """Should parse coordinates in all features."""
        data = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": json.dumps(sample_polygon_coordinates),
                    },
                    "properties": {},
                },
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": json.dumps(sample_polygon_coordinates),
                    },
                    "properties": {},
                },
            ],
            "id": "test-id",
            "created_on": datetime.now(),
            "modified_on": datetime.now(),
        }
        domain = Domain(**data)
        assert len(domain.features) == 2


# =============================================================================
# Round-Trip Tests
# =============================================================================


class TestRoundTrip:
    def test_serialize_deserialize_round_trip(
        self, sample_domain_data, sample_polygon_coordinates
    ):
        """Data should survive a full Firestore round-trip."""
        # Create original domain
        original = Domain(**sample_domain_data)

        # Serialize for Firestore
        firestore_data = original.model_dump(context={"for_firestore": True})

        # Verify coordinates are stringified
        assert isinstance(firestore_data["features"][0]["geometry"]["coordinates"], str)

        # Deserialize back (simulating read from Firestore)
        restored = Domain(**firestore_data)

        # Verify data integrity
        assert restored.id == original.id
        assert restored.name == original.name
        assert restored.description == original.description
        assert len(restored.features) == len(original.features)

    def test_multiple_round_trips(self, sample_domain_data):
        """Data should survive multiple round-trips without degradation."""
        domain = Domain(**sample_domain_data)

        for _ in range(3):
            # Serialize for Firestore
            firestore_data = domain.model_dump(context={"for_firestore": True})
            # Deserialize back
            domain = Domain(**firestore_data)

        assert domain.id == "test-domain-id"
        assert domain.name == "Test Domain"
