"""Regression tests for the generated OpenAPI contract."""

import re
from collections import defaultdict

from api.app import app


def test_schema_component_titles_are_unique():
    """Every component title must map to exactly one generated client model."""
    schemas = app.openapi()["components"]["schemas"]
    components_by_title = defaultdict(list)

    for component_name, schema in schemas.items():
        if title := schema.get("title"):
            components_by_title[title].append(component_name)

    duplicates = {
        title: component_names
        for title, component_names in components_by_title.items()
        if len(component_names) > 1
    }

    assert not duplicates, f"Duplicate OpenAPI schema titles: {duplicates}"


def test_generator_sensitive_schema_titles_are_stable():
    """Keep the disambiguated names consumed by generated clients stable."""
    schemas = app.openapi()["components"]["schemas"]
    titles = {schema.get("title") for schema in schemas.values()}

    assert "GeoJsonFeature" in titles
    assert schemas["CreateDomainRequestBody"]["title"] == "GeoJsonFeatureCollection"
    assert schemas["Domain"]["title"] == "Domain"
    assert "PointCloudThreeDepCoverageResponse" in titles
    assert "TopographyThreeDepCoverageResponse" in titles


def test_input_requirements_are_documented_and_name_real_fields():
    """Every requirement names a field of its request, using documented keys."""
    from api.resources.input_requirements import (
        INPUT_REQUIREMENTS_DOC,
        X_INPUT_REQUIREMENTS,
    )

    schemas = app.openapi()["components"]["schemas"]

    def resolve(schema, path):
        for part in path.split("."):
            if "$ref" in schema:
                schema = schemas[schema["$ref"].split("/")[-1]]
            variants = schema.get("oneOf") or schema.get("anyOf") or [schema]
            schema = next(
                v["properties"][part]
                for v in (
                    schemas[x["$ref"].split("/")[-1]] if "$ref" in x else x
                    for x in variants
                )
                if part in v.get("properties", {})
            )
        return schema

    entries = [
        (name, entry)
        for name, schema in schemas.items()
        for entry in schema.get(X_INPUT_REQUIREMENTS, [])
    ]
    assert entries
    for name, entry in entries:
        resolve(schemas[name], entry["field"])
        for path in entry.get("when", {}):
            resolve(schemas[name], path)
        assert all(f"`{key}`" in INPUT_REQUIREMENTS_DOC for key in entry), entry


def _band_enums(schemas: dict) -> dict[str, dict]:
    from lib.band_quantities import BAND_KEY_PATTERN

    return {
        name: schema
        for name, schema in schemas.items()
        if schema.get("enum")
        and all(re.fullmatch(BAND_KEY_PATTERN, str(v)) for v in schema["enum"])
    }


def test_every_band_enum_value_has_metadata_from_its_definitions():
    """Each band-key enum labels every value with its code band definition."""
    from api.resources.band_metadata import (
        _BAND_DEFS,
        ENUM_METADATA_FIELDS,
        X_BAND_QUANTITIES,
        X_ENUM_METADATA,
    )

    schemas = app.openapi()["components"]["schemas"]
    quantities = schemas["Band"][X_BAND_QUANTITIES]
    defs_by_name = {enum.__name__: (enum, defs) for enum, defs in _BAND_DEFS.items()}
    band_enums = _band_enums(schemas)

    assert set(band_enums) == set(defs_by_name)
    for name, schema in band_enums.items():
        metadata = schema[X_ENUM_METADATA]
        assert set(metadata) == set(schema["enum"]), name
        enum, defs = defs_by_name[name]
        for value, entry in metadata.items():
            band_def = defs[enum(value)]
            assert entry == {
                k: str(band_def[k])
                for k in ENUM_METADATA_FIELDS
                if band_def.get(k) is not None
            }, (name, value)
            assert {"name", "type"} <= entry.keys(), (name, value)
            assert {k: entry[k] for k in ("type", "unit") if k in entry} == (
                quantities[value.partition(".")[0]]
            ), (name, value)


def test_band_quantities_are_published_on_band():
    from api.resources.band_metadata import X_BAND_QUANTITIES

    from lib.band_quantities import BAND_QUANTITIES

    published = app.openapi()["components"]["schemas"]["Band"][X_BAND_QUANTITIES]
    assert published == {
        name: {"type": q.type, **({"unit": q.unit} if q.unit else {})}
        for name, q in BAND_QUANTITIES.items()
    }


def test_band_metadata_keys_are_documented():
    from api.resources.band_metadata import (
        BAND_METADATA_DOC,
        X_BAND_QUANTITIES,
        X_ENUM_METADATA,
    )

    assert f"`{X_ENUM_METADATA}`" in BAND_METADATA_DOC
    assert f"`{X_BAND_QUANTITIES}`" in BAND_METADATA_DOC
    assert BAND_METADATA_DOC in app.openapi()["info"]["description"]
