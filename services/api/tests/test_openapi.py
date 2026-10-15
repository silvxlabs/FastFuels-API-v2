"""Regression tests for the generated OpenAPI contract."""

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
            if part == "*":
                schema = schema["items"]
                continue
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
        paths = [*entry.get("when", {})]
        paths += [
            entry[k] for k in ("same_lattice_as", "of", "derived_from") if k in entry
        ]
        for path in paths:
            resolve(schemas[name], path)
        assert all(f"`{key}`" in INPUT_REQUIREMENTS_DOC for key in entry), entry
