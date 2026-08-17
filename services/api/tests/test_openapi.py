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
