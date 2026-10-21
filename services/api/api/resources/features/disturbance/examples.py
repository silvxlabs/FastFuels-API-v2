"""
api/v2/resources/features/disturbance/examples.py

Example request bodies for Disturbance endpoints.

These examples are used in OpenAPI/Swagger documentation and tested in
integration tests, so the documentation stays accurate. domain_id comes from
the URL path parameter, not the request body.
"""

# Two overlapping polygons drawn inside the Blackfoot test domain
# (lib/tests/shared_data/domains/blackfoot.json, UTM 12N meters). The first is a
# moderate-severity fire 2-5 years ago (FDist 122); the second is a mechanical
# removal within the last year (FDist 311). Where they overlap, the more recent
# one (last digit 1) wins.
_FIRE_GEOMETRY = {
    "type": "Polygon",
    "coordinates": [
        [
            [294150.0, 5199050.0],
            [294500.0, 5199050.0],
            [294500.0, 5199450.0],
            [294150.0, 5199450.0],
            [294150.0, 5199050.0],
        ]
    ],
}
_THINNING_GEOMETRY = {
    "type": "Polygon",
    "coordinates": [
        [
            [294350.0, 5199300.0],
            [294700.0, 5199300.0],
            [294700.0, 5199700.0],
            [294350.0, 5199700.0],
            [294350.0, 5199300.0],
        ]
    ],
}
_CRS = {"type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::32612"}}

# Only the required property: the FDist code.
EXAMPLE_DISTURBANCE_MINIMAL = {
    "type": "FeatureCollection",
    "crs": _CRS,
    "mapping_year": 2025,
    "features": [
        {"type": "Feature", "properties": {"fdist": 122}, "geometry": _FIRE_GEOMETRY},
        {
            "type": "Feature",
            "properties": {"fdist": 311},
            "geometry": _THINNING_GEOMETRY,
        },
    ],
}

# The same polygons with the optional descriptive labels.
EXAMPLE_DISTURBANCE_WITH_LABELS = {
    "type": "FeatureCollection",
    "crs": _CRS,
    "mapping_year": 2025,
    "features": [
        {
            "type": "Feature",
            "properties": {
                "fdist": 122,
                "name": "Example Fire",
                "type": "wildfire",
                "year": 2022,
                "severity": "moderate",
            },
            "geometry": _FIRE_GEOMETRY,
        },
        {
            "type": "Feature",
            "properties": {
                "fdist": 311,
                "name": "Example Thinning",
                "type": "mechanical removal",
                "year": 2025,
                "severity": "low",
            },
            "geometry": _THINNING_GEOMETRY,
        },
    ],
}

EXAMPLE_DISTURBANCE_WITH_METADATA = {
    **EXAMPLE_DISTURBANCE_MINIMAL,
    "name": "Fire and thinning history",
    "tags": ["disturbance", "fire", "thinning"],
}

CREATE_DISTURBANCE_OPENAPI_EXAMPLES = {
    "minimal": {
        "value": EXAMPLE_DISTURBANCE_MINIMAL,
        "summary": "Minimal request",
        "description": (
            "Uploads two overlapping disturbance polygons with only the "
            "required `fdist` code on each, measured from a `mapping_year` "
            "of 2025."
        ),
    },
    "with_labels": {
        "value": EXAMPLE_DISTURBANCE_WITH_LABELS,
        "summary": "With descriptive labels",
        "description": (
            "Adds the optional `name`, `type`, `year` and `severity` to each "
            "polygon. They are stored as given and don't affect the result."
        ),
    },
    "with_metadata": {
        "value": EXAMPLE_DISTURBANCE_WITH_METADATA,
        "summary": "With name and tags",
        "description": "Uploads named, tagged disturbance polygons.",
    },
}

DISTURBANCE_EXAMPLE_VALUES = [
    ("minimal", EXAMPLE_DISTURBANCE_MINIMAL),
    ("with_labels", EXAMPLE_DISTURBANCE_WITH_LABELS),
    ("with_metadata", EXAMPLE_DISTURBANCE_WITH_METADATA),
]
