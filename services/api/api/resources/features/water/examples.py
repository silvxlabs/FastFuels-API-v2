"""
api/v2/resources/features/water/examples.py

Example request bodies for Water feature endpoints.

These examples are used in:
1. OpenAPI/Swagger documentation - Users see these as example payloads
2. Integration tests - Each example is tested to ensure documentation stays accurate

domain_id comes from the URL path parameter, not the request body.
"""

EXAMPLE_WATER_MINIMAL = {"type": "water"}

EXAMPLE_WATER_WITH_METADATA = {
    "type": "water",
    "name": "OSM Water Bodies and Streams",
    "description": "OpenStreetMap lakes, ponds, rivers, and streams. Linear waterways are automatically buffered into polygons.",
    "tags": ["hydrology", "osm", "water"],
}

CREATE_WATER_OPENAPI_EXAMPLES = {
    "minimal": {
        "value": EXAMPLE_WATER_MINIMAL,
        "summary": "Minimal request",
        "description": (
            "Kicks off a background task to extract, buffer, and clip water "
            "features for this domain using default OpenStreetMap data."
        ),
    },
    "with_metadata": {
        "value": EXAMPLE_WATER_WITH_METADATA,
        "summary": "With name and tags",
        "description": (
            "Creates a named water feature with custom tags for organization. "
            "Useful for distinguishing between different hydrology layers or scenarios."
        ),
    },
}

WATER_EXAMPLE_VALUES = [
    ("minimal", EXAMPLE_WATER_MINIMAL),
    ("with_metadata", EXAMPLE_WATER_WITH_METADATA),
]
