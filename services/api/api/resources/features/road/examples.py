"""
api/v2/resources/features/road/examples.py

Example request bodies for Road feature endpoints.

These examples are used in:
1. OpenAPI/Swagger documentation - Users see these as example payloads
2. Integration tests - Each example is tested to ensure documentation stays accurate

domain_id comes from the URL path parameter, not the request body.
"""

EXAMPLE_ROAD_MINIMAL = {"type": "road"}

EXAMPLE_ROAD_WITH_METADATA = {
    "type": "road",
    "name": "OSM Road Network",
    "description": "OpenStreetMap roads automatically buffered to realistic widths based on highway classification.",
    "tags": ["infrastructure", "osm", "roads"],
}

CREATE_ROAD_OPENAPI_EXAMPLES = {
    "minimal": {
        "value": EXAMPLE_ROAD_MINIMAL,
        "summary": "Minimal request",
        "description": (
            "Kicks off a background task to extract, buffer, and clip the road "
            "network for this domain using default OpenStreetMap data."
        ),
    },
    "with_metadata": {
        "value": EXAMPLE_ROAD_WITH_METADATA,
        "summary": "With name and tags",
        "description": (
            "Creates a named road feature with custom tags for organization. "
            "Useful when managing multiple infrastructure layers."
        ),
    },
}

ROAD_EXAMPLE_VALUES = [
    ("minimal", EXAMPLE_ROAD_MINIMAL),
    ("with_metadata", EXAMPLE_ROAD_WITH_METADATA),
]
