"""
api/v2/resources/features/building/examples.py

Example request bodies for Building feature endpoints.

These examples are used in:
1. OpenAPI/Swagger documentation - Users see these as example payloads
2. Integration tests - Each example is tested to ensure documentation stays accurate

domain_id comes from the URL path parameter, not the request body.
"""

EXAMPLE_BUILDING_MINIMAL = {"type": "building"}

EXAMPLE_BUILDING_WITH_METADATA = {
    "type": "building",
    "name": "Overture Buildings",
    "description": "Building footprints with heights from Overture Maps.",
    "tags": ["infrastructure", "overture", "buildings"],
}

EXAMPLE_BUILDING_WITH_BUFFER = {
    "type": "building",
    "name": "Overture Buildings (with overhang)",
    "description": "Buildings clipped to the domain plus 25 m.",
    "tags": ["infrastructure", "overture", "buildings"],
    "extent_buffer_m": 25,
}

CREATE_BUILDING_OPENAPI_EXAMPLES = {
    "minimal": {
        "value": EXAMPLE_BUILDING_MINIMAL,
        "summary": "Minimal request",
        "description": "Building footprints clipped to the domain boundary.",
    },
    "with_metadata": {
        "value": EXAMPLE_BUILDING_WITH_METADATA,
        "summary": "With name and tags",
        "description": "A named building feature with tags for organization.",
    },
    "with_buffer": {
        "value": EXAMPLE_BUILDING_WITH_BUFFER,
        "summary": "With clip-extent buffer",
        "description": (
            "Expands the clip extent by 25 m, so buildings that cross the "
            "domain boundary are not cut at the edge."
        ),
    },
}

BUILDING_EXAMPLE_VALUES = [
    ("minimal", EXAMPLE_BUILDING_MINIMAL),
    ("with_metadata", EXAMPLE_BUILDING_WITH_METADATA),
    ("with_buffer", EXAMPLE_BUILDING_WITH_BUFFER),
]
