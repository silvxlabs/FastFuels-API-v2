"""
Example request bodies for FBFM40 endpoints.

These examples are used in:
1. OpenAPI/Swagger documentation - Users see these as example payloads
2. Integration tests - Each example is tested to ensure documentation stays accurate

domain_id comes from the URL path parameter, not the request body.
"""

EXAMPLE_FBFM40_MINIMAL = {}

EXAMPLE_FBFM40_WITH_METADATA = {
    "name": "FBFM40 fuel model codes",
    "description": "Baseline fuel model codes for scenario comparison",
    "tags": ["baseline", "surface-fuel"],
    "version": "2022",
}

CREATE_LANDFIRE_FBFM40_OPENAPI_EXAMPLES = {
    "minimal": {
        "value": EXAMPLE_FBFM40_MINIMAL,
        "summary": "Minimal request",
        "description": (
            "Creates a grid with default settings. Returns FBFM40 fuel model codes "
            "at 30m resolution. Use /grids/lookup/fbfm40 to convert codes to "
            "fuel parameters."
        ),
    },
    "with_metadata": {
        "value": EXAMPLE_FBFM40_WITH_METADATA,
        "summary": "With name and tags",
        "description": (
            "Creates a named grid with tags for organization. Useful when "
            "maintaining multiple grids for scenario comparison."
        ),
    },
}

ALL_FBFM40_EXAMPLE_VALUES = [
    ("minimal", EXAMPLE_FBFM40_MINIMAL),
    ("with_metadata", EXAMPLE_FBFM40_WITH_METADATA),
]
