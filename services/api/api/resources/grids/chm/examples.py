"""
Example request bodies for CHM Meta endpoints.

These examples are used in:
1. OpenAPI/Swagger documentation - Users see these as example payloads
2. Integration tests - Each example is tested to ensure documentation stays accurate

domain_id comes from the URL path parameter, not the request body.
"""

EXAMPLE_META_CHM_MINIMAL = {}

EXAMPLE_META_CHM_WITH_METADATA = {
    "name": "Meta canopy height",
    "description": "Global canopy height model for forest inventory",
    "tags": ["chm", "meta"],
    "version": "2",
}

CREATE_META_CHM_OPENAPI_EXAMPLES = {
    "minimal": {
        "value": EXAMPLE_META_CHM_MINIMAL,
        "summary": "Minimal request",
        "description": (
            "Creates a grid with default settings. Returns canopy height "
            "at ~1m resolution."
        ),
    },
    "with_metadata": {
        "value": EXAMPLE_META_CHM_WITH_METADATA,
        "summary": "With name and tags",
        "description": (
            "Creates a named grid with tags for organization. Useful when "
            "maintaining multiple grids for scenario comparison."
        ),
    },
}

META_CHM_EXAMPLE_VALUES = [
    ("minimal", EXAMPLE_META_CHM_MINIMAL),
    ("with_metadata", EXAMPLE_META_CHM_WITH_METADATA),
]

EXAMPLE_NAIP_CHM_MINIMAL = {}

EXAMPLE_NAIP_CHM_WITH_METADATA = {
    "name": "NAIP canopy height",
    "description": "High-resolution 0.6m canopy height model for CONUS",
    "tags": ["chm", "naip", "high-res"],
}

CREATE_NAIP_CHM_OPENAPI_EXAMPLES = {
    "minimal": {
        "value": EXAMPLE_NAIP_CHM_MINIMAL,
        "summary": "Minimal request",
        "description": (
            "Creates a grid with default settings. Returns NAIP canopy height "
            "at ~0.6m resolution."
        ),
    },
    "with_metadata": {
        "value": EXAMPLE_NAIP_CHM_WITH_METADATA,
        "summary": "With name and tags",
        "description": (
            "Creates a named high-res NAIP grid with tags for organization. "
            "Useful for detailed stand-level analysis."
        ),
    },
}


NAIP_CHM_EXAMPLE_VALUES = [
    ("naip_minimal", EXAMPLE_NAIP_CHM_MINIMAL),
    ("naip_with_metadata", EXAMPLE_NAIP_CHM_WITH_METADATA),
]
