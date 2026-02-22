"""
Example request bodies for Topography endpoints.

These examples are used in:
1. OpenAPI/Swagger documentation - Users see these as example payloads
2. Integration tests - Each example is tested to ensure documentation stays accurate

domain_id comes from the URL path parameter, not the request body.
"""

EXAMPLE_TOPOGRAPHY_MINIMAL = {}

EXAMPLE_TOPOGRAPHY_ELEVATION_ONLY = {
    "bands": ["elevation"],
}

EXAMPLE_TOPOGRAPHY_WITH_METADATA = {
    "name": "Terrain data",
    "description": "Elevation, slope, and aspect for fire behavior modeling",
    "tags": ["topography", "terrain"],
    "version": "2020",
    "bands": ["elevation", "slope", "aspect"],
}

CREATE_LANDFIRE_TOPOGRAPHY_OPENAPI_EXAMPLES = {
    "minimal": {
        "value": EXAMPLE_TOPOGRAPHY_MINIMAL,
        "summary": "Minimal request (all bands)",
        "description": (
            "Creates a grid with default settings. Returns all three topography "
            "bands (elevation, slope, aspect) at 30m resolution."
        ),
    },
    "elevation_only": {
        "value": EXAMPLE_TOPOGRAPHY_ELEVATION_ONLY,
        "summary": "Elevation only",
        "description": (
            "Creates a grid with only the elevation band. Useful when slope "
            "and aspect are not needed."
        ),
    },
    "with_metadata": {
        "value": EXAMPLE_TOPOGRAPHY_WITH_METADATA,
        "summary": "With name and tags",
        "description": (
            "Creates a named grid with all topography bands and tags for organization."
        ),
    },
}

ALL_TOPOGRAPHY_EXAMPLE_VALUES = [
    ("minimal", EXAMPLE_TOPOGRAPHY_MINIMAL),
    ("elevation_only", EXAMPLE_TOPOGRAPHY_ELEVATION_ONLY),
    ("with_metadata", EXAMPLE_TOPOGRAPHY_WITH_METADATA),
]
