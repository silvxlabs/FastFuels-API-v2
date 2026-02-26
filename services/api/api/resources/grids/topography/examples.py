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

# 3DEP examples

EXAMPLE_3DEP_MINIMAL = {}

EXAMPLE_3DEP_ALL_BANDS = {
    "bands": ["elevation", "slope", "aspect"],
}

EXAMPLE_3DEP_1M = {
    "resolution": 1,
    "bands": ["elevation"],
}

EXAMPLE_3DEP_WITH_METADATA = {
    "name": "High-res terrain",
    "description": "10m elevation for wind flow modeling",
    "tags": ["topography", "3dep"],
    "resolution": 10,
    "bands": ["elevation", "slope", "aspect"],
}

CREATE_3DEP_TOPOGRAPHY_OPENAPI_EXAMPLES = {
    "minimal": {
        "value": EXAMPLE_3DEP_MINIMAL,
        "summary": "Minimal request (10m elevation)",
        "description": (
            "Creates a grid with default settings. Returns elevation at 10m resolution."
        ),
    },
    "all_bands": {
        "value": EXAMPLE_3DEP_ALL_BANDS,
        "summary": "All bands",
        "description": (
            "Creates a grid with elevation, slope, and aspect at default "
            "10m resolution."
        ),
    },
    "1m_elevation": {
        "value": EXAMPLE_3DEP_1M,
        "summary": "1m elevation",
        "description": (
            "Creates a grid with 1m elevation data from 3DEP seamless 1-meter "
            "(S1M) or legacy project tiles."
        ),
    },
    "with_metadata": {
        "value": EXAMPLE_3DEP_WITH_METADATA,
        "summary": "With name and tags",
        "description": ("Creates a named 10m grid with all topography bands and tags."),
    },
}

ALL_3DEP_EXAMPLE_VALUES = [
    ("minimal", EXAMPLE_3DEP_MINIMAL),
    ("all_bands", EXAMPLE_3DEP_ALL_BANDS),
    ("1m_elevation", EXAMPLE_3DEP_1M),
    ("with_metadata", EXAMPLE_3DEP_WITH_METADATA),
]
