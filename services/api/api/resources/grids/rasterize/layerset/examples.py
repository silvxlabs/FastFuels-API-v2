"""
api/v2/resources/grids/rasterize/layerset/examples.py

Example request bodies for the layerset rasterize endpoint.

These examples are used in:
1. OpenAPI/Swagger documentation - Users see these as example payloads
2. Integration tests - Each example is tested to ensure documentation stays accurate

domain_id comes from the URL path parameter, not the request body.
"""

# Placeholder layerset Feature ID. Integration tests substitute a real
# uploaded layerset's ID at runtime; the docs render the placeholder so
# users see the expected shape.
_EXAMPLE_LAYERSET_ID = "LAYERSET_ID_GOES_HERE"

EXAMPLE_LAYERSET_MINIMAL = {
    "layerset_id": _EXAMPLE_LAYERSET_ID,
}

EXAMPLE_LAYERSET_WITH_METADATA = {
    "layerset_id": _EXAMPLE_LAYERSET_ID,
    "name": "Custom surface fuels — Blackfoot site",
    "description": "Rasterized layerset for the Blackfoot example domain.",
    "tags": ["layerset", "custom", "blackfoot"],
}

EXAMPLE_LAYERSET_MAX_OVERLAP = {
    "layerset_id": _EXAMPLE_LAYERSET_ID,
    "overlap_method": "max",
}

EXAMPLE_LAYERSET_WITH_BUFFER = {
    "layerset_id": _EXAMPLE_LAYERSET_ID,
    "extent_buffer_cells": 4,
}

CREATE_LAYERSET_RASTERIZE_OPENAPI_EXAMPLES = {
    "minimal": {
        "value": EXAMPLE_LAYERSET_MINIMAL,
        "summary": "Minimal request",
        "description": (
            "Rasterizes the named layerset using the default mean overlap method "
            "and the domain alignment target."
        ),
    },
    "with_metadata": {
        "value": EXAMPLE_LAYERSET_WITH_METADATA,
        "summary": "With name, description, and tags",
        "description": (
            "Adds discoverability metadata to the rasterized grid. Useful when "
            "maintaining multiple fuel scenarios per domain."
        ),
    },
    "max_overlap": {
        "value": EXAMPLE_LAYERSET_MAX_OVERLAP,
        "summary": "Use max reduction for overlapping polygons",
        "description": (
            "Where multiple fuelbed polygons overlap a single output cell, the "
            "cell takes the maximum value rather than the default mean."
        ),
    },
    "with_buffer": {
        "value": EXAMPLE_LAYERSET_WITH_BUFFER,
        "summary": "With output buffer",
        "description": (
            "Includes 4 result-grid cells of buffer around the domain extent. "
            "Cells inside the buffered extent that fall outside polygon coverage "
            "are populated with the rasterizer's fill value."
        ),
    },
}

ALL_LAYERSET_RASTERIZE_EXAMPLE_VALUES = [
    ("minimal", EXAMPLE_LAYERSET_MINIMAL),
    ("with_metadata", EXAMPLE_LAYERSET_WITH_METADATA),
    ("max_overlap", EXAMPLE_LAYERSET_MAX_OVERLAP),
    ("with_buffer", EXAMPLE_LAYERSET_WITH_BUFFER),
]
