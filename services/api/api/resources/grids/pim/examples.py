"""
Example request bodies for TreeMap PIM endpoints.

These examples are used in:
1. OpenAPI/Swagger documentation - Users see these as example payloads
2. Integration tests - Each example is tested to ensure documentation stays accurate

domain_id comes from the URL path parameter, not the request body.
"""

EXAMPLE_TREEMAP_MINIMAL = {}

EXAMPLE_TREEMAP_WITH_METADATA = {
    "name": "TreeMap plot IDs",
    "description": "Plot imputation map for tree inventory creation",
    "tags": ["treemap", "pim"],
    "version": "2022",
}

EXAMPLE_TREEMAP_BOTH_BANDS = {
    "bands": ["tm_id", "plt_cn"],
}

EXAMPLE_TREEMAP_WITH_BUFFER = {
    "extent_buffer_cells": 8,
}

CREATE_TREEMAP_OPENAPI_EXAMPLES = {
    "minimal": {
        "value": EXAMPLE_TREEMAP_MINIMAL,
        "summary": "Minimal request",
        "description": (
            "Creates a grid with default settings. Returns TreeMap TM_ID values "
            "at 30m resolution."
        ),
    },
    "with_metadata": {
        "value": EXAMPLE_TREEMAP_WITH_METADATA,
        "summary": "With name and tags",
        "description": (
            "Creates a named grid with tags for organization. Useful when "
            "maintaining multiple grids for scenario comparison."
        ),
    },
    "both_bands": {
        "value": EXAMPLE_TREEMAP_BOTH_BANDS,
        "summary": "Both TM_ID and PLT_CN bands",
        "description": (
            "Creates a grid with both TreeMap ID and FIA plot condition number. "
            "PLT_CN is derived from the tree table lookup."
        ),
    },
    "with_buffer": {
        "value": EXAMPLE_TREEMAP_WITH_BUFFER,
        "summary": "With output buffer",
        "description": (
            "Includes 8 native-resolution cells of buffer beyond the domain "
            "extent. Useful when downstream resampling, reprojection, or "
            "edge-sensitive processing needs context past the domain edge."
        ),
    },
}

ALL_TREEMAP_EXAMPLE_VALUES = [
    ("minimal", EXAMPLE_TREEMAP_MINIMAL),
    ("with_metadata", EXAMPLE_TREEMAP_WITH_METADATA),
    ("both_bands", EXAMPLE_TREEMAP_BOTH_BANDS),
    ("with_buffer", EXAMPLE_TREEMAP_WITH_BUFFER),
]
