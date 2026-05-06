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

EXAMPLE_FBFM40_REMOVE_NON_BURNABLE = {
    "name": "FBFM40 with non-burnable removal",
    "remove_non_burnable": ["NB1", "NB2", "NB8", "NB9"],
}

EXAMPLE_FBFM40_WITH_BUFFER = {
    "name": "FBFM40 with buffer",
    "extent_buffer_cells": 8,
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
    "remove_non_burnable": {
        "value": EXAMPLE_FBFM40_REMOVE_NON_BURNABLE,
        "summary": "Remove non-burnable blocks",
        "description": (
            "Removes coarse 30m-resolution non-burnable blocks (urban, snow/ice, "
            "water, bare ground) by replacing them with the most frequent "
            "neighboring burnable fuel model. This clears the way for masking "
            "in high-resolution 1-2m road, water, and building features from "
            "sources such as OpenStreetMap."
        ),
    },
    "with_buffer": {
        "value": EXAMPLE_FBFM40_WITH_BUFFER,
        "summary": "With output buffer",
        "description": (
            "Includes 8 native-resolution cells of buffer beyond the domain "
            "extent. Useful when the grid will later be resampled, reprojected, "
            "or processed by focal filters / derivative calculations that are "
            "sensitive to edges."
        ),
    },
}

ALL_FBFM40_EXAMPLE_VALUES = [
    ("minimal", EXAMPLE_FBFM40_MINIMAL),
    ("with_metadata", EXAMPLE_FBFM40_WITH_METADATA),
    ("remove_non_burnable", EXAMPLE_FBFM40_REMOVE_NON_BURNABLE),
    ("with_buffer", EXAMPLE_FBFM40_WITH_BUFFER),
]
