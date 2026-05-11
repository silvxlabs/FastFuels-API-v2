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

EXAMPLE_META_CHM_WITH_BUFFER = {
    "extent_buffer_cells": 4,
}

EXAMPLE_META_CHM_DOMAIN_2M = {
    "alignment": {"target": "domain", "resolution": 2.0},
    "name": "Meta CHM at 2m on domain origin",
}

EXAMPLE_META_CHM_TARGET_GRID = {
    "alignment": {"target": "grid", "grid_id": "grid_xyz789"},
    "name": "Meta CHM aligned to existing grid",
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
    "with_buffer": {
        "value": EXAMPLE_META_CHM_WITH_BUFFER,
        "summary": "With output buffer",
        "description": (
            "Includes 4 result-grid cells of buffer beyond the domain extent. "
            "Useful when downstream resampling, reprojection, or "
            "edge-sensitive processing needs context past the domain edge."
        ),
    },
    "domain_aligned_2m": {
        "value": EXAMPLE_META_CHM_DOMAIN_2M,
        "summary": "2m output anchored to domain origin",
        "description": (
            "Resamples CHM to 2m on the domain-origin lattice. "
            "Composes with other 2m domain-anchored grids."
        ),
    },
    "target_grid": {
        "value": EXAMPLE_META_CHM_TARGET_GRID,
        "summary": "Align to an existing grid",
        "description": (
            "Aligns CHM to the exact CRS, transform, and shape of the "
            "named target grid. Useful for composing with an existing "
            "lattice."
        ),
    },
}

META_CHM_EXAMPLE_VALUES = [
    ("minimal", EXAMPLE_META_CHM_MINIMAL),
    ("with_metadata", EXAMPLE_META_CHM_WITH_METADATA),
    ("with_buffer", EXAMPLE_META_CHM_WITH_BUFFER),
    ("domain_aligned_2m", EXAMPLE_META_CHM_DOMAIN_2M),
    ("target_grid", EXAMPLE_META_CHM_TARGET_GRID),
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
