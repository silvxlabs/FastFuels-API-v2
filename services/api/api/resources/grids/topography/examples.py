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

EXAMPLE_TOPOGRAPHY_WITH_BUFFER = {
    "bands": ["elevation", "slope", "aspect"],
    "extent_buffer_cells": 8,
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
    "with_buffer": {
        "value": EXAMPLE_TOPOGRAPHY_WITH_BUFFER,
        "summary": "With output buffer",
        "description": (
            "Includes 8 result-grid cells of buffer beyond the domain extent. "
            "Useful when downstream resampling, reprojection, or derivative "
            "recomputation needs context beyond the domain edge."
        ),
    },
}

ALL_TOPOGRAPHY_EXAMPLE_VALUES = [
    ("minimal", EXAMPLE_TOPOGRAPHY_MINIMAL),
    ("elevation_only", EXAMPLE_TOPOGRAPHY_ELEVATION_ONLY),
    ("with_metadata", EXAMPLE_TOPOGRAPHY_WITH_METADATA),
    ("with_buffer", EXAMPLE_TOPOGRAPHY_WITH_BUFFER),
]

# 3DEP examples

EXAMPLE_3DEP_MINIMAL = {}

EXAMPLE_3DEP_ALL_BANDS = {
    "bands": ["elevation", "slope", "aspect"],
}

EXAMPLE_3DEP_1M = {
    "source_resolution": 1,
    "bands": ["elevation"],
}

EXAMPLE_3DEP_WITH_METADATA = {
    "name": "High-res terrain",
    "description": "10m elevation for wind flow modeling",
    "tags": ["topography", "3dep"],
    "source_resolution": 10,
    "bands": ["elevation", "slope", "aspect"],
}

EXAMPLE_3DEP_WITH_BUFFER = {
    "source_resolution": 10,
    "bands": ["elevation", "slope", "aspect"],
    "extent_buffer_cells": 10,
}

EXAMPLE_3DEP_DOMAIN_2M = {
    "source_resolution": 1,
    "bands": ["elevation"],
    "alignment": {"target": "domain", "resolution": 2.0},
    "name": "1m 3DEP downsampled to 2m at domain origin",
    "description": "Composes with other 2m grids on the same domain.",
}

EXAMPLE_3DEP_NATIVE = {
    "source_resolution": 1,
    "bands": ["elevation"],
    "alignment": {"target": "native"},
    "name": "1m elevation at native source pixel anchor",
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
    "with_buffer": {
        "value": EXAMPLE_3DEP_WITH_BUFFER,
        "summary": "With output buffer for derivatives",
        "description": (
            "Includes 10 result-grid cells of buffer around the domain. "
            "Helps reduce edge artifacts in slope/aspect computed from the DEM."
        ),
    },
    "domain_aligned_2m": {
        "value": EXAMPLE_3DEP_DOMAIN_2M,
        "summary": "1m source resampled to 2m at the domain origin",
        "description": (
            "Picks the 1m source product, then aligns the output to the "
            "domain origin at a 2m cell size. Composes with other 2m "
            "domain-anchored grids."
        ),
    },
    "native_anchor": {
        "value": EXAMPLE_3DEP_NATIVE,
        "summary": "Preserve source pixel anchor",
        "description": (
            "Disables the default domain anchor. Output preserves the "
            "source raster's pixel grid; will not compose with "
            "domain-anchored grids without further alignment."
        ),
    },
}

ALL_3DEP_EXAMPLE_VALUES = [
    ("minimal", EXAMPLE_3DEP_MINIMAL),
    ("all_bands", EXAMPLE_3DEP_ALL_BANDS),
    ("1m_elevation", EXAMPLE_3DEP_1M),
    ("with_metadata", EXAMPLE_3DEP_WITH_METADATA),
    ("with_buffer", EXAMPLE_3DEP_WITH_BUFFER),
    ("domain_aligned_2m", EXAMPLE_3DEP_DOMAIN_2M),
    ("native_anchor", EXAMPLE_3DEP_NATIVE),
]
