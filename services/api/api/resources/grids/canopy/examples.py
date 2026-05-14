"""
Example request bodies for canopy endpoints (Meta CHM, NAIP CHM, LANDFIRE).

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

# LANDFIRE canopy examples

EXAMPLE_LANDFIRE_CANOPY_MINIMAL = {}

EXAMPLE_LANDFIRE_CANOPY_CROWN_FIRE_INPUTS = {
    "bands": ["cbd", "cbh"],
}

EXAMPLE_LANDFIRE_CANOPY_COVER_ONLY = {
    "bands": ["cc"],
}

EXAMPLE_LANDFIRE_CANOPY_WITH_METADATA = {
    "name": "LANDFIRE canopy fuels",
    "description": "Canopy bulk density, base height, height, and cover for crown fire modeling",
    "tags": ["canopy", "landfire"],
    "version": "2024",
    "bands": ["chm", "cbd", "cbh", "cc"],
}

EXAMPLE_LANDFIRE_CANOPY_WITH_BUFFER = {
    "bands": ["chm", "cbd", "cbh", "cc"],
    "extent_buffer_cells": 6,
}

EXAMPLE_LANDFIRE_CANOPY_NATIVE_ALIGNMENT = {
    "alignment": {"target": "native"},
    "name": "LANDFIRE canopy preserving native pixel anchor",
    "bands": ["chm", "cbd", "cbh", "cc"],
}

CREATE_LANDFIRE_CANOPY_OPENAPI_EXAMPLES = {
    "minimal": {
        "value": EXAMPLE_LANDFIRE_CANOPY_MINIMAL,
        "summary": "Minimal request (all bands)",
        "description": (
            "Creates a grid with default settings. Returns all four canopy "
            "bands (chm, cbd, cbh, cc) at 30m resolution."
        ),
    },
    "crown_fire_inputs": {
        "value": EXAMPLE_LANDFIRE_CANOPY_CROWN_FIRE_INPUTS,
        "summary": "Crown fire inputs (cbd + cbh)",
        "description": (
            "Returns canopy bulk density and canopy base height — the "
            "canopy fuel inputs most relevant to crown fire propagation."
        ),
    },
    "cover_only": {
        "value": EXAMPLE_LANDFIRE_CANOPY_COVER_ONLY,
        "summary": "Canopy cover only",
        "description": (
            "Returns just the canopy cover band (percent), useful for "
            "horizontal masking and overstory-vs-understory partitioning."
        ),
    },
    "with_metadata": {
        "value": EXAMPLE_LANDFIRE_CANOPY_WITH_METADATA,
        "summary": "With name and tags",
        "description": (
            "Creates a named grid with all four canopy bands and tags for organization."
        ),
    },
    "with_buffer": {
        "value": EXAMPLE_LANDFIRE_CANOPY_WITH_BUFFER,
        "summary": "With output buffer",
        "description": (
            "Includes 6 result-grid cells of buffer beyond the domain extent. "
            "Useful when downstream resampling or focal operations need "
            "context past the domain edge."
        ),
    },
    "native_alignment": {
        "value": EXAMPLE_LANDFIRE_CANOPY_NATIVE_ALIGNMENT,
        "summary": "Preserve the native LANDFIRE pixel anchor",
        "description": (
            'Sets `alignment.target="native"` so the output keeps the '
            "LANDFIRE source raster's pixel anchor instead of snapping to "
            "the domain origin. Choose this when faithful representation "
            "of LANDFIRE cell positions matters more than composing with "
            "other domain-aligned grids — e.g. for cross-version LANDFIRE "
            "comparisons or to minimize resampling artifacts in the "
            "canopy bands."
        ),
    },
}

ALL_LANDFIRE_CANOPY_EXAMPLE_VALUES = [
    ("minimal", EXAMPLE_LANDFIRE_CANOPY_MINIMAL),
    ("crown_fire_inputs", EXAMPLE_LANDFIRE_CANOPY_CROWN_FIRE_INPUTS),
    ("cover_only", EXAMPLE_LANDFIRE_CANOPY_COVER_ONLY),
    ("with_metadata", EXAMPLE_LANDFIRE_CANOPY_WITH_METADATA),
    ("with_buffer", EXAMPLE_LANDFIRE_CANOPY_WITH_BUFFER),
    ("native_alignment", EXAMPLE_LANDFIRE_CANOPY_NATIVE_ALIGNMENT),
]
