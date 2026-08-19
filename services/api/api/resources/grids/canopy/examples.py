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


EXAMPLE_POINT_CLOUD_CHM_MINIMAL = {
    "source_point_cloud_id": "8fc4dcad181944fd9cb594af32b58432",
}

EXAMPLE_POINT_CLOUD_CHM_NAMED = {
    "source_point_cloud_id": "8fc4dcad181944fd9cb594af32b58432",
    "name": "Blackfoot ALS canopy height",
    "description": "1 m CHM rasterized from the 3DEP point cloud.",
    "tags": ["als", "chm"],
}

EXAMPLE_POINT_CLOUD_CHM_5M = {
    "source_point_cloud_id": "8fc4dcad181944fd9cb594af32b58432",
    "alignment": {"target": "domain", "resolution": 5.0},
}

EXAMPLE_POINT_CLOUD_CHM_10M = {
    "source_point_cloud_id": "8fc4dcad181944fd9cb594af32b58432",
    "alignment": {"target": "domain", "resolution": 10.0},
}

EXAMPLE_POINT_CLOUD_CHM_30M = {
    "source_point_cloud_id": "8fc4dcad181944fd9cb594af32b58432",
    "alignment": {"target": "domain", "resolution": 30.0},
}

EXAMPLE_POINT_CLOUD_CHM_ALIGNED_TO_GRID = {
    "source_point_cloud_id": "8fc4dcad181944fd9cb594af32b58432",
    "alignment": {"target": "grid", "grid_id": "REPLACE_WITH_GRID_ID"},
}

EXAMPLE_POINT_CLOUD_CHM_ALIGNED_TO_GRID_AT_1M = {
    "source_point_cloud_id": "8fc4dcad181944fd9cb594af32b58432",
    "alignment": {
        "target": "grid",
        "grid_id": "REPLACE_WITH_GRID_ID",
        "resolution": 1.0,
    },
}

EXAMPLE_POINT_CLOUD_CHM_NO_SPIKE_FILTER = {
    "source_point_cloud_id": "8fc4dcad181944fd9cb594af32b58432",
    "spike_filter": None,
}

EXAMPLE_POINT_CLOUD_CHM_SPIKE_FILTER = {
    "source_point_cloud_id": "8fc4dcad181944fd9cb594af32b58432",
    "spike_filter": {"min_canopy_footprint_m": 5.0, "min_prominence_m": 15.0},
}

CREATE_POINT_CLOUD_CHM_OPENAPI_EXAMPLES: dict = {
    "minimal": {
        "value": EXAMPLE_POINT_CLOUD_CHM_MINIMAL,
        "summary": "1 m cells (the default)",
        "description": (
            "The only required field is the point cloud to rasterize. Each "
            "cell holds the height above ground of the tallest return that "
            "falls in it, in meters.\n\n"
            "Cells are 1 m unless you ask for something else. The point cloud "
            "must be an airborne (`als`) cloud in this domain, with status "
            "`completed`."
        ),
    },
    "named": {
        "value": EXAMPLE_POINT_CLOUD_CHM_NAMED,
        "summary": "With metadata",
        "description": "Name, description, and tags for organizing grids.",
    },
    "resolution_5m": {
        "value": EXAMPLE_POINT_CLOUD_CHM_5M,
        "summary": "5 m cells",
        "description": (
            "`alignment.resolution` sets the cell size, in meters. Larger "
            "cells give a smaller grid that smooths over individual tree "
            "crowns. The smallest cell size accepted is 1 m."
        ),
    },
    "resolution_10m": {
        "value": EXAMPLE_POINT_CLOUD_CHM_10M,
        "summary": "10 m cells",
        "description": (
            "At 10 m and coarser the grid describes the height of the canopy "
            "as a whole rather than of individual trees."
        ),
    },
    "resolution_30m": {
        "value": EXAMPLE_POINT_CLOUD_CHM_30M,
        "summary": "30 m cells, matching LANDFIRE",
        "description": (
            "30 m is the cell size of the LANDFIRE canopy products. Note that "
            "the same cell size does not by itself put two grids on the same "
            "cells — to line up with a particular LANDFIRE grid, align to it "
            "by id instead (see the next two examples)."
        ),
    },
    "aligned_to_existing_grid": {
        "value": EXAMPLE_POINT_CLOUD_CHM_ALIGNED_TO_GRID,
        "summary": "Match an existing grid exactly",
        "description": (
            "Naming another grid in this domain produces a CHM on exactly "
            "that grid's cells — same origin, same cell size, same shape — so "
            "the two can be composed or exported together without "
            "resampling.\n\n"
            "The target must be a completed grid in this domain, using this "
            "domain's CRS."
        ),
    },
    "aligned_to_existing_grid_at_1m": {
        "value": EXAMPLE_POINT_CLOUD_CHM_ALIGNED_TO_GRID_AT_1M,
        "summary": "Match an existing grid, at your own cell size",
        "description": (
            "Adding `resolution` keeps the target grid's position and extent "
            "but uses the cell size you give. Use this for a finer CHM that "
            "still lines up with a coarser grid — for example 1 m canopy "
            "heights on the cells of a 30 m LANDFIRE grid."
        ),
    },
    "spike_filter": {
        "value": EXAMPLE_POINT_CLOUD_CHM_SPIKE_FILTER,
        "summary": "Tune the removal of spurious returns",
        "description": (
            "A cell holds the tallest return that falls in it, so one bad "
            "return — a bird, haze — becomes the height of that cell unless "
            "the cloud classified it as noise, and many clouds do not. Such a "
            "return leaves a shape real canopy cannot: a single cell towering "
            "over everything around it.\n\n"
            "`min_canopy_footprint_m` is the narrowest ground footprint real "
            "canopy can occupy. A cell is judged against everything within "
            "that distance, and only a cell narrower than it can be rejected — "
            "so the filter does not run once `alignment.resolution` reaches "
            "this value, where one cell holds a stand rather than a crown. "
            "`min_prominence_m` is how far above every neighbour a cell must "
            "rise. Both are in meters and default to 3 and 25."
        ),
    },
    "no_spike_filter": {
        "value": EXAMPLE_POINT_CLOUD_CHM_NO_SPIKE_FILTER,
        "summary": "Keep every return",
        "description": (
            "`null` turns the filter off, so nothing is removed from the "
            "finished grid. Use it when the cloud's noise is already "
            "classified, or when you would rather inspect the raw heights."
        ),
    },
}

POINT_CLOUD_CHM_EXAMPLE_VALUES = [
    ("spike_filter", EXAMPLE_POINT_CLOUD_CHM_SPIKE_FILTER),
    ("no_spike_filter", EXAMPLE_POINT_CLOUD_CHM_NO_SPIKE_FILTER),
    ("minimal", EXAMPLE_POINT_CLOUD_CHM_MINIMAL),
    ("named", EXAMPLE_POINT_CLOUD_CHM_NAMED),
    ("resolution_5m", EXAMPLE_POINT_CLOUD_CHM_5M),
    ("resolution_10m", EXAMPLE_POINT_CLOUD_CHM_10M),
    ("resolution_30m", EXAMPLE_POINT_CLOUD_CHM_30M),
    ("aligned_to_existing_grid", EXAMPLE_POINT_CLOUD_CHM_ALIGNED_TO_GRID),
]
