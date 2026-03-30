"""
api/v2/resources/inventories/chm/examples.py

Example request bodies for CHM inventory endpoints.

These examples are used in:
1. OpenAPI/Swagger documentation - Users see these as example payloads
2. Integration tests - Each example is tested to ensure documentation stays accurate

domain_id comes from the URL path parameter, not the request body.
"""

# Minimal request with only the required field (uses default LMF parameters)
EXAMPLE_CHM_MINIMAL = {
    "source_chm_grid_id": "PLACEHOLDER_GRID_ID",
}

# Explicitly defining the Local Maximum Filter (LMF) algorithm parameters and metadata
EXAMPLE_CHM_LMF_FULL = {
    "source_chm_grid_id": "PLACEHOLDER_GRID_ID",
    "algorithm": {
        "name": "lmf",
        "min_height": 2.5,
        "footprint_size": 5,
    },
    "type": "tree",
    "name": "LMF CHM extraction inventory",
    "description": "Tree inventory extracted from high-res NAIP CHM using fixed-window LMF.",
}

# Explicitly defining the Variable Window Filter (VWF) algorithm parameters
EXAMPLE_CHM_VWF = {
    "source_chm_grid_id": "PLACEHOLDER_GRID_ID",
    "algorithm": {
        "name": "vwf",
        "min_height": 3.0,
        "crown_ratio": 0.15,
        "crown_offset": 1.0,
    },
    "type": "tree",
    "name": "VWF CHM extraction inventory",
    "description": "Tree inventory using dynamic Variable Window Filtering for mixed canopy structures.",
}

# Request with modifications (remove trees under 5m)
EXAMPLE_CHM_WITH_MODIFICATIONS = {
    "source_chm_grid_id": "PLACEHOLDER_GRID_ID",
    "algorithm": {
        "name": "lmf",
        "min_height": 2.0,
        "footprint_size": 3,
    },
    "name": "CHM inventory with strict height filter",
    "modifications": [
        {
            "conditions": {
                "attribute": "height",
                "operator": "lt",
                "value": 5.0,
            },
            "actions": {"modifier": "remove"},
        }
    ],
}

CREATE_CHM_OPENAPI_EXAMPLES = {
    "minimal": {
        "value": EXAMPLE_CHM_MINIMAL,
        "summary": "Minimal request (Defaults to LMF)",
        "description": (
            "Creates an inventory from a CHM grid using default Local Maximum "
            "Filtering (LMF) parameters (2m min_height, 3px footprint)."
        ),
    },
    "lmf_full": {
        "value": EXAMPLE_CHM_LMF_FULL,
        "summary": "Custom LMF extraction",
        "description": (
            "Creates an inventory from a specific CHM grid with a custom "
            "fixed-window LMF configuration and optional metadata fields."
        ),
    },
    "vwf_extraction": {
        "value": EXAMPLE_CHM_VWF,
        "summary": "Custom VWF extraction (Dynamic Window)",
        "description": (
            "Creates an inventory using the Variable Window Filter (VWF) algorithm. "
            "This algorithm dynamically scales the search window based on pixel height "
            "using the crown_ratio and crown_offset parameters."
        ),
    },
    "with_modifications": {
        "value": EXAMPLE_CHM_WITH_MODIFICATIONS,
        "summary": "With modifications (remove short trees)",
        "description": (
            "Creates an inventory from a CHM grid and subsequently removes "
            "any trees with height < 5.0 meters using the post-processing modifications array."
        ),
    },
}

ALL_CHM_EXAMPLE_VALUES = [
    ("minimal", EXAMPLE_CHM_MINIMAL),
    ("lmf_full", EXAMPLE_CHM_LMF_FULL),
    ("vwf_extraction", EXAMPLE_CHM_VWF),
    ("with_modifications", EXAMPLE_CHM_WITH_MODIFICATIONS),
]
