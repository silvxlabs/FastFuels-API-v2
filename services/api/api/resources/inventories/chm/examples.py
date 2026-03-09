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

# Full request explicitly defining the LMF algorithm parameters and metadata
EXAMPLE_CHM_FULL = {
    "source_chm_grid_id": "PLACEHOLDER_GRID_ID",
    "algorithm": {
        "name": "lmf",
        "min_height": 2.5,
        "footprint_size": 5,
    },
    "type": "tree",
    "name": "CHM extraction inventory",
    "description": "Tree inventory extracted from high-res NAIP CHM",
    "tags": ["lidar", "baseline"],
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
        "summary": "Minimal request",
        "description": (
            "Creates an inventory from a CHM grid using default Local Maximum "
            "Filtering (LMF) parameters (2m min_height, 3px footprint)."
        ),
    },
    "full": {
        "value": EXAMPLE_CHM_FULL,
        "summary": "Full request with custom algorithm parameters",
        "description": (
            "Creates an inventory from a specific CHM grid with a custom "
            "LMF configuration and all optional metadata fields."
        ),
    },
    "with_modifications": {
        "value": EXAMPLE_CHM_WITH_MODIFICATIONS,
        "summary": "With modifications (remove short trees)",
        "description": (
            "Creates an inventory from a CHM grid and subsequently removes "
            "any trees with height < 5.0 meters."
        ),
    },
}

ALL_CHM_EXAMPLE_VALUES = [
    ("minimal", EXAMPLE_CHM_MINIMAL),
    ("full", EXAMPLE_CHM_FULL),
    ("with_modifications", EXAMPLE_CHM_WITH_MODIFICATIONS),
]
