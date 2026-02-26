"""
Example request bodies for PIM inventory endpoints.

These examples are used in:
1. OpenAPI/Swagger documentation - Users see these as example payloads
2. Integration tests - Each example is tested to ensure documentation stays accurate

domain_id comes from the URL path parameter, not the request body.
"""

# Minimal request with only the required field
EXAMPLE_PIM_MINIMAL = {
    "source_pim_grid_id": "PLACEHOLDER_GRID_ID",
    "seed": 42,
}

# Full request with all optional fields
EXAMPLE_PIM_FULL = {
    "source_pim_grid_id": "PLACEHOLDER_GRID_ID",
    "seed": 12345,
    "point_process": "inhomogeneous_poisson",
    "type": "tree",
    "name": "PIM expansion inventory",
    "description": "Tree inventory from PIM grid expansion",
    "tags": ["baseline"],
}

# Request with modifications (remove small trees from microplot expansion)
EXAMPLE_PIM_WITH_MODIFICATIONS = {
    "source_pim_grid_id": "PLACEHOLDER_GRID_ID",
    "seed": 42,
    "name": "PIM inventory with small tree removal",
    "modifications": [
        {
            "conditions": {
                "attribute": "dbh",
                "operator": "le",
                "value": 12.7,
            },
            "actions": {"modifier": "remove"},
        }
    ],
}

CREATE_PIM_OPENAPI_EXAMPLES = {
    "minimal": {
        "value": EXAMPLE_PIM_MINIMAL,
        "summary": "Minimal request",
        "description": (
            "Creates an inventory from a PIM grid with a specific seed. "
            "Seed controls reproducibility; omit for a random seed."
        ),
    },
    "full": {
        "value": EXAMPLE_PIM_FULL,
        "summary": "Full request with all options",
        "description": (
            "Creates an inventory from a specific PIM grid with all "
            "optional fields specified."
        ),
    },
    "with_modifications": {
        "value": EXAMPLE_PIM_WITH_MODIFICATIONS,
        "summary": "With modifications (remove small trees)",
        "description": (
            "Creates an inventory from a PIM grid and removes trees "
            "with dbh <= 12.7 cm after expansion. This is a common "
            "fix for unrealistic microplot tree densities."
        ),
    },
}

ALL_PIM_EXAMPLE_VALUES = [
    ("minimal", EXAMPLE_PIM_MINIMAL),
    ("full", EXAMPLE_PIM_FULL),
    ("with_modifications", EXAMPLE_PIM_WITH_MODIFICATIONS),
]
