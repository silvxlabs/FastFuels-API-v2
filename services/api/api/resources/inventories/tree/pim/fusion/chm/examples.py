"""
Example request bodies for PIM-CHM fusion inventory endpoints.

These examples are used in:
1. OpenAPI/Swagger documentation - Users see these as example payloads
2. Integration tests - Each example is tested to ensure documentation stays accurate

domain_id comes from the URL path parameter, not the request body. The two grid
IDs are placeholders the router tests replace with real fixtures.
"""

# Minimal request: both sources plus a seed. Method defaults to reimputation.
EXAMPLE_PIM_CHM_MINIMAL = {
    "source_pim_grid_id": "PLACEHOLDER_PIM_GRID_ID",
    "source_chm_grid_id": "PLACEHOLDER_CHM_GRID_ID",
    "seed": 42,
}

# Full request with the reimputation method knobs and all metadata.
EXAMPLE_PIM_CHM_FULL = {
    "source_pim_grid_id": "PLACEHOLDER_PIM_GRID_ID",
    "source_chm_grid_id": "PLACEHOLDER_CHM_GRID_ID",
    "method": {
        "name": "reimputation",
        "resolution": 7.5,
        "min_height": 2.0,
        "cover_threshold": 0.2,
    },
    "point_process": "inhomogeneous_poisson",
    "seed": 12345,
    "type": "tree",
    "name": "Kaibab PIM x CHM inventory",
    "description": "TreeMap plots conditioned on ALS canopy cover",
    "tags": ["fusion"],
}

# Request with modifications (remove small trees from microplot expansion).
EXAMPLE_PIM_CHM_WITH_MODIFICATIONS = {
    "source_pim_grid_id": "PLACEHOLDER_PIM_GRID_ID",
    "source_chm_grid_id": "PLACEHOLDER_CHM_GRID_ID",
    "seed": 42,
    "name": "Fusion inventory with small tree removal",
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

CREATE_PIM_CHM_FUSION_OPENAPI_EXAMPLES = {
    "minimal": {
        "value": EXAMPLE_PIM_CHM_MINIMAL,
        "summary": "Minimal request",
        "description": (
            "Fuses a PIM with a CHM using the default reimputation method. "
            "Seed controls reproducibility; omit for a random seed."
        ),
    },
    "full": {
        "value": EXAMPLE_PIM_CHM_FULL,
        "summary": "Full request with reimputation knobs",
        "description": (
            "Sets the reimputation resolution, canopy-height threshold, and "
            "cover threshold explicitly, along with all optional metadata."
        ),
    },
    "with_modifications": {
        "value": EXAMPLE_PIM_CHM_WITH_MODIFICATIONS,
        "summary": "With modifications (remove small trees)",
        "description": (
            "Fuses a PIM with a CHM and removes trees with dbh <= 12.7 cm after "
            "expansion, a common fix for unrealistic microplot tree densities."
        ),
    },
}

ALL_PIM_CHM_FUSION_EXAMPLE_VALUES = [
    ("minimal", EXAMPLE_PIM_CHM_MINIMAL),
    ("full", EXAMPLE_PIM_CHM_FULL),
    ("with_modifications", EXAMPLE_PIM_CHM_WITH_MODIFICATIONS),
]
