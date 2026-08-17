"""
Example request bodies for the FCCS lookup endpoint.

These examples are used in:
1. OpenAPI/Swagger documentation - Users see these as example payloads
2. Integration tests - Each example is tested to ensure documentation stays accurate

All examples assume a valid source_grid_id exists. The domain_id is
propagated from the source grid automatically.
Replace placeholder IDs with actual values when testing.
"""

EXAMPLE_FCCS_LOOKUP_MINIMAL = {
    "source_grid_id": "grid_abc123",
    "bands": ["fuel_load.1hr", "duff_depth"],
}

EXAMPLE_FCCS_LOOKUP_ALL_BANDS = {
    "source_grid_id": "grid_abc123",
    "name": "Surface fuels from FCCS",
    "description": "All FOFEM FCCS fuel parameters for baseline scenario",
    "tags": ["baseline", "surface-fuel"],
    "bands": [
        "fuel_load.litter",
        "fuel_load.duff",
        "duff_depth",
        "fuel_load.live_shrub",
        "fuel_load.live_herb",
        "fuel_load.1hr",
        "fuel_load.10hr",
        "fuel_load.100hr",
        "fuel_load.1000hr_sound",
        "fuel_load.1000hr_rotten",
        "fuel_load.live_foliage",
        "fuel_load.live_branch",
    ],
}

CREATE_FCCS_LOOKUP_OPENAPI_EXAMPLES = {
    "minimal": {
        "value": EXAMPLE_FCCS_LOOKUP_MINIMAL,
        "summary": "Minimal request",
        "description": (
            "Looks up just 1-hr fuel load and duff depth from an FCCS "
            "source grid. The source grid must have status 'completed' and "
            "contain an 'fccs' band."
        ),
    },
    "all_bands": {
        "value": EXAMPLE_FCCS_LOOKUP_ALL_BANDS,
        "summary": "All fuel bands",
        "description": (
            "Looks up all 12 FOFEM FCCS fuel parameters: ground fuels "
            "(litter, duff, duff depth), live surface fuels (shrub, herb), "
            "dead fuels (1hr-1000hr), and live crown fuels (foliage, "
            "branch)."
        ),
    },
}

ALL_FCCS_LOOKUP_EXAMPLE_VALUES = [
    ("minimal", EXAMPLE_FCCS_LOOKUP_MINIMAL),
    ("all_bands", EXAMPLE_FCCS_LOOKUP_ALL_BANDS),
]
