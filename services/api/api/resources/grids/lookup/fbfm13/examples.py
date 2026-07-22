"""
Example request bodies for the FBFM13 lookup endpoint.

These examples are used in:
1. OpenAPI/Swagger documentation - Users see these as example payloads
2. Integration tests - Each example is tested to ensure documentation stays accurate

All examples assume a valid source_grid_id exists. The domain_id is
propagated from the source grid automatically.
Replace placeholder IDs with actual values when testing.
"""

EXAMPLE_FBFM13_LOOKUP_MINIMAL = {
    "source_grid_id": "grid_abc123",
    "bands": ["fuel_load.1hr", "fuel_depth"],
}

EXAMPLE_FBFM13_LOOKUP_ALL_BANDS = {
    "source_grid_id": "grid_abc123",
    "name": "Surface fuels from FBFM13",
    "description": "All Anderson 13 fuel parameters for baseline scenario",
    "tags": ["baseline", "surface-fuel"],
    "bands": [
        "fuel_load.1hr",
        "fuel_load.10hr",
        "fuel_load.100hr",
        "fuel_load.live_foliage",
        "savr.1hr",
        "savr.10hr",
        "savr.100hr",
        "savr.live_foliage",
        "fuel_depth",
    ],
}

CREATE_FBFM13_LOOKUP_OPENAPI_EXAMPLES = {
    "minimal": {
        "value": EXAMPLE_FBFM13_LOOKUP_MINIMAL,
        "summary": "Minimal request",
        "description": (
            "Looks up just 1-hr fuel load and fuel depth from an FBFM13 "
            "source grid. The source grid must have status 'completed' and "
            "contain an 'fbfm13' band."
        ),
    },
    "all_bands": {
        "value": EXAMPLE_FBFM13_LOOKUP_ALL_BANDS,
        "summary": "All fuel bands",
        "description": (
            "Looks up all available Anderson 13 fuel parameters: fuel loads "
            "(4 size classes), SAV ratios (4 size classes), and fuel depth."
        ),
    },
}

ALL_FBFM13_LOOKUP_EXAMPLE_VALUES = [
    ("minimal", EXAMPLE_FBFM13_LOOKUP_MINIMAL),
    ("all_bands", EXAMPLE_FBFM13_LOOKUP_ALL_BANDS),
]
