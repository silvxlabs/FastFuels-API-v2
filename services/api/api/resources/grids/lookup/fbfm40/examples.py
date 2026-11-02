"""
Example request bodies for the FBFM40 lookup endpoint.

These examples are used in:
1. OpenAPI/Swagger documentation - Users see these as example payloads
2. Integration tests - Each example is tested to ensure documentation stays accurate

All examples assume a valid source_grid_id exists. The domain_id is
propagated from the source grid automatically.
Replace placeholder IDs with actual values when testing.
"""

EXAMPLE_FBFM40_LOOKUP_MINIMAL = {
    "source_grid_id": "grid_abc123",
    "bands": ["fuel_load.1hr", "fuel_depth"],
}

EXAMPLE_FBFM40_LOOKUP_ALL_BANDS = {
    "source_grid_id": "grid_abc123",
    "name": "Surface fuels from FBFM40",
    "description": "All SB40 fuel parameters for baseline scenario",
    "tags": ["baseline", "surface-fuel"],
    "bands": [
        "fuel_load.1hr",
        "fuel_load.10hr",
        "fuel_load.100hr",
        "fuel_load.live_herb",
        "fuel_load.live_woody",
        "fuel_load.dead_herb",
        "fuel_load.fine_dead",
        "savr.1hr",
        "savr.10hr",
        "savr.100hr",
        "savr.live_herb",
        "savr.live_woody",
        "savr.dead_herb",
        "savr.fine_dead",
        "fuel_depth",
    ],
}

EXAMPLE_FBFM40_LOOKUP_FINE_DEAD = {
    "source_grid_id": "grid_abc123",
    "name": "Fully cured surface fuels",
    "herbaceous_curing": 1.0,
    "bands": ["fuel_load.fine_dead", "savr.fine_dead", "fuel_depth"],
}

CREATE_FBFM40_LOOKUP_OPENAPI_EXAMPLES = {
    "minimal": {
        "value": EXAMPLE_FBFM40_LOOKUP_MINIMAL,
        "summary": "Minimal request",
        "description": (
            "Looks up just 1-hr fuel load and fuel depth from an FBFM40 "
            "source grid. The source grid must have status 'completed' and "
            "contain an 'fbfm' band."
        ),
    },
    "all_bands": {
        "value": EXAMPLE_FBFM40_LOOKUP_ALL_BANDS,
        "summary": "All fuel bands",
        "description": (
            "Looks up all available SB40 fuel parameters: fuel loads and SAV "
            "ratios for every fuel class, including dead herbaceous and fine "
            "dead fuel at the default curing, and fuel depth."
        ),
    },
    "fine_dead": {
        "value": EXAMPLE_FBFM40_LOOKUP_FINE_DEAD,
        "summary": "Fine dead fuel, fully cured",
        "description": (
            "Looks up 1-hr plus dead herbaceous fuel as a single fine dead "
            "fuel class, with all herbaceous fuel cured. Suits models with one "
            "surface fuel class, such as the QUIC-Fire export."
        ),
    },
}

ALL_FBFM40_LOOKUP_EXAMPLE_VALUES = [
    ("minimal", EXAMPLE_FBFM40_LOOKUP_MINIMAL),
    ("all_bands", EXAMPLE_FBFM40_LOOKUP_ALL_BANDS),
    ("fine_dead", EXAMPLE_FBFM40_LOOKUP_FINE_DEAD),
]
