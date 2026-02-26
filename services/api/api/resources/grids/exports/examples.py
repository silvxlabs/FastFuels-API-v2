"""
Example request bodies for grid export endpoints.

These examples are used in:
1. OpenAPI/Swagger documentation - Users see these as example payloads
2. Integration tests - Each example is tested to ensure documentation stays accurate
"""

# Per-grid endpoint: POST /domains/{domain_id}/grids/{grid_id}/exports/{format}

EXAMPLE_GRID_EXPORT_ALL_BANDS = {
    "name": "Surface fuels export",
    "tags": ["surface-fuel"],
}

EXAMPLE_GRID_EXPORT_BAND_SUBSET = {
    "bands": ["fuel_load.1hr", "fuel_load.10hr", "fuel_depth"],
    "name": "Fuel loads only",
    "description": "Subset of surface fuel bands for analysis",
    "tags": ["surface-fuel", "analysis"],
}

EXAMPLE_GRID_EXPORT_MINIMAL = {}

CREATE_GRID_EXPORT_OPENAPI_EXAMPLES = {
    "all_bands": {
        "value": EXAMPLE_GRID_EXPORT_ALL_BANDS,
        "summary": "Export all bands",
        "description": "Exports all bands from the grid.",
    },
    "band_subset": {
        "value": EXAMPLE_GRID_EXPORT_BAND_SUBSET,
        "summary": "Export specific bands",
        "description": "Exports only the specified bands from the grid.",
    },
    "minimal": {
        "value": EXAMPLE_GRID_EXPORT_MINIMAL,
        "summary": "Minimal request",
        "description": (
            "Exports all bands with no name or tags. The request body can be empty."
        ),
    },
}

ALL_GRID_EXPORT_EXAMPLE_VALUES = [
    ("all_bands", EXAMPLE_GRID_EXPORT_ALL_BANDS),
    ("band_subset", EXAMPLE_GRID_EXPORT_BAND_SUBSET),
    ("minimal", EXAMPLE_GRID_EXPORT_MINIMAL),
]
