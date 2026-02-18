"""
Example request bodies for grid export endpoints.

These examples are used in:
1. OpenAPI/Swagger documentation - Users see these as example payloads
2. Integration tests - Each example is tested to ensure documentation stays accurate
"""

# Domain-level multi-grid endpoint: POST /domains/{domain_id}/grids/exports/geotiff

EXAMPLE_GEOTIFF_ALL_BANDS = {
    "grid_ids": ["abc123def456"],
    "name": "Surface fuels GeoTIFF",
    "tags": ["surface-fuel"],
}

EXAMPLE_GEOTIFF_MULTI_GRID = {
    "grid_ids": ["abc123def456", "ghi789jkl012"],
    "name": "Surface fuels + topography",
    "description": "Combined export of surface fuel and terrain grids",
    "tags": ["combined"],
}

EXAMPLE_GEOTIFF_BAND_SUBSET = {
    "grid_ids": ["abc123def456"],
    "bands": ["fuel_load.1hr", "fuel_load.10hr", "fuel_depth"],
    "name": "Fuel loads only",
    "description": "Subset of surface fuel bands for analysis",
    "tags": ["surface-fuel", "analysis"],
}

EXAMPLE_GEOTIFF_MINIMAL = {
    "grid_ids": ["abc123def456"],
}

CREATE_GEOTIFF_OPENAPI_EXAMPLES = {
    "all_bands": {
        "value": EXAMPLE_GEOTIFF_ALL_BANDS,
        "summary": "Export all bands from one grid",
        "description": ("Exports all bands from a single grid to a GeoTIFF file."),
    },
    "multi_grid": {
        "value": EXAMPLE_GEOTIFF_MULTI_GRID,
        "summary": "Export multiple grids",
        "description": (
            "Exports bands from multiple grids into a single GeoTIFF file."
        ),
    },
    "band_subset": {
        "value": EXAMPLE_GEOTIFF_BAND_SUBSET,
        "summary": "Export specific bands",
        "description": (
            "Exports only the specified bands from the grid. "
            "Useful for reducing file size when only certain quantities are needed."
        ),
    },
    "minimal": {
        "value": EXAMPLE_GEOTIFF_MINIMAL,
        "summary": "Minimal request",
        "description": (
            "Exports all bands with no name or tags. "
            "Only the grid_ids list is required."
        ),
    },
}

ALL_GEOTIFF_EXAMPLE_VALUES = [
    ("all_bands", EXAMPLE_GEOTIFF_ALL_BANDS),
    ("multi_grid", EXAMPLE_GEOTIFF_MULTI_GRID),
    ("band_subset", EXAMPLE_GEOTIFF_BAND_SUBSET),
    ("minimal", EXAMPLE_GEOTIFF_MINIMAL),
]


# Per-grid endpoint: POST /domains/{domain_id}/grids/{grid_id}/exports/geotiff

EXAMPLE_SINGLE_GRID_ALL_BANDS = {
    "name": "Surface fuels GeoTIFF",
    "tags": ["surface-fuel"],
}

EXAMPLE_SINGLE_GRID_BAND_SUBSET = {
    "bands": ["fuel_load.1hr", "fuel_load.10hr", "fuel_depth"],
    "name": "Fuel loads only",
    "description": "Subset of surface fuel bands for analysis",
    "tags": ["surface-fuel", "analysis"],
}

EXAMPLE_SINGLE_GRID_MINIMAL = {}

CREATE_SINGLE_GRID_GEOTIFF_OPENAPI_EXAMPLES = {
    "all_bands": {
        "value": EXAMPLE_SINGLE_GRID_ALL_BANDS,
        "summary": "Export all bands",
        "description": ("Exports all bands from the grid to a GeoTIFF file."),
    },
    "band_subset": {
        "value": EXAMPLE_SINGLE_GRID_BAND_SUBSET,
        "summary": "Export specific bands",
        "description": ("Exports only the specified bands from the grid."),
    },
    "minimal": {
        "value": EXAMPLE_SINGLE_GRID_MINIMAL,
        "summary": "Minimal request",
        "description": (
            "Exports all bands with no name or tags. The request body can be empty."
        ),
    },
}

ALL_SINGLE_GRID_GEOTIFF_EXAMPLE_VALUES = [
    ("all_bands", EXAMPLE_SINGLE_GRID_ALL_BANDS),
    ("band_subset", EXAMPLE_SINGLE_GRID_BAND_SUBSET),
    ("minimal", EXAMPLE_SINGLE_GRID_MINIMAL),
]
