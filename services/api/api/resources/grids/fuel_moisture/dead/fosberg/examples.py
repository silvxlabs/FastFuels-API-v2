"""
Example request bodies for the Fosberg 1-hr DFMC endpoint.

These examples are used in:
1. OpenAPI/Swagger documentation - Users see these as example payloads.
2. Integration tests - Each example is tested to keep the docs accurate.

The grid-id placeholders (TOPOGRAPHY_GRID_ID / IRRADIANCE_GRID_ID) are
substituted with real fixture grids in the router tests.
"""

EXAMPLE_FOSBERG_MINIMAL = {
    "source_topography_grid_id": "TOPOGRAPHY_GRID_ID",
    "source_irradiance_grid_id": "IRRADIANCE_GRID_ID",
    "dry_bulb_temp": 75,
    "relative_humidity": 30,
    "time": 1200,
    "month": "June",
}

EXAMPLE_FOSBERG_FULL = {
    "name": "Peak burn-period 1-hr DFMC",
    "description": "Midday June dead fuel moisture for the burn window.",
    "tags": ["fuel-moisture", "surface-fuel"],
    "source_topography_grid_id": "TOPOGRAPHY_GRID_ID",
    "source_irradiance_grid_id": "IRRADIANCE_GRID_ID",
    "dry_bulb_temp": 82,
    "relative_humidity": 20,
    "time": 1400,
    "month": "August",
    "elevation": "above",
}

CREATE_FOSBERG_FUEL_MOISTURE_OPENAPI_EXAMPLES = {
    "minimal": {
        "value": EXAMPLE_FOSBERG_MINIMAL,
        "summary": "Minimal request",
        "description": (
            "Computes midday June 1-hr dead fuel moisture from a topography "
            "grid (slope + aspect) and a leaflux irradiance grid. Elevation "
            "defaults to `near` (no station correction)."
        ),
    },
    "full": {
        "value": EXAMPLE_FOSBERG_FULL,
        "summary": "With metadata and elevation correction",
        "description": (
            "A named, tagged grid for a hot, dry August afternoon on a site "
            "above the reference weather station."
        ),
    },
}

ALL_FOSBERG_EXAMPLE_VALUES = [
    ("minimal", EXAMPLE_FOSBERG_MINIMAL),
    ("full", EXAMPLE_FOSBERG_FULL),
]
