"""
api/v2/resources/features/crown/examples.py

Example request bodies for Crown feature endpoints.

These examples are used in:
1. OpenAPI/Swagger documentation - Users see these as example payloads
2. Integration tests - Each example is tested to ensure documentation stays accurate

domain_id comes from the URL path parameter, not the request body. The
placeholder source IDs are replaced with real IDs in tests.
"""

EXAMPLE_CROWN_MINIMAL = {
    "source_inventory_id": "PLACEHOLDER_INVENTORY_ID",
    "source_chm_grid_id": "PLACEHOLDER_CHM_GRID_ID",
}

EXAMPLE_CROWN_WITH_SEGMENTATION = {
    "name": "Hitchiti crowns",
    "source_inventory_id": "PLACEHOLDER_INVENTORY_ID",
    "source_chm_grid_id": "PLACEHOLDER_CHM_GRID_ID",
    "crown_segmentation": {
        "method": "dalponte2016",
        "min_height": 2.0,
        "max_height": 120.0,
        "min_relative_height": 0.45,
        "min_relative_crown_height": 0.55,
        "max_crown_radius": 10.0,
    },
}

CREATE_CROWN_OPENAPI_EXAMPLES = {
    "minimal": {
        "value": EXAMPLE_CROWN_MINIMAL,
        "summary": "Minimal request",
        "description": (
            "Grows one crown per inventory tree on the CHM with the default "
            "segmentation settings."
        ),
    },
    "with_segmentation": {
        "value": EXAMPLE_CROWN_WITH_SEGMENTATION,
        "summary": "With segmentation settings",
        "description": "Sets every segmentation parameter explicitly.",
    },
}

CROWN_EXAMPLE_VALUES = [
    ("minimal", EXAMPLE_CROWN_MINIMAL),
    ("with_segmentation", EXAMPLE_CROWN_WITH_SEGMENTATION),
]
