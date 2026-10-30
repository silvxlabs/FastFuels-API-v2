"""
Example request bodies for crown base height inventory endpoints.

These examples are used in:
1. OpenAPI/Swagger documentation - Users see these as example payloads
2. Integration tests - Each example is tested to ensure documentation stays accurate

domain_id comes from the URL path parameter, not the request body.
"""

# Minimal request with only the required field
EXAMPLE_CROWN_BASE_HEIGHT_MINIMAL = {
    "source_tree_inventory_id": "PLACEHOLDER_INVENTORY_ID",
}

# Full request with all optional fields
EXAMPLE_CROWN_BASE_HEIGHT_FULL = {
    "source_tree_inventory_id": "PLACEHOLDER_INVENTORY_ID",
    "model": {"name": "toney_reeves_2009"},
    "overwrite": True,
    "name": "Crown base height inventory",
    "description": "Uploaded inventory with crown base heights from Toney and Reeves",
    "tags": ["upload"],
}

CREATE_CROWN_BASE_HEIGHT_OPENAPI_EXAMPLES = {
    "minimal": {
        "value": EXAMPLE_CROWN_BASE_HEIGHT_MINIMAL,
        "summary": "Minimal request",
        "description": (
            "Fills each empty crown base height of a completed tree inventory "
            "from its species, dbh, crown ratio and height with the default "
            "model. Existing crown base heights are kept."
        ),
    },
    "full": {
        "value": EXAMPLE_CROWN_BASE_HEIGHT_FULL,
        "summary": "Full request with all options",
        "description": (
            "Recomputes every tree's crown base height with the Toney and "
            "Reeves (2009) model, with the inventory metadata (name, "
            "description, tags) specified."
        ),
    },
}

ALL_CROWN_BASE_HEIGHT_EXAMPLE_VALUES = [
    ("minimal", EXAMPLE_CROWN_BASE_HEIGHT_MINIMAL),
    ("full", EXAMPLE_CROWN_BASE_HEIGHT_FULL),
]
