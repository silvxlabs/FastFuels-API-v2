"""
Example request bodies for GDAM allometry inventory endpoints.

These examples are used in:
1. OpenAPI/Swagger documentation - Users see these as example payloads
2. Integration tests - Each example is tested to ensure documentation stays accurate

domain_id comes from the URL path parameter, not the request body.
"""

# Minimal request with only the required field
EXAMPLE_GDAM_MINIMAL = {
    "source_tree_inventory_id": "PLACEHOLDER_INVENTORY_ID",
}

# Full request with all optional metadata fields
EXAMPLE_GDAM_FULL = {
    "source_tree_inventory_id": "PLACEHOLDER_INVENTORY_ID",
    "type": "tree",
    "name": "GDAM allometry inventory",
    "description": "Position+height inventory filled in with GDAM-predicted morphology",
    "tags": ["baseline"],
}

CREATE_GDAM_OPENAPI_EXAMPLES = {
    "minimal": {
        "value": EXAMPLE_GDAM_MINIMAL,
        "summary": "Minimal request",
        "description": (
            "Fills in a completed tree inventory's missing morphology columns "
            "(dbh, crown ratio, species) using GDAM. Existing values are "
            "preserved; only missing cells are imputed."
        ),
    },
    "full": {
        "value": EXAMPLE_GDAM_FULL,
        "summary": "Full request with all options",
        "description": (
            "Same as the minimal request, with inventory metadata "
            "(name, description, tags) specified."
        ),
    },
}

ALL_GDAM_EXAMPLE_VALUES = [
    ("minimal", EXAMPLE_GDAM_MINIMAL),
    ("full", EXAMPLE_GDAM_FULL),
]
