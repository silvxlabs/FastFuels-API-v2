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

# Full request with all optional fields
EXAMPLE_GDAM_FULL = {
    "source_tree_inventory_id": "PLACEHOLDER_INVENTORY_ID",
    "impute_columns": ["dbh", "crown_ratio", "fia_species_code", "crown_base_height"],
    "overwrite": False,
    "crown_base_height_model": {"name": "fia_national_logit_2026"},
    "name": "GDAM allometry inventory",
    "description": "Position+height inventory filled in with GDAM-predicted morphology",
    "tags": ["baseline"],
}

# Re-predict crown ratio for every tree, ignoring the source's values, and
# recompute crown base height from the new crown ratios.
EXAMPLE_GDAM_OVERWRITE = {
    "source_tree_inventory_id": "PLACEHOLDER_INVENTORY_ID",
    "impute_columns": ["crown_ratio", "crown_base_height"],
    "overwrite": True,
    "crown_base_height_model": {"name": "toney_reeves_2009"},
}

CREATE_GDAM_OPENAPI_EXAMPLES = {
    "minimal": {
        "value": EXAMPLE_GDAM_MINIMAL,
        "summary": "Minimal request",
        "description": (
            "Fills in a completed tree inventory's empty morphology cells "
            "(dbh, crown ratio, species) using GDAM, then computes crown base "
            "height from them with the default model. Existing values are kept "
            "and given to GDAM."
        ),
    },
    "full": {
        "value": EXAMPLE_GDAM_FULL,
        "summary": "Full request with all options",
        "description": (
            "Same as the minimal request, with every option and the inventory "
            "metadata (name, description, tags) specified."
        ),
    },
    "overwrite": {
        "value": EXAMPLE_GDAM_OVERWRITE,
        "summary": "Replace crown ratio and recompute crown base height",
        "description": (
            "Replaces every tree's crown ratio with a GDAM prediction that does "
            "not see the old values, then recomputes every crown base height "
            "from the new crown ratios with the Toney and Reeves (2009) model. "
            "The source needs dbh and species, or add them to `impute_columns`."
        ),
    },
}

ALL_GDAM_EXAMPLE_VALUES = [
    ("minimal", EXAMPLE_GDAM_MINIMAL),
    ("full", EXAMPLE_GDAM_FULL),
    ("overwrite", EXAMPLE_GDAM_OVERWRITE),
]
