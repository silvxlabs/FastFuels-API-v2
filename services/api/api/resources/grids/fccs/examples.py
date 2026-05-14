"""
Example request bodies for FCCS endpoints.

These examples are used in:
1. OpenAPI/Swagger documentation - Users see these as example payloads
2. Integration tests - Each example is tested to ensure documentation stays accurate

domain_id comes from the URL path parameter, not the request body.
"""

EXAMPLE_FCCS_MINIMAL = {}

EXAMPLE_FCCS_WITH_METADATA = {
    "name": "FCCS fuelbed IDs",
    "description": "Baseline fuelbed IDs for scenario comparison",
    "tags": ["baseline", "surface-fuel"],
    "version": "2023",
}

EXAMPLE_FCCS_REMOVE_BARE_GROUND = {
    "name": "FCCS with bare ground removal",
    "remove_bare_ground": True,
}

CREATE_LANDFIRE_FCCS_OPENAPI_EXAMPLES = {
    "minimal": {
        "value": EXAMPLE_FCCS_MINIMAL,
        "summary": "Minimal request",
        "description": (
            "Creates a grid with default settings. Returns FCCS fuelbed IDs "
            "at 30m resolution. Use /grids/lookup/fccs to convert codes to "
            "fuel parameters."
        ),
    },
    "with_metadata": {
        "value": EXAMPLE_FCCS_WITH_METADATA,
        "summary": "With name and tags",
        "description": (
            "Creates a named grid with tags for organization. Useful when "
            "maintaining multiple grids for scenario comparison."
        ),
    },
    "remove_bare_ground": {
        "value": EXAMPLE_FCCS_REMOVE_BARE_GROUND,
        "summary": "Remove bare ground blocks",
        "description": (
            "Removes coarse 30m-resolution bare ground blocks by replacing "
            "them with the most frequent neighboring burnable fuel model. "
            "This clears the way for masking in high-resolution 1-2m road, "
            "water, and building features from sources such as OpenStreetMap."
        ),
    },
}

ALL_FCCS_EXAMPLE_VALUES = [
    ("minimal", EXAMPLE_FCCS_MINIMAL),
    ("with_metadata", EXAMPLE_FCCS_WITH_METADATA),
    ("remove_bare_ground", EXAMPLE_FCCS_REMOVE_BARE_GROUND),
]
