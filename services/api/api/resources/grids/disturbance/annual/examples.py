"""
Example request bodies for LANDFIRE Limited Annual Disturbance endpoints.

These examples are used in:
1. OpenAPI/Swagger documentation - Users see these as example payloads
2. Integration tests - Each example is tested to ensure documentation stays accurate

domain_id comes from the URL path parameter, not the request body.

Every example here is fetched on demand from LANDFIRE Product Service.
"""

EXAMPLE_DISTURBANCE_MINIMAL = {}

EXAMPLE_DISTURBANCE_WITH_METADATA = {
    "name": "Limited Annual Disturbance codes",
    "description": "Recent disturbance codes for scenario comparison",
    "tags": ["baseline", "disturbance"],
}


CREATE_LANDFIRE_DISTURBANCE_OPENAPI_EXAMPLES = {
    "minimal": {
        "value": EXAMPLE_DISTURBANCE_MINIMAL,
        "summary": "Minimal request",
        "description": (
            "Creates a grid with default settings. Returns LANDFIRE Limited "
            "Annual Disturbance codes at 30m resolution, fetched on demand "
            "from LANDFIRE Product Service."
        ),
    },
    "with_metadata": {
        "value": EXAMPLE_DISTURBANCE_WITH_METADATA,
        "summary": "With name and tags",
        "description": (
            "Creates a named grid with tags for organization. Useful when "
            "maintaining multiple grids for scenario comparison."
        ),
    },
}

# annual_disturbance has no staged national release -- every example here is
# fetched on demand from LANDFIRE Product Service, so (unlike fbfm40/fccs/
# fbfm13) there is only one list, and tests using it need a domain with known
# LFPS coverage.
LFPS_DISTURBANCE_EXAMPLE_VALUES = [
    ("minimal", EXAMPLE_DISTURBANCE_MINIMAL),
    ("with_metadata", EXAMPLE_DISTURBANCE_WITH_METADATA),
]
