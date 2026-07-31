"""
Example request bodies for the 3DEP point cloud endpoint.

These appear in the OpenAPI/Swagger documentation and are exercised by the
router tests so the documented payloads stay valid. Every example omits an
acquisition type: 3DEP is airborne, so the resulting cloud is always `als`.
"""

# The common case: let the backend pick the acquisitions.
EXAMPLE_3DEP_AUTOMATIC = {
    "name": "Bridger ALS",
    "description": "3DEP airborne lidar over the Bridger study area.",
    "tags": ["bridger", "als"],
}

# Pin the fetch to specific acquisitions, e.g. to force a denser or newer
# survey where several overlap. Names come from the coverage endpoint.
EXAMPLE_3DEP_PINNED = {
    "name": "Bridger ALS 2020",
    "datasets": ["WY_Southwest_1_2020"],
}

CREATE_3DEP_OPENAPI_EXAMPLES = {
    "automatic": {
        "value": EXAMPLE_3DEP_AUTOMATIC,
        "summary": "Automatic acquisition choice",
        "description": (
            "Fetch 3DEP lidar for the domain, letting the backend choose which "
            "acquisitions to read."
        ),
    },
    "pinned": {
        "value": EXAMPLE_3DEP_PINNED,
        "summary": "Pinned acquisition",
        "description": (
            "Read a specific 3DEP acquisition. Use the coverage endpoint to "
            "see which acquisitions are available for the domain."
        ),
    },
}

ALL_3DEP_EXAMPLE_VALUES = [
    ("automatic", EXAMPLE_3DEP_AUTOMATIC),
    ("pinned", EXAMPLE_3DEP_PINNED),
]
