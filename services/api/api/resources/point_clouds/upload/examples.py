"""
Example request bodies for the point cloud upload endpoint.

These appear in the OpenAPI/Swagger documentation and are exercised by the
router tests so the documented payloads stay valid. The upload flow is two
steps: POST one of these bodies to create the point cloud and receive a signed
PUT URL, then PUT the file (LAS or LAZ — detected automatically) to that URL
with the matching Content-Type.
"""

# Airborne laser scan.
EXAMPLE_UPLOAD_ALS = {
    "type": "als",
    "name": "Bridger ALS 2020",
    "description": "Airborne lidar over the Bridger study area.",
    "tags": ["bridger", "als"],
}

# Terrestrial plot scan.
EXAMPLE_UPLOAD_TLS = {
    "type": "tls",
    "name": "Plot 3 TLS",
    "tags": ["plot-3"],
}

CREATE_UPLOAD_OPENAPI_EXAMPLES = {
    "als": {
        "value": EXAMPLE_UPLOAD_ALS,
        "summary": "Airborne (ALS)",
        "description": "Upload an airborne laser scan (LAS or LAZ file).",
    },
    "tls": {
        "value": EXAMPLE_UPLOAD_TLS,
        "summary": "Terrestrial (TLS)",
        "description": ("Upload a terrestrial (tripod) laser scan (LAS or LAZ file)."),
    },
}

ALL_UPLOAD_EXAMPLE_VALUES = [
    ("als", EXAMPLE_UPLOAD_ALS),
    ("tls", EXAMPLE_UPLOAD_TLS),
]
