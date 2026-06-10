"""
Example request bodies for the point cloud upload endpoint.

These appear in the OpenAPI/Swagger documentation and are exercised by the
router tests so the documented payloads stay valid. The upload flow is two
steps: POST one of these bodies to create the point cloud and receive a signed
PUT URL, then PUT the file to that URL with the matching Content-Type.
"""

# Airborne scan uploaded as compressed LAZ.
EXAMPLE_UPLOAD_ALS_LAZ = {
    "type": "als",
    "format": "laz",
    "name": "Bridger ALS 2020",
    "description": "Airborne lidar over the Bridger study area.",
    "tags": ["bridger", "als"],
}

# Terrestrial plot scan uploaded as compressed LAZ.
EXAMPLE_UPLOAD_TLS_LAZ = {
    "type": "tls",
    "format": "laz",
    "name": "Plot 3 TLS",
    "tags": ["plot-3"],
}

# Airborne scan already in Cloud Optimized Point Cloud form.
EXAMPLE_UPLOAD_ALS_COPC = {
    "type": "als",
    "format": "copc",
    "name": "Bridger ALS (COPC)",
}

CREATE_UPLOAD_OPENAPI_EXAMPLES = {
    "als_laz": {
        "value": EXAMPLE_UPLOAD_ALS_LAZ,
        "summary": "Airborne (ALS) — LAZ",
        "description": "Upload an airborne laser scan as a compressed LAZ file.",
    },
    "tls_laz": {
        "value": EXAMPLE_UPLOAD_TLS_LAZ,
        "summary": "Terrestrial (TLS) — LAZ",
        "description": (
            "Upload a terrestrial (tripod) laser scan as a compressed LAZ file."
        ),
    },
    "als_copc": {
        "value": EXAMPLE_UPLOAD_ALS_COPC,
        "summary": "Airborne (ALS) — COPC",
        "description": "Upload an airborne scan already in COPC (.copc.laz) format.",
    },
}

ALL_UPLOAD_EXAMPLE_VALUES = [
    ("als_laz", EXAMPLE_UPLOAD_ALS_LAZ),
    ("tls_laz", EXAMPLE_UPLOAD_TLS_LAZ),
    ("als_copc", EXAMPLE_UPLOAD_ALS_COPC),
]
