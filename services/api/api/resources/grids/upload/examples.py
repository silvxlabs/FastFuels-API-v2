"""
Example request bodies for the grid upload endpoint.

These examples are used in:
1. OpenAPI/Swagger documentation
2. Integration tests — each example is tested to ensure documentation stays accurate
"""

EXAMPLE_UPLOAD_SINGLE_BAND = {
    "format": "geotiff",
    "bands": [{"key": "fbfm", "type": "categorical"}],
    "name": "Custom FBFM40",
}

EXAMPLE_UPLOAD_MULTI_BAND = {
    "format": "geotiff",
    "bands": [
        {"key": "bulk_density.foliage", "type": "continuous", "unit": "kg/m3"},
        {"key": "bulk_density.branchwood", "type": "continuous", "unit": "kg/m3"},
    ],
    "name": "Custom bulk density",
    "tags": ["lidar"],
}

CREATE_GRID_UPLOAD_OPENAPI_EXAMPLES = {
    "single_band": {
        "value": EXAMPLE_UPLOAD_SINGLE_BAND,
        "summary": "Single-band categorical (FBFM40)",
        "description": (
            "Upload a single-band GeoTIFF with fuel model codes. "
            "The band key maps to the variable name in the output Zarr store."
        ),
    },
    "multi_band": {
        "value": EXAMPLE_UPLOAD_MULTI_BAND,
        "summary": "Multi-band continuous (bulk density)",
        "description": (
            "Upload a multi-band GeoTIFF. bands[i] maps to GeoTIFF band i+1. "
            "Each band becomes a separate variable in the output Zarr store."
        ),
    },
}

ALL_GRID_UPLOAD_EXAMPLE_VALUES = [
    ("single_band", EXAMPLE_UPLOAD_SINGLE_BAND),
    ("multi_band", EXAMPLE_UPLOAD_MULTI_BAND),
]
