"""
Example request bodies for the netCDF grid upload endpoint.

These examples are used in:
1. OpenAPI/Swagger documentation
2. Integration tests — each example is tested to ensure documentation stays accurate
"""

EXAMPLE_NETCDF_MINIMAL: dict = {}

EXAMPLE_NETCDF_WITH_METADATA = {
    "name": "Custom 3D fuel grid",
    "description": "Voxelized bulk density from external LiDAR pipeline.",
    "tags": ["lidar", "external"],
}

EXAMPLE_NETCDF_WITH_BUFFER = {
    "name": "FBFM with 2-cell buffer",
    "num_buffer_cells": 2,
}

CREATE_NETCDF_UPLOAD_OPENAPI_EXAMPLES = {
    "minimal": {
        "value": EXAMPLE_NETCDF_MINIMAL,
        "summary": "Minimal request — no metadata",
        "description": (
            "All variable names, units, dtypes, CRS, and z-axis spacing come "
            "from the netCDF file itself. The request body only carries "
            "resource-level metadata."
        ),
    },
    "with_metadata": {
        "value": EXAMPLE_NETCDF_WITH_METADATA,
        "summary": "Name, description, tags",
    },
    "with_buffer": {
        "value": EXAMPLE_NETCDF_WITH_BUFFER,
        "summary": "Keep a 2-cell buffer around the domain",
    },
}

ALL_NETCDF_UPLOAD_EXAMPLE_VALUES = [
    ("minimal", EXAMPLE_NETCDF_MINIMAL),
    ("with_metadata", EXAMPLE_NETCDF_WITH_METADATA),
    ("with_buffer", EXAMPLE_NETCDF_WITH_BUFFER),
]
