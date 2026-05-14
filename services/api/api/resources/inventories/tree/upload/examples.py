"""
Example request bodies for the upload inventory endpoint.

These examples are used in:
1. OpenAPI/Swagger documentation - Users see these as example payloads
2. Integration tests - Each example is tested to ensure documentation stays accurate

The upload flow is two-step: POST this body to create the inventory and receive a
signed PUT URL, then PUT the file to that URL with the matching Content-Type header.
"""

# Minimal CSV request — standard column names, no mapping needed
EXAMPLE_UPLOAD_CSV_MINIMAL = {
    "format": "csv",
}

# CSV with custom column names mapped to v2 names
EXAMPLE_UPLOAD_CSV_WITH_MAPPING = {
    "format": "csv",
    "columns": {
        "height": "HT",
        "fia_species_code": "SPCD",
        "dbh": "DIA",
    },
    "name": "Field survey 2024",
    "description": "Plot measurements from summer field campaign",
    "tags": ["field", "2024"],
}

# GeoJSON upload — coordinates must be EPSG:4326 per the GeoJSON spec
EXAMPLE_UPLOAD_GEOJSON = {
    "format": "geojson",
    "name": "LiDAR-derived tree list",
    "tags": ["lidar"],
}

# GeoPackage upload — any CRS accepted, reprojected to domain CRS automatically
EXAMPLE_UPLOAD_GEOPACKAGE = {
    "format": "geopackage",
    "columns": {
        "height": "tree_height_m",
    },
    "name": "Survey plot trees",
    "tags": ["survey"],
}

CREATE_UPLOAD_OPENAPI_EXAMPLES = {
    "csv_minimal": {
        "value": EXAMPLE_UPLOAD_CSV_MINIMAL,
        "summary": "CSV — minimal",
        "description": (
            "Upload a CSV file whose columns already use v2 names "
            "(x, y, height). No column mapping required."
        ),
    },
    "csv_with_mapping": {
        "value": EXAMPLE_UPLOAD_CSV_WITH_MAPPING,
        "summary": "CSV — custom column names",
        "description": (
            "Upload a CSV file with non-standard column names. "
            "The `columns` mapping translates file column names to v2 names. "
            "CSV coordinates must already be in the domain's projected CRS."
        ),
    },
    "geojson": {
        "value": EXAMPLE_UPLOAD_GEOJSON,
        "summary": "GeoJSON",
        "description": (
            "Upload a GeoJSON FeatureCollection with Point or MultiPoint geometries. "
            "Coordinates must be in EPSG:4326 (WGS84) per the GeoJSON spec; "
            "the uploader reprojects to the domain CRS automatically."
        ),
    },
    "geopackage": {
        "value": EXAMPLE_UPLOAD_GEOPACKAGE,
        "summary": "GeoPackage",
        "description": (
            "Upload an OGC GeoPackage with Point or MultiPoint geometries. "
            "Any CRS is accepted; the uploader reprojects to the domain CRS automatically."
        ),
    },
}

ALL_UPLOAD_EXAMPLE_VALUES = [
    ("csv_minimal", EXAMPLE_UPLOAD_CSV_MINIMAL),
    ("csv_with_mapping", EXAMPLE_UPLOAD_CSV_WITH_MAPPING),
    ("geojson", EXAMPLE_UPLOAD_GEOJSON),
    ("geopackage", EXAMPLE_UPLOAD_GEOPACKAGE),
]
