"""
Example request bodies for inventory export endpoints.

These examples are used in:
1. OpenAPI/Swagger documentation - Users see these as example payloads
2. Integration tests - Each example is tested to ensure documentation stays accurate
"""

EXAMPLE_EXPORT_ALL_COLUMNS = {
    "name": "Tree inventory export",
    "tags": ["trees"],
}

EXAMPLE_EXPORT_COLUMN_SUBSET = {
    "columns": ["x", "y", "dbh", "height"],
    "name": "Coordinates and dimensions only",
    "description": "Export subset of columns for spatial analysis",
    "tags": ["analysis"],
}

EXAMPLE_EXPORT_MINIMAL = {}

CREATE_INVENTORY_EXPORT_OPENAPI_EXAMPLES = {
    "all_columns": {
        "value": EXAMPLE_EXPORT_ALL_COLUMNS,
        "summary": "Export all columns",
        "description": "Exports all columns from the inventory.",
    },
    "column_subset": {
        "value": EXAMPLE_EXPORT_COLUMN_SUBSET,
        "summary": "Export specific columns",
        "description": (
            "Exports only the specified columns. "
            "Useful for reducing file size when only certain attributes are needed."
        ),
    },
    "minimal": {
        "value": EXAMPLE_EXPORT_MINIMAL,
        "summary": "Minimal request",
        "description": "Exports all columns with no name or tags.",
    },
}

ALL_INVENTORY_EXPORT_EXAMPLE_VALUES = [
    ("all_columns", EXAMPLE_EXPORT_ALL_COLUMNS),
    ("column_subset", EXAMPLE_EXPORT_COLUMN_SUBSET),
    ("minimal", EXAMPLE_EXPORT_MINIMAL),
]
