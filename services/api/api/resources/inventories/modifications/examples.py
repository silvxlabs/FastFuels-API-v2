"""
Example request bodies for standalone inventory modification endpoints.

These examples are used in:
1. OpenAPI/Swagger documentation
2. Integration tests
"""

# Remove small trees (common microplot fix)
EXAMPLE_REMOVE_SMALL_TREES = {
    "modifications": [
        {
            "conditions": {
                "attribute": "dbh",
                "operator": "lt",
                "value": 12.7,
            },
            "actions": {"modifier": "remove"},
        }
    ],
}

# Remove small trees using inches (unit conversion)
EXAMPLE_REMOVE_SMALL_TREES_INCHES = {
    "modifications": [
        {
            "conditions": {
                "attribute": "dbh",
                "operator": "lt",
                "value": 5.0,
                "unit": "in",
            },
            "actions": {"modifier": "remove"},
        }
    ],
}

# Remove by species code
EXAMPLE_REMOVE_BY_SPECIES = {
    "modifications": [
        {
            "conditions": {
                "attribute": "fia_species_code",
                "operator": "eq",
                "value": [93, 15],
            },
            "actions": {"modifier": "remove"},
        }
    ],
}

# Remove by expression
EXAMPLE_REMOVE_BY_EXPRESSION = {
    "modifications": [
        {
            "conditions": {"expression": "height * crown_ratio < 1.0"},
            "actions": {"modifier": "remove"},
        }
    ],
}

# Reduce tall tree heights
EXAMPLE_REDUCE_TALL_TREES = {
    "modifications": [
        {
            "conditions": {
                "attribute": "height",
                "operator": "gt",
                "value": 65,
                "unit": "ft",
            },
            "actions": {
                "attribute": "height",
                "modifier": "multiply",
                "value": 0.9,
            },
        }
    ],
}

# Combined conditions: species + dbh filter
EXAMPLE_COMBINED_CONDITIONS = {
    "modifications": [
        {
            "conditions": [
                {
                    "attribute": "fia_species_code",
                    "operator": "eq",
                    "value": 202,
                },
                {
                    "attribute": "dbh",
                    "operator": "lt",
                    "value": 12.7,
                },
            ],
            "actions": {"modifier": "remove"},
        }
    ],
}

# Remove all trees inside a water Feature (e.g., a lake or river polygon)
EXAMPLE_REMOVE_TREES_IN_WATER_FEATURE = {
    "modifications": [
        {
            "conditions": {
                "source": "feature",
                "operator": "within",
                "feature_id": "feat_water_xyz",
            },
            "actions": {"modifier": "remove"},
        }
    ],
}

# Remove trees within a buffered road Feature (linestring + buffer is typical)
EXAMPLE_REMOVE_TREES_IN_ROAD_BUFFERED = {
    "modifications": [
        {
            "conditions": {
                "source": "feature",
                "operator": "within",
                "feature_id": "feat_road_abc",
                "buffer_m": 3,
            },
            "actions": {"modifier": "remove"},
        }
    ],
}

# Compound: only large trees inside a road buffer
EXAMPLE_REMOVE_LARGE_TREES_IN_ROAD_BUFFER = {
    "modifications": [
        {
            "conditions": [
                {
                    "source": "feature",
                    "operator": "within",
                    "feature_id": "feat_road_abc",
                    "buffer_m": 4,
                },
                {"attribute": "dbh", "operator": "gt", "value": 30},
            ],
            "actions": {"modifier": "remove"},
        }
    ],
}

# Inline geometry variant (no Feature resource needed)
EXAMPLE_REMOVE_TREES_IN_INLINE_POLYGON = {
    "modifications": [
        {
            "conditions": {
                "source": "geometry",
                "operator": "within",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [-120.0, 38.0],
                            [-119.5, 38.0],
                            [-119.5, 38.5],
                            [-120.0, 38.5],
                            [-120.0, 38.0],
                        ]
                    ],
                },
                "buffer_m": 5,
            },
            "actions": {"modifier": "remove"},
        }
    ],
}

# Multiple modifications: remove small + reduce tall
EXAMPLE_MULTIPLE_MODIFICATIONS = {
    "modifications": [
        {
            "conditions": {
                "attribute": "dbh",
                "operator": "lt",
                "value": 2.54,
            },
            "actions": {"modifier": "remove"},
        },
        {
            "conditions": {
                "attribute": "height",
                "operator": "gt",
                "value": 50,
            },
            "actions": {
                "attribute": "height",
                "modifier": "multiply",
                "value": 0.9,
            },
        },
    ],
}

APPLY_MODIFICATIONS_OPENAPI_EXAMPLES = {
    "remove_small_trees": {
        "value": EXAMPLE_REMOVE_SMALL_TREES,
        "summary": "Remove small trees (microplot fix)",
        "description": "Remove trees with dbh < 12.7 cm to fix microplot density issues.",
    },
    "remove_small_trees_inches": {
        "value": EXAMPLE_REMOVE_SMALL_TREES_INCHES,
        "summary": "Remove small trees (with unit conversion)",
        "description": (
            "Remove trees with dbh < 5 inch. The unit field converts "
            "the value to cm before comparison."
        ),
    },
    "remove_by_species": {
        "value": EXAMPLE_REMOVE_BY_SPECIES,
        "summary": "Remove by species code",
        "description": "Remove trees matching specific FIA species codes.",
    },
    "remove_by_expression": {
        "value": EXAMPLE_REMOVE_BY_EXPRESSION,
        "summary": "Remove by expression",
        "description": (
            "Remove trees where height * crown_ratio < 1.0. "
            "Expressions always use native units (cm, m, 0-1 fraction)."
        ),
    },
    "reduce_tall_trees": {
        "value": EXAMPLE_REDUCE_TALL_TREES,
        "summary": "Reduce tall tree heights",
        "description": "Multiply height by 0.9 for trees taller than 65 feet.",
    },
    "multiple_modifications": {
        "value": EXAMPLE_MULTIPLE_MODIFICATIONS,
        "summary": "Multiple modifications",
        "description": "Remove small trees and reduce tall tree heights in one request.",
    },
    "remove_trees_in_water_feature": {
        "value": EXAMPLE_REMOVE_TREES_IN_WATER_FEATURE,
        "summary": "Remove trees inside a water Feature",
        "description": (
            "Remove every tree whose point falls inside the geometry of the "
            "referenced water Feature (lake, river, ...). The Feature must "
            "belong to the same domain as the inventory."
        ),
    },
    "remove_trees_in_road_buffered": {
        "value": EXAMPLE_REMOVE_TREES_IN_ROAD_BUFFERED,
        "summary": "Remove trees within a 3 m road buffer",
        "description": (
            "Roads are typically linestrings; tree points almost never "
            "intersect a bare linestring, so `buffer_m` is effectively "
            "required for road Features. Here we strip every tree within "
            "3 metres of the road centerline."
        ),
    },
    "remove_large_trees_in_road_buffer": {
        "value": EXAMPLE_REMOVE_LARGE_TREES_IN_ROAD_BUFFER,
        "summary": "Remove large trees inside a road buffer",
        "description": (
            "Compound spatial + attribute condition (AND): remove only "
            "trees with `dbh > 30` (cm) that fall inside a 4 m road buffer."
        ),
    },
    "remove_trees_in_inline_polygon": {
        "value": EXAMPLE_REMOVE_TREES_IN_INLINE_POLYGON,
        "summary": "Remove trees inside an inline polygon (with buffer)",
        "description": (
            "Use `source: geometry` to pass GeoJSON directly when you don't "
            "have a persisted Feature resource. `buffer_m` works the same "
            "way: the geometry is expanded outward by 5 m before testing."
        ),
    },
}

ALL_MODIFICATIONS_EXAMPLE_VALUES = [
    ("remove_small_trees", EXAMPLE_REMOVE_SMALL_TREES),
    ("remove_small_trees_inches", EXAMPLE_REMOVE_SMALL_TREES_INCHES),
    ("remove_by_species", EXAMPLE_REMOVE_BY_SPECIES),
    ("remove_by_expression", EXAMPLE_REMOVE_BY_EXPRESSION),
    ("reduce_tall_trees", EXAMPLE_REDUCE_TALL_TREES),
    ("combined_conditions", EXAMPLE_COMBINED_CONDITIONS),
    ("multiple_modifications", EXAMPLE_MULTIPLE_MODIFICATIONS),
    ("remove_trees_in_water_feature", EXAMPLE_REMOVE_TREES_IN_WATER_FEATURE),
    ("remove_trees_in_road_buffered", EXAMPLE_REMOVE_TREES_IN_ROAD_BUFFERED),
    (
        "remove_large_trees_in_road_buffer",
        EXAMPLE_REMOVE_LARGE_TREES_IN_ROAD_BUFFER,
    ),
    ("remove_trees_in_inline_polygon", EXAMPLE_REMOVE_TREES_IN_INLINE_POLYGON),
]
