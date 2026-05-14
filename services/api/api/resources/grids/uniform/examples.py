"""
Example request bodies for uniform grid endpoints.

These examples are used in:
1. OpenAPI/Swagger documentation - Users see these as example payloads
2. Integration tests - Each example is tested to ensure documentation stays accurate

domain_id comes from the URL path parameter, not the request body.
"""

EXAMPLE_UNIFORM_SINGLE_BAND = {
    "resolution": 2.0,
    "bands": [
        {"key": "fuel_moisture.1hr", "value": 6.0},
    ],
}

EXAMPLE_UNIFORM_MOISTURE_SCENARIO = {
    "resolution": 2.0,
    "bands": [
        {"key": "fuel_moisture.1hr", "value": 6.0},
        {"key": "fuel_moisture.10hr", "value": 8.0},
        {"key": "fuel_moisture.100hr", "value": 12.0},
        {"key": "fuel_moisture.live_herb", "value": 60.0},
        {"key": "fuel_moisture.live_woody", "value": 90.0},
    ],
    "name": "Dry fuel moisture scenario",
    "tags": ["fuel-moisture", "dry-scenario"],
}

EXAMPLE_UNIFORM_FUEL_LOADS = {
    "resolution": 5.0,
    "bands": [
        {"key": "fuel_load.1hr", "value": 0.15},
        {"key": "fuel_load.10hr", "value": 0.10},
        {"key": "fuel_depth", "value": 0.3},
    ],
    "name": "Custom fuel load",
    "description": "Uniform fuel load for sensitivity analysis",
    "tags": ["fuel-load", "sensitivity"],
}

CREATE_UNIFORM_OPENAPI_EXAMPLES = {
    "single_band": {
        "value": EXAMPLE_UNIFORM_SINGLE_BAND,
        "summary": "Single band (1hr fuel moisture)",
        "description": (
            "Creates a uniform grid with a single band of 1-hour fuel "
            "moisture at 6% across the entire domain at 2m resolution."
        ),
    },
    "moisture_scenario": {
        "value": EXAMPLE_UNIFORM_MOISTURE_SCENARIO,
        "summary": "Multiple moisture bands",
        "description": (
            "Creates a uniform grid with all five fuel moisture bands "
            "for a dry scenario. Each band is filled with a constant value."
        ),
    },
    "fuel_loads": {
        "value": EXAMPLE_UNIFORM_FUEL_LOADS,
        "summary": "Fuel loads and depth",
        "description": (
            "Creates a uniform grid with fuel loads and fuel depth at "
            "5m resolution for sensitivity analysis."
        ),
    },
}

ALL_UNIFORM_EXAMPLE_VALUES = [
    ("single_band", EXAMPLE_UNIFORM_SINGLE_BAND),
    ("moisture_scenario", EXAMPLE_UNIFORM_MOISTURE_SCENARIO),
    ("fuel_loads", EXAMPLE_UNIFORM_FUEL_LOADS),
]
