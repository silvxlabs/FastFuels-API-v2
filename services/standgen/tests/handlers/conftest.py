"""Shared fixtures and constants for standgen handler tests."""

BASE_INVENTORY_COLUMNS = [
    {"key": "x", "type": "continuous", "unit": "m"},
    {"key": "y", "type": "continuous", "unit": "m"},
    {"key": "fia_species_code", "type": "categorical"},
    {"key": "fia_status_code", "type": "categorical"},
    {"key": "dbh", "type": "continuous", "unit": "cm"},
    {"key": "height", "type": "continuous", "unit": "m"},
    {"key": "crown_ratio", "type": "continuous"},
]

CHM_INVENTORY_COLUMNS = [
    {"key": "x", "type": "continuous", "unit": "m"},
    {"key": "y", "type": "continuous", "unit": "m"},
    {"key": "height", "type": "continuous", "unit": "m"},
]
