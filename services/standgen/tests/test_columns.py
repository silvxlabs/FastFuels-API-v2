"""Tests for standgen column definitions."""

from standgen.columns import BASE_COLUMNS, RENAME_MAP


def test_rename_map_covers_base_columns():
    """Verify RENAME_MAP output values match BASE_COLUMNS."""
    renamed_values = set(RENAME_MAP.values())
    for col in BASE_COLUMNS:
        assert col in renamed_values, f"Base column '{col}' not in RENAME_MAP values"


def test_base_columns_count():
    assert len(BASE_COLUMNS) == 7
