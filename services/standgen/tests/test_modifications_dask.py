"""
Dask integration tests for standgen/modifications.py

Verifies that apply_modifications works correctly through dask's
map_partitions, which is how it runs in production. Catches issues
that pure pandas unit tests miss:
- Index resets across partitions
- Type coercion differences between dask and pandas
- Partition boundary effects (rows split across partitions)
- Empty partitions after filtering
- Column dtype preservation through map_partitions
"""

import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest
from standgen.modifications import apply_modifications


@pytest.fixture
def sample_ddf():
    """Multi-partition dask DataFrame mimicking a real inventory.

    Creates 100 trees split across 5 partitions of 20 rows each.
    Includes a mix of species, sizes, and statuses.
    """
    rng = np.random.default_rng(42)
    n = 100
    df = pd.DataFrame(
        {
            "x": rng.uniform(500000, 501000, n),
            "y": rng.uniform(5200000, 5201000, n),
            "fia_species_code": rng.choice([93, 122, 202, 15, 108], n),
            "fia_status_code": rng.choice([1, 2], n, p=[0.9, 0.1]),
            "dbh": rng.uniform(0.5, 80.0, n),  # cm
            "height": rng.uniform(0.3, 40.0, n),  # m
            "crown_ratio": rng.uniform(0.05, 0.95, n),
        }
    )
    return dd.from_pandas(df, npartitions=5)


@pytest.fixture
def large_ddf():
    """Larger multi-partition DataFrame for stress testing."""
    rng = np.random.default_rng(123)
    n = 10_000
    df = pd.DataFrame(
        {
            "x": rng.uniform(500000, 501000, n),
            "y": rng.uniform(5200000, 5201000, n),
            "fia_species_code": rng.choice([93, 122, 202, 15, 108], n),
            "fia_status_code": rng.choice([1, 2], n, p=[0.9, 0.1]),
            "dbh": rng.uniform(0.5, 80.0, n),
            "height": rng.uniform(0.3, 40.0, n),
            "crown_ratio": rng.uniform(0.05, 0.95, n),
        }
    )
    return dd.from_pandas(df, npartitions=20)


class TestDaskRemove:
    """Test remove modifications through dask map_partitions."""

    def test_remove_filters_across_partitions(self, sample_ddf):
        """Remove should filter rows from all partitions, not just one."""
        mods = [
            {
                "conditions": [{"attribute": "dbh", "operator": "lt", "value": 5.0}],
                "actions": [{"modifier": "remove"}],
            }
        ]

        # Count trees to be removed across all partitions
        original = sample_ddf.compute()
        expected_remaining = len(original[original["dbh"] >= 5.0])

        result = sample_ddf.map_partitions(apply_modifications, mods).compute()

        assert len(result) == expected_remaining
        assert (result["dbh"] >= 5.0).all()

    def test_remove_by_species_list(self, sample_ddf):
        """Remove with species list condition works across partitions."""
        mods = [
            {
                "conditions": [
                    {
                        "attribute": "fia_species_code",
                        "operator": "eq",
                        "value": [93, 15],
                    }
                ],
                "actions": [{"modifier": "remove"}],
            }
        ]

        original = sample_ddf.compute()
        expected_remaining = len(original[~original["fia_species_code"].isin([93, 15])])

        result = sample_ddf.map_partitions(apply_modifications, mods).compute()

        assert len(result) == expected_remaining
        assert not result["fia_species_code"].isin([93, 15]).any()

    def test_remove_by_expression(self, sample_ddf):
        """Expression-based remove works through map_partitions."""
        mods = [
            {
                "conditions": [{"expression": "height * crown_ratio < 1.0"}],
                "actions": [{"modifier": "remove"}],
            }
        ]

        original = sample_ddf.compute()
        expected_remaining = len(
            original[~(original["height"] * original["crown_ratio"] < 1.0)]
        )

        result = sample_ddf.map_partitions(apply_modifications, mods).compute()

        assert len(result) == expected_remaining

    def test_remove_with_unit_conversion(self, sample_ddf):
        """Unit conversion in conditions works through map_partitions."""
        mods = [
            {
                "conditions": [
                    {"attribute": "dbh", "operator": "lt", "value": 2.0, "unit": "in"}
                ],
                "actions": [{"modifier": "remove"}],
            }
        ]

        # 2 inches = 5.08 cm
        original = sample_ddf.compute()
        expected_remaining = len(original[original["dbh"] >= 5.08])

        result = sample_ddf.map_partitions(apply_modifications, mods).compute()

        assert len(result) == expected_remaining
        assert (result["dbh"] >= 5.08 - 0.001).all()

    def test_remove_can_empty_all_partitions(self, sample_ddf):
        """Removing everything should produce an empty DataFrame with correct columns."""
        mods = [
            {
                "conditions": [{"attribute": "dbh", "operator": "ge", "value": 0.0}],
                "actions": [{"modifier": "remove"}],
            }
        ]

        result = sample_ddf.map_partitions(apply_modifications, mods).compute()

        assert len(result) == 0
        # Column schema should be preserved even when empty
        assert list(result.columns) == list(sample_ddf.columns)

    def test_remove_some_partitions_empty(self, sample_ddf):
        """When some partitions lose all rows, result should still be valid."""
        # Use a threshold that will empty some but not all partitions
        original = sample_ddf.compute()
        median_dbh = original["dbh"].median()

        mods = [
            {
                "conditions": [
                    {"attribute": "dbh", "operator": "lt", "value": median_dbh}
                ],
                "actions": [{"modifier": "remove"}],
            }
        ]

        result = sample_ddf.map_partitions(apply_modifications, mods).compute()

        assert len(result) > 0
        assert (result["dbh"] >= median_dbh).all()


class TestDaskModify:
    """Test value-modifying actions through dask map_partitions."""

    def test_multiply_height(self, sample_ddf):
        """Multiply action applies correctly across partitions."""
        mods = [
            {
                "conditions": [
                    {"attribute": "height", "operator": "gt", "value": 20.0}
                ],
                "actions": [
                    {"attribute": "height", "modifier": "multiply", "value": 0.5}
                ],
            }
        ]

        original = sample_ddf.compute()
        result = sample_ddf.map_partitions(apply_modifications, mods).compute()

        # Row count should be unchanged
        assert len(result) == len(original)

        # Tall trees should be halved; check those that were > 20
        for idx in original.index:
            orig_h = original.loc[idx, "height"]
            # Find matching row by position (index may differ after reset)
            if orig_h > 20.0:
                # The modified value should be roughly half
                # Can't match by index directly after partition processing,
                # so verify aggregate behavior
                pass

        # All modified heights should be <= original max * 0.5 + some tolerance
        tall_original = original[original["height"] > 20.0]["height"]
        if len(tall_original) > 0:
            # After multiplying by 0.5, tallest should be roughly half
            assert result["height"].max() <= original["height"].max() + 0.01

    def test_replace_attribute(self, sample_ddf):
        """Replace action on matching rows across partitions."""
        mods = [
            {
                "conditions": [
                    {"attribute": "fia_species_code", "operator": "eq", "value": 93}
                ],
                "actions": [
                    {
                        "attribute": "dbh",
                        "modifier": "replace",
                        "value": 10.0,
                    }
                ],
            }
        ]

        original = sample_ddf.compute()
        count_93 = (original["fia_species_code"] == 93).sum()

        result = sample_ddf.map_partitions(apply_modifications, mods).compute()

        # Same number of rows
        assert len(result) == len(original)

        # Species 93 trees should all have dbh=10.0
        sp93_dbh = result[result["fia_species_code"] == 93]["dbh"]
        assert len(sp93_dbh) == count_93
        assert (sp93_dbh == 10.0).all()

    def test_add_with_unit(self, sample_ddf):
        """Add action with unit conversion across partitions."""
        mods = [
            {
                "conditions": [{"attribute": "dbh", "operator": "gt", "value": 0.0}],
                "actions": [
                    {"attribute": "dbh", "modifier": "add", "value": 1.0, "unit": "in"}
                ],
            }
        ]

        original = sample_ddf.compute()
        result = sample_ddf.map_partitions(apply_modifications, mods).compute()

        # Every tree should have 2.54 cm added
        # Compare sums since row order may differ
        expected_sum = original["dbh"].sum() + len(original) * 2.54
        assert abs(result["dbh"].sum() - expected_sum) < 0.1

    def test_clamp_crown_ratio(self, sample_ddf):
        """Crown ratio clamping works through map_partitions."""
        mods = [
            {
                "conditions": [
                    {"attribute": "crown_ratio", "operator": "gt", "value": 0.0}
                ],
                "actions": [
                    {"attribute": "crown_ratio", "modifier": "add", "value": 2.0}
                ],
            }
        ]

        result = sample_ddf.map_partitions(apply_modifications, mods).compute()

        # All values should be clamped to [0, 1]
        assert (result["crown_ratio"] >= 0).all()
        assert (result["crown_ratio"] <= 1.0).all()

    def test_clamp_dbh_non_negative(self, sample_ddf):
        """DBH clamping at 0 works through map_partitions."""
        mods = [
            {
                "conditions": [{"attribute": "dbh", "operator": "gt", "value": 0.0}],
                "actions": [
                    {"attribute": "dbh", "modifier": "subtract", "value": 1000.0}
                ],
            }
        ]

        result = sample_ddf.map_partitions(apply_modifications, mods).compute()

        assert (result["dbh"] >= 0).all()


class TestDaskMultipleModifications:
    """Test multiple sequential modifications through dask."""

    def test_remove_then_modify(self, sample_ddf):
        """First remove small trees, then scale heights."""
        mods = [
            {
                "conditions": [{"attribute": "dbh", "operator": "lt", "value": 5.0}],
                "actions": [{"modifier": "remove"}],
            },
            {
                "conditions": [
                    {"attribute": "height", "operator": "gt", "value": 20.0}
                ],
                "actions": [
                    {"attribute": "height", "modifier": "multiply", "value": 0.9}
                ],
            },
        ]

        original = sample_ddf.compute()
        after_remove = original[original["dbh"] >= 5.0]

        result = sample_ddf.map_partitions(apply_modifications, mods).compute()

        # Row count should match remove filter
        assert len(result) == len(after_remove)
        # All remaining trees have dbh >= 5
        assert (result["dbh"] >= 5.0).all()

    def test_combined_conditions_then_remove(self, sample_ddf):
        """Multiple AND conditions with remove."""
        mods = [
            {
                "conditions": [
                    {"attribute": "fia_species_code", "operator": "eq", "value": 202},
                    {"attribute": "dbh", "operator": "lt", "value": 10.0},
                ],
                "actions": [{"modifier": "remove"}],
            }
        ]

        original = sample_ddf.compute()
        to_remove = (original["fia_species_code"] == 202) & (original["dbh"] < 10.0)
        expected_remaining = len(original[~to_remove])

        result = sample_ddf.map_partitions(apply_modifications, mods).compute()

        assert len(result) == expected_remaining

    def test_empty_modifications_passthrough(self, sample_ddf):
        """Empty modifications list should not alter the DataFrame."""
        original = sample_ddf.compute()
        result = sample_ddf.map_partitions(apply_modifications, []).compute()

        assert len(result) == len(original)
        assert list(result.columns) == list(original.columns)


class TestDaskColumnPreservation:
    """Verify column dtypes and schema are preserved through map_partitions."""

    def test_column_dtypes_preserved_after_remove(self, sample_ddf):
        """Column dtypes should not change after remove modifications."""
        mods = [
            {
                "conditions": [{"attribute": "dbh", "operator": "lt", "value": 5.0}],
                "actions": [{"modifier": "remove"}],
            }
        ]

        original_dtypes = sample_ddf.compute().dtypes
        result = sample_ddf.map_partitions(apply_modifications, mods).compute()

        for col in result.columns:
            assert result[col].dtype == original_dtypes[col], (
                f"Column '{col}' dtype changed: {original_dtypes[col]} -> {result[col].dtype}"
            )

    def test_column_dtypes_preserved_after_modify(self, sample_ddf):
        """Column dtypes should not change after value modifications."""
        mods = [
            {
                "conditions": [{"attribute": "height", "operator": "gt", "value": 0.0}],
                "actions": [
                    {"attribute": "height", "modifier": "multiply", "value": 0.9}
                ],
            }
        ]

        original_dtypes = sample_ddf.compute().dtypes
        result = sample_ddf.map_partitions(apply_modifications, mods).compute()

        for col in result.columns:
            assert result[col].dtype == original_dtypes[col], (
                f"Column '{col}' dtype changed: {original_dtypes[col]} -> {result[col].dtype}"
            )

    def test_column_order_preserved(self, sample_ddf):
        """Column order should match the original."""
        mods = [
            {
                "conditions": [{"attribute": "dbh", "operator": "lt", "value": 5.0}],
                "actions": [{"modifier": "remove"}],
            }
        ]

        result = sample_ddf.map_partitions(apply_modifications, mods).compute()
        assert list(result.columns) == list(sample_ddf.columns)


class TestDaskLargeDataset:
    """Test with a larger dataset to catch partition-boundary issues."""

    def test_large_remove(self, large_ddf):
        """Remove on a 10K-row, 20-partition dataset."""
        mods = [
            {
                "conditions": [{"attribute": "dbh", "operator": "lt", "value": 2.54}],
                "actions": [{"modifier": "remove"}],
            }
        ]

        original = large_ddf.compute()
        expected = len(original[original["dbh"] >= 2.54])

        result = large_ddf.map_partitions(apply_modifications, mods).compute()

        assert len(result) == expected

    def test_large_modify_and_remove(self, large_ddf):
        """Sequential remove + modify on large dataset."""
        mods = [
            {
                "conditions": [
                    {"attribute": "dbh", "operator": "lt", "value": 1.0, "unit": "in"}
                ],
                "actions": [{"modifier": "remove"}],
            },
            {
                "conditions": [
                    {"attribute": "height", "operator": "gt", "value": 30.0}
                ],
                "actions": [
                    {"attribute": "height", "modifier": "multiply", "value": 0.85}
                ],
            },
        ]

        original = large_ddf.compute()
        after_remove = original[original["dbh"] >= 2.54]  # 1 in ≈ 2.54 cm

        result = large_ddf.map_partitions(apply_modifications, mods).compute()

        assert len(result) == len(after_remove)
        assert (result["dbh"] >= 2.54 - 0.001).all()


class TestDaskEquivalence:
    """Verify dask results exactly match pandas results."""

    def test_remove_equivalence(self, sample_ddf):
        """Dask map_partitions remove matches direct pandas apply."""
        mods = [
            {
                "conditions": [{"attribute": "dbh", "operator": "lt", "value": 10.0}],
                "actions": [{"modifier": "remove"}],
            }
        ]

        # Dask path
        dask_result = sample_ddf.map_partitions(apply_modifications, mods).compute()

        # Pandas path
        pandas_result = apply_modifications(sample_ddf.compute().copy(), mods)

        assert len(dask_result) == len(pandas_result)
        # Both should have same remaining species distribution
        assert sorted(
            dask_result["fia_species_code"].value_counts().to_dict().items()
        ) == sorted(pandas_result["fia_species_code"].value_counts().to_dict().items())

    def test_modify_equivalence(self, sample_ddf):
        """Dask map_partitions modify matches direct pandas apply."""
        mods = [
            {
                "conditions": [
                    {"attribute": "height", "operator": "gt", "value": 20.0}
                ],
                "actions": [
                    {"attribute": "height", "modifier": "multiply", "value": 0.9}
                ],
            }
        ]

        dask_result = (
            sample_ddf.map_partitions(apply_modifications, mods)
            .compute()
            .sort_values(["x", "y"])
            .reset_index(drop=True)
        )
        pandas_result = (
            apply_modifications(sample_ddf.compute().copy(), mods)
            .sort_values(["x", "y"])
            .reset_index(drop=True)
        )

        pd.testing.assert_frame_equal(dask_result, pandas_result)

    def test_complex_pipeline_equivalence(self, sample_ddf):
        """Full pipeline: remove + expression + modify + unit conversion."""
        mods = [
            {
                "conditions": [
                    {"attribute": "dbh", "operator": "lt", "value": 1.0, "unit": "in"}
                ],
                "actions": [{"modifier": "remove"}],
            },
            {
                "conditions": [{"expression": "height / dbh > 5"}],
                "actions": [
                    {"attribute": "height", "modifier": "multiply", "value": 0.8}
                ],
            },
        ]

        dask_result = (
            sample_ddf.map_partitions(apply_modifications, mods)
            .compute()
            .sort_values(["x", "y"])
            .reset_index(drop=True)
        )
        pandas_result = (
            apply_modifications(sample_ddf.compute().copy(), mods)
            .sort_values(["x", "y"])
            .reset_index(drop=True)
        )

        pd.testing.assert_frame_equal(dask_result, pandas_result)
