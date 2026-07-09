"""
Per-column summary statistics for inventory DataFrames.

Builds a dict of dask expression-backed lazy scalars that can be fused with a
concurrent to_parquet graph in a single dask.compute call, so each partition is
materialized exactly once.
"""

import dask.dataframe as dd


def _build_column_stats_graph(
    ddf: dd.DataFrame, columns: list[dict]
) -> dict[str, dict]:
    """Build a dict of per-column lazy scalar reductions keyed by column key.

    Returns a nested dict where each value is a dict of dask lazy scalars
    (not yet computed). Because these are dask expression-backed scalars derived
    directly from ``ddf``, they share the same expression graph as a concurrent
    ``to_parquet`` call and can be fused into a single ``dask.compute`` call,
    so each partition is materialized exactly once.

    Args:
        ddf: Lazy dask DataFrame.
        columns: List of column dicts with 'key' and 'type' fields.

    Returns:
        Dict keyed by column key. Each value is a dict of lazy scalars for
        that column's stats. Columns absent from ``ddf`` are silently skipped.
    """
    parts = {}
    for col in columns:
        key = col["key"]
        col_type = col["type"]
        if key not in ddf.columns:
            continue
        series = ddf[key]
        if col_type == "continuous":
            parts[key] = {
                "type": "continuous",
                "count": series.count(),
                "null_count": series.isna().sum(),
                "min": series.min(),
                "max": series.max(),
                "mean": series.mean(),
                "std": series.std(),
            }
        else:
            parts[key] = {
                "type": "categorical",
                "count": series.count(),
                "null_count": series.isna().sum(),
                "unique_count": series.nunique(),
            }
    return parts
