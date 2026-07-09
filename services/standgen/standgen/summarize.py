"""
Per-column and stand-level summary reductions for inventory DataFrames.

Builds dask expression-backed lazy scalars that can be fused with a
concurrent to_parquet graph in a single dask.compute call, so each partition is
materialized exactly once.
"""

import math
from pathlib import Path

import dask
import dask.dataframe as dd
import geopandas as gpd
import pandas as pd
import pint

_FIA_SPECIES_GROUPS = pd.read_csv(
    Path(__file__).parent / "data" / "fia_species_groups.csv", index_col="SPCD"
)


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


def _build_tree_forestry_graph(
    ddf: dd.DataFrame,
    domain_gdf: gpd.GeoDataFrame,
    top_species_groups: int = 5,
) -> dask.delayed:
    """ "Build a dask.delayed of stand-level forestry metric reductions.

    Returns a dask.delayed that resolves to a forestry metrics dict shaped like
    TreeForestryMetrics. Because the internal reductions are dask expression-backed
    scalars derived directly from ``ddf``, they share the same expression graph as a
    concurrent ``to_parquet`` call and can be fused into a single ``dask.compute``
    call, so each partition is materialized exactly once.

    Args:
        ddf: Lazy dask DataFrame with 'dbh' (cm) and 'fia_species_code' columns.
        domain_gdf: Domain geometry for per-area metric computation. Reprojected
            to UTM if not already projected.
        top_species_groups: Number of top FIA species groups by basal area share
            to include. Defaults to 5.

    Returns:
        dask.delayed resolving to a forestry metrics dict, or a zero-tree dict
        if the inventory is empty.
    """
    ba_per_tree_m2 = math.pi * (ddf["dbh"] / 200.0) ** 2
    total_ba_m2 = ba_per_tree_m2.sum()
    n_trees = ddf["dbh"].count()
    sum_dbh_sq = (ddf["dbh"] ** 2).sum()
    ba_by_spcd = ba_per_tree_m2.groupby(ddf["fia_species_code"]).sum()

    @dask.delayed
    def _compute(total_ba_m2, n_trees, sum_dbh_sq, ba_by_spcd):
        if n_trees == 0:
            return {
                "type": "tree",
                "tree_count": 0,
                "basal_area_per_area": None,
                "tree_density": None,
                "quadratic_mean_diameter": None,
                "dominant_species_groups": [],
            }

        # Domain area in m²
        if domain_gdf.crs is None or not domain_gdf.crs.is_projected:
            area_gdf = domain_gdf.to_crs(domain_gdf.estimate_utm_crs())
        else:
            area_gdf = domain_gdf
        domain_area_m2 = float(area_gdf.area.sum())

        # Unit conversions via pint
        ureg = pint.UnitRegistry()
        Q_ = ureg.Quantity
        basal_area_per_area = float(
            Q_(total_ba_m2 / domain_area_m2, "m**2/m**2").to("ft**2/acre").magnitude
        )
        tree_density = float(
            Q_(n_trees / domain_area_m2, "1/m**2").to("1/acre").magnitude
        )
        qmd = float(Q_(math.sqrt(sum_dbh_sq / n_trees), "cm").to("in").magnitude)

        # Species group rollup
        ba_by_spcd_mapped = ba_by_spcd.copy()
        ba_by_spcd_mapped.index = (
            ba_by_spcd_mapped.index.map(_FIA_SPECIES_GROUPS["JENKINS_SPGRPCD"])
            .fillna(-1)
            .astype(int)
        )
        ba_by_spgrp = ba_by_spcd_mapped.groupby(level=0).sum()
        shares = (ba_by_spgrp / total_ba_m2).sort_values(ascending=False)
        top = shares.iloc[:top_species_groups]
        dominant_species_groups = [
            {
                "spgrpcd": int(spgrpcd),
                "name": _FIA_SPECIES_GROUPS.loc[
                    _FIA_SPECIES_GROUPS["JENKINS_SPGRPCD"] == spgrpcd, "JENKINS_NAME"
                ].iloc[0],
                # Clamp to 1.0: the group sum and total are two independent dask
                # reductions, so FP rounding can push a dominant group's share a
                # hair over 1.0 — which the API's le=1.0 validator rejects on read.
                "basal_area_share": min(1.0, float(share)),
            }
            for spgrpcd, share in top.items()
            if spgrpcd != -1
        ]

        return {
            "type": "tree",
            "tree_count": int(n_trees),
            "basal_area_per_area": basal_area_per_area,
            "tree_density": tree_density,
            "quadratic_mean_diameter": qmd,
            "dominant_species_groups": dominant_species_groups,
        }

    return _compute(total_ba_m2, n_trees, sum_dbh_sq, ba_by_spcd)
