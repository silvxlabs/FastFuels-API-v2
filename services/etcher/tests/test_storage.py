"""Tests for feature storage."""

import json
from unittest.mock import patch

import geopandas as gpd
import pyarrow.parquet as pq
from etcher.storage import save_features
from shapely.geometry import Polygon

ATTRIBUTIONS = [
    {
        "dataset": "OpenStreetMap",
        "license_name": "ODbL-1.0",
        "license_url": "https://opendatacommons.org/licenses/odbl/1-0/",
        "citation": "© OpenStreetMap contributors",
        "access_url": "https://www.openstreetmap.org/copyright",
        "accessed_on": "2026-11-05",
    }
]


def _sample_gdf(n=2500):
    polys = [Polygon([(i, 0), (i + 1, 0), (i + 1, 1), (i, 1)]) for i in range(n)]
    return gpd.GeoDataFrame(
        {"name": [f"f{i}" for i in range(n)], "geometry": polys}, crs="EPSG:32610"
    )


def _save_local(tmp_path, gdf, **kwargs):
    """Run save_features, redirecting the gs:// write to a local file."""
    local = tmp_path / "feature.parquet"
    real_write = pq.write_table
    calls = []

    def write_local(table, path, **write_kwargs):
        calls.append((path, write_kwargs))
        real_write(table, str(local), **write_kwargs)

    with patch("etcher.storage.pq.write_table", side_effect=write_local):
        returned = save_features("dom-1", "feat-1", gdf, **kwargs)

    return local, returned, calls


def test_attributions_in_metadata(tmp_path):
    local, _, _ = _save_local(tmp_path, _sample_gdf(), attributions=ATTRIBUTIONS)

    metadata = pq.read_schema(local).metadata
    assert json.loads(metadata[b"attributions"]) == ATTRIBUTIONS


def test_geo_metadata_intact(tmp_path):
    gdf = _sample_gdf()
    local, _, _ = _save_local(tmp_path, gdf, attributions=ATTRIBUTIONS)

    geo = json.loads(pq.read_schema(local).metadata[b"geo"])
    assert geo["primary_column"] == "geometry"

    result = gpd.read_parquet(local)
    assert isinstance(result, gpd.GeoDataFrame)
    assert result.crs == gdf.crs
    assert result.geometry.equals(gdf.geometry)
    assert list(result["name"]) == list(gdf["name"])


def test_format_unchanged(tmp_path):
    local, returned, calls = _save_local(
        tmp_path, _sample_gdf(), attributions=ATTRIBUTIONS
    )

    ((path, write_kwargs),) = calls
    assert path == returned
    assert path.startswith("gs://") and path.endswith("/dom-1/feat-1.parquet")
    assert write_kwargs == {"compression": "zstd", "row_group_size": 1000}

    meta = pq.ParquetFile(local).metadata
    assert meta.num_row_groups == 3
    assert meta.row_group(0).column(0).compression == "ZSTD"


def test_attributions_default_empty(tmp_path):
    local, _, _ = _save_local(tmp_path, _sample_gdf(10))

    assert json.loads(pq.read_schema(local).metadata[b"attributions"]) == []
