"""Tests for the CHM crown feature handler."""

from unittest.mock import MagicMock, patch

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rioxarray  # noqa: F401
import xarray as xr
from affine import Affine
from etcher.handlers.crown import (
    handle_chm,
    margin_cells,
    resolve_seeds,
    segment,
    trace_crowns,
)
from rasterio.features import rasterize
from shapely.geometry import Polygon

CRS = "EPSG:32617"
ORIGIN = (500000.0, 3600000.0)

SETTINGS = {
    "method": "dalponte2016",
    "min_height": 2.0,
    "max_height": 120.0,
    "min_relative_height": 0.45,
    "min_relative_crown_height": 0.55,
    "max_crown_radius": 10.0,
}


def make_chm(values: np.ndarray, cell: float = 1.0) -> xr.DataArray:
    nrows, ncols = values.shape
    x0, y0 = ORIGIN
    da = xr.DataArray(
        values,
        dims=("y", "x"),
        coords={
            "y": y0 - (np.arange(nrows) + 0.5) * cell,
            "x": x0 + (np.arange(ncols) + 0.5) * cell,
        },
        name="chm",
    )
    return da.rio.write_crs(CRS).rio.write_transform(Affine(cell, 0, x0, 0, -cell, y0))


def synthetic_forest(shape=(240, 260), n_trees=180, seed=7):
    """A CHM of overlapping Gaussian crowns, and the cell of each crown's top."""
    rng = np.random.default_rng(seed)
    nrows, ncols = shape
    rows = rng.integers(3, nrows - 3, n_trees)
    cols = rng.integers(3, ncols - 3, n_trees)
    heights = rng.uniform(8, 35, n_trees)
    widths = rng.uniform(1.5, 4.5, n_trees)
    rr, cc = np.mgrid[0:nrows, 0:ncols]
    chm = np.zeros(shape)
    for r, c, h, w in zip(rows, cols, heights, widths):
        chm = np.maximum(
            chm, h * np.exp(-((rr - r) ** 2 + (cc - c) ** 2) / (2 * w * w))
        )
    chm = chm.astype(np.float32)
    cells = pd.DataFrame({"row": rows, "col": cols}).drop_duplicates()
    return chm, cells


def trees_at(cells: pd.DataFrame, chm: np.ndarray, cell: float = 1.0) -> pd.DataFrame:
    x0, y0 = ORIGIN
    return pd.DataFrame(
        {
            "tree_id": np.arange(1, len(cells) + 1, dtype=np.int32),
            "x": x0 + (cells["col"].to_numpy() + 0.25) * cell,
            "y": y0 - (cells["row"].to_numpy() + 0.75) * cell,
            "height": chm[cells["row"], cells["col"]].astype(float),
        }
    )


def run(chm: xr.DataArray, trees: pd.DataFrame, chunk_size: int):
    seeds = resolve_seeds(trees, chm)
    labels = segment(chm, seeds, SETTINGS)
    transform = chm.rio.transform()
    geometries = trace_crowns(
        labels,
        seeds["row"].to_numpy(),
        seeds["col"].to_numpy(),
        transform,
        chunk_size,
        margin_cells(SETTINGS["max_crown_radius"], transform),
    )
    return seeds, labels, geometries


def burn(geometries, shape, transform) -> np.ndarray:
    """Rasterize polygon i as label i + 1, summing to expose overlaps."""
    out = np.zeros(shape, dtype=np.int64)
    counts = np.zeros(shape, dtype=np.int64)
    for i, g in enumerate(geometries):
        mask = rasterize([(g, 1)], out_shape=shape, transform=transform, fill=0)
        out += mask.astype(np.int64) * (i + 1)
        counts += mask
    return out, counts


class TestResolveSeeds:
    def setup_method(self):
        self.chm = make_chm(np.full((10, 10), 10.0, dtype=np.float32))

    def tree(self, tree_id, row, col, height):
        x0, y0 = ORIGIN
        return {
            "tree_id": tree_id,
            "x": x0 + col + 0.5,
            "y": y0 - row - 0.5,
            "height": height,
        }

    def test_tallest_tree_keeps_shared_cell(self):
        trees = pd.DataFrame(
            [self.tree(1, 2, 2, 10.0), self.tree(2, 2, 2, 12.0), self.tree(3, 5, 5, 9)]
        )
        seeds = resolve_seeds(trees, self.chm)
        assert seeds["tree_id"].tolist() == [2, 3]

    def test_height_tie_goes_to_lower_tree_id(self):
        trees = pd.DataFrame([self.tree(9, 2, 2, 10.0), self.tree(4, 2, 2, 10.0)])
        assert resolve_seeds(trees, self.chm)["tree_id"].tolist() == [4]

    def test_missing_height_loses_shared_cell(self):
        trees = pd.DataFrame([self.tree(1, 2, 2, np.nan), self.tree(2, 2, 2, 3.0)])
        assert resolve_seeds(trees, self.chm)["tree_id"].tolist() == [2]

    def test_outside_chm_dropped(self):
        trees = pd.DataFrame(
            [
                self.tree(1, -1, 3, 10.0),
                self.tree(2, 3, 10, 10.0),
                self.tree(3, 10, 3, 10.0),
                self.tree(4, 3, -1, 10.0),
                self.tree(5, 9, 9, 10.0),
            ]
        )
        assert resolve_seeds(trees, self.chm)["tree_id"].tolist() == [5]

    def test_nodata_cell_dropped(self):
        values = np.full((10, 10), 10.0, dtype=np.float32)
        values[4, 4] = np.nan
        trees = pd.DataFrame([self.tree(1, 4, 4, 10.0), self.tree(2, 1, 1, 10.0)])
        assert resolve_seeds(trees, make_chm(values))["tree_id"].tolist() == [2]

    def test_seed_moves_to_cell_center(self):
        trees = pd.DataFrame(
            [{"tree_id": 1, "x": 500003.9, "y": 3599997.2, "height": 1}]
        )
        seeds = resolve_seeds(trees, self.chm)
        assert (seeds.loc[0, "row"], seeds.loc[0, "col"]) == (2, 3)
        assert (seeds.loc[0, "x"], seeds.loc[0, "y"]) == (500003.5, 3599997.5)


class TestTraceCrowns:
    def test_holes_are_kept(self):
        labels = np.zeros((7, 7), dtype=np.int32)
        labels[1:6, 1:6] = 1
        labels[3, 3] = 0
        transform = Affine(1, 0, 0, 0, -1, 7)
        [polygon] = trace_crowns(labels, np.array([1]), np.array([1]), transform, 4, 5)
        assert isinstance(polygon, Polygon)
        assert len(polygon.interiors) == 1
        assert polygon.area == 24

    def test_split_crown_keeps_seed_piece(self):
        labels = np.zeros((6, 6), dtype=np.int32)
        labels[0:2, 0:2] = 1
        labels[4, 4] = 1
        transform = Affine(1, 0, 0, 0, -1, 6)
        [polygon] = trace_crowns(labels, np.array([0]), np.array([0]), transform, 6, 6)
        assert polygon.area == 4


class TestSegmentAndTrace:
    @pytest.fixture(scope="class")
    def forest(self):
        values, cells = synthetic_forest()
        return make_chm(values), trees_at(cells, values)

    def test_polygons_reproduce_labels(self, forest):
        chm, trees = forest
        seeds, labels, geometries = run(chm, trees, chunk_size=4096)
        assert all(isinstance(g, Polygon) and g.is_valid for g in geometries)
        burned, counts = burn(geometries, labels.shape, chm.rio.transform())
        assert counts.max() == 1
        np.testing.assert_array_equal(burned, labels)

    def test_area_matches_cell_count(self, forest):
        chm, trees = forest
        seeds, labels, geometries = run(chm, trees, chunk_size=4096)
        cell_counts = np.bincount(labels.ravel(), minlength=len(seeds) + 1)[1:]
        areas = np.array([g.area for g in geometries])
        np.testing.assert_allclose(areas, cell_counts)
        assert (cell_counts >= 1).all()

    def test_chunked_matches_unchunked(self, forest):
        chm, trees = forest
        _, labels, whole = run(chm, trees, chunk_size=4096)
        _, chunked_labels, chunked = run(chm.chunk(64), trees, chunk_size=64)
        np.testing.assert_array_equal(chunked_labels, labels)
        assert all(a.equals(b) for a, b in zip(whole, chunked))
        # Some crowns span a chunk boundary.
        spans = [
            g
            for g in whole
            if int(g.bounds[0] - ORIGIN[0]) // 64
            != int(g.bounds[2] - ORIGIN[0] - 1e-9) // 64
        ]
        assert spans


@patch("etcher.handlers.crown.save_features")
@patch("etcher.handlers.crown.load_chm")
@patch("etcher.handlers.crown.load_trees")
def test_handle_chm_writes_one_row_per_crown(mock_trees, mock_chm, mock_save):
    values, cells = synthetic_forest(shape=(80, 90), n_trees=30, seed=3)
    trees = trees_at(cells, values)
    # A duplicate that loses its cell, and a tree off the CHM.
    extra = pd.DataFrame(
        {
            "tree_id": [1000, 1001],
            "x": [trees.loc[0, "x"], ORIGIN[0] - 5],
            "y": [trees.loc[0, "y"], ORIGIN[1] + 5],
            "height": [0.0, 20.0],
        }
    )
    mock_trees.return_value = pd.concat([extra, trees], ignore_index=True)
    mock_chm.return_value = make_chm(values)
    domain = gpd.GeoDataFrame(
        geometry=[Polygon([(500000, 3599920), (500090, 3599920), (500090, 3600000)])],
        crs=CRS,
    )
    feature = {"id": "f", "domain_id": "d"}
    source = {
        "product": "chm",
        "source_inventory_id": "inv",
        "source_chm_grid_id": "grid",
        "crown_segmentation": SETTINGS,
    }

    result = handle_chm(feature, source, domain, MagicMock())

    domain_id, feature_id, gdf = mock_save.call_args.args
    assert (domain_id, feature_id) == ("d", "f")
    assert list(gdf.columns) == ["tree_id", "geometry"]
    assert gdf["tree_id"].dtype == np.int32
    assert gdf["tree_id"].tolist() == trees["tree_id"].tolist()
    assert gdf.crs == CRS
    assert (gdf.geom_type == "Polygon").all()
    assert result["georeference"]["crs"] == CRS
