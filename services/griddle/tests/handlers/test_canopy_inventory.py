"""Tests for the inventory canopy handler.

The science is fastfuels-core's and is tested there; these tests cover the ETL
this handler owns: translating the persisted ``source`` into the exact core
kwargs, resolving the output lattice, initialising and finalising the band
Dataset, and turning bad input into terminal ProcessingErrors. Trees are
placed at known coordinates so cell membership is known by construction, and
``read_inventory`` is replaced so nothing touches GCS.
"""

from unittest.mock import patch

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from griddle.handlers import canopy_inventory as ci
from shapely.geometry import box

from lib.errors import ProcessingError

CRS = "EPSG:5070"  # meters, so cell coordinates are the tree coordinates


def _source(**overrides) -> dict:
    """A fully-resolved persisted inventory-canopy source (zero-config path).

    Mirrors what the API router persists after resolving defaults: every
    modeling choice is concrete and every requested band carries a method.
    """
    source = {
        "name": "canopy",
        "product": "inventory",
        "source_inventory_id": "inv_test",
        "alignment": {"target": "domain", "resolution": 30.0},
        "bands": ["cbd", "cbh", "chm", "cc"],
        "biomass_source": {"type": "allometry", "equations": "nsvb"},
        "available_fuel": {
            "foliage_fraction": 1.0,
            "branchwood": {"size_partition": "none", "fraction": 0.075},
        },
        "species_inclusion": "all_species",
        "crown_class_adjustment": {"method": "none"},
        "min_tree_height": 0.0,
        "vertical_distribution": "reinhardt_2006",
        "layer_depth": 0.3048,
        "horizontal_distribution": "crown_projected",
        "max_crown_radius_source": {"type": "allometry", "equations": "purves"},
        "cbd": {
            "method": "maximum_running_mean",
            "window": 3.0,
            "edge": "ground_clamped",
        },
        "cbh": {
            "method": "bulk_density_threshold",
            "threshold": 0.012,
            "relative_threshold_fraction": 0.1,
            "smoothing_window": None,
            "smoothing_edge": "ground_clamped",
        },
        "chm": {
            "method": "bulk_density_threshold",
            "threshold": 0.012,
            "relative_threshold_fraction": 0.1,
            "smoothing_window": None,
            "smoothing_edge": "ground_clamped",
        },
        "cc": {"method": "crown_union"},
    }
    source.update(overrides)
    return source


def _roi(size_m: float = 120.0) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(geometry=[box(0.0, 0.0, size_m, size_m)], crs=CRS)


def _trees(n: int = 40, species: int = 122, seed: int = 0, **cols) -> pd.DataFrame:
    """Live conifers clustered inside a 120 m box (ponderosa pine by default)."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "x": rng.uniform(10, 110, n),
            "y": rng.uniform(10, 110, n),
            "fia_species_code": np.full(n, species),
            "fia_status_code": np.ones(n, dtype=int),
            "dbh": rng.uniform(15, 45, n),
            "height": rng.uniform(8, 25, n),
            "crown_ratio": rng.uniform(0.3, 0.6, n),
        }
    )
    for key, value in cols.items():
        df[key] = value
    return df


def _run(source, df, **kwargs):
    with patch.object(ci, "read_inventory", return_value=df):
        return ci.fetch_canopy_inventory(
            roi=_roi(),
            source=source,
            alignment=source["alignment"],
            target_grid_doc=None,
            progress=lambda *a, **k: None,
            **kwargs,
        )


class TestTranslator:
    def test_default_source_maps_to_fastfuels_native_kwargs(self):
        kw = ci._core_kwargs(_source())
        assert kw["equations"] == "nsvb"
        assert kw["branchwood_size_partition"] == "none"
        assert kw["foliage_fraction"] == 1.0
        assert kw["branchwood_fraction"] == 0.075
        assert kw["exclude_hardwoods"] is False
        assert kw["crown_class_adjustment"] == "none"
        assert "crown_class_column" not in kw
        assert kw["crown_radius_equations"] == "purves"
        assert kw["horizontal_distribution"] == "crown_projected"
        assert kw["vertical_distribution"] == "reinhardt_2006"
        assert kw["cover_method"] == "crown_union"

    def test_running_mean_edge_is_translated(self):
        # schema edge names -> core edge names
        for schema_edge, core_edge in [
            ("fixed_depth", "slab"),
            ("ground_clamped", "fuelcalc"),
            ("truncated", "truncate"),
        ]:
            src = _source(
                cbd={
                    "method": "maximum_running_mean",
                    "window": 3.0,
                    "edge": schema_edge,
                }
            )
            assert ci._core_kwargs(src)["cbd_window_edge"] == core_edge

    def test_species_inclusion_maps_to_exclude_hardwoods(self):
        assert ci._core_kwargs(_source())["exclude_hardwoods"] is False
        assert (
            ci._core_kwargs(_source(species_inclusion="fuelcalc_default"))[
                "exclude_hardwoods"
            ]
            is True
        )

    def test_inventory_column_biomass_sets_fuel_column(self):
        src = _source(
            biomass_source={
                "type": "inventory_column",
                "column": "my_fuel",
                "unit": "kg",
            },
            available_fuel=None,
        )
        kw = ci._core_kwargs(src)
        assert kw["fuel_column"] == "my_fuel"
        assert "equations" not in kw
        assert "branchwood_size_partition" not in kw
        assert ci._inventory_columns(src)[0] == "my_fuel"

    def test_inventory_column_crown_radius_sets_column(self):
        src = _source(
            max_crown_radius_source={
                "type": "inventory_column",
                "column": "lidar_r",
                "unit": "m",
            }
        )
        kw = ci._core_kwargs(src)
        assert kw["crown_radius_column"] == "lidar_r"
        assert "crown_radius_equations" not in kw
        assert ci._inventory_columns(src)[1] == "lidar_r"

    def test_crown_class_fuelcalc_table_names_synthetic_column(self):
        src = _source(
            crown_class_adjustment={
                "method": "fuelcalc_table",
                "missing_crown_class": "other_none",
            }
        )
        kw = ci._core_kwargs(src)
        assert kw["crown_class_adjustment"] == "fuelcalc_table"
        assert kw["crown_class_column"] == ci._CROWN_CLASS_COLUMN

    def test_brown_equations_partition_maps_to_brown_proportions(self):
        src = _source(
            biomass_source={"type": "allometry", "equations": "brown_1978"},
            available_fuel={
                "foliage_fraction": 1.0,
                "branchwood": {"size_partition": "equations", "fraction": 0.5},
            },
        )
        kw = ci._core_kwargs(src)
        assert kw["equations"] == "brown_1978"
        assert kw["branchwood_size_partition"] == "brown_proportions"

    def test_load_over_depth_cbd_sets_depth_not_window(self):
        src = _source(cbd={"method": "load_over_depth", "depth": "canopy_depth"})
        kw = ci._core_kwargs(src)
        assert kw["cbd_method"] == "load_over_depth"
        assert kw["cbd_depth"] == "canopy_depth"
        assert "cbd_window" not in kw

    def test_alternate_cbh_chm_cc_methods(self):
        src = _source(
            bands=["cbh", "chm", "cc"],
            cbd=None,
            cbh={"method": "mean_crown_base"},
            chm={"method": "height_percentile", "percentile": 95.0},
            cc={"method": "cover_fraction", "height_threshold": 3.0},
        )
        kw = ci._core_kwargs(src)
        assert kw["cbh_method"] == "mean_crown_base"
        assert "cbh_threshold" not in kw
        assert kw["chm_method"] == "height_percentile"
        assert kw["chm_percentile"] == 95.0
        assert kw["cover_method"] == "cover_fraction"
        assert kw["cover_height_threshold"] == 3.0

    def test_unrequested_bands_omit_method_kwargs(self):
        # Only cc requested: no cbd/cbh/chm method kwargs, so core's valid
        # defaults stand in and those bands are not computed.
        src = _source(bands=["cc"], cbd=None, cbh=None, chm=None)
        kw = ci._core_kwargs(src)
        assert not any(k.startswith(("cbd_", "cbh_", "chm_")) for k in kw)
        assert kw["cover_method"] == "crown_union"


class TestLattice:
    def test_domain_target_resolves_expected_grid(self):
        transform, shape = ci._resolve_lattice(
            _roi(120.0), {"target": "domain", "resolution": 30.0}, None, 0
        )
        assert shape == (4, 4)
        assert transform.a == 30.0 and transform.e == -30.0

    def test_native_target_is_rejected(self):
        with pytest.raises(ProcessingError) as exc:
            ci._resolve_lattice(_roi(), {"target": "native"}, None, 0)
        assert exc.value.code == "UNSUPPORTED_ALIGNMENT"

    def test_crs_mismatch_is_rejected(self):
        target_grid_doc = {
            "georeference": {
                "crs": "EPSG:32612",  # different from the domain CRS
                "transform": [30.0, 0.0, 0.0, 0.0, -30.0, 120.0],
                "shape": [4, 4],
            }
        }
        with pytest.raises(ProcessingError) as exc:
            ci._resolve_lattice(
                _roi(), {"target": "grid", "resolution": None}, target_grid_doc, 0
            )
        assert exc.value.code == "ALIGNMENT_CRS_MISMATCH"


class TestEndToEnd:
    def test_default_bands_are_georeferenced_float32_non_forest_zero(self):
        ds = _run(_source(bands=["cbd", "cbh", "chm", "cc", "cfl"]), _trees())
        assert list(ds.data_vars) == ["cbd", "cbh", "chm", "cc", "cfl"]
        assert dict(ds.sizes) == {"y": 4, "x": 4}
        assert str(ds.rio.crs) == CRS
        for key in ds.data_vars:
            arr = ds[key].data
            assert arr.dtype == np.float32
            # NaN was filled to 0, so every cell is finite (non-forest -> 0).
            assert np.isfinite(arr).all()
        # Something got fuel; the grid is not uniformly zero.
        assert ds["cbd"].data.max() > 0.0
        assert ds["cfl"].data.max() > 0.0

    def test_empty_lattice_cells_are_zero(self):
        # All trees in the lower-left cell; the far corner cell must be 0.
        df = _trees(n=20)
        df["x"] = np.linspace(2, 28, len(df))
        df["y"] = np.linspace(2, 28, len(df))
        ds = _run(_source(), df)
        # Row 0 (top, high y) col 3 (right) is far from the cluster -> empty.
        assert ds["cbd"].data[0, 3] == 0.0
        assert ds["cbh"].data[0, 3] == 0.0

    def test_crown_class_adjustment_lowers_bulk_density(self):
        # The fuelcalc_table arm attaches a uniform "N" crown class, whose
        # Other/none factor (~0.5 for ponderosa) halves crown weight, so CBD
        # must come out below the unadjusted run on the same stand.
        df = _trees()
        base = _run(_source(), df.copy())
        adjusted = _run(
            _source(
                crown_class_adjustment={
                    "method": "fuelcalc_table",
                    "missing_crown_class": "other_none",
                }
            ),
            df.copy(),
        )
        assert adjusted["cbd"].data.max() < base["cbd"].data.max()

    def test_crown_class_synthetic_column_reaches_core(self):
        # The handler must attach the crown-class column to the frame it hands
        # core; without it core raises on the fuelcalc_table arm.
        captured = {}

        def fake_compute(trees, dataset, **kwargs):
            captured["columns"] = list(trees.columns)
            captured["crown_class_column"] = kwargs.get("crown_class_column")
            return dataset

        src = _source(
            crown_class_adjustment={
                "method": "fuelcalc_table",
                "missing_crown_class": "other_none",
            }
        )
        with patch.object(ci, "compute_canopy_metrics", side_effect=fake_compute):
            _run(src, _trees())
        assert ci._CROWN_CLASS_COLUMN in captured["columns"]
        assert captured["crown_class_column"] == ci._CROWN_CLASS_COLUMN


class TestErrorPaths:
    def test_empty_inventory_raises_terminal_error(self):
        empty = _trees(n=0)
        with pytest.raises(ProcessingError) as exc:
            _run(_source(), empty)
        assert exc.value.code == "EMPTY_INVENTORY"

    def test_unpriceable_species_becomes_input_error(self):
        # Under the FuelCalc fine-share crosswalk (brown_proportions), a species
        # outside the table raises in core; the handler turns that terminal.
        df = _trees(species=999)
        src = _source(
            available_fuel={
                "foliage_fraction": 1.0,
                "branchwood": {"size_partition": "brown_proportions", "fraction": 0.5},
            }
        )
        with pytest.raises(ProcessingError) as exc:
            _run(src, df)
        assert exc.value.code == "CANOPY_FUEL_INPUT_ERROR"
        assert exc.value.suggestion is not None


class TestRequiredColumns:
    """The handler reads and requires non-null only the columns the selected
    methods consume — the same set the API router validates — so an inventory
    without dbh / species is not rejected or thinned on a path that ignores them.
    """

    def _capture(self, source, df):
        """Run the handler, returning the required_columns passed to the reader
        and the null-row drop."""
        captured: dict = {}
        real_drop = ci.drop_null_rows

        def cap_read(*args, **kwargs):
            captured["read"] = kwargs.get("required_columns")
            return df

        def cap_drop(frame, *args, **kwargs):
            captured["drop"] = kwargs.get("required_columns")
            return real_drop(frame, *args, **kwargs)

        with (
            patch.object(ci, "read_inventory", side_effect=cap_read),
            patch.object(ci, "drop_null_rows", side_effect=cap_drop),
        ):
            ci.fetch_canopy_inventory(
                roi=_roi(),
                source=source,
                alignment=source["alignment"],
                target_grid_doc=None,
                progress=lambda *a, **k: None,
            )
        return captured

    def test_column_fuel_source_requires_neither_dbh_nor_species(self):
        rng = np.random.default_rng(0)
        n = 40
        # A frame with no dbh / fia_species_code, as the trimmed read returns.
        df = pd.DataFrame(
            {
                "x": rng.uniform(10, 110, n),
                "y": rng.uniform(10, 110, n),
                "height": rng.uniform(8, 25, n),
                "crown_ratio": rng.uniform(0.3, 0.6, n),
                "fia_status_code": np.ones(n, dtype=int),
                "acf_kg": np.full(n, 6.0),
                "crad_m": np.full(n, 2.0),
            }
        )
        src = _source(
            biomass_source={
                "type": "inventory_column",
                "column": "acf_kg",
                "unit": "kg",
            },
            vertical_distribution="uniform",
            species_inclusion="all_species",
            crown_class_adjustment={"method": "none"},
            max_crown_radius_source={
                "type": "inventory_column",
                "column": "crad_m",
                "unit": "m",
            },
        )
        captured = self._capture(src, df)
        assert set(captured["read"]) == {"x", "y", "height", "crown_ratio"}
        assert set(captured["drop"]) == {"x", "y", "height", "crown_ratio"}

    def test_allometry_source_requires_dbh_and_species(self):
        captured = self._capture(_source(), _trees())
        assert {"dbh", "fia_species_code"} <= set(captured["read"])
        assert {"dbh", "fia_species_code"} <= set(captured["drop"])
