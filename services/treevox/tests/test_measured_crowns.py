"""Measured crowns (#617): footprints, crown shape, mass, fallbacks, chunking.

Crown shape is checked through treevox's own profile construction
(`build_tree`), so the API profile names reach core's measured-crown function
with the right geometry. Core's own tests cover the function in isolation.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import rioxarray  # noqa: F401
import shapely
import xarray as xr
from affine import Affine
from fastfuels_core.voxelization import voxelize_measured_crown
from treevox import measured_crowns, storage
from treevox import voxelize as v
from treevox._worker import run as worker_run
from treevox.handlers import voxelize as handler

HEIGHT = 20.0
CROWN_RATIO = 0.5
CBH = HEIGHT * (1 - CROWN_RATIO)  # 10 m
MIDPOINT = (CBH + HEIGHT) / 2  # 15 m: where the dual profiles are widest

ALL_PROFILES = list(v.CROWN_PROFILE_MODEL_MAP)
BASE_WIDEST_PROFILES = [
    "purves",
    "cone",
    "cylinder",
    "single_ellipsoid",
    "single_paraboloid",
]

BANDS = [
    "volume_fraction",
    "bulk_density.foliage.live",
    "leaf_area_density",
    "spcd",
    "tree_id",
]


def _source(profile="purves", geometry="crown_feature", **extra):
    source = {
        "crown_profile_model": profile,
        "biomass_source": {
            "type": "allometry",
            "equations": "nsvb",
            "components": ["foliage"],
            "component_states": {"foliage": {"live": 1.0, "dead": 0.0}},
        },
        "crown_geometry_source": (
            {"type": "crown_feature", "feature_id": "f1"}
            if geometry == "crown_feature"
            else {"type": "parametric"}
        ),
        "seed": 42,
        **extra,
    }
    return source


def _row(tree_id=1, x=20.0, y=20.0, height=HEIGHT, crown_ratio=CROWN_RATIO):
    return pd.Series(
        {
            "tree_id": tree_id,
            "x": x,
            "y": y,
            "fia_species_code": 122,
            "fia_status_code": 1,
            "dbh": 35.0,
            "height": height,
            "crown_ratio": crown_ratio,
        }
    )


def _trees(*rows):
    return pd.DataFrame([r.to_dict() for r in rows])


def _disc(cx, cy, radius, res=1.0, top=16.0, shift=(0.0, 0.0)):
    """A disc of CHM cells around (cx, cy). `top` is a constant or f(x, y)."""
    n = int(np.ceil(radius / res)) + 1
    x_west = (np.floor(cx / res) - n) * res + shift[0]
    y_north = (np.floor(cy / res) + n + 1) * res + shift[1]
    size = 2 * n + 2
    cols, rows = np.meshgrid(np.arange(size), np.arange(size))
    xc = x_west + (cols + 0.5) * res
    yc = y_north - (rows + 0.5) * res
    mask = np.hypot(xc - cx, yc - cy) <= radius
    heights = top(xc, yc) if callable(top) else np.full(mask.shape, float(top))
    return v.Footprint(
        mask=mask,
        top=np.where(mask, heights, np.nan),
        origin=(float(x_west), float(y_north)),
        resolution=float(res),
    )


def _crown(row, footprint, profile, hr=1.0, vr=1.0, origin=(0.0, 40.0)):
    """Core's measured crown for a treevox-built tree."""
    tree = v.build_tree(row, _source(profile))
    return voxelize_measured_crown(
        footprint.mask,
        footprint.top,
        footprint.origin,
        footprint.resolution,
        (tree.x, tree.y),
        tree.crown_base_height,
        tree.crown_profile_model,
        origin,
        (hr, vr),
    )


def _measured(row, footprint, profile="purves", hr=1.0, vr=1.0, seed=1):
    tree = v.build_tree(row, _source(profile))
    return tree, v.build_measured_biomass(
        tree, footprint, hr, vr, 0.0, 40.0, "foliage", seed
    )


def _column_z(biomass, offset, footprint, cell, hr=1.0, origin=(0.0, 40.0)):
    """Occupied voxel z indices of the output column under a footprint cell."""
    r, c = cell
    x = footprint.origin[0] + (c + 0.5) * footprint.resolution
    y = footprint.origin[1] - (r + 0.5) * footprint.resolution
    col = int(np.floor((x - origin[0]) / hr)) - offset[2]
    row = int(np.floor((origin[1] - y) / hr)) - offset[1]
    if not (0 <= row < biomass.shape[1] and 0 <= col < biomass.shape[2]):
        return set()
    return set((np.flatnonzero(biomass[:, row, col] > 0) + offset[0]).tolist())


@pytest.fixture
def keep_every_voxel(monkeypatch):
    """Occupy every crown voxel, so occupied voxels show the crown's shape."""
    monkeypatch.setattr(
        v, "sample_occupancy", lambda vf, field, n, seed=None: vf.copy()
    )


class TestCrownShape:
    """Measured crown geometry through treevox's profile construction."""

    @pytest.mark.parametrize("profile", BASE_WIDEST_PROFILES)
    def test_widest_at_base_gives_flat_bottom_at_crown_base(self, profile):
        footprint = _disc(20.0, 20.0, 4.0)
        crown = _crown(_row(), footprint, profile)
        np.testing.assert_allclose(crown.bottom[footprint.mask], CBH)

    @pytest.mark.parametrize(
        "profile,inverse",
        [
            (
                "dual_ellipsoid",
                lambda rho: MIDPOINT - (MIDPOINT - CBH) * np.sqrt(1 - rho**2),
            ),
            ("dual_paraboloid", lambda rho: CBH + (MIDPOINT - CBH) * rho**2),
        ],
    )
    def test_dual_profile_bottom_inverts_lower_branch(self, profile, inverse):
        footprint = _disc(20.5, 20.5, 5.0, top=40.0)
        crown = _crown(_row(x=20.5, y=20.5), footprint, profile)
        rho = _rho(footprint, (20.5, 20.5))
        step = (MIDPOINT - CBH) / 2048
        np.testing.assert_allclose(
            crown.bottom[footprint.mask], inverse(rho), atol=step
        )
        # Flat at the stem, rising toward the edge.
        assert crown.bottom[footprint.mask].min() == pytest.approx(CBH)
        assert crown.bottom[footprint.mask].max() > CBH + 0.5 * (MIDPOINT - CBH)

    def test_beta_bottom_inverts_lower_branch(self):
        footprint = _disc(20.5, 20.5, 5.0, top=40.0)
        tree = v.build_tree(_row(x=20.5, y=20.5), _source("beta"))
        profile = tree.crown_profile_model
        hd = profile.get_max_radius_height()
        z = np.linspace(CBH, hd, 20001)
        ratio = np.maximum.accumulate(
            np.asarray(profile.get_radius_at_height(z)) / profile.get_max_radius()
        )
        rho = _rho(footprint, (20.5, 20.5))
        expected = np.interp(rho, ratio, z)
        crown = _crown(_row(x=20.5, y=20.5), footprint, "beta")
        np.testing.assert_allclose(
            crown.bottom[footprint.mask], expected, atol=(hd - CBH) / 1000
        )
        assert crown.bottom[footprint.mask].max() > CBH

    def test_flat_top(self, keep_every_voxel):
        footprint = _disc(20.0, 20.0, 3.0, top=16.0)
        _, (biomass, offset) = _measured(_row(), footprint)
        for cell in zip(*np.nonzero(footprint.mask)):
            assert _column_z(biomass, offset, footprint, cell) == set(range(10, 16))
        assert np.count_nonzero(biomass.any(axis=0)) == footprint.mask.sum()

    def test_stepped_top(self, keep_every_voxel):
        footprint = _disc(
            20.0, 20.0, 3.0, top=lambda x, y: np.where(x < 20.0, 16.0, 13.0)
        )
        _, (biomass, offset) = _measured(_row(), footprint)
        for r, c in zip(*np.nonzero(footprint.mask)):
            x = footprint.origin[0] + (c + 0.5)
            expected = set(range(10, 16 if x < 20.0 else 13))
            assert _column_z(biomass, offset, footprint, (r, c)) == expected

    def test_sloped_top(self, keep_every_voxel):
        footprint = _disc(20.0, 20.0, 3.0, top=lambda x, y: 12.0 + 0.5 * (x - 16.0))
        _, (biomass, offset) = _measured(_row(), footprint)
        for r, c in zip(*np.nonzero(footprint.mask)):
            top = footprint.top[r, c]
            expected = set(range(10, int(np.ceil(top))))
            assert _column_z(biomass, offset, footprint, (r, c)) == expected

    def test_min_thickness_rule(self, keep_every_voxel):
        """Edge columns, whose profile bottom is above top - 1 voxel, stay one
        voxel thick; the column under the stem reaches the crown base."""
        footprint = _disc(20.5, 20.5, 5.0, top=12.0)
        row = _row(x=20.5, y=20.5)
        crown = _crown(row, footprint, "dual_ellipsoid")
        bottoms = crown.bottom[footprint.mask]
        np.testing.assert_allclose(bottoms.max(), 11.0)
        assert bottoms.min() == pytest.approx(CBH)
        _, (biomass, offset) = _measured(row, footprint, "dual_ellipsoid")
        for r, c in zip(*np.nonzero(footprint.mask)):
            z = _column_z(biomass, offset, footprint, (r, c))
            assert z == ({11} if crown.bottom[r, c] >= 11.0 else {10, 11})

    def test_columns_at_or_below_crown_base_are_empty(self, keep_every_voxel):
        footprint = _disc(
            20.0, 20.0, 3.0, top=lambda x, y: np.where(y < 20.0, CBH, 16.0)
        )
        _, (biomass, offset) = _measured(_row(), footprint)
        for r, c in zip(*np.nonzero(footprint.mask)):
            z = _column_z(biomass, offset, footprint, (r, c))
            assert z == (set() if footprint.top[r, c] <= CBH else set(range(10, 16)))

    def test_crown_entirely_below_base_has_no_volume(self):
        footprint = _disc(20.0, 20.0, 3.0, top=CBH)
        assert not v.footprint_has_volume(footprint, CBH, 1.0)
        _, built = _measured(_row(), footprint)
        assert built is None

    @pytest.mark.parametrize(
        "chm_res,hr,vr,shift",
        [
            (1.0, 1.0, 1.0, (0.0, 0.0)),
            (0.5, 1.0, 0.7, (0.25, -0.15)),
            (1.0, 2.0, 0.5, (0.3, 0.6)),
            (2.0, 0.75, 1.0, (0.1, 0.0)),
        ],
    )
    @pytest.mark.parametrize("profile", ["purves", "dual_ellipsoid", "beta"])
    def test_resolution_and_alignment_preserve_volume(
        self, chm_res, hr, vr, shift, profile
    ):
        footprint = _disc(
            20.0,
            20.0,
            4.0,
            res=chm_res,
            shift=shift,
            top=lambda x, y: 14.0 + 0.3 * (x - 16.0) - 0.2 * (y - 16.0),
        )
        crown = _crown(_row(), footprint, profile, hr=hr, vr=vr)
        active = footprint.mask & (footprint.top > CBH)
        columns = (footprint.top[active] - crown.bottom[active]).sum() * chm_res**2
        assert crown.volume_fraction.sum() * hr * hr * vr == pytest.approx(columns)


def _rho(footprint, stem):
    """rho = d / (d + e) for each footprint cell, as the issue defines it."""
    from scipy.ndimage import distance_transform_edt

    rows, cols = np.nonzero(footprint.mask)
    res = footprint.resolution
    x = footprint.origin[0] + (cols + 0.5) * res
    y = footprint.origin[1] - (rows + 0.5) * res
    d = np.hypot(x - stem[0], y - stem[1])
    edt = distance_transform_edt(np.pad(footprint.mask, 1))[1:-1, 1:-1]
    e = (edt[rows, cols] - 0.5) * res
    return d / (d + e)


class TestMass:
    """A measured crown holds exactly its parametric crown's mass."""

    @pytest.mark.parametrize("profile", ALL_PROFILES)
    def test_measured_mass_equals_parametric_mass(self, profile):
        row = _row()
        footprint = _disc(
            20.0, 20.0, 4.0, top=lambda x, y: 18.0 - 0.3 * np.hypot(x - 20, y - 20)
        )
        tree, (biomass, _) = _measured(row, footprint, profile, hr=1.0, vr=0.5)
        measured_mass = biomass.sum() * 1.0 * 1.0 * 0.5

        df = _trees(row).assign(_cache_key=0)
        cache = v.build_chunk_cache(
            df, 1.0, 0.5, _source(profile, "parametric"), np.random.default_rng(0)
        )
        parametric = cache[0].biomass_arrays[0].sum() * 1.0 * 1.0 * 0.5

        assert measured_mass == pytest.approx(tree.foliage_biomass, rel=1e-9)
        assert measured_mass == pytest.approx(parametric, rel=1e-9)

    def test_inventory_column_biomass_is_kept(self):
        row = _row()
        row["foliage_kg"] = 7.5
        source = _source("dual_paraboloid")
        source["biomass_source"] = {
            "type": "inventory_columns",
            "columns": {"foliage": {"column": "foliage_kg", "unit": "kg"}},
            "components": ["foliage"],
            "component_states": {"foliage": {"live": 1.0, "dead": 0.0}},
        }
        tree = v.build_tree(row, source)
        biomass, _ = v.build_measured_biomass(
            tree, _disc(20.0, 20.0, 3.0), 1.0, 1.0, 0.0, 40.0, "foliage", 3
        )
        assert biomass.sum() == pytest.approx(7.5, rel=1e-9)


def _payload(trees, footprints, source, hr=1.0, vr=1.0, shape=(24, 40, 40)):
    trees = trees.assign(_cache_key=v.compute_cache_keys(trees, source))
    trees = trees.sort_values("height", kind="stable", ignore_index=True)
    buffers = {
        k: np.full(shape, storage.BAND_SPECS[k][1], dtype=storage.BAND_SPECS[k][0])
        for k in BANDS
    }
    payload = {
        "chunk_location": (0, 0),
        "buffers": buffers,
        "trees": trees,
        "hr": hr,
        "vr": vr,
        "x_origin": 0.0,
        "y_origin": 40.0,
        "source_config": source,
        "chunk_y_start": 0,
        "chunk_x_start": 0,
        "y_slice": slice(0, shape[1]),
        "x_slice": slice(0, shape[2]),
        "rng_seed": 7,
    }
    if footprints is not None:
        payload["footprints"] = footprints
        payload["base_seed"] = 42
    return payload


class TestStemInCrown:
    def test_stem_cell_in_crown(self):
        assert v.stem_in_crown(20.0, 20.0, _disc(20.0, 20.0, 3.0))

    def test_stem_inside_box_but_not_crown(self):
        # A disc's box corner is outside the disc.
        footprint = _disc(20.0, 20.0, 3.0)
        x0, y0 = footprint.origin
        assert not v.stem_in_crown(x0 + 0.5, y0 - 0.5, footprint)

    def test_stem_outside_box(self):
        assert not v.stem_in_crown(5.0, 5.0, _disc(20.0, 20.0, 3.0))

    @pytest.mark.parametrize("res", [0.5, 1.0, 2.0])
    def test_cell_edges_follow_floor(self, res):
        footprint = _disc(20.0, 20.0, 3.0, res=res, shift=(0.3, -0.2))
        rows, cols = np.nonzero(footprint.mask)
        x0, y0 = footprint.origin
        for r, c in zip(rows, cols):
            # The west and north edges belong to the cell, like etcher's seeds.
            assert v.stem_in_crown(x0 + c * res, y0 - r * res - 1e-9, footprint)


class TestFallbackAndMatching:
    def test_plan_counts_each_reason(self):
        trees = _trees(_row(1), _row(2, x=10.0), _row(3, x=30.0))
        footprints = {
            1: _disc(20.0, 20.0, 3.0, top=16.0),
            3: _disc(30.0, 20.0, 3.0, top=CBH - 1.0),
            99: _disc(5.0, 5.0, 2.0),  # crown of a tree not in the chunk
        }
        measured, reasons = v.plan_measured_crowns(trees, footprints, 1.0)
        assert set(measured) == {1}
        assert reasons == {
            1: "measured",
            2: "parametric_no_crown",
            3: "parametric_no_volume",
        }

    def test_stem_outside_crown_is_drawn_parametrically(self):
        trees = _trees(_row(1), _row(2, x=8.0, y=8.0)).assign(
            _stem_outside_crown=[False, True]
        )
        footprints = {1: _disc(20.0, 20.0, 3.0, top=16.0)}
        measured, reasons = v.plan_measured_crowns(trees, footprints, 1.0)
        assert set(measured) == {1}
        assert reasons == {1: "measured", 2: "parametric_stem_outside_crown"}

        result = worker_run(_payload(trees, footprints, _source("purves")))
        assert "error" not in result, result.get("error")
        assert result["crown_geometry_counts"] == {
            "measured": 1,
            "parametric_no_crown": 0,
            "parametric_stem_outside_crown": 1,
            "parametric_no_volume": 0,
        }
        # Drawn around its own stem.
        assert (result["buffers"]["tree_id"][:, 27:38, 3:14] == 2).any()

    def test_fallback_trees_are_drawn_parametrically(self):
        trees = _trees(_row(1), _row(2, x=8.0, y=8.0), _row(3, x=31.0, y=31.0))
        footprints = {
            1: _disc(20.0, 20.0, 3.0, top=16.0),
            3: _disc(31.0, 31.0, 3.0, top=CBH - 1.0),
            99: _disc(5.0, 35.0, 2.0),
        }
        result = worker_run(_payload(trees, footprints, _source("purves")))
        assert "error" not in result, result.get("error")
        assert result["crown_geometry_counts"] == {
            "measured": 1,
            "parametric_no_crown": 1,
            "parametric_stem_outside_crown": 0,
            "parametric_no_volume": 1,
        }
        tree_ids = result["buffers"]["tree_id"]
        # Tree 1's voxels lie exactly over its footprint cells.
        mask_1 = (tree_ids == 1).any(axis=0)
        expected = np.zeros_like(mask_1)
        rows, cols = np.nonzero(footprints[1].mask)
        x0, y0 = footprints[1].origin
        expected[(40.0 - y0 + rows).astype(int), (x0 + cols).astype(int)] = True
        assert mask_1.any() and not (mask_1 & ~expected).any()
        # The no-crown and no-volume trees are drawn, around their stems.
        assert (tree_ids[:, 27:38, 3:14] == 2).any()
        assert (tree_ids[:, 4:15, 26:37] == 3).any()
        # Tree 99 is not in the inventory: nothing is drawn under its crown.
        assert not tree_ids[:, 4:7, 4:7].any()

    def test_every_tree_keeps_its_mass(self):
        trees = _trees(_row(1), _row(2, x=8.0, y=8.0), _row(3, x=31.0, y=31.0))
        footprints = {
            1: _disc(20.0, 20.0, 3.0, top=16.0),
            3: _disc(31.0, 31.0, 3.0, top=CBH - 1.0),
        }
        source = _source("dual_ellipsoid")
        result = worker_run(_payload(trees, footprints, source))
        density = result["buffers"]["bulk_density.foliage.live"]
        tree_ids = result["buffers"]["tree_id"]
        for _, row in trees.iterrows():
            mass = density[tree_ids == row["tree_id"]].sum()
            assert mass == pytest.approx(
                v.build_tree(row, source).foliage_biomass, rel=1e-5
            )

    def test_trees_that_cannot_be_built_are_not_counted(self):
        """An unknown species is skipped on either path, and not counted."""
        unknown = _row(2, x=10.5, y=10.5).copy()
        unknown["fia_species_code"] = 999999
        no_crown = _row(3, x=30.5, y=30.5).copy()
        no_crown["fia_species_code"] = 999999
        trees = _trees(_row(1), unknown, no_crown)
        footprints = {
            1: _disc(20.0, 20.0, 3.0, top=16.0),
            2: _disc(10.5, 10.5, 3.0, top=16.0),
        }
        result = worker_run(_payload(trees, footprints, _source("purves")))
        assert "error" not in result, result.get("error")
        assert result["crown_geometry_counts"] == {
            "measured": 1,
            "parametric_no_crown": 0,
            "parametric_stem_outside_crown": 0,
            "parametric_no_volume": 0,
        }
        assert set(np.unique(result["buffers"]["tree_id"])) == {0, 1}

    def test_empty_footprints_draw_everything_parametrically(self):
        trees = _trees(_row(1), _row(2, x=8.0, y=8.0))
        result = worker_run(_payload(trees, {}, _source()))
        assert result["crown_geometry_counts"] == {
            "measured": 0,
            "parametric_no_crown": 2,
            "parametric_stem_outside_crown": 0,
            "parametric_no_volume": 0,
        }


class TestBufferClipping:
    def test_crown_is_clipped_to_the_chunk_buffer(self, keep_every_voxel):
        """A crown reaching past the buffer keeps exactly its in-buffer part."""
        footprint = _disc(20.0, 20.0, 5.0, top=16.0)
        trees = _trees(_row(1))
        full = worker_run(_payload(trees, {1: footprint}, _source()))
        # A buffer starting at absolute (row 17, col 18), 6 x 7 cells.
        payload = _payload(trees, {1: footprint}, _source(), shape=(24, 6, 7))
        payload["chunk_y_start"], payload["chunk_x_start"] = 17, 18
        clipped = worker_run(payload)
        assert "error" not in clipped, clipped.get("error")
        for key in ("tree_id", "volume_fraction", "bulk_density.foliage.live"):
            np.testing.assert_array_equal(
                clipped["buffers"][key], full["buffers"][key][:, 17:23, 18:25]
            )

    def test_crown_outside_the_buffer_writes_nothing(self):
        payload = _payload(
            _trees(_row(1)), {1: _disc(20.0, 20.0, 3.0)}, _source(), shape=(24, 5, 5)
        )
        result = worker_run(payload)
        assert "error" not in result
        assert not result["buffers"]["tree_id"].any()


class TestParametricUnchanged:
    def test_parametric_source_matches_absent_source(self):
        trees = _trees(_row(1), _row(2, x=8.0, y=8.0), _row(3, x=31.0, y=31.0))
        absent = _source()
        del absent["crown_geometry_source"]
        a = worker_run(_payload(trees, None, absent))
        b = worker_run(_payload(trees, None, _source(geometry="parametric")))
        assert "crown_geometry_counts" not in a and "crown_geometry_counts" not in b
        for key in BANDS:
            assert a["buffers"][key].tobytes() == b["buffers"][key].tobytes()

    def test_parametric_payloads_carry_no_footprints(self):
        grid = _grid(_source(geometry="parametric"))
        df = _trees(_row(1))
        layout = handler._plan_grid_layout(grid, _domain(), df)
        df = handler._prepare_tree_chunks(df, layout, grid["source"])
        union = xr.Dataset(
            {
                k: (("z", "y", "x"), np.zeros((layout.dims["nz"], 40, 40), "float32"))
                for k in BANDS
            }
        )
        payloads = handler._build_payloads(
            [(0, 0)],
            union,
            slice(0, 40),
            slice(0, 40),
            df,
            df.groupby(["row_chunk", "col_chunk"]).indices,
            layout,
            grid["source"],
            grid["id"],
        )
        assert "footprints" not in payloads[0] and "base_seed" not in payloads[0]


class TestFootprintsFromCrowns:
    def _chm(self):
        rng = np.random.default_rng(0)
        chm = rng.uniform(5.0, 25.0, (12, 16))
        chm[3, 4] = np.nan
        return chm, Affine(0.5, 0, 100.0, 0, -0.5, 206.0)

    def _cells_polygon(self, transform, cells):
        boxes = [
            shapely.box(*(transform * (c, r + 1)), *(transform * (c + 1, r)))
            for r, c in cells
        ]
        return shapely.union_all(boxes)

    def test_recovers_exact_cells_and_tops(self):
        chm, transform = self._chm()
        cells_a = [(r, c) for r in range(2, 6) for c in range(3, 7)]
        ring = [(r, c) for r in range(7, 12) for c in range(8, 14)]
        hole = {(9, 10), (9, 11)}
        cells_b = [rc for rc in ring if rc not in hole]
        geoms = [
            self._cells_polygon(transform, cells_a),
            self._cells_polygon(transform, cells_b),
        ]
        footprints = measured_crowns.footprints_from_crowns(
            np.array([11, 22]), np.array(geoms, dtype=object), chm, transform
        )
        for tree_id, cells in ((11, cells_a), (22, cells_b)):
            fp = footprints[tree_id]
            r0 = min(r for r, _ in cells)
            c0 = min(c for _, c in cells)
            got = {(r + r0, c + c0) for r, c in zip(*np.nonzero(fp.mask))}
            assert got == {rc for rc in cells if np.isfinite(chm[rc])}
            for r, c in got:
                assert fp.top[r - r0, c - c0] == chm[r, c]
            assert fp.origin == transform * (c0, r0)
            assert fp.resolution == 0.5
        assert (3, 4) not in {  # the no-data cell is left out
            (r + 2, c + 3) for r, c in zip(*np.nonzero(footprints[11].mask))
        }

    def test_crown_without_cells_gets_empty_footprint(self):
        chm, transform = self._chm()
        sliver = shapely.box(100.01, 200.01, 100.02, 200.02)
        footprints = measured_crowns.footprints_from_crowns(
            np.array([5]), np.array([sliver], dtype=object), chm, transform
        )
        assert not footprints[5].mask.any()
        assert not v.footprint_has_volume(footprints[5], 0.0, 1.0)


class TestLoadFootprints:
    def _patch(self, monkeypatch):
        transform = Affine(1.0, 0, 0.0, 0, -1.0, 40.0)
        heights = np.full((40, 40), 18.0)
        heights[0, 0] = -9999.0
        chm = xr.DataArray(
            heights,
            dims=("y", "x"),
            coords={"y": 40.0 - np.arange(40) - 0.5, "x": np.arange(40) + 0.5},
        )
        chm = chm.rio.write_transform(transform).rio.write_nodata(-9999.0)
        crowns = pd.DataFrame(
            {
                "tree_id": np.array([1, 2, 3], dtype="int32"),
                "geometry": [
                    shapely.box(10, 10, 14, 14),
                    shapely.box(20, 20, 23, 22),
                    shapely.box(30, 30, 32, 32),
                ],
            }
        )
        monkeypatch.setattr(
            measured_crowns,
            "load_feature",
            lambda fid: {
                "domain_id": "d1",
                "source": {"source_chm_grid_id": "chm1"},
            },
        )
        monkeypatch.setattr(
            measured_crowns.gpd,
            "read_parquet",
            lambda path, columns: measured_crowns.gpd.GeoDataFrame(
                crowns, geometry="geometry"
            ),
        )
        monkeypatch.setattr(
            measured_crowns, "load_zarr", lambda path: xr.Dataset({"chm": chm})
        )

    def test_matches_by_tree_id_and_ignores_removed_trees(self, monkeypatch):
        self._patch(monkeypatch)
        trees = pd.DataFrame(
            {
                "tree_id": np.array([1, 3, 4], dtype="int32"),
                "x": [12.0, 31.0, 5.0],
                "y": [12.0, 31.0, 5.0],
            }
        )
        result = measured_crowns.load_footprints("f1", trees)
        assert set(result.footprints) == {1, 3}
        assert result.ignored_crowns == 1  # tree 2 is not in the inventory
        assert result.footprints[1].mask.sum() == 16
        assert result.footprints[3].mask.sum() == 4
        assert result.max_top == 18.0
        assert result.stem_outside == frozenset()

    def test_tiles_do_not_change_footprints(self, monkeypatch):
        """Crowns straddling tile edges are cut out as from one window."""
        self._patch(monkeypatch)
        trees = pd.DataFrame(
            {
                "tree_id": np.array([1, 2, 3], dtype="int32"),
                "x": [12.0, 21.0, 31.0],
                "y": [12.0, 21.0, 31.0],
            }
        )
        whole = measured_crowns.load_footprints("f1", trees)
        monkeypatch.setattr(measured_crowns, "TILE_CELLS", 7)
        tiled = measured_crowns.load_footprints("f1", trees)
        assert set(tiled.footprints) == set(whole.footprints) == {1, 2, 3}
        for tree_id, footprint in whole.footprints.items():
            other = tiled.footprints[tree_id]
            np.testing.assert_array_equal(other.mask, footprint.mask)
            np.testing.assert_array_equal(other.top, footprint.top)
            assert other.origin == footprint.origin
        assert tiled.max_top == whole.max_top

    def test_sets_aside_a_crown_that_does_not_hold_its_stem(self, monkeypatch):
        """Tree 3's crown is at (31, 31), but its stem is at (5, 5)."""
        self._patch(monkeypatch)
        trees = pd.DataFrame(
            {
                "tree_id": np.array([1, 3], dtype="int32"),
                "x": [12.0, 5.0],
                "y": [12.0, 5.0],
            }
        )
        result = measured_crowns.load_footprints("f1", trees)
        assert set(result.footprints) == {1}
        assert result.stem_outside == frozenset({3})


def _domain(n=40):
    return SimpleNamespace(
        total_bounds=np.array([0.0, 0.0, float(n), float(n)]), crs="EPSG:32610"
    )


def _grid(source):
    return {
        "id": "g-measured",
        "domain_id": "d1",
        "source": {
            "operation": "voxelize",
            "input": "inventory",
            "entity": "tree",
            "source_inventory_id": "inv1",
            "resolution": {"horizontal": 1.0, "vertical": 1.0},
            **source,
        },
        "bands": [{"key": k} for k in BANDS],
    }


class _SerialPool:
    def map(self, fn, items):
        return [fn(item) for item in items]


def _run_job(tmp_path, monkeypatch, df, footprints, source, chunk_xy, name):
    """Run every batch of a job against a local zarr store; return the grid."""
    monkeypatch.setattr(handler.voxelize, "CHUNK_SIZE_HORIZONTAL", chunk_xy)
    grid = _grid(source)
    max_top = max(np.nanmax(f.top) for f in footprints.values())
    layout = handler._plan_grid_layout(
        grid, _domain(), df, min_top=max_top, footprints=footprints
    )
    path = str(tmp_path / name)
    storage.init_store(
        path,
        x_coords=layout.dims["x_coords"],
        y_coords=layout.dims["y_coords"],
        z_coords=layout.dims["z_coords"],
        hr=layout.dims["hr"],
        vr=layout.dims["vr"],
        crs=layout.dims["crs"],
        z_origin=layout.dims["z_origin"],
        requested_keys=layout.requested_keys,
        chunk_shape=layout.chunk_shape,
    )
    df = handler._prepare_tree_chunks(df, layout, grid["source"])
    chunk_indices = df.groupby(["row_chunk", "col_chunk"], sort=False).indices
    counts = []
    for i in range(0, len(layout.chunk_locations), 2):
        batch = layout.chunk_locations[i : i + 2]
        stats = handler._process_batch(
            _SerialPool(),
            batch,
            i,
            0,
            df,
            chunk_indices,
            layout,
            grid["source"],
            grid["id"],
            path,
            footprints,
        )
        counts.append(stats["crown_geometry_counts"])
    ds = xr.open_zarr(path, consolidated=True).load()
    return ds, handler._sum_counts(counts), layout


class TestChunking:
    def _stand(self):
        """Measured trees whose crowns cross the chunk seams at 20 m."""
        stems = [(18.5, 21.5), (21.5, 18.5), (19.5, 19.5), (10.5, 30.5), (30.5, 10.5)]
        rows, footprints = [], {}
        for i, (x, y) in enumerate(stems, start=1):
            rows.append(_row(i, x=x, y=y, height=18.0 + i))
            footprints[i] = _disc(
                x,
                y,
                3.0,
                top=lambda xx, yy, h=18.0 + i: h - 0.4 * np.hypot(xx - x, yy - y),
            )
        # Split overlapping discs so no cell belongs to two crowns.
        claimed = set()
        for i, fp in footprints.items():
            rows_i, cols_i = np.nonzero(fp.mask)
            for r, c in zip(rows_i, cols_i):
                key = (round(fp.origin[0] + c), round(fp.origin[1] - r))
                if key in claimed:
                    fp.mask[r, c] = False
                    fp.top[r, c] = np.nan
                claimed.add(key)
        return _trees(*rows), footprints

    @pytest.mark.parametrize("profile", ["dual_ellipsoid", "purves"])
    def test_chunked_equals_unchunked(self, tmp_path, monkeypatch, profile):
        df, footprints = self._stand()
        source = _source(profile)
        one, counts_one, layout_one = _run_job(
            tmp_path, monkeypatch, df, footprints, source, 500, "one"
        )
        four, counts_four, layout_four = _run_job(
            tmp_path, monkeypatch, df, footprints, source, 20, "four"
        )
        assert len(layout_one.chunk_locations) == 1
        assert len(layout_four.chunk_locations) == 4
        assert (
            counts_one
            == counts_four
            == {
                "measured": 5,
                "parametric_no_crown": 0,
                "parametric_stem_outside_crown": 0,
                "parametric_no_volume": 0,
            }
        )
        for key in BANDS:
            np.testing.assert_array_equal(one[key].values, four[key].values)
        # The seam crowns really did cross into neighbouring chunks.
        tree_ids = one["tree_id"].values.any(axis=0)
        assert tree_ids[:20, :20].any() and tree_ids[20:, 20:].any()

    def test_crown_wider_than_default_halo_is_not_clipped(self, tmp_path, monkeypatch):
        """A crown reaching 14 cells from a stem on a chunk's first cell."""
        df = _trees(_row(1, x=20.5, y=20.5, height=22.0))
        footprints = {
            1: _disc(
                20.5,
                20.5,
                14.0,
                top=lambda xx, yy: 22.0 - 0.3 * np.hypot(xx - 20.5, yy - 20.5),
            )
        }
        source = _source("dual_ellipsoid")
        one, _, _ = _run_job(tmp_path, monkeypatch, df, footprints, source, 500, "one")
        four, _, layout = _run_job(
            tmp_path, monkeypatch, df, footprints, source, 20, "four"
        )
        assert layout.halo_cells > handler.voxelize.OVERLAP_CELLS
        occupied = np.flatnonzero(one["volume_fraction"].values.any(axis=(0, 1)))
        assert 20 - occupied.min() > handler.voxelize.OVERLAP_CELLS
        for key in BANDS:
            np.testing.assert_array_equal(one[key].values, four[key].values)

    def test_grid_reaches_highest_chm_top(self):
        df = _trees(_row(1, height=12.0))
        layout = handler._plan_grid_layout(
            _grid(_source()), _domain(), df, min_top=21.3
        )
        assert layout.dims["nz"] == 22


class TestVoxelizeInventoryWithCrownFeature:
    def test_job_reports_counts_on_the_grid(self, tmp_path, monkeypatch, caplog):
        df = _trees(_row(1), _row(2, x=8.0, y=8.0), _row(3, x=31.0, y=31.0))
        footprints = {
            1: _disc(20.0, 20.0, 3.0, top=22.4),
            3: _disc(31.0, 31.0, 3.0, top=CBH - 1.0),
        }
        loaded = []

        def fake_load_footprints(feature_id, trees):
            loaded.append((feature_id, sorted(trees["tree_id"])))
            # Tree 2's crown was set aside: it does not hold tree 2's stem.
            return measured_crowns.MeasuredFootprints(
                footprints=footprints,
                max_top=22.4,
                ignored_crowns=4,
                stem_outside=frozenset({2}),
            )

        monkeypatch.setattr(handler, "read_inventory", lambda *a, **k: df.copy())
        monkeypatch.setattr(
            handler.measured_crowns, "load_footprints", fake_load_footprints
        )
        monkeypatch.setattr(
            handler.storage, "gcs_path", lambda grid_id: str(tmp_path / grid_id)
        )
        monkeypatch.setattr(
            handler.multiprocessing, "get_context", lambda method: _SerialContext()
        )

        # treevox.main stops the package logger propagating to caplog's handler.
        monkeypatch.setattr(logging.getLogger("treevox"), "propagate", True)
        with caplog.at_level("INFO", logger=handler.logger.name):
            result = handler.voxelize_inventory(
                _grid(_source("dual_ellipsoid")), _domain(), lambda *a, **k: None
            )

        assert loaded == [("f1", [1, 2, 3])]
        assert "raise the grid top from 20 m, the tallest tree, to 22.4 m" in (
            caplog.text
        )
        assert result.source_updates == {
            "crown_geometry_counts": {
                "measured": 1,
                "parametric_no_crown": 0,
                "parametric_stem_outside_crown": 1,
                "parametric_no_volume": 1,
            }
        }
        # The grid reaches the measured crown's CHM top, above every tree.
        assert result.georeference["shape"][0] == 23
        ds = xr.open_zarr(result.gcs_path, consolidated=True).load()
        assert set(np.unique(ds["tree_id"].values)) == {0, 1, 2, 3}

    def test_parametric_job_has_no_source_updates(self, tmp_path, monkeypatch):
        df = _trees(_row(1))
        monkeypatch.setattr(handler, "read_inventory", lambda *a, **k: df.copy())
        monkeypatch.setattr(
            handler.storage, "gcs_path", lambda grid_id: str(tmp_path / grid_id)
        )
        monkeypatch.setattr(
            handler.multiprocessing, "get_context", lambda method: _SerialContext()
        )

        def fail(*a, **k):
            raise AssertionError("parametric grids read no crown Feature")

        monkeypatch.setattr(handler.measured_crowns, "load_footprints", fail)
        result = handler.voxelize_inventory(
            _grid(_source(geometry="parametric")), _domain(), lambda *a, **k: None
        )
        assert result.source_updates is None


class _PoolContext(_SerialPool):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _SerialContext:
    def Pool(self, processes):
        return _PoolContext()


class TestMeasuredHaloCells:
    def _dims(self, hr=1.0, n=40):
        return {"hr": hr, "nx": n, "ny": n, "x_origin": 0.0, "y_origin": float(n)}

    def test_reach_from_stem_cell(self):
        df = _trees(_row(1, x=20.5, y=20.5))
        # Disc cells span x = 7 .. 34, so 13 cells west and 13 east of col 20.
        footprints = {1: _disc(20.5, 20.5, 13.0)}
        assert v.measured_halo_cells(df, footprints, self._dims()) == 13

    def test_trees_without_a_footprint_are_ignored(self):
        df = _trees(_row(1, x=20.5, y=20.5))
        assert v.measured_halo_cells(df, {2: _disc(5.5, 5.5, 30.0)}, self._dims()) == 0

    @pytest.mark.parametrize("hr", [0.3, 0.5, 1.0, 2.0])
    @pytest.mark.parametrize("shift", [(0.0, 0.0), (0.37, -0.81), (0.5, 0.5)])
    def test_covers_core_placement(self, hr, shift):
        """Every voxel core places lies within the reach of the stem's cell."""
        dims = self._dims(hr=hr, n=int(round(40 / hr)))
        dims["x_origin"], dims["y_origin"] = 0.0, 40.0
        rng = np.random.default_rng(0)
        for i in range(20):
            x, y = rng.uniform(12.0, 28.0, size=2)
            row = _row(1, x=x, y=y, height=20.0)
            footprint = _disc(
                x + rng.uniform(-1, 1),
                y + rng.uniform(-1, 1),
                rng.uniform(1.0, 9.0),
                res=float(rng.choice([0.5, 1.0, 2.0])),
                top=20.0,
                shift=shift,
            )
            crown = _crown(row, footprint, "purves", hr=hr, origin=(0.0, 40.0))
            if crown is None:
                continue
            reach = v.measured_halo_cells(_trees(row), {1: footprint}, dims)
            stem_col = int(np.floor(x / hr))
            stem_row = int(np.floor((40.0 - y) / hr))
            _, r0, c0 = crown.offset
            _, ny, nx = crown.volume_fraction.shape
            assert stem_col - c0 <= reach and c0 + nx - 1 - stem_col <= reach
            assert stem_row - r0 <= reach and r0 + ny - 1 - stem_row <= reach
