"""
Tests for the point-cloud CHM handler.

Points are placed at exact coordinates so the expected canopy height of each
cell is known by construction rather than asserted against a recorded output.
Most tests replace the point reader so the algorithm is exercised without GCS;
`TestReadingFromStorage` drives the real one against a dataset on disk.
"""

from contextlib import ExitStack
from unittest.mock import patch

import geopandas as gpd
import numpy as np
import pyproj
import pytest
from griddle.handlers import chm_point_cloud
from pyarrow.fs import LocalFileSystem
from scipy.ndimage import distance_transform_edt
from shapely.geometry import box

from lib.errors import ProcessingError
from lib.pointcloud.reader import open_dataset, read_points
from lib.pointcloud.schema import tile_span

CRS = "EPSG:32612"


def _chunks(x, y, z, classification):
    """Build a ``_read_points`` replacement over an in-memory cloud.

    Applies the class filter the real reader pushes into Parquet, and ignores
    the bounds: a block drops what falls outside it by cell index anyway, so
    returning the whole cloud exercises the same code path a real partition
    read would.
    """

    def _read(_dataset, _manifest, _bounds, classes):
        xs = np.asarray(x, dtype=float)
        ys = np.asarray(y, dtype=float)
        zs = np.asarray(z, dtype=float)
        cs = np.asarray(classification, dtype=np.uint8)
        if classes is not None:
            keep = np.isin(cs, list(classes))
            return xs[keep], ys[keep], zs[keep], cs[keep]
        return xs, ys, zs, cs

    return _read


# Stands in for the dataset manifest: millimetre scaling anchored at the origin,
# with one tile wide enough that the fixtures never straddle a partition.
_MANIFEST = {
    "tile_m": 500.0,
    "mins": [0.0, 0.0, 0.0],
    "scales": [0.001, 0.001, 0.001],
    "offsets": [0.0, 0.0, 0.0],
}


def _roi(size: float = 4.0) -> gpd.GeoDataFrame:
    """A square domain anchored at the origin, `size` metres on a side."""
    return gpd.GeoDataFrame(geometry=[box(0.0, 0.0, size, size)], crs=CRS)


def _flat_ground(size: int = 4, elevation: float = 10.0):
    """One ground return at the centre of every cell of a `size`x`size` grid."""
    xs, ys, zs, classes = [], [], [], []
    for row in range(size):
        for col in range(size):
            xs.append(col + 0.5)
            ys.append(size - row - 0.5)
            zs.append(elevation)
            classes.append(2)
    return xs, ys, zs, classes


def _target_grid(transform, shape, crs=CRS):
    """A stored grid document standing in as an alignment target."""
    return {"georeference": {"crs": crs, "transform": list(transform), "shape": shape}}


def _run(
    read_points,
    point_classes,
    resolution=1.0,
    roi=None,
    alignment=None,
    target_grid_doc=None,
    extent_buffer_cells=0,
    block_cells=None,
):
    # Storage and the dataset handle are stubbed so the algorithm runs without
    # GCS. `block_cells` forces a blocking; left alone the handler picks one.
    patches = [
        patch.object(chm_point_cloud, "read_manifest", return_value=_MANIFEST),
        patch.object(chm_point_cloud, "open_dataset", return_value=None),
        patch.object(chm_point_cloud, "read_points", read_points),
    ]
    if block_cells is not None:
        patches.append(
            patch.object(
                chm_point_cloud, "_compute_block_cells", return_value=block_cells
            )
        )
    with ExitStack() as stack:
        for one in patches:
            stack.enter_context(one)
        return chm_point_cloud.fetch_point_cloud_chm(
            roi=roi if roi is not None else _roi(),
            point_cloud_id="test-cloud",
            point_classes=point_classes,
            alignment=alignment or {"target": "domain", "resolution": resolution},
            target_grid_doc=target_grid_doc,
            progress=lambda message, percent=None: None,
            extent_buffer_cells=extent_buffer_cells,
        )


class TestCanopyHeights:
    """The band is height above ground, maximum per cell."""

    def test_height_is_measured_above_the_ground_surface(self):
        x, y, z, c = _flat_ground()
        # A canopy return 15 m above ground in the top-left cell.
        x, y, z, c = [*x, 0.5], [*y, 3.5], [*z, 25.0], [*c, 5]

        ds, _ = _run(_chunks(x, y, z, c), [2, 5])

        assert ds["chm"].values[0, 0] == pytest.approx(15.0)

    def test_cell_takes_the_tallest_return(self):
        x, y, z, c = _flat_ground()
        x = [*x, 0.2, 0.8]
        y = [*y, 3.2, 3.8]
        z = [*z, 18.0, 22.0]
        c = [*c, 5, 5]

        ds, _ = _run(_chunks(x, y, z, c), [2, 5])

        assert ds["chm"].values[0, 0] == pytest.approx(12.0)

    def test_cells_with_only_ground_are_zero(self):
        ds, _ = _run(_chunks(*_flat_ground()), [2])

        assert ds["chm"].values == pytest.approx(0.0)

    def test_cells_with_no_returns_are_nodata(self):
        # Ground everywhere, one canopy return; every other cell has ground
        # (height 0) so use a domain larger than the points to get empty cells.
        x, y, z, c = _flat_ground(size=2)
        ds, _ = _run(_chunks(x, y, z, c), [2], roi=_roi(size=4.0))

        chm = ds["chm"].values
        # The 2x2 block of ground sits in the lower-left of a 4x4 lattice.
        assert np.isnan(chm[0, 0])
        assert not np.isnan(chm[3, 0])

    def test_returns_below_ground_are_dropped(self):
        x, y, z, c = _flat_ground()
        x, y, z, c = [*x, 0.5], [*y, 3.5], [*z, 5.0], [*c, 5]

        ds, _ = _run(_chunks(x, y, z, c), [2, 5])

        # The sub-ground return is discarded, leaving only the ground itself.
        assert ds["chm"].values[0, 0] == pytest.approx(0.0)

    def test_returns_above_the_height_ceiling_are_dropped(self):
        x, y, z, c = _flat_ground()
        x, y, z, c = [*x, 0.5], [*y, 3.5], [*z, 10.0 + 150.0], [*c, 5]

        ds, _ = _run(_chunks(x, y, z, c), [2, 5])

        assert ds["chm"].values[0, 0] == pytest.approx(0.0)

    def test_points_outside_the_domain_are_ignored(self):
        x, y, z, c = _flat_ground()
        x, y, z, c = [*x, 99.0], [*y, 99.0], [*z, 40.0], [*c, 5]

        ds, _ = _run(_chunks(x, y, z, c), [2, 5])

        assert ds["chm"].shape == (4, 4)
        assert np.nanmax(ds["chm"].values) == pytest.approx(0.0)


class TestOutlierHandling:
    """A single spurious return must not become a tree."""

    def test_noise_classes_are_excluded(self):
        x, y, z, c = _flat_ground()
        # ASPRS 18 is high noise — a bird, not canopy.
        x, y, z, c = [*x, 0.5], [*y, 3.5], [*z, 60.0], [*c, 18]

        ds, _ = _run(_chunks(x, y, z, c), [2, 18])

        assert ds["chm"].values[0, 0] == pytest.approx(0.0)

    def test_isolated_spike_is_removed(self):
        x, y, z, c = _flat_ground()
        # Unclassified noise the class filter cannot catch, towering far above
        # its neighbours.
        x, y, z, c = [*x, 0.5], [*y, 3.5], [*z, 10.0 + 40.0], [*c, 1]

        ds, _ = _run(_chunks(x, y, z, c), [1, 2])

        assert np.isnan(ds["chm"].values[0, 0])

    def test_real_canopy_is_not_treated_as_a_spike(self):
        x, y, z, c = _flat_ground()
        # A crown spanning several cells at a plausible height survives.
        for col in range(3):
            x, y, z, c = [*x, col + 0.5], [*y, 3.5], [*z, 24.0], [*c, 5]

        ds, _ = _run(_chunks(x, y, z, c), [2, 5])

        assert ds["chm"].values[0, :3] == pytest.approx(14.0)


class TestGroundSource:
    """Ground comes from the classification when present, else from the data."""

    def test_classification_is_used_when_ground_returns_exist(self):
        _, provenance = _run(_chunks(*_flat_ground()), [2])

        assert provenance["ground_source"] == "classification"
        assert provenance["ground_coverage"] == 1.0

    def test_never_classified_cloud_still_produces_a_chm(self):
        """A genuinely unclassified upload is all ASPRS class 0.

        Excluding class 0 would yield an empty CHM for exactly the clouds the
        derived-ground path exists to serve.
        """
        size = 8
        xs, ys, zs, classes = [], [], [], []
        for row in range(size):
            for col in range(size):
                xs.append(col + 0.5)
                ys.append(size - row - 0.5)
                zs.append(10.0)
                classes.append(0)
        # One canopy return, also unclassified.
        xs, ys, zs, classes = [*xs, 0.5], [*ys, 7.5], [*zs, 22.0], [*classes, 0]

        ds, provenance = _run(
            _chunks(xs, ys, zs, classes), [0], roi=_roi(size=float(size))
        )

        assert provenance["ground_source"] == "derived"
        assert np.isfinite(ds["chm"].values).any()
        assert np.nanmax(ds["chm"].values) == pytest.approx(12.0, abs=0.5)

    def test_provenance_reports_how_well_constrained_the_ground_was(self):
        _, provenance = _run(_chunks(*_flat_ground()), [2])

        assert set(provenance) == {
            "ground_source",
            "ground_coverage",
            "max_ground_distance_m",
        }
        assert provenance["max_ground_distance_m"] == 0.0

    def test_no_usable_returns_raises_processing_error(self):
        with pytest.raises(ProcessingError) as excinfo:
            _run(_chunks([99.0], [99.0], [10.0], [2]), [2])

        assert excinfo.value.code == "EMPTY_POINT_CLOUD"


class TestGroundGapFill:
    """How far a ground surface is carried into cells with no ground return."""

    @pytest.mark.parametrize("resolution", [1.0, 2.0, 5.0, 10.0, 30.0])
    def test_the_reach_is_the_same_distance_at_every_cell_size(self, resolution):
        """The reach is a physical distance, so it converts to cells at run
        time. A fixed cell count would interpolate 30 m across at 1 m cells and
        900 m across at 30 m cells — the same argument `_pmf` makes for its
        windows."""
        cells = chm_point_cloud._fill_cells(resolution)

        assert cells * resolution == pytest.approx(
            chm_point_cloud.GROUND_FILL_MAX_M, abs=resolution / 2
        )

    def test_one_metre_cells_keep_the_established_reach(self):
        """1 m is the default, and the derived-ground accuracy in the module
        docstring was measured there — it must not shift."""
        assert chm_point_cloud._fill_cells(1.0) == 30

    def test_a_cell_coarser_than_the_reach_still_fills_its_neighbour(self):
        """Rounding to zero would disable the fill outright."""
        assert chm_point_cloud._fill_cells(100.0) == 1

    def test_gaps_within_the_reach_are_interpolated(self):
        surface = np.full((1, 12), np.nan, dtype=np.float32)
        surface[0, 0] = 10.0

        filled = chm_point_cloud._fill_gaps(surface, 4)

        assert np.isfinite(filled[0, :5]).all()

    def test_gaps_beyond_the_reach_stay_nodata(self):
        """A cell too far from any real ground return is left unknown rather
        than extrapolated across a void the data says nothing about."""
        surface = np.full((1, 12), np.nan, dtype=np.float32)
        surface[0, 0] = 10.0

        filled = chm_point_cloud._fill_gaps(surface, 4)

        assert np.isnan(filled[0, 5:]).all()


class TestDatasetContract:
    """What downstream consumers of a CHM grid rely on."""

    def test_dataset_shape_dtype_and_nodata(self):
        ds, _ = _run(_chunks(*_flat_ground()), [2])

        assert list(ds["chm"].dims) == ["y", "x"]
        assert ds["chm"].dtype == np.float32
        assert np.isnan(ds["chm"].rio.nodata)

    def test_dataset_is_georeferenced_on_the_domain_lattice(self):
        ds, _ = _run(_chunks(*_flat_ground()), [2])

        assert ds.rio.crs == pyproj.CRS.from_user_input(CRS)
        transform = ds.rio.transform()
        assert transform.a == pytest.approx(1.0)
        assert transform.c == pytest.approx(0.0)
        assert transform.f == pytest.approx(4.0)

    def test_resolution_sets_the_cell_size(self):
        ds, _ = _run(_chunks(*_flat_ground()), [2], resolution=2.0)

        assert ds["chm"].shape == (2, 2)
        assert ds.rio.transform().a == pytest.approx(2.0)


class TestAlignment:
    """Which lattice the CHM lands on.

    The target grid here is deliberately off the domain lattice — origin at
    (0.5, 4.5) rather than (0.0, 4.0) — so a result anchored on the domain is
    distinguishable from one anchored on the target.
    """

    TARGET_TRANSFORM = (2.0, 0.0, 0.5, 0.0, -2.0, 4.5)
    TARGET_SHAPE = (2, 2)

    def _target(self, crs=CRS):
        return _target_grid(self.TARGET_TRANSFORM, self.TARGET_SHAPE, crs=crs)

    def test_domain_target_is_anchored_on_the_domain(self):
        """The default path, unchanged: the domain's own lattice."""
        ds, _ = _run(_chunks(*_flat_ground()), [2])

        transform = ds.rio.transform()
        assert (transform.c, transform.f) == pytest.approx((0.0, 4.0))
        assert ds["chm"].shape == (4, 4)

    def test_grid_target_reproduces_the_target_lattice_exactly(self):
        """`resolution: null` against a grid means cell-for-cell match.

        Origin, cell size and shape all come from the target, so the two grids
        compose without resampling.
        """
        ds, _ = _run(
            _chunks(*_flat_ground()),
            [2],
            alignment={"target": "grid", "grid_id": "g-1", "resolution": None},
            target_grid_doc=self._target(),
        )

        assert tuple(ds.rio.transform())[:6] == pytest.approx(self.TARGET_TRANSFORM)
        assert ds["chm"].shape == self.TARGET_SHAPE

    def test_grid_target_with_a_resolution_keeps_the_target_origin(self):
        """The second alignment mode: same anchor, different cell size."""
        ds, _ = _run(
            _chunks(*_flat_ground()),
            [2],
            alignment={"target": "grid", "grid_id": "g-1", "resolution": 1.0},
            target_grid_doc=self._target(),
        )

        transform = ds.rio.transform()
        assert (transform.c, transform.f) == pytest.approx((0.5, 4.5))
        assert transform.a == pytest.approx(1.0)
        assert ds["chm"].shape == (4, 4)

    def test_grid_target_in_another_crs_is_rejected(self):
        """Points are stored in the domain CRS and this handler cannot
        reproject them, so a foreign target lattice is a clean failure rather
        than a silently misplaced raster."""
        with pytest.raises(ProcessingError) as excinfo:
            _run(
                _chunks(*_flat_ground()),
                [2],
                alignment={"target": "grid", "grid_id": "g-1", "resolution": None},
                target_grid_doc=self._target(crs="EPSG:32613"),
            )

        assert excinfo.value.code == "ALIGNMENT_CRS_MISMATCH"

    def test_extent_buffer_grows_the_lattice_on_every_side(self):
        ds, _ = _run(_chunks(*_flat_ground()), [2], extent_buffer_cells=2)

        transform = ds.rio.transform()
        assert (transform.c, transform.f) == pytest.approx((-2.0, 6.0))
        assert ds["chm"].shape == (8, 8)


class TestReadingFromStorage:
    """The dataset read path, against a real Parquet dataset on disk."""

    def _write_dataset(self, root):
        """Two partitions, so a bounds-pruned read has something to exclude."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        for tile_x, xs in ((0, [1.0, 2.0, 3.0]), (1, [501.0, 502.0])):
            directory = root / f"tile_x={tile_x}" / "tile_y=0"
            directory.mkdir(parents=True)
            count = len(xs)
            table = pa.table(
                {
                    "lod": pa.array([0] * count, pa.uint8()),
                    "X": pa.array([round(v * 1000) for v in xs], pa.int32()),
                    "Y": pa.array([1000] * count, pa.int32()),
                    "Z": pa.array(
                        [round(v * 1000) for v in ([10.0, 20.0, 30.0][:count])],
                        pa.int32(),
                    ),
                    "intensity": pa.array([0] * count, pa.uint16()),
                    "classification": pa.array(([2, 5, 5][:count]), pa.uint8()),
                }
            )
            pq.write_table(table, directory / "part-00000.parquet")

    def test_reads_only_the_partitions_a_block_overlaps(self, tmp_path):
        root = tmp_path / "cloud.parquet"
        self._write_dataset(root)
        dataset = open_dataset(str(root), filesystem=LocalFileSystem())

        x, y, z, classification = read_points(
            dataset, _MANIFEST, (0.0, 0.0, 100.0, 100.0), None
        )

        # The second tile starts at 500 m and is pruned by the partition filter.
        assert sorted(x.tolist()) == pytest.approx([1.0, 2.0, 3.0])
        assert z.tolist() == pytest.approx([10.0, 20.0, 30.0])
        assert classification.tolist() == [2, 5, 5]

    def test_class_filter_is_pushed_into_the_read(self, tmp_path):
        root = tmp_path / "cloud.parquet"
        self._write_dataset(root)
        dataset = open_dataset(str(root), filesystem=LocalFileSystem())

        _, _, z, classification = read_points(
            dataset, _MANIFEST, (0.0, 0.0, 100.0, 100.0), (2,)
        )

        assert classification.tolist() == [2]
        assert z.tolist() == pytest.approx([10.0])

    def test_a_block_over_empty_ground_reads_nothing(self, tmp_path):
        root = tmp_path / "cloud.parquet"
        self._write_dataset(root)
        dataset = open_dataset(str(root), filesystem=LocalFileSystem())

        x, _, _, _ = read_points(
            dataset, _MANIFEST, (2000.0, 2000.0, 2100.0, 2100.0), None
        )

        assert x.size == 0


class TestBlockingIsInvisible:
    """Blocking must not change the answer, only how it is computed.

    The point passes are commutative scatter-reductions and every raster step
    runs through a halo at least as wide as its dependency radius, so a blocked
    run has to agree with an unblocked one cell for cell — not approximately.

    Run at 10 m cells so the halos (2 cells for the filter, 3 for the fill, 6
    for the ground-distance cap) fit inside a block small enough to tile a
    fixture of a few thousand points. At 1 m the filter alone reaches 62 cells,
    which would need a 252-cell grid and half a million points to exercise
    honestly.
    """

    RESOLUTION = 10.0
    SIZE_M = 320.0
    BLOCK_CELLS = 16  # 2x2 blocks over the 32-cell grid

    def _cloud(self, spacing=5.0):
        """Pitched ground with canopy on it, and a ground void on a block seam.

        The void sits at x = 160 m, exactly where the blocks meet, so a halo too
        narrow to see across the seam would fill it differently on each side.
        """
        rng = np.random.default_rng(0)
        steps = int(self.SIZE_M / spacing)
        xs, ys, zs, classes = [], [], [], []
        for row in range(steps):
            for col in range(steps):
                px = col * spacing + spacing / 2
                py = row * spacing + spacing / 2
                ground = 10.0 + 0.05 * px + 0.03 * py
                if not 140.0 < px < 180.0:
                    xs.append(px)
                    ys.append(py)
                    zs.append(ground)
                    classes.append(2)
                if rng.random() < 0.6:
                    xs.append(px)
                    ys.append(py)
                    zs.append(ground + rng.uniform(1.0, 18.0))
                    classes.append(5)
        return xs, ys, zs, classes

    # [2] takes the classified path, whose only blocked stage is the point
    # pass, so it pins the scatter-reduction. [1] takes the derived path, where
    # the fill and the morphological filter are blocked behind halos, so it is
    # the case that actually exercises them — verified by forcing the halo to
    # zero and watching this fail.
    @pytest.mark.parametrize("point_classes", ([2], [1]))
    def test_blocked_matches_unblocked(self, point_classes):
        cloud = self._cloud()
        roi = _roi(self.SIZE_M)

        whole, whole_provenance = _run(
            _chunks(*cloud),
            point_classes,
            resolution=self.RESOLUTION,
            roi=roi,
            block_cells=4096,
        )
        blocked, blocked_provenance = _run(
            _chunks(*cloud),
            point_classes,
            resolution=self.RESOLUTION,
            roi=roi,
            block_cells=self.BLOCK_CELLS,
        )

        # Guard the guard: a fixture that degenerated to one block would make
        # the comparison vacuous.
        assert whole["chm"].shape[0] // self.BLOCK_CELLS >= 2
        assert np.isfinite(whole["chm"].values).any()

        np.testing.assert_array_equal(whole["chm"].values, blocked["chm"].values)
        assert whole_provenance == blocked_provenance

    def test_block_size_covers_the_widest_halo(self):
        """A block has to be able to feed the halo, or the answer changes."""
        # 2 * (400 + 1) = 802 cells needed, which takes two 500-cell tiles.
        assert chm_point_cloud._compute_block_cells(1.0, 500.0, halo_cells=400) == 1000
        assert chm_point_cloud._compute_block_cells(1.0, 500.0, halo_cells=600) == 1500

    def test_block_size_is_a_whole_number_of_cloud_tiles(self):
        """A block that stops mid-partition decodes that partition anyway."""
        # One 500-cell tile clears both the halo and the 512-cell default,
        # which is rounded to the nearest tile rather than up: rounding up
        # would double the block's area for a storage-chunk match `save_zarr`
        # re-does on write.
        assert chm_point_cloud._compute_block_cells(1.0, 500.0, halo_cells=0) == 500
        # At 0.5 m one tile is already 1000 cells.
        assert chm_point_cloud._compute_block_cells(0.5, 500.0, halo_cells=0) == 1000
        # At 10 m a tile is 50 cells, so the default decides how many.
        assert chm_point_cloud._compute_block_cells(10.0, 500.0, halo_cells=0) == 500

    def test_block_edges_land_on_cloud_tile_boundaries(self):
        """Cut on the tiles, so a block reads each partition once.

        The lattice origin and the cloud's origin have no reason to agree, and
        an offset block reads two tiles per axis where it needs one.
        """
        # Lattice starting 120 m east of the cloud's own origin, 1 m cells.
        cuts = chm_point_cloud._tile_cuts(2000, 1120.0, 1.0, 1000.0, 500.0)
        assert cuts == [380, 880, 1380, 1880]
        # 380 is skipped because a 380-cell leading block could not feed the
        # halo, and 1880 because it would leave a 120-cell tail. Both merge
        # into their neighbour, which keeps every edge on a tile boundary.
        assert chm_point_cloud._block_slices(2000, 500, cuts) == [
            (0, 880),
            (880, 1380),
            (1380, 2000),
        ]

    def test_tile_cuts_run_ascending_on_the_row_axis(self):
        """Rows count downward while northings count up; the cuts still sort."""
        # transform.f is the north edge, transform.e negative.
        assert chm_point_cloud._tile_cuts(2000, 3000.0, -1.0, 1000.0, 500.0) == [
            500,
            1000,
            1500,
        ]

    def test_a_block_stops_short_of_the_next_partition(self):
        """Every edge but the origin is exclusive, or an aligned block reads twice."""
        min_x, min_y, max_x, max_y = chm_point_cloud._block_bounds(
            (1000.0, 2000.0, 500, 500), 1.0
        )
        assert min_x == 1000.0
        assert max_x < 1500.0
        assert min_y > 1500.0
        # The row axis runs downward, so the top edge is the one that lands on a
        # tile origin. Left closed it pulled in the tile above, on every block.
        assert max_y < 2000.0

    def test_an_aligned_block_reads_exactly_one_partition(self):
        """The whole point of cutting on tile boundaries, asserted end to end.

        Measured at 2.0 partitions per block on the 64 km2 cloud before the top
        edge was made exclusive -- the x axis read one tile and the y axis two.
        """
        tile_m, cloud_origin = 500.0, (1000.0, 1500.0)
        for row in range(4):
            for col in range(4):
                bounds = chm_point_cloud._block_bounds(
                    (1000.0 + col * 500.0, 3500.0 - row * 500.0, 500, 500), 1.0
                )
                tx0, tx1 = tile_span(bounds[0], bounds[2], cloud_origin[0], tile_m)
                ty0, ty1 = tile_span(bounds[1], bounds[3], cloud_origin[1], tile_m)
                assert (tx1 - tx0 + 1, ty1 - ty0 + 1) == (1, 1)

    def test_blocks_are_divided_evenly(self):
        """A greedy split leaves a remainder block narrower than the halo."""
        sizes = [
            stop - start for start, stop in chm_point_cloud._block_slices(1025, 512)
        ]
        assert sizes == [342, 341, 342]


class TestFillGapsIsBlockInvariant:
    """Blocking the fill must not let the chunking into the answer.

    `_fill_gaps` propagates one cell per iteration, so its reach is `max_cells`
    and a halo that wide should feed a block everything the whole-grid pass saw.
    `uniform_filter`'s `mode="nearest"` also perturbs the outermost cell of
    whatever array it is handed, which then travels inward one cell per
    iteration -- so this asserts the equality rather than trusting the argument.
    """

    @staticmethod
    def _surface(seed=0):
        rng = np.random.default_rng(seed)
        surface = rng.random((600, 600)).astype(np.float32) * 20.0
        # Scattered dropouts, which interpolate, plus one void wider than the
        # reach, whose interior must stay NaN however it is blocked.
        surface[rng.random((600, 600)) < 0.35] = np.nan
        surface[200:320, 150:290] = np.nan
        return surface

    def test_matches_the_whole_grid_fill(self):
        surface = self._surface()
        expected = chm_point_cloud._fill_gaps(surface, 30)
        actual = chm_point_cloud._blocked_fill_gaps(surface, 200, 30)
        assert np.array_equal(np.isnan(expected), np.isnan(actual))
        finite = ~np.isnan(expected)
        np.testing.assert_allclose(expected[finite], actual[finite], rtol=1e-6)

    def test_same_answer_at_every_block_size(self):
        surface = self._surface(seed=3)
        answers = [
            chm_point_cloud._blocked_fill_gaps(surface, block, 30)
            for block in (150, 200, 300, 600)
        ]
        for other in answers[1:]:
            assert np.array_equal(np.isnan(answers[0]), np.isnan(other))
            finite = ~np.isnan(answers[0])
            np.testing.assert_allclose(answers[0][finite], other[finite], rtol=1e-6)

    def test_a_void_wider_than_the_reach_keeps_its_core(self):
        """The bound is the point of `_fill_gaps`; blocking must not relax it."""
        surface = np.full((400, 400), 5.0, dtype=np.float32)
        surface[100:300, 100:300] = np.nan
        filled = chm_point_cloud._blocked_fill_gaps(surface, 200, 30)
        # 30 cells of reach from each edge leaves the middle untouched.
        assert np.isnan(filled[150:250, 150:250]).all()
        assert not np.isnan(filled[100:130, 200]).any()


class TestGroundDistanceIsBlockInvariant:
    """The reported distance must describe the data, not the chunking.

    It reduced over the halo before, so a cell only as well served as a halo it
    did not itself have could set the maximum. The same ground read 40 m at a
    512-cell block and 38 m at 500.
    """

    @staticmethod
    def _known(seed=0):
        rng = np.random.default_rng(seed)
        known = rng.random((600, 600)) < 0.02
        # One wide void, which is the thing the metric exists to find.
        known[200:320, 150:290] = False
        return known

    def test_same_answer_at_every_block_size(self):
        known = self._known()
        answers = {
            block: chm_point_cloud._blocked_ground_distance(known, block, 1.0)
            for block in (150, 200, 256, 300, 512, 600)
        }
        assert len(set(answers.values())) == 1, answers

    def test_matches_the_unblocked_transform(self):
        known = self._known()
        cap = chm_point_cloud.GROUND_DISTANCE_CAP_M
        expected = min(float(distance_transform_edt(~known).max()), cap)
        assert chm_point_cloud._blocked_ground_distance(
            known, 256, 1.0
        ) == pytest.approx(expected)

    def test_saturates_where_no_ground_is_in_reach(self):
        known = np.zeros((300, 300), dtype=bool)
        assert chm_point_cloud._blocked_ground_distance(known, 128, 1.0) == (
            chm_point_cloud.GROUND_DISTANCE_CAP_M
        )

    def test_zero_when_every_cell_has_ground(self):
        known = np.ones((300, 300), dtype=bool)
        assert chm_point_cloud._blocked_ground_distance(known, 128, 1.0) == 0.0
