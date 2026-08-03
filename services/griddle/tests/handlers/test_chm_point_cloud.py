"""
Tests for the point-cloud CHM handler.

Points are placed at exact coordinates so the expected canopy height of each
cell is known by construction rather than asserted against a recorded output.
Most tests replace ``_iter_points`` so the algorithm is exercised without GCS;
one test drives the real LAZ reader against a file on disk.
"""

import io
from unittest.mock import MagicMock, patch

import geopandas as gpd
import laspy
import numpy as np
import pyproj
import pytest
from griddle.handlers import chm_point_cloud
from shapely.geometry import box

from lib.errors import ProcessingError

CRS = "EPSG:32612"


def _chunks(x, y, z, classification):
    """Build an ``_iter_points`` replacement yielding one chunk.

    Takes the downloaded buffer the real one does, and ignores it.
    """

    def _iter(_cloud):
        yield (
            np.asarray(x, dtype=float),
            np.asarray(y, dtype=float),
            np.asarray(z, dtype=float),
            np.asarray(classification, dtype=np.uint8),
        )

    return _iter


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
    iter_points,
    point_classes,
    resolution=1.0,
    roi=None,
    alignment=None,
    target_grid_doc=None,
    extent_buffer_cells=0,
):
    # The handler downloads the cloud once and replays the buffer; both are
    # stubbed so the algorithm runs without GCS.
    with (
        patch.object(chm_point_cloud, "_open_cloud", return_value=lambda: io.BytesIO()),
        patch.object(chm_point_cloud, "_iter_points", iter_points),
    ):
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
    """The GCS read path, against a real LAZ on disk."""

    def _write_laz(self, path):
        header = laspy.LasHeader(version="1.4", point_format=6)
        header.offsets = [0.0, 0.0, 0.0]
        header.scales = [0.01, 0.01, 0.01]
        header.add_crs(pyproj.CRS.from_epsg(32612))
        las = laspy.LasData(header)
        las.x = np.array([1.0, 2.0, 3.0])
        las.y = np.array([1.0, 2.0, 3.0])
        las.z = np.array([10.0, 20.0, 30.0])
        las.classification = np.array([2, 5, 5], dtype=np.uint8)
        las.write(str(path))

    def test_small_cloud_is_buffered_and_read_from_memory(self, tmp_path):
        path = tmp_path / "cloud.laz"
        self._write_laz(path)
        client = MagicMock()
        client.size.return_value = path.stat().st_size
        client.cat.return_value = path.read_bytes()

        with patch.object(chm_point_cloud, "get_gcsfs_client", return_value=client):
            cloud = chm_point_cloud._open_cloud("some-cloud-id", cells=1_000_000)

        assert client.cat.call_args[0][0].endswith("/some-cloud-id/cloud.laz")
        x, y, z, classification = next(chm_point_cloud._iter_points(cloud))
        assert len(x) == 3
        assert z.tolist() == pytest.approx([10.0, 20.0, 30.0])
        assert classification.tolist() == [2, 5, 5]

    def test_large_cloud_is_streamed_rather_than_buffered(self, tmp_path, monkeypatch):
        """Above the threshold the object is re-opened per pass.

        A full-resolution 3DEP cloud over a large domain runs to gigabytes —
        1.04 billion points and 7.3 GB compressed over 64 km2, measured — which
        no worker should hold in memory.
        """
        path = tmp_path / "cloud.laz"
        self._write_laz(path)
        monkeypatch.setattr(chm_point_cloud, "MAX_BUFFERED_LAZ_BYTES", 1)

        client = MagicMock()
        client.size.return_value = 10_000_000_000
        client.open.side_effect = lambda *a, **k: open(path, "rb")

        with patch.object(chm_point_cloud, "get_gcsfs_client", return_value=client):
            cloud = chm_point_cloud._open_cloud("big-cloud-id", cells=1_000_000)
            first = [c[2].tolist() for c in chm_point_cloud._iter_points(cloud)]
            second = [c[2].tolist() for c in chm_point_cloud._iter_points(cloud)]

        client.cat.assert_not_called()
        assert client.open.call_count == 2
        assert first == second

    def test_large_lattice_leaves_no_room_to_buffer_a_small_cloud(self, tmp_path):
        """The buffer and the rasters share one worker, so the grid gets a vote.

        A cloud well under the standalone threshold is still streamed when the
        output lattice already claims most of the memory budget — buffering it
        would push the two working sets past the worker's limit together.
        """
        path = tmp_path / "cloud.laz"
        self._write_laz(path)

        size = 1_500_000_000
        client = MagicMock()
        client.size.return_value = size
        client.open.side_effect = lambda *a, **k: open(path, "rb")

        # A lattice large enough to leave less headroom than the cloud needs,
        # while the cloud stays under the standalone threshold that would
        # otherwise buffer it.
        headroom = size // 2
        cells = (chm_point_cloud.MEMORY_BUDGET_BYTES - headroom) // (
            chm_point_cloud.RASTER_BYTES_PER_CELL
        )
        assert size < chm_point_cloud.MAX_BUFFERED_LAZ_BYTES

        with patch.object(chm_point_cloud, "get_gcsfs_client", return_value=client):
            chm_point_cloud._open_cloud("big-grid", cells=cells)

        client.cat.assert_not_called()

    def test_a_cloud_that_only_fits_once_is_streamed(self, tmp_path):
        """The gate compares the peak of the fetch, not the object size.

        gcsfs reads the response in chunks and joins them, so a download costs
        about twice the object while both copies are live. A cloud that fits
        the remaining budget once but not twice is streamed instead.
        """
        path = tmp_path / "cloud.laz"
        self._write_laz(path)

        headroom = chm_point_cloud.MEMORY_BUDGET_BYTES // 2
        cells = (chm_point_cloud.MEMORY_BUDGET_BYTES - headroom) // (
            chm_point_cloud.RASTER_BYTES_PER_CELL
        )
        size = int(headroom * 0.75)
        assert size < min(headroom, chm_point_cloud.MAX_BUFFERED_LAZ_BYTES)
        assert 2 * size > headroom

        client = MagicMock()
        client.size.return_value = size
        client.open.side_effect = lambda *a, **k: open(path, "rb")

        with patch.object(chm_point_cloud, "get_gcsfs_client", return_value=client):
            chm_point_cloud._open_cloud("only-fits-once", cells=cells)

        client.cat.assert_not_called()

    def test_buffer_is_replayable_across_passes(self, tmp_path):
        """The algorithm reads the cloud two or three times off one download."""
        path = tmp_path / "cloud.laz"
        self._write_laz(path)
        buffer = chm_point_cloud._ReplayableBuffer(path.read_bytes())

        first = [c[2].tolist() for c in chm_point_cloud._iter_points(lambda: buffer)]
        second = [c[2].tolist() for c in chm_point_cloud._iter_points(lambda: buffer)]

        assert first == second
        assert first[0] == pytest.approx([10.0, 20.0, 30.0])

    def test_cloud_is_downloaded_once_for_the_whole_job(self, tmp_path):
        """Deriving ground takes three passes; all three read one download."""
        path = tmp_path / "cloud.laz"
        self._write_laz(path)
        client = MagicMock()
        client.size.return_value = path.stat().st_size
        client.cat.return_value = path.read_bytes()

        with patch.object(chm_point_cloud, "get_gcsfs_client", return_value=client):
            # No ground class in the census, so the derived path runs.
            ds, provenance = chm_point_cloud.fetch_point_cloud_chm(
                roi=_roi(),
                point_cloud_id="some-cloud-id",
                point_classes=[5],
                alignment={"target": "domain", "resolution": 1.0},
                progress=lambda message, percent=None: None,
            )

        assert provenance["ground_source"] == "derived"
        assert np.isfinite(ds["chm"].values).any()
        assert client.cat.call_count == 1
