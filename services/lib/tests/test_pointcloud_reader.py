"""Unit tests for the point cloud reader's selection.

Everything here runs against a small Parquet dataset written to a tmp_path and
read through the local filesystem — no GCS. What these guard is which stored
data a read touches: the partition prune, the LOD cut that skips whole row
groups, and the batching the block callers reduce over.
"""

import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fsspec.implementations.local import LocalFileSystem as FsspecLocalFileSystem
from pyarrow.fs import LocalFileSystem

from lib.pointcloud.reader import (
    POINT_COLUMNS,
    _lod_row_groups,
    find_tile,
    iter_points,
    open_dataset,
    open_point_cloud,
    read_manifest,
    read_points,
    read_tile,
)
from lib.pointcloud.schema import LOD_LEVELS

# Millimetre scaling anchored at the origin, one 500 m tile per partition.
MANIFEST = {
    "tile_m": 500.0,
    "mins": [0.0, 0.0, 0.0],
    "scales": [0.001, 0.001, 0.001],
    "offsets": [0.0, 0.0, 0.0],
}
WHOLE_FIRST_TILE = (0.0, 0.0, 499.0, 499.0)


def write_tile(
    root,
    tile_x,
    lods,
    classes=None,
    x0=0.0,
    group_per_lod=True,
    part_sequence=0,
):
    """One partition whose row groups are `lods`, a level per group by default.

    `group_per_lod` off writes every level into a single row group, which is the
    layout whose statistics straddle any cut — the case the reader must not get
    wrong just because the writer happens not to produce it today.
    """
    directory = root / f"tile_x={tile_x}" / "tile_y=0"
    directory.mkdir(parents=True, exist_ok=True)

    columns = {
        name: [] for name in ("lod", "X", "Y", "Z", "intensity", "classification")
    }
    for level, count in enumerate(lods):
        for i in range(count):
            columns["lod"].append(level)
            columns["X"].append(round((x0 + 1.0 + i) * 1000))
            columns["Y"].append(1000)
            columns["Z"].append(round((10.0 + level) * 1000))
            columns["intensity"].append(0)
            columns["classification"].append(
                2 if classes is None else classes[level % len(classes)]
            )

    table = pa.table(
        {
            "lod": pa.array(columns["lod"], pa.uint8()),
            "X": pa.array(columns["X"], pa.int32()),
            "Y": pa.array(columns["Y"], pa.int32()),
            "Z": pa.array(columns["Z"], pa.int32()),
            "intensity": pa.array(columns["intensity"], pa.uint16()),
            "classification": pa.array(columns["classification"], pa.uint8()),
        }
    )
    path = directory / f"part-{part_sequence:05d}.parquet"
    if group_per_lod:
        with pq.ParquetWriter(path, table.schema) as writer:
            offset = 0
            for count in lods:
                if count:
                    writer.write_table(table.slice(offset, count), row_group_size=count)
                offset += count
    else:
        pq.write_table(table, path, row_group_size=table.num_rows)
    write_global_metadata(root)
    return path


def write_global_metadata(root):
    """Rebuild the standard aggregate footer for the local test dataset."""
    footers = []
    for path in sorted(root.glob("tile_x=*/tile_y=*/part-*.parquet")):
        footer = pq.read_metadata(path)
        footer.set_file_path(str(path.relative_to(root)))
        footers.append(footer)
    combined = footers[0]
    for footer in footers[1:]:
        combined.append_row_groups(footer)
    combined.write_metadata_file(root / "_metadata")


def full_manifest(tiles=1, points=4):
    return {
        "tiles": tiles,
        "tile_m": 500.0,
        "points": points,
        "lod_levels": LOD_LEVELS,
        "scales": [0.001, 0.001, 0.001],
        "offsets": [0.0, 0.0, 0.0],
        "mins": [0.0, 0.0, 0.0],
        "maxs": [1000.0, 500.0, 100.0],
    }


def open_storage(root, *, tiles=1, points=4):
    (root / "_manifest.json").write_text(json.dumps(full_manifest(tiles, points)))
    return open_point_cloud(str(root), LocalFileSystem())


@pytest.fixture
def pyramid(tmp_path):
    """A tile with the writer's own shape: a row group per level, 4x each."""
    root = tmp_path / "cloud.parquet"
    write_tile(root, 0, [1, 4, 16, 64])
    return open_dataset(str(root), filesystem=LocalFileSystem()), LocalFileSystem()


class TestLodCut:
    """`max_lod` is the one predicate the stored layout answers by skipping."""

    @pytest.mark.parametrize(
        "max_lod,expected",
        [(0, 1), (1, 5), (2, 21), (3, 85), (None, 85)],
    )
    def test_cut_keeps_every_level_up_to_it(self, pyramid, max_lod, expected):
        dataset, filesystem = pyramid

        _, _, z, _ = read_points(
            dataset, MANIFEST, WHOLE_FIRST_TILE, None, filesystem, max_lod
        )

        assert z.size == expected

    def test_cut_skips_the_row_groups_it_excludes(self, pyramid):
        """The point of the cut: excluded levels are never decoded."""
        dataset, _ = pyramid
        parquet_file = pq.ParquetFile(next(dataset.get_fragments()).path)

        assert _lod_row_groups(parquet_file, 1) == ([0, 1], False)
        assert _lod_row_groups(parquet_file, 3) == ([0, 1, 2, 3], False)

    def test_a_straddling_row_group_is_kept_and_its_rows_filtered(self, tmp_path):
        """Correctness must not rest on the writer's group-per-level layout."""
        root = tmp_path / "cloud.parquet"
        path = write_tile(root, 0, [1, 4, 16], group_per_lod=False)
        dataset = open_dataset(str(root), filesystem=LocalFileSystem())

        parquet_file = pq.ParquetFile(path)
        assert _lod_row_groups(parquet_file, 1) == ([0], True)

        _, _, z, _ = read_points(
            dataset, MANIFEST, WHOLE_FIRST_TILE, None, LocalFileSystem(), 1
        )

        assert z.size == 5

    def test_the_cut_composes_with_the_class_filter(self, tmp_path):
        root = tmp_path / "cloud.parquet"
        write_tile(root, 0, [2, 2, 2], classes=[2, 5, 6])
        dataset = open_dataset(str(root), filesystem=LocalFileSystem())

        _, _, _, classification = read_points(
            dataset, MANIFEST, WHOLE_FIRST_TILE, (2, 5), LocalFileSystem(), 1
        )

        # Levels 0 and 1 survive the cut; of those only classes 2 and 5 are kept.
        assert sorted(classification.tolist()) == [2, 2, 5, 5]

    def test_lod_is_not_read_when_no_row_group_straddles(self, pyramid):
        """The extra column is a cost of the straddling case, not of every read."""
        dataset, filesystem = pyramid
        seen = []
        original = pq.ParquetFile.iter_batches

        def spy(self, *args, **kwargs):
            seen.append(kwargs.get("columns"))
            return original(self, *args, **kwargs)

        pq.ParquetFile.iter_batches = spy
        try:
            read_points(dataset, MANIFEST, WHOLE_FIRST_TILE, None, filesystem, 2)
        finally:
            pq.ParquetFile.iter_batches = original

        assert seen == [POINT_COLUMNS]


class TestFetchStrategy:
    """Whole-file GET or ranged reads, chosen by how much of the file is wanted.

    Invisible in the returned points and, from outside the region, invisible in
    wall clock too -- latency swamps it. What it moves is bytes: on a real
    40.6 MiB tile, `lod <= 2` needs 0.56 MiB, so a whole-file GET to serve it is
    72x the transfer.
    """

    def _spy_on_whole_file_fetch(self, monkeypatch):
        from lib.pointcloud import reader

        calls = []
        original = reader._read_whole

        def spy(filesystem, path):
            calls.append(path)
            return original(filesystem, path)

        monkeypatch.setattr(reader, "_read_whole", spy)
        return calls

    def test_no_cut_fetches_the_file_whole(self, pyramid, monkeypatch):
        dataset, filesystem = pyramid
        calls = self._spy_on_whole_file_fetch(monkeypatch)

        read_points(dataset, MANIFEST, WHOLE_FIRST_TILE, None, filesystem)

        assert len(calls) == 1

    def test_a_cut_that_excludes_row_groups_reads_ranges_instead(
        self, pyramid, monkeypatch
    ):
        dataset, filesystem = pyramid
        calls = self._spy_on_whole_file_fetch(monkeypatch)

        read_points(dataset, MANIFEST, WHOLE_FIRST_TILE, None, filesystem, 1)

        assert calls == []

    def test_a_cut_that_excludes_nothing_falls_back_to_the_whole_file(
        self, pyramid, monkeypatch
    ):
        """Ranged reads would only buy round trips when nothing is skipped."""
        dataset, filesystem = pyramid
        calls = self._spy_on_whole_file_fetch(monkeypatch)

        read_points(dataset, MANIFEST, WHOLE_FIRST_TILE, None, filesystem, 3)

        assert len(calls) == 1


class TestBatching:
    """A caller that reduces must see every point exactly once, in pieces."""

    def test_batches_partition_the_selection(self, pyramid):
        dataset, filesystem = pyramid

        batches = list(
            iter_points(dataset, MANIFEST, WHOLE_FIRST_TILE, None, filesystem)
        )
        whole = read_points(dataset, MANIFEST, WHOLE_FIRST_TILE, None, filesystem)

        assert sum(x.size for x, _, _, _ in batches) == whole[0].size
        for column, gathered in enumerate(whole):
            assert np.array_equal(
                np.concatenate([b[column] for b in batches]), gathered
            )

    def test_a_selection_with_no_points_yields_nothing(self, pyramid):
        dataset, filesystem = pyramid

        assert (
            list(
                iter_points(
                    dataset,
                    MANIFEST,
                    (2000.0, 2000.0, 2100.0, 2100.0),
                    None,
                    filesystem,
                )
            )
            == []
        )

    def test_empty_selection_still_returns_four_arrays(self, pyramid):
        dataset, filesystem = pyramid

        x, y, z, classification = read_points(
            dataset, MANIFEST, (2000.0, 2000.0, 2100.0, 2100.0), None, filesystem
        )

        assert (x.size, y.size, z.size, classification.size) == (0, 0, 0, 0)


class TestPartitionPrune:
    """Tiles the bounds do not touch are never opened."""

    def test_only_overlapping_partitions_are_read(self, tmp_path):
        root = tmp_path / "cloud.parquet"
        write_tile(root, 0, [3], x0=0.0)
        write_tile(root, 1, [2], x0=500.0)
        dataset = open_dataset(str(root), filesystem=LocalFileSystem())

        x, _, _, _ = read_points(
            dataset, MANIFEST, WHOLE_FIRST_TILE, None, LocalFileSystem()
        )

        assert sorted(x.tolist()) == pytest.approx([1.0, 2.0, 3.0])


class TestManifest:
    def test_reads_and_validates_the_current_format(self, tmp_path):
        root = tmp_path / "cloud.parquet"
        root.mkdir()
        expected = full_manifest()
        (root / "_manifest.json").write_text(json.dumps(expected))

        assert read_manifest(str(root), LocalFileSystem()) == expected

    @pytest.mark.parametrize(
        "change,detail",
        [
            (lambda manifest: manifest.update(manifest_version=2), "manifest_version"),
            (lambda manifest: manifest.update(tiles=[]), "tiles must be an integer"),
            (lambda manifest: manifest.update(lod_levels=5), "lod_levels must equal"),
            (lambda manifest: manifest.pop("tile_m"), "missing required fields"),
        ],
    )
    def test_rejects_any_other_layout(self, tmp_path, change, detail):
        root = tmp_path / "cloud.parquet"
        root.mkdir()
        manifest = full_manifest()
        change(manifest)
        (root / "_manifest.json").write_text(json.dumps(manifest))

        with pytest.raises(Exception) as error:
            read_manifest(str(root), LocalFileSystem())

        assert error.value.code == "POINT_CLOUD_UNREADABLE"
        assert detail in error.value.traceback


class TestIndexedStorage:
    def test_missing_metadata_is_not_a_compatibility_format(self, tmp_path):
        root = tmp_path / "cloud.parquet"
        root.mkdir()
        (root / "_manifest.json").write_text(json.dumps(full_manifest()))

        with pytest.raises(Exception) as error:
            open_point_cloud(str(root), LocalFileSystem())

        assert error.value.code == "POINT_CLOUD_UNREADABLE"
        assert "_metadata" in error.value.traceback

    def test_parquet_totals_must_match_the_commit_manifest(self, tmp_path):
        root = tmp_path / "cloud.parquet"
        write_tile(root, 0, [1, 0, 0, 0, 0, 3])

        with pytest.raises(Exception) as error:
            open_storage(root, points=5)

        assert error.value.code == "POINT_CLOUD_UNREADABLE"
        assert "Parquet has 4 points" in error.value.traceback

    def test_catalogue_requires_exact_lod_row_group_statistics(self, tmp_path):
        root = tmp_path / "cloud.parquet"
        write_tile(root, 0, [1, 3], group_per_lod=False)

        with pytest.raises(Exception) as error:
            open_storage(root)

        assert error.value.code == "POINT_CLOUD_UNREADABLE"
        assert "one exact lod value" in error.value.traceback


class TestExactTileRead:
    @pytest.fixture
    def sparse_multipart_tile(self, tmp_path):
        root = tmp_path / "cloud.parquet"
        write_tile(root, 0, [1, 0, 0, 0, 0, 0], part_sequence=0)
        write_tile(root, 0, [0, 0, 0, 0, 0, 3], part_sequence=1)
        storage = open_storage(root)
        return storage, storage.tiles[0]

    def test_reads_indexed_parts_without_listing(self, tmp_path):
        root = tmp_path / "cloud.parquet"
        write_tile(root, 0, [1, 0, 0, 0, 0, 0], part_sequence=0)
        write_tile(root, 0, [0, 0, 0, 0, 0, 3], part_sequence=1)
        (root / "_manifest.json").write_text(json.dumps(full_manifest()))
        filesystem = FsspecLocalFileSystem()

        def no_listing(*_args, **_kwargs):
            raise AssertionError("indexed point-cloud reads must not list storage")

        filesystem.ls = no_listing
        filesystem.find = no_listing
        filesystem.glob = no_listing
        storage = open_point_cloud(str(root), filesystem)
        result = read_tile(
            storage,
            0,
            0,
            ["X", "classification"],
        )

        assert result.column_names == ["X", "classification"]
        assert result.schema.field("X").type == pa.int32()
        assert result.schema.field("classification").type == pa.uint8()
        assert result.num_rows == 4

    @pytest.mark.parametrize("max_lod,expected", [(0, 1), (4, 1), (5, 4)])
    def test_sparse_lod_counts_match_the_catalogue(
        self, sparse_multipart_tile, max_lod, expected
    ):
        storage, tile = sparse_multipart_tile

        result = read_tile(
            storage,
            0,
            0,
            ["X"],
            max_lod=max_lod,
        )

        assert result.num_rows == expected
        assert expected == tile["points_by_lod"][max_lod]

    def test_projection_and_class_filter_compose(self, tmp_path):
        root = tmp_path / "cloud.parquet"
        write_tile(root, 0, [2, 2], classes=[2, 5])
        storage = open_storage(root)

        result = read_tile(
            storage,
            0,
            0,
            ["Z"],
            classes=[5],
        )

        assert result.column_names == ["Z"]
        assert result.num_rows == 2
        assert result.column("Z").to_pylist() == [11_000, 11_000]

    def test_find_tile_uses_the_catalogue(self):
        first = {
            "tile_x": -1,
            "tile_y": 3,
            "points": 1,
            "points_by_lod": [1] * LOD_LEVELS,
        }
        second = {
            "tile_x": 2,
            "tile_y": 4,
            "points": 1,
            "points_by_lod": [1] * LOD_LEVELS,
        }
        tiles = (first, second)

        assert find_tile(tiles, 2, 4) is second
        assert find_tile(tiles, 1, 4) is None
