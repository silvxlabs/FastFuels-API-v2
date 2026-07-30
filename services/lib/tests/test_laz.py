"""
Unit tests for lib.laz canonical LAZ output.

Everything runs in memory against synthetic records — no network, no files.
The cross-format cases are the point of this module: merging point clouds from
different sources fails loudly in laspy unless records are normalized first.

Run with: uv run --extra pointcloud pytest tests/test_laz.py -v
"""

import io

import laspy
import numpy as np
import pytest
from laspy.header import GpsTimeType
from laspy.point.record import ScaleAwarePointRecord
from pyproj import CRS

from lib.laz import (
    CANONICAL_POINT_FORMAT_ID,
    LazAccumulator,
    build_output_header,
    merged_point_format_id,
    normalize_record,
)

DOMAIN_CRS = CRS.from_user_input("EPSG:32612")
# A domain-sized extent in UTM 12N metres.
BOUNDS = (500000.0, 4300000.0, 501000.0, 4301000.0)


def make_source(
    point_format_id: int,
    count: int = 10,
    *,
    version: str = "1.2",
    extra: str | None = None,
    scales=(0.01, 0.01, 0.01),
    offsets=(-12700000.0, 4600000.0, 0.0),
) -> ScaleAwarePointRecord:
    """Build a source record, by default in EPSG:3857-magnitude coordinates.

    The default offsets mimic a 3DEP EPT node: coordinates around -1.27e7 that
    cannot be re-encoded against a UTM header's offsets without overflowing.
    """
    header = laspy.LasHeader(
        version=version, point_format=laspy.PointFormat(point_format_id)
    )
    header.scales = list(scales)
    header.offsets = list(offsets)
    if extra:
        header.add_extra_dim(laspy.ExtraBytesParams(name=extra, type=np.uint64))

    record = ScaleAwarePointRecord.zeros(count, header=header)
    record.x = np.full(count, offsets[0] + 100.0)
    record.y = np.full(count, offsets[1] + 100.0)
    record.z = np.linspace(1500.0, 1600.0, count)
    record.intensity = np.arange(count, dtype=np.uint16)
    record.classification = np.array([2] * (count // 2) + [5] * (count - count // 2))
    record.point_source_id = np.full(count, 7, dtype=np.uint16)
    if extra:
        record[extra] = np.arange(count, dtype=np.uint64)
    return record


def output_coords(count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Coordinates inside BOUNDS, standing in for a reprojection result."""
    return (
        np.linspace(BOUNDS[0] + 1, BOUNDS[0] + 500, count),
        np.linspace(BOUNDS[1] + 1, BOUNDS[1] + 500, count),
        np.linspace(1500.0, 1600.0, count),
    )


class TestBuildOutputHeader:
    """Tests for the canonical header."""

    def test_uses_canonical_format(self):
        header = build_output_header(DOMAIN_CRS, BOUNDS)
        assert header.point_format.id == CANONICAL_POINT_FORMAT_ID
        assert str(header.version) == "1.4"
        assert header.point_format.num_extra_bytes == 0

    def test_gps_time_type_is_standard_by_default(self):
        """A fresh laspy header claims Week Time; 3DEP is Adjusted Standard.

        Getting this wrong misdates every point by decades and is invisible
        until someone reads the timestamps.
        """
        header = build_output_header(DOMAIN_CRS, BOUNDS)
        assert header.global_encoding.gps_time_type == GpsTimeType.STANDARD

    def test_gps_time_type_can_be_week_time(self):
        header = build_output_header(DOMAIN_CRS, BOUNDS, gps_standard_time=False)
        assert header.global_encoding.gps_time_type == GpsTimeType.WEEK_TIME

    def test_offsets_anchor_to_the_extent(self):
        header = build_output_header(DOMAIN_CRS, BOUNDS)
        assert header.offsets[0] == 500000.0
        assert header.offsets[1] == 4300000.0
        assert header.offsets[2] == 0.0

    def test_crs_round_trips(self):
        header = build_output_header(DOMAIN_CRS, BOUNDS)
        assert header.parse_crs().equals(DOMAIN_CRS)

    def test_a_supplied_format_preserves_its_extra_dimensions(self):
        """Passing a format reproduces it; passing an id rebuilds a bare one.

        Rewriting a single file has no format conflict to resolve, so the
        source's own format is reproduced verbatim. Scanner exports routinely
        add dimensions like amplitude or reflectance, and an id alone would
        drop every one of them.
        """
        source = make_source(3, extra="Amplitude")

        preserved = build_output_header(
            DOMAIN_CRS, BOUNDS, point_format=source.point_format
        )
        assert "Amplitude" in preserved.point_format.extra_dimension_names

        by_id = build_output_header(
            DOMAIN_CRS, BOUNDS, point_format=source.point_format.id
        )
        assert by_id.point_format.num_extra_bytes == 0

    def test_extra_dimension_values_round_trip(self):
        source = make_source(3, count=5, extra="Amplitude")
        header = build_output_header(
            DOMAIN_CRS, BOUNDS, point_format=source.point_format
        )

        acc = LazAccumulator(header)
        acc.append(normalize_record(source, header, *output_coords(5)))
        buffer, _, _ = acc.finish()

        reread = laspy.read(buffer)
        assert np.array_equal(
            np.asarray(reread["Amplitude"]), np.asarray(source["Amplitude"])
        )


class TestMergedPointFormatId:
    """Tests for choosing an output format that loses nothing."""

    def test_colourless_sources_use_the_canonical_format(self):
        assert (
            merged_point_format_id(laspy.PointFormat(1), laspy.PointFormat(6))
            == CANONICAL_POINT_FORMAT_ID
        )

    def test_rgb_source_promotes_the_output(self):
        """Point format 6 has no colour, so an RGB source must promote to 7."""
        assert merged_point_format_id(laspy.PointFormat(1), laspy.PointFormat(3)) == 7

    def test_nir_source_promotes_furthest(self):
        assert merged_point_format_id(laspy.PointFormat(1), laspy.PointFormat(8)) == 8

    def test_colour_survives_a_promoted_merge(self):
        """The promotion is only worth anything if the values come through."""
        coloured = make_source(3, count=4)
        coloured.red = np.array([100, 200, 300, 400], dtype=np.uint16)
        plain = make_source(1, count=2)

        header = build_output_header(
            DOMAIN_CRS,
            BOUNDS,
            point_format=merged_point_format_id(
                coloured.point_format, plain.point_format
            ),
        )
        acc = LazAccumulator(header)
        acc.append(normalize_record(coloured, header, *output_coords(4)))
        acc.append(normalize_record(plain, header, *output_coords(2)))
        buffer, stats, _ = acc.finish()

        reread = laspy.read(buffer)
        assert stats["point_count"] == 6
        assert list(np.asarray(reread.red)[:4]) == [100, 200, 300, 400]


class TestNormalizeRecord:
    """Tests for converting a source record into the canonical format."""

    def test_converts_legacy_format_to_canonical(self):
        source = make_source(1)
        header = build_output_header(DOMAIN_CRS, BOUNDS)
        out = normalize_record(source, header, *output_coords(len(source)))
        assert out.point_format.id == CANONICAL_POINT_FORMAT_ID
        assert len(out) == len(source)

    def test_writes_the_supplied_coordinates_not_the_source_ones(self):
        """The source is in 3857-magnitude coordinates; output must be UTM."""
        source = make_source(1)
        header = build_output_header(DOMAIN_CRS, BOUNDS)
        x, y, z = output_coords(len(source))
        out = normalize_record(source, header, x, y, z)
        assert np.allclose(np.asarray(out.x), x, atol=0.01)
        assert np.allclose(np.asarray(out.y), y, atol=0.01)
        assert np.allclose(np.asarray(out.z), z, atol=0.01)

    def test_preserves_attributes(self):
        source = make_source(1)
        header = build_output_header(DOMAIN_CRS, BOUNDS)
        out = normalize_record(source, header, *output_coords(len(source)))
        assert np.array_equal(np.asarray(out.intensity), np.asarray(source.intensity))
        assert np.array_equal(
            np.asarray(out.classification), np.asarray(source.classification)
        )
        assert np.array_equal(
            np.asarray(out.point_source_id), np.asarray(source.point_source_id)
        )

    def test_widens_classification_flags(self):
        """A withheld class-2 point must survive the 5-bit to full-byte move."""
        source = make_source(1, count=4)
        source.classification = np.array([2, 2, 5, 5])
        source.withheld = np.array([1, 0, 0, 0])
        header = build_output_header(DOMAIN_CRS, BOUNDS)
        out = normalize_record(source, header, *output_coords(4))
        assert list(np.asarray(out.classification)) == [2, 2, 5, 5]
        assert list(np.asarray(out.withheld)) == [1, 0, 0, 0]

    def test_maps_scan_angle_rank_to_scan_angle(self):
        """The dimension is renamed between formats and would silently vanish."""
        source = make_source(1, count=3)
        source.scan_angle_rank = np.array([-30, 0, 30], dtype=np.int8)
        header = build_output_header(DOMAIN_CRS, BOUNDS)
        out = normalize_record(source, header, *output_coords(3))
        # Point format 6 stores 0.006-degree increments.
        assert list(np.asarray(out["scan_angle"])) == [-5000, 0, 5000]

    def test_drops_extra_dimensions(self):
        """EPT files carry OriginId, which is meaningless after a merge."""
        source = make_source(1, extra="OriginId")
        header = build_output_header(DOMAIN_CRS, BOUNDS)
        out = normalize_record(source, header, *output_coords(len(source)))
        assert "OriginId" not in out.point_format.dimension_names
        assert out.point_format.num_extra_bytes == 0

    def test_out_of_range_coordinates_raise(self):
        """Coordinates outside the header's extent must fail loudly.

        Re-encoding a coordinate far from the header offset overflows int32.
        The scaled assignment path range-checks; the raw path would wrap
        silently, so this test pins us to the safe one.
        """
        source = make_source(1, count=3)
        header = build_output_header(DOMAIN_CRS, BOUNDS)
        far = np.full(3, -12700000.0)
        with pytest.raises(OverflowError):
            normalize_record(source, header, far, far, np.zeros(3))


class TestCrossFormatMerge:
    """Tests for writing records from different sources into one file."""

    def test_writes_normalized_legacy_record(self):
        """Guards the LaspyException raised on a point-format mismatch."""
        source = make_source(1, extra="OriginId")
        header = build_output_header(DOMAIN_CRS, BOUNDS)
        acc = LazAccumulator(header)
        acc.append(normalize_record(source, header, *output_coords(len(source))))
        _, stats, _ = acc.finish()
        assert stats["point_count"] == len(source)

    def test_merges_different_point_formats(self):
        """Point format 1 and point format 6 sources into one output file.

        This is the multi-acquisition case: two 3DEP acquisitions need not
        agree on point format, and laspy will not write mismatched records.
        """
        legacy = make_source(1, count=6, extra="OriginId")
        modern = make_source(6, count=4, version="1.4")
        header = build_output_header(DOMAIN_CRS, BOUNDS)

        acc = LazAccumulator(header)
        acc.append(normalize_record(legacy, header, *output_coords(6)))
        acc.append(normalize_record(modern, header, *output_coords(4)))
        buffer, stats, _ = acc.finish()

        assert stats["point_count"] == 10
        reread = laspy.read(buffer)
        assert len(reread.points) == 10

    def test_raw_append_of_mismatched_record_still_fails(self):
        """Sanity check that normalization is what makes the merge work."""
        header = build_output_header(DOMAIN_CRS, BOUNDS)
        acc = LazAccumulator(header)
        with pytest.raises(laspy.LaspyException):
            acc.append(make_source(1))


class TestLazAccumulator:
    """Tests for the streaming writer and its statistics."""

    def test_round_trips_through_a_real_laz(self):
        source = make_source(1, count=20)
        header = build_output_header(DOMAIN_CRS, BOUNDS)
        x, y, z = output_coords(20)

        acc = LazAccumulator(header)
        acc.append(normalize_record(source, header, x, y, z))
        buffer, stats, bounds = acc.finish()

        reread = laspy.read(buffer)
        assert len(reread.points) == 20
        assert reread.header.parse_crs().equals(DOMAIN_CRS)
        assert np.allclose(np.asarray(reread.x), x, atol=0.01)
        assert bounds[0] == pytest.approx(x.min(), abs=0.01)
        assert bounds[3] == pytest.approx(x.max(), abs=0.01)
        assert stats["point_count"] == 20

    def test_statistics_describe_what_was_written(self):
        source = make_source(1, count=10)
        source.classification = np.array([2] * 5 + [5] * 3 + [1] * 2)
        header = build_output_header(DOMAIN_CRS, BOUNDS)
        acc = LazAccumulator(header)
        acc.append(normalize_record(source, header, *output_coords(10)))
        _, stats, _ = acc.finish()

        assert stats["point_classes"] == [1, 2, 5]
        assert stats["point_count"] == 10
        assert stats["density"] > 0

    def test_classes_accumulate_across_appends(self):
        header = build_output_header(DOMAIN_CRS, BOUNDS)
        acc = LazAccumulator(header)
        for classification in (2, 5):
            source = make_source(1, count=4)
            source.classification = np.full(4, classification)
            acc.append(normalize_record(source, header, *output_coords(4)))
        _, stats, _ = acc.finish()
        assert stats["point_classes"] == [2, 5]
        assert stats["point_count"] == 8

    def test_empty_cloud_finishes_cleanly(self):
        """No points is a valid, if useless, outcome — it must not crash."""
        header = build_output_header(DOMAIN_CRS, BOUNDS)
        buffer, stats, bounds = LazAccumulator(header).finish()
        assert stats["point_count"] == 0
        assert stats["point_classes"] == []
        assert stats["density"] == 0.0
        assert all(np.isfinite(bounds))
        assert len(laspy.read(buffer).points) == 0

    def test_appending_an_empty_record_is_a_noop(self):
        header = build_output_header(DOMAIN_CRS, BOUNDS)
        acc = LazAccumulator(header)
        acc.append(ScaleAwarePointRecord.zeros(0, header=header))
        assert acc.point_count == 0
        _, stats, _ = acc.finish()
        assert stats["point_count"] == 0

    def test_output_is_compressed(self):
        source = make_source(1, count=1000)
        header = build_output_header(DOMAIN_CRS, BOUNDS)
        acc = LazAccumulator(header)
        acc.append(normalize_record(source, header, *output_coords(1000)))
        buffer, _, _ = acc.finish()
        assert laspy.open(io.BytesIO(buffer.getvalue())).header.are_points_compressed
