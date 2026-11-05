"""Offline unit tests for the Overture Maps buildings reader.

These never touch S3: the filesystem and ``pyarrow.dataset.dataset`` are mocked,
and reads run against an in-memory dataset shaped like Overture's buildings
theme (``id``, WKB ``geometry``, ``height``, ``sources``, ``bbox`` struct).
"""

from datetime import date
from unittest.mock import MagicMock, patch

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.fs as pafs
import pytest
import shapely
from etcher import overture_source
from etcher.errors import ProcessingError
from shapely.geometry import box

ROI = (0.0, 0.0, 10.0, 10.0)


def _dir(name):
    return pafs.FileInfo(
        f"overturemaps-us-west-2/release/{name}", pafs.FileType.Directory
    )


def _fake_fs(infos):
    fs = MagicMock()
    fs.get_file_info.return_value = infos
    return fs


def _buildings_dataset(geoms):
    """In-memory dataset with one building per geometry."""
    bounds = [g.bounds for g in geoms]
    sources_type = pa.list_(
        pa.struct(
            [
                ("property", pa.string()),
                ("dataset", pa.string()),
                ("license", pa.string()),
            ]
        )
    )
    table = pa.table(
        {
            "id": [f"b{i}" for i in range(len(geoms))],
            "geometry": pa.array(shapely.to_wkb(geoms).tolist(), pa.binary()),
            "height": pa.array([5.0 + i for i in range(len(geoms))], pa.float64()),
            "sources": pa.array(
                [[{"property": "", "dataset": "OpenStreetMap", "license": "ODbL-1.0"}]]
                * len(geoms),
                sources_type,
            ),
            "bbox": pa.array(
                [
                    {"xmin": b[0], "xmax": b[2], "ymin": b[1], "ymax": b[3]}
                    for b in bounds
                ],
                pa.struct(
                    [(k, pa.float64()) for k in ("xmin", "xmax", "ymin", "ymax")]
                ),
            ),
        }
    )
    return ds.InMemoryDataset(table)


@pytest.mark.parametrize(
    "names, expected",
    [
        (["2026-09-23.0", "2026-09-23.1"], "2026-09-23.1"),
        (["2026-09-23.1", "2026-10-21.0", "2026-09-23.0"], "2026-10-21.0"),
        # Numeric N, not string order: .10 is newer than .9.
        (["2026-09-23.9", "2026-09-23.10"], "2026-09-23.10"),
        (["2026-09-23.0", "not-a-release", "2026-13-01.0"], "2026-09-23.0"),
        (["junk", "2026-02-30.0"], None),
        ([], None),
    ],
)
def test_newest_release_orders_by_date_then_n(names, expected):
    assert overture_source.newest_release(names) == expected


def test_parse_release():
    assert overture_source.parse_release("2026-10-21.0") == (date(2026, 10, 21), 0)
    assert overture_source.parse_release("2026-10-21") is None
    assert overture_source.parse_release("2026-10-21.0.1") is None


def test_latest_release_lists_release_dirs():
    fs = _fake_fs([_dir("2026-09-23.0"), _dir("2026-10-21.0"), _dir("2026-09-23.1")])
    with patch.object(overture_source, "s3_filesystem", return_value=fs):
        assert overture_source.latest_release() == "2026-10-21.0"
    selector = fs.get_file_info.call_args.args[0]
    assert selector.base_dir == "overturemaps-us-west-2/release"
    assert not selector.recursive


def test_latest_release_ignores_files():
    file_info = pafs.FileInfo(
        "overturemaps-us-west-2/release/2027-01-01.0", pafs.FileType.File
    )
    fs = _fake_fs([_dir("2026-10-21.0"), file_info])
    with patch.object(overture_source, "s3_filesystem", return_value=fs):
        assert overture_source.latest_release() == "2026-10-21.0"


def test_latest_release_listing_failure_is_processing_error():
    fs = MagicMock()
    fs.get_file_info.side_effect = OSError("connection reset")
    with patch.object(overture_source, "s3_filesystem", return_value=fs):
        with pytest.raises(ProcessingError) as exc:
            overture_source.latest_release()
    assert exc.value.code == "OVERTURE_SOURCE_READ_FAILED"
    assert isinstance(exc.value.__cause__, OSError)


def test_latest_release_none_published():
    fs = _fake_fs([_dir("junk")])
    with patch.object(overture_source, "s3_filesystem", return_value=fs):
        with pytest.raises(ProcessingError) as exc:
            overture_source.latest_release()
    assert exc.value.code == "OVERTURE_RELEASE_NOT_FOUND"


def test_bbox_filter_expression():
    expr = overture_source.bbox_filter((1.0, 2.0, 3.0, 4.0))
    expected = (
        (pc.field("bbox", "xmin") <= 3.0)
        & (pc.field("bbox", "xmax") >= 1.0)
        & (pc.field("bbox", "ymin") <= 4.0)
        & (pc.field("bbox", "ymax") >= 2.0)
    )
    assert expr.equals(expected)


@pytest.mark.parametrize(
    "geom, keep",
    [
        (box(1, 1, 2, 2), True),  # inside
        (box(9, 9, 11, 11), True),  # straddles the NE corner
        (box(-1, 4, 0.5, 5), True),  # straddles the west edge
        (box(11, 1, 12, 2), False),  # east of the ROI
        (box(1, -3, 2, -1), False),  # south of the ROI
        (box(1, 11, 2, 12), False),  # north of the ROI
        (box(-3, 1, -1, 2), False),  # west of the ROI
    ],
)
def test_bbox_filter_selects_overlapping_rows(geom, keep):
    dataset = _buildings_dataset([geom])
    table = dataset.to_table(filter=overture_source.bbox_filter(ROI))
    assert table.num_rows == (1 if keep else 0)


def test_read_overture_buildings_decodes_and_clips():
    # An L-shape whose bbox overlaps the ROI but whose geometry does not.
    l_shape = shapely.Polygon(
        [(11, -5), (20, -5), (20, 20), (-5, 20), (-5, 11), (11, 11)]
    )
    dataset = _buildings_dataset([box(1, 1, 2, 2), l_shape, box(30, 30, 31, 31)])
    with (
        patch.object(overture_source, "s3_filesystem", return_value=MagicMock()),
        patch.object(overture_source.ds, "dataset", return_value=dataset) as open_ds,
    ):
        gdf = overture_source.read_overture_buildings(ROI, "2026-10-21.0")

    assert open_ds.call_args.args[0] == (
        "overturemaps-us-west-2/release/2026-10-21.0/theme=buildings/type=building/"
    )
    assert gdf.crs == "EPSG:4326"
    assert list(gdf["id"]) == ["b0"]
    assert set(gdf.columns) == {"id", "height", "sources", "geometry"}
    assert gdf.geometry.iloc[0].equals(box(1, 1, 2, 2))
    assert gdf["height"].iloc[0] == 5.0
    assert gdf["sources"].iloc[0][0]["dataset"] == "OpenStreetMap"


def test_read_overture_buildings_empty():
    dataset = _buildings_dataset([box(30, 30, 31, 31)])
    with (
        patch.object(overture_source, "s3_filesystem", return_value=MagicMock()),
        patch.object(overture_source.ds, "dataset", return_value=dataset),
    ):
        gdf = overture_source.read_overture_buildings(ROI, "2026-10-21.0")
    assert gdf.empty
    assert gdf.crs == "EPSG:4326"


def test_read_failure_is_processing_error():
    with (
        patch.object(overture_source, "s3_filesystem", return_value=MagicMock()),
        patch.object(overture_source.ds, "dataset", side_effect=OSError("timeout")),
    ):
        with pytest.raises(ProcessingError) as exc:
            overture_source.read_overture_buildings(ROI, "2026-10-21.0")
    assert exc.value.code == "OVERTURE_SOURCE_READ_FAILED"
    assert "2026-10-21.0" in exc.value.message
    assert isinstance(exc.value.__cause__, OSError)


def test_overture_attribution():
    attribution = overture_source.overture_attribution()
    assert attribution == {
        "dataset": "Overture Maps Foundation",
        "license_name": "ODbL-1.0",
        "license_url": "https://opendatacommons.org/licenses/odbl/1-0/",
        "citation": "© OpenStreetMap contributors, Overture Maps Foundation",
        "access_url": "https://docs.overturemaps.org/attribution/",
        "accessed_on": date.today().isoformat(),
    }
