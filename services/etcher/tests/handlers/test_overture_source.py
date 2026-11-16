"""Offline unit tests for the Overture Maps buildings reader.

These never touch the network: the STAC catalog fetch and
``pyarrow.dataset.dataset`` are mocked. The file index is a small
``collections.parquet`` shaped like Overture's, and reads run against an
in-memory dataset shaped like Overture's buildings theme (``id``, WKB
``geometry``, ``height``, ``sources``, ``bbox`` struct).
"""

import json
from datetime import date
from unittest.mock import MagicMock, patch

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pytest
import shapely
from etcher import overture_source
from etcher.errors import ProcessingError
from shapely.geometry import box

ROI = (0.0, 0.0, 10.0, 10.0)
RELEASE = "2026-10-21.0"
INDEX_URL = f"https://stac.overturemaps.org/{RELEASE}/collections.parquet"
CATALOG_URL = "https://stac.overturemaps.org/catalog.json"
BUILDINGS = "overturemaps-us-west-2/release/2026-10-21.0/theme=buildings/type=building"


def _bbox_struct(rows):
    return pa.array(
        [{"xmin": b[0], "ymin": b[1], "xmax": b[2], "ymax": b[3]} for b in rows],
        pa.struct([(k, pa.float64()) for k in ("xmin", "ymin", "xmax", "ymax")]),
    )


def _index_bytes(files):
    """A ``collections.parquet`` with one row per (collection, bbox, s3 path)."""
    table = pa.table(
        {
            "collection": pa.array([c for c, _, _ in files]).dictionary_encode(),
            "assets": pa.array(
                [
                    {
                        "aws": {
                            "href": "https://unused",
                            "alternate": {"s3": {"href": f"s3://{p}"}},
                        }
                    }
                    for _, _, p in files
                ]
            ),
            "bbox": _bbox_struct([b for _, b, _ in files]),
        }
    )
    sink = pa.BufferOutputStream()
    pq.write_table(table, sink)
    return sink.getvalue().to_pybytes()


INDEX = _index_bytes(
    [
        ("building", (-1.0, -1.0, 5.0, 5.0), f"{BUILDINGS}/part-00001.parquet"),
        ("building", (4.0, 4.0, 20.0, 20.0), f"{BUILDINGS}/part-00000.parquet"),
        ("building", (30.0, 30.0, 40.0, 40.0), f"{BUILDINGS}/part-00002.parquet"),
        ("building_part", (0.0, 0.0, 10.0, 10.0), "other/part-00000.parquet"),
        ("address", (0.0, 0.0, 10.0, 10.0), "other/part-00001.parquet"),
    ]
)


def _fake_fetch(responses):
    def fetch(url):
        if url not in responses:
            raise OSError(f"404 {url}")
        return responses[url]

    return fetch


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


def test_parse_release():
    assert overture_source.parse_release("2026-10-21.0") == (date(2026, 10, 21), 0)
    assert overture_source.parse_release("2026-09-23.10") == (date(2026, 9, 23), 10)
    assert overture_source.parse_release("2026-10-21") is None
    assert overture_source.parse_release("2026-10-21.0.1") is None
    assert overture_source.parse_release("2026-13-01.0") is None


def test_latest_release_reads_stac_root():
    catalog = json.dumps({"type": "Catalog", "latest": RELEASE}).encode()
    fetch = _fake_fetch({CATALOG_URL: catalog})
    with patch.object(overture_source, "fetch", side_effect=fetch):
        assert overture_source.latest_release() == RELEASE


def test_latest_release_fetch_failure_is_processing_error():
    with patch.object(overture_source, "fetch", side_effect=OSError("reset")):
        with pytest.raises(ProcessingError) as exc:
            overture_source.latest_release()
    assert exc.value.code == "OVERTURE_SOURCE_READ_FAILED"
    assert isinstance(exc.value.__cause__, OSError)


@pytest.mark.parametrize(
    "body", [b"{}", b'{"latest": "junk"}', b'{"latest": 5}', b"[]"]
)
def test_latest_release_none_published(body):
    with patch.object(
        overture_source, "fetch", side_effect=_fake_fetch({CATALOG_URL: body})
    ):
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


@pytest.mark.parametrize(
    "bbox, expected",
    [
        (ROI, ["part-00000.parquet", "part-00001.parquet"]),
        ((0.0, 0.0, 1.0, 1.0), ["part-00001.parquet"]),
        ((35.0, 35.0, 36.0, 36.0), ["part-00002.parquet"]),
        ((50.0, 50.0, 51.0, 51.0), []),
    ],
)
def test_building_files_picks_overlapping_building_files(bbox, expected):
    with patch.object(
        overture_source, "fetch", side_effect=_fake_fetch({INDEX_URL: INDEX})
    ):
        paths = overture_source.building_files(bbox, RELEASE)
    assert paths == [f"{BUILDINGS}/{name}" for name in expected]


def test_read_overture_buildings_decodes_and_clips():
    # An L-shape whose bbox overlaps the ROI but whose geometry does not.
    l_shape = shapely.Polygon(
        [(11, -5), (20, -5), (20, 20), (-5, 20), (-5, 11), (11, 11)]
    )
    dataset = _buildings_dataset([box(1, 1, 2, 2), l_shape, box(30, 30, 31, 31)])
    with (
        patch.object(
            overture_source, "fetch", side_effect=_fake_fetch({INDEX_URL: INDEX})
        ),
        patch.object(overture_source, "s3_filesystem", return_value=MagicMock()),
        patch.object(overture_source.ds, "dataset", return_value=dataset) as open_ds,
    ):
        gdf = overture_source.read_overture_buildings(ROI, RELEASE)

    # Only the files the index places over the ROI are opened.
    assert open_ds.call_args.args[0] == [
        f"{BUILDINGS}/part-00000.parquet",
        f"{BUILDINGS}/part-00001.parquet",
    ]
    assert gdf.crs == "EPSG:4326"
    assert list(gdf["id"]) == ["b0"]
    assert set(gdf.columns) == {"id", "height", "sources", "geometry"}
    assert gdf.geometry.iloc[0].equals(box(1, 1, 2, 2))
    assert gdf["height"].iloc[0] == 5.0
    assert gdf["sources"].iloc[0][0]["dataset"] == "OpenStreetMap"


def test_read_overture_buildings_no_rows_in_bbox():
    dataset = _buildings_dataset([box(30, 30, 31, 31)])
    with (
        patch.object(
            overture_source, "fetch", side_effect=_fake_fetch({INDEX_URL: INDEX})
        ),
        patch.object(overture_source, "s3_filesystem", return_value=MagicMock()),
        patch.object(overture_source.ds, "dataset", return_value=dataset),
    ):
        gdf = overture_source.read_overture_buildings(ROI, RELEASE)
    assert gdf.empty
    assert gdf.crs == "EPSG:4326"


def test_read_overture_buildings_no_files_in_bbox():
    with (
        patch.object(
            overture_source, "fetch", side_effect=_fake_fetch({INDEX_URL: INDEX})
        ),
        patch.object(overture_source.ds, "dataset") as open_ds,
    ):
        gdf = overture_source.read_overture_buildings((50.0, 50.0, 51.0, 51.0), RELEASE)
    open_ds.assert_not_called()
    assert gdf.empty
    assert gdf.crs == "EPSG:4326"
    assert set(gdf.columns) == {"id", "height", "sources", "geometry"}


def test_index_failure_is_processing_error():
    with patch.object(overture_source, "fetch", side_effect=OSError("timeout")):
        with pytest.raises(ProcessingError) as exc:
            overture_source.read_overture_buildings(ROI, RELEASE)
    assert exc.value.code == "OVERTURE_SOURCE_READ_FAILED"
    assert RELEASE in exc.value.message
    assert isinstance(exc.value.__cause__, OSError)


def test_read_failure_is_processing_error():
    with (
        patch.object(
            overture_source, "fetch", side_effect=_fake_fetch({INDEX_URL: INDEX})
        ),
        patch.object(overture_source, "s3_filesystem", return_value=MagicMock()),
        patch.object(overture_source.ds, "dataset", side_effect=OSError("timeout")),
    ):
        with pytest.raises(ProcessingError) as exc:
            overture_source.read_overture_buildings(ROI, RELEASE)
    assert exc.value.code == "OVERTURE_SOURCE_READ_FAILED"
    assert RELEASE in exc.value.message
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
