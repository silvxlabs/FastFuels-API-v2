"""
Batched readers for uploaded tree-inventory files.

Each reader streams the upload straight from GCS (or a local path) without a
local copy and yields pandas DataFrames of about ``batch_size`` rows. Frames
carry v2 column names and are indexed by the row's position in the parsed
file, so pandera failure-case indices refer to file rows.
"""

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import pyarrow.fs as pafs
import pyogrio
import pyproj
import shapely

from lib.errors import ProcessingError

BATCH_SIZE = 65_536

# pyarrow's null markers plus the two more pandas has, so missing values match
# pd.read_csv.
_CSV_NULL_VALUES = [*pacsv.ConvertOptions().null_values, "None", "<NA>"]

_POINT, _MULTIPOINT = 0, 4


def check_exists(src: str) -> None:
    """Raise FileNotFoundError if ``src`` (``gs://`` URI or local path) is missing."""
    fs, path = pafs.FileSystem.from_uri(src)
    if fs.get_file_info(path).type == pafs.FileType.NotFound:
        raise FileNotFoundError(src)


def csv_batches(
    src: str,
    rename: dict,
    columns: set[str],
    as_strings: bool = False,
    batch_size: int = BATCH_SIZE,
):
    """Yield the CSV's v2 columns in batches.

    By default every kept column is read as float64; a cell that is not a
    number raises ``pyarrow.ArrowInvalid``. With ``as_strings`` the columns
    are read as strings and converted the way ``pd.read_csv`` would: numeric
    columns to float64, a column with an unparseable cell left as strings, so
    pandera reports that cell. A header-only file yields one empty frame.

    Args:
        src: ``gs://`` URI or local path of the upload.
        rename: Map of file column name to v2 column name.
        columns: v2 column names to keep.
        as_strings: Read the kept columns as strings.
        batch_size: Approximate rows per batch.

    Raises:
        ProcessingError: ``SCHEMA_VALIDATION_ERROR`` with ``as_strings`` for a
            malformed row, such as one with too few or too many fields.
    """
    fs, path = pafs.FileSystem.from_uri(src)
    with fs.open_input_stream(path) as f:
        head = f.read(1 << 20)
    header = head.split(b"\n", 1)[0] + b"\n"
    names = pacsv.read_csv(pa.BufferReader(header)).column_names
    keep = [c for c in names if rename.get(c, c) in columns]
    row_bytes = max(1, len(head) // max(1, head.count(b"\n")))
    read_options = pacsv.ReadOptions(block_size=max(batch_size * row_bytes, 1 << 16))
    convert_options = pacsv.ConvertOptions(
        include_columns=keep,
        column_types=dict.fromkeys(keep, pa.string() if as_strings else pa.float64()),
        null_values=_CSV_NULL_VALUES,
        strings_can_be_null=True,
    )

    offset = 0
    # Without a line break the file is its header alone, which open_csv rejects.
    if b"\n" in head or b"\r" in head:
        with fs.open_input_stream(path) as f:
            try:
                reader = pacsv.open_csv(
                    f, read_options=read_options, convert_options=convert_options
                )
                for batch in reader:
                    df = _strings_to_numbers(batch) if as_strings else batch.to_pandas()
                    df.columns = [rename.get(c, c) for c in df.columns]
                    df.index = pd.RangeIndex(offset, offset + len(df))
                    offset += len(df)
                    yield df
            except pa.ArrowInvalid as e:
                # Reading strings cannot fail on a cell, so this is a malformed
                # row, such as one with too few or too many fields.
                if not as_strings:
                    raise
                raise ProcessingError(
                    code="SCHEMA_VALIDATION_ERROR",
                    message=f"The CSV file could not be parsed: {e}",
                    suggestion="Give every row one value per header column.",
                )
    if offset == 0:
        yield pd.DataFrame(
            {rename.get(c, c): pd.Series([], dtype="float64") for c in keep}
        )


def _strings_to_numbers(batch: pa.RecordBatch) -> pd.DataFrame:
    """Convert a string batch to float64 columns; a column with an unparseable
    cell keeps its strings, as ``pd.read_csv`` does."""
    out = {}
    for name, arr in zip(batch.schema.names, batch.columns):
        try:
            out[name] = pc.cast(arr, pa.float64()).to_numpy(zero_copy_only=False)
        except pa.ArrowInvalid:
            out[name] = arr.to_pandas()
    return pd.DataFrame(out)


def geo_batches(
    fmt: str,
    src: str,
    rename: dict,
    columns: set[str],
    domain_crs: str,
    batch_size: int = BATCH_SIZE,
):
    """Yield a GeoJSON / GeoPackage upload's points in batches.

    x and y come from the geometry, reprojected to the domain CRS; GeoJSON
    without a CRS is EPSG:4326 per spec. MultiPoints are exploded into one row
    per point.

    Args:
        fmt: ``geojson`` or ``geopackage``.
        src: ``gs://`` URI or local path of the upload.
        rename: Map of file column name to v2 column name.
        columns: v2 column names to keep.
        domain_crs: CRS to reproject x and y into.
        batch_size: Features per batch.

    Raises:
        ProcessingError: ``INVALID_GEOMETRY_TYPE`` for a geometry other than a
            Point or MultiPoint, reported from the first batch that has one;
            ``EMPTY_FILE`` for a file without features.
    """
    path = "/vsigs/" + src.removeprefix("gs://") if src.startswith("gs://") else src
    with pyogrio.open_arrow(path, batch_size=batch_size, use_pyarrow=True) as (
        meta,
        reader,
    ):
        crs = meta["crs"] or ("EPSG:4326" if fmt == "geojson" else None)
        transformer = (
            pyproj.Transformer.from_crs(crs, domain_crs, always_xy=True)
            if crs is not None
            else None
        )
        geom_col = meta["geometry_name"] or "wkb_geometry"
        attrs = [
            c
            for c in reader.schema.names
            if c != geom_col
            and rename.get(c, c) in columns
            and rename.get(c, c) not in ("x", "y")
        ]
        offset = 0
        features = 0
        for batch in reader:
            features += batch.num_rows
            geoms = shapely.from_wkb(
                batch.column(geom_col).to_numpy(zero_copy_only=False)
            )
            type_ids = shapely.get_type_id(geoms)
            bad = ~np.isin(type_ids, (_POINT, _MULTIPOINT))
            if bad.any():
                found = {None if g is None else g.geom_type for g in geoms[bad]}
                raise ProcessingError(
                    code="INVALID_GEOMETRY_TYPE",
                    message=f"Expected Point or MultiPoint geometries, found: {found}.",
                )
            props = batch.select(attrs).to_pandas()
            if (type_ids == _MULTIPOINT).any():
                geoms, parts = shapely.get_parts(geoms, return_index=True)
                props = props.iloc[parts].reset_index(drop=True)
            x, y = shapely.get_x(geoms), shapely.get_y(geoms)
            if transformer is not None:
                x, y = transformer.transform(x, y)
            df = pd.DataFrame({"x": np.asarray(x), "y": np.asarray(y)})
            for col in attrs:
                df[rename.get(col, col)] = props[col].values
            df.index = pd.RangeIndex(offset, offset + len(df))
            offset += len(df)
            yield df
    if features == 0:
        raise ProcessingError(
            code="EMPTY_FILE",
            message="The uploaded file contains no features.",
        )
