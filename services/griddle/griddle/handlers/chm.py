"""
CHM (Canopy Height Model) source handlers.

Pure functions that fetch CHM data for a domain extent.
All handlers return xr.Dataset where each variable name is a band name.
"""

import io
import logging
import traceback
from collections.abc import Callable

import gcsfs
import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from rioxarray.merge import merge_arrays

from griddle.handlers.tiles import TileMetadata
from griddle.utils import to_dataset
from lib.alignment import RESAMPLING_METHOD_MAP, resolve_alignment_destination
from lib.config import TABLES_BUCKET
from lib.errors import ProcessingError
from lib.raster import RasterConnection, cog_env

META_VERSION_CONFIG = {
    "1": {
        "s3_base": "s3://dataforgood-fb-data/forests/v1/alsgedi_global_v6_float/chm",
        "tile_index": f"{TABLES_BUCKET}/Meta2024_chm_index_optimized.parquet",
    },
    "2": {
        "s3_base": "s3://dataforgood-fb-data/forests/v2/global/dinov3_global_chm_v2_ml3/chm",
        "tile_index": f"{TABLES_BUCKET}/Meta_chmv2_index_optimized.parquet",
    },
}
NAIP_INDEX_PATH = f"{TABLES_BUCKET}/naip_chm_index_optimized.parquet"


def _query_tile_index(index_path: str, roi: gpd.GeoDataFrame) -> pd.DataFrame:
    """Download a tile index and return rows whose bbox intersects the ROI.

    The index is a plain parquet with flat bbox columns (bbox_xmin, bbox_ymin,
    bbox_xmax, bbox_ymax) produced by scripts/optimize_tile_index.py.
    """
    roi_4326 = roi.to_crs("EPSG:4326")
    xmin_q, ymin_q, xmax_q, ymax_q = roi_4326.total_bounds

    fs = gcsfs.GCSFileSystem()
    raw = fs.cat(index_path)
    df = pd.read_parquet(io.BytesIO(raw))
    mask = (
        (df["bbox_xmax"] >= xmin_q)
        & (df["bbox_xmin"] <= xmax_q)
        & (df["bbox_ymax"] >= ymin_q)
        & (df["bbox_ymin"] <= ymax_q)
    )
    return df[mask]


def _process_intersecting_tiles(
    roi: gpd.GeoDataFrame,
    fetch_urls: list[str],
    scale_factors: list[float],
    gdal_env: dict[str, str],
    progress: Callable[[str, int | None], None],
    extent_buffer_cells: int,
    alignment: dict,
    target_grid_doc: dict | None,
) -> tuple[xr.Dataset, TileMetadata]:
    """Shared core processing logic for CHM extractions.

    Each tile's ``extract_window`` is called with the alignment destination
    so all tiles share the same output transform/shape; ``merge_arrays``
    then mosaics them with nodata-aware compositing.
    """
    n_tiles = len(fetch_urls)
    tile_arrays = []
    method_name = alignment.get("method") or "bilinear"
    # All tiles in a single fetch share the same scale (Meta = 1.0,
    # NAIP = 100.0). The scale is applied once after merging.
    if len(set(scale_factors)) != 1:
        raise ProcessingError(
            code="INTERNAL_ERROR",
            message="Mixed scale factors across tiles are not supported.",
        )

    with cog_env(**gdal_env):
        for i, url in enumerate(fetch_urls):
            progress(
                f"Fetching CHM tile {i + 1}/{n_tiles}...",
                10 + int(60 * (i + 1) / n_tiles),
            )

            raster = RasterConnection(url, connection_type="rioxarray", cache=True)
            dest = resolve_alignment_destination(
                alignment,
                roi,
                target_grid_doc,
                raster.target_native_resolution(roi)[0],
                extent_buffer_cells=extent_buffer_cells,
            )
            data = raster.extract_window(
                roi=roi,
                interpolation_padding_cells=extent_buffer_cells,
                resampling=RESAMPLING_METHOD_MAP[method_name],
                destination_resolution=alignment.get("resolution")
                if alignment["target"] == "native"
                else None,
                **dest,
            )

            # Squeeze band dimension to satisfy strict 2D requirements.
            # Keep the tile in its native dtype so merge_arrays can compose
            # with the source nodata sentinel — applying the scale factor
            # here would convert the sentinel into an indistinguishable
            # float value (e.g. NAIP 65535/100 = 655.35) and merge_arrays
            # would then overwrite valid pixels with scaled-nodata garbage
            # across tile boundaries.
            tile_arrays.append(data.squeeze("band", drop=True))

    progress("Building dataset...", 80)

    # Mosaic if the ROI spanned multiple tiles
    if len(tile_arrays) == 1:
        chm_da = tile_arrays[0]
    else:
        chm_da = merge_arrays(tile_arrays)

    # Now that compositing is done, mask the source nodata sentinel to NaN
    # and apply the scale factor (e.g. NAIP UInt16 -> float meters). Both
    # happen post-merge so the sentinel never survives into output pixels.
    nodata = chm_da.rio.nodata
    chm_da = chm_da.astype("float32")
    if nodata is not None:
        chm_da = chm_da.where(chm_da != np.float32(nodata))
    scale = scale_factors[0]
    if scale != 1.0:
        chm_da = chm_da / np.float32(scale)
    chm_da = chm_da.rio.write_nodata(np.float32("nan"))

    # Create dataset with CRS and transform
    variables = {"chm": chm_da}
    ds = to_dataset(variables)

    tile_metadata: TileMetadata = {
        "tiles": fetch_urls,
        "tile_source": None,
        "tile_count": len(fetch_urls),
        "native_crs": str(chm_da.rio.crs) if chm_da.rio.crs else None,
        "acquisition_dates": None,
    }

    return ds, tile_metadata


def fetch_meta_chm(
    roi: gpd.GeoDataFrame,
    version: str,
    progress: Callable[[str, int | None], None],
    extent_buffer_cells: int = 0,
    alignment: dict | None = None,
    target_grid_doc: dict | None = None,
) -> tuple[xr.Dataset, TileMetadata]:
    """Fetch Meta global canopy height model data.

    Args:
        roi: GeoDataFrame defining the region of interest
        version: Meta CHM version ("1" or "2")
        progress: Progress callback
        extent_buffer_cells: Result-grid cells of buffer around the ROI
        alignment: Alignment specification dict. Defaults to
            ``{"target": "domain"}`` when omitted.
        target_grid_doc: Loaded grid document used when
            ``alignment["target"] == "grid"``.

    Returns:
        Tuple of (Dataset with the ``chm`` variable, tile metadata dict).
    """
    alignment = alignment or {"target": "domain"}
    progress("Loading Meta CHM parquet index...", 10)

    config = META_VERSION_CONFIG[version]
    try:
        intersecting = _query_tile_index(config["tile_index"], roi)
    except Exception as e:
        tb = traceback.format_exc()
        logging.getLogger(__name__).error("Meta CHM index lookup failed: %s\n%s", e, tb)
        raise ProcessingError(
            code="INDEX_FETCH_FAILED",
            message="Failed to load Meta CHM file index.",
            traceback=tb,
        )

    if intersecting.empty:
        raise ProcessingError(
            code="COVERAGE_ERROR",
            message="No Meta CHM tiles found for the given domain extent.",
            suggestion="The domain may be outside the dataset coverage area.",
        )

    fetch_urls = [f"{config['s3_base']}/{tile}.tif" for tile in intersecting["tile"]]
    scale_factors = [1.0] * len(fetch_urls)

    return _process_intersecting_tiles(
        roi=roi,
        fetch_urls=fetch_urls,
        scale_factors=scale_factors,
        gdal_env={"AWS_NO_SIGN_REQUEST": "YES"},
        progress=progress,
        extent_buffer_cells=extent_buffer_cells,
        alignment=alignment,
        target_grid_doc=target_grid_doc,
    )


def fetch_naip_chm(
    roi: gpd.GeoDataFrame,
    progress: Callable[[str, int | None], None],
    extent_buffer_cells: int = 0,
    alignment: dict | None = None,
    target_grid_doc: dict | None = None,
) -> tuple[xr.Dataset, TileMetadata]:
    """Fetch NAIP high-resolution canopy height model data.

    Args:
        roi: GeoDataFrame defining the region of interest
        progress: Progress callback
        extent_buffer_cells: Result-grid cells of buffer around the ROI
        alignment: Alignment specification dict. Defaults to
            ``{"target": "domain"}`` when omitted.
        target_grid_doc: Loaded grid document used when
            ``alignment["target"] == "grid"``.

    Returns:
        Tuple of (Dataset with the ``chm`` variable, tile metadata dict).
    """
    alignment = alignment or {"target": "domain"}
    progress("Loading NAIP CHM parquet index...", 10)

    try:
        intersecting = _query_tile_index(NAIP_INDEX_PATH, roi)
    except Exception as e:
        tb = traceback.format_exc()
        logging.getLogger(__name__).error("NAIP CHM index lookup failed: %s\n%s", e, tb)
        raise ProcessingError(
            code="INDEX_FETCH_FAILED",
            message="Failed to load NAIP CHM file index.",
            traceback=tb,
        )

    if intersecting.empty:
        raise ProcessingError(
            code="COVERAGE_ERROR",
            message="No NAIP CHM tiles found for the given domain extent.",
            suggestion="The domain may be outside the CONUS coverage area.",
        )

    # Extract exactly what the helper needs into explicit Python lists
    fetch_urls = intersecting["chm_url"].tolist()
    if "scale_factor" in intersecting.columns:
        scale_factors = intersecting["scale_factor"].tolist()
    else:
        scale_factors = [100.0] * len(fetch_urls)

    return _process_intersecting_tiles(
        roi=roi,
        fetch_urls=fetch_urls,
        scale_factors=scale_factors,
        gdal_env={},  # Standard HTTP requests require no auth config
        progress=progress,
        extent_buffer_cells=extent_buffer_cells,
        alignment=alignment,
        target_grid_doc=target_grid_doc,
    )
