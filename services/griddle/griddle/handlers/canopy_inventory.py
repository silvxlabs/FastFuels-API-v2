"""Canopy fuel grids derived directly from a tree inventory.

Reproduces the FuelCalc / LANDFIRE canopy-profile method per output cell —
allometric crown fuel -> available fuel -> vertical distribution -> 1 ft
layers -> per-band reduction, with no voxelization. The science lives in
``fastfuels_core.canopy_fuel``; this handler is the ETL around it: resolve the
output lattice, read the inventory parquet, translate the persisted grid
``source`` into the science kwargs, and hand core a georeferenced band Dataset
to fill in place.

The bands share keys and units with the LANDFIRE canopy source, so the result
drops into anything that accepts one — the landscape export above all. Cells
with no canopy are written as 0, matching LANDFIRE's non-forest encoding.
"""

from collections.abc import Callable

import geopandas as gpd
import numpy as np
import pandas as pd
import rioxarray  # noqa: F401  (registers the .rio accessor)
import xarray as xr
from affine import Affine
from fastfuels_core.canopy_fuel import compute_canopy_metrics

from lib.alignment import resolve_alignment_destination
from lib.crs import crs_equal
from lib.errors import ProcessingError
from lib.inventory_io import (
    CROWN_CLASS_COLUMN,
    canopy_required_columns,
    drop_null_rows,
    read_inventory,
)

# An inventory has no native raster cell size. The API resolves the
# domain-target default (30 m) before persisting, and a grid target inherits
# its target's cell size, so this fallback is only reached by a source that
# bypassed the API's resolution step.
FALLBACK_RESOLUTION_M = 30.0

# Per-tree column handed to core carrying each tree's crown class as one of
# FuelCalc's letters. Built from the inventory's ``fia_crown_class_code`` (FIA
# CCLCD) when it has one; a tree with no code — a null, or an inventory that
# carries no crown class at all — takes ``"N"``, which core folds onto the
# table's Other/none column, the schema's ``missing_crown_class`` fallback.
_CROWN_CLASS_COLUMN = "_crown_class"
_CROWN_CLASS_FILL = "N"

# FIA CCLCD (crown class code) -> the FuelCalc/Reinhardt letters core reads.
# 1 Open grown folds onto Dominant: its crowns receive full light from above,
# like a dominant's. 2 Dominant, 3 Codominant, 4 Intermediate, 5 Overtopped map
# straight through. Core owns letters+Other/none and invites callers carrying a
# different coding to map onto them (available_fuel.CROWN_CLASS_REMAP); this is
# that map for FIA. Any code not here (or null) falls to _CROWN_CLASS_FILL.
_CCLCD_TO_CROWN_CLASS = {1: "D", 2: "D", 3: "C", 4: "I", 5: "S"}

# schema CanopyRunningMeanEdge -> core edge (bulk_density.py). The schema names
# each edge for the reading it produces; core names each for its arithmetic.
_EDGE = {
    "fixed_depth": "slab",
    "ground_clamped": "fuelcalc",
    "truncated": "truncate",
}

# schema CanopyBranchwoodSizePartition -> core branchwood_size_partition. Core
# has no "equations" value: under the brown_1978 arm its "brown_proportions"
# path *is* Brown's own fine-branchwood share, which is what the schema's
# "equations" (the family's native size class) means for Brown.
_SIZE_PARTITION = {
    "equations": "brown_proportions",
    "brown_proportions": "brown_proportions",
    "none": "none",
}


def _resolve_lattice(
    roi: gpd.GeoDataFrame,
    alignment: dict,
    target_grid_doc: dict | None,
    extent_buffer_cells: int,
) -> tuple[Affine, tuple[int, int]]:
    """Return (transform, (height, width)) for the output lattice.

    Goes through the shared resolver every other source handler uses, so the
    same request produces the same lattice here as it would from a raster
    source: ``target="domain"`` on the domain lattice, ``target="grid"``
    cell-for-cell on another grid.

    Raises:
        ProcessingError: If the alignment names no lattice this handler can
            rasterize onto, or names one in another CRS.
    """
    destination = resolve_alignment_destination(
        alignment,
        roi,
        target_grid_doc,
        FALLBACK_RESOLUTION_M,
        extent_buffer_cells=extent_buffer_cells,
    )

    # ``target="native"`` returns a bare CRS override (or nothing) — "reproject,
    # keep the source raster's pixel anchor" — but there is no source raster to
    # take an anchor from. The API rejects native at create time; this covers a
    # source that reached storage some other way.
    if "destination_transform" not in destination:
        raise ProcessingError(
            code="UNSUPPORTED_ALIGNMENT",
            message=(
                f"alignment.target '{alignment['target']}' does not describe a "
                f"lattice a tree inventory can be rasterized onto."
            ),
            suggestion="Recreate the grid with alignment.target 'domain' or 'grid'.",
        )

    # Trees are stored in the domain CRS and this handler does not reproject
    # them, so a target lattice in another CRS would place every stem wrongly.
    destination_crs = destination["destination_crs"]
    if not crs_equal(str(destination_crs), str(roi.crs)):
        raise ProcessingError(
            code="ALIGNMENT_CRS_MISMATCH",
            message=(
                f"The alignment target grid is in {destination_crs}, but this "
                f"domain's inventories are stored in {roi.crs}."
            ),
            suggestion=(
                "Align to a grid in this domain's CRS, or use "
                "alignment.target 'domain'."
            ),
        )

    return destination["destination_transform"], destination["destination_shape"]


def _init_dataset(
    bands: list[str], transform: Affine, crs: str, shape: tuple[int, int]
) -> xr.Dataset:
    """Georeferenced Dataset with one empty float32 band per requested key.

    Core fills each band's cells in place, so the initial fill is only a
    sentinel; cell-centre coordinates are read off the transform so they and
    the transform written beside them cannot disagree.
    """
    height, width = shape
    x_coords = transform.c + (np.arange(width) + 0.5) * transform.a
    y_coords = transform.f + (np.arange(height) + 0.5) * transform.e

    data_vars = {
        key: xr.DataArray(
            np.full(shape, np.nan, dtype=np.float32),
            dims=["y", "x"],
            coords={"y": y_coords, "x": x_coords},
        )
        for key in bands
    }
    ds = xr.Dataset(data_vars)
    ds = ds.rio.write_crs(crs)
    ds = ds.rio.write_transform(transform)
    return ds


def _inventory_columns(source: dict) -> tuple[str | None, str | None]:
    """Optional per-tree columns to project from the parquet, if the source
    reads fuel or crown radius from a column rather than computing them."""
    biomass = source["biomass_source"]
    fuel_column = biomass["column"] if biomass["type"] == "inventory_column" else None
    radius = source["max_crown_radius_source"]
    radius_column = radius["column"] if radius["type"] == "inventory_column" else None
    return fuel_column, radius_column


def _core_kwargs(source: dict) -> dict:
    """Translate a persisted inventory-canopy source into the exact
    ``compute_canopy_metrics`` kwargs.

    The API schema resolves every modeling choice onto the source at write
    time, and core's own defaults are FuelCalc's — the opposite of this
    endpoint's FastFuels-native defaults — so the translation is total: every
    stage kwarg core reads for a computed band is set here from the source,
    never left to a core default. Per-band method kwargs are set only for
    requested bands (source method is non-null); for the rest core's valid
    defaults stand in and the band is not computed.
    """
    kwargs: dict = {}

    # Biomass / available fuel.
    biomass = source["biomass_source"]
    if biomass["type"] == "inventory_column":
        # The column already holds available canopy fuel; core returns it as-is
        # and bypasses allometry, available fuel, and crown-class adjustment.
        kwargs["fuel_column"] = biomass["column"]
    else:
        kwargs["equations"] = biomass["equations"]
        available = source["available_fuel"]  # non-null with an allometry source
        kwargs["foliage_fraction"] = available["foliage_fraction"]
        kwargs["branchwood_fraction"] = available["branchwood"]["fraction"]
        kwargs["branchwood_size_partition"] = _SIZE_PARTITION[
            available["branchwood"]["size_partition"]
        ]

    # Species inclusion (broadleaf drop from the bulk-density profile only).
    kwargs["exclude_hardwoods"] = source["species_inclusion"] == "fuelcalc_default"

    # Crown-class adjustment. The fuelcalc_table arm needs a per-tree crown
    # class; v2 inventories carry none, so a uniform Other/none column is
    # attached to the frame in the handler (see _CROWN_CLASS_COLUMN).
    crown_class = source["crown_class_adjustment"]
    if crown_class["method"] == "fuelcalc_table":
        kwargs["crown_class_adjustment"] = "fuelcalc_table"
        kwargs["crown_class_column"] = _CROWN_CLASS_COLUMN
    else:
        kwargs["crown_class_adjustment"] = "none"

    # Max crown radius source (drives crown_projected attribution and the
    # geometric cover methods).
    radius = source["max_crown_radius_source"]
    if radius["type"] == "inventory_column":
        kwargs["crown_radius_column"] = radius["column"]
    else:
        kwargs["crown_radius_equations"] = radius["equations"]

    # Profile assembly.
    kwargs["min_tree_height"] = source["min_tree_height"]
    kwargs["vertical_distribution"] = source["vertical_distribution"]
    kwargs["horizontal_distribution"] = source["horizontal_distribution"]
    kwargs["layer_depth"] = source["layer_depth"]

    # Per-band reduction methods — set only for requested bands.
    cbd = source.get("cbd")
    if cbd is not None:
        kwargs["cbd_method"] = cbd["method"]
        if cbd["method"] == "load_over_depth":
            kwargs["cbd_depth"] = cbd["depth"]
        else:
            kwargs["cbd_window"] = cbd["window"]
            kwargs["cbd_window_edge"] = _EDGE[cbd["edge"]]

    cbh = source.get("cbh")
    if cbh is not None:
        kwargs["cbh_method"] = cbh["method"]
        if cbh["method"] == "bulk_density_threshold":
            kwargs["cbh_threshold"] = cbh["threshold"]
            kwargs["cbh_relative_fraction"] = cbh["relative_threshold_fraction"]
            kwargs["cbh_smoothing_window"] = cbh["smoothing_window"]
            kwargs["cbh_smoothing_edge"] = _EDGE[cbh["smoothing_edge"]]

    chm = source.get("chm")
    if chm is not None:
        kwargs["chm_method"] = chm["method"]
        if chm["method"] == "bulk_density_threshold":
            kwargs["chm_threshold"] = chm["threshold"]
            kwargs["chm_relative_fraction"] = chm["relative_threshold_fraction"]
            kwargs["chm_smoothing_window"] = chm["smoothing_window"]
            kwargs["chm_smoothing_edge"] = _EDGE[chm["smoothing_edge"]]
        elif chm["method"] == "height_percentile":
            kwargs["chm_percentile"] = chm["percentile"]

    cc = source.get("cc")
    if cc is not None:
        kwargs["cover_method"] = cc["method"]
        if cc["method"] == "cover_fraction":
            kwargs["cover_height_threshold"] = cc["height_threshold"]

    return kwargs


def _crown_class_letters(df: pd.DataFrame) -> "pd.Series | str":
    """Each tree's crown class as a FuelCalc letter for core.

    Translates the inventory's FIA CCLCD codes onto core's D/C/I/S; a tree with
    no code — a null value, or a frame with no ``fia_crown_class_code`` column
    at all — takes ``_CROWN_CLASS_FILL`` (``"N"`` -> Other/none). Returns the
    scalar fill when the column is absent so the caller can assign it directly.
    """
    if CROWN_CLASS_COLUMN not in df.columns:
        return _CROWN_CLASS_FILL
    codes = pd.to_numeric(df[CROWN_CLASS_COLUMN], errors="coerce").astype("Int64")
    letters = codes.map(_CCLCD_TO_CROWN_CLASS)
    return letters.where(letters.notna(), _CROWN_CLASS_FILL)


def fetch_canopy_inventory(
    roi: gpd.GeoDataFrame,
    source: dict,
    alignment: dict,
    target_grid_doc: dict | None,
    progress: Callable[[str, int | None], None],
    extent_buffer_cells: int = 0,
) -> xr.Dataset:
    """Derive a 2D canopy fuel grid from a tree inventory.

    Resolves the output lattice, reads the inventory's live trees, translates
    the persisted ``source`` into the science kwargs, and fills a band Dataset
    via ``fastfuels_core.canopy_fuel.compute_canopy_metrics``. Non-forest cells
    are written as 0 to match the LANDFIRE canopy source.

    Raises:
        ProcessingError: For a lattice this handler cannot rasterize onto, an
            inventory with no usable live trees, or a tree-frame input the
            science rejects (e.g. a species it cannot price).
    """
    bands = source["bands"]
    inventory_id = source["source_inventory_id"]

    progress("Resolving output lattice...", 15)
    transform, shape = _resolve_lattice(
        roi, alignment, target_grid_doc, extent_buffer_cells
    )

    progress("Reading inventory...", 25)
    fuel_column, radius_column = _inventory_columns(source)
    # Read (and require non-null) only the morphology columns the selected
    # methods actually consume — the same set the API router validated against.
    # A request taking fuel and crown radius from columns needs neither dbh nor
    # fia_species_code, so an inventory lacking them must not fail or be silently
    # thinned here.
    required = list(canopy_required_columns(source))
    # The FuelCalc crown-class adjustment reads each tree's crown class; project
    # it from the inventory when that arm is selected (and the inventory has it).
    use_crown_class = source["crown_class_adjustment"]["method"] == "fuelcalc_table"
    df = read_inventory(
        inventory_id,
        fuel_column,
        radius_column,
        include_crown_class=use_crown_class,
        required_columns=required,
    )
    df = drop_null_rows(df, fuel_column, radius_column, required_columns=required)
    if df.empty:
        raise ProcessingError(
            code="EMPTY_INVENTORY",
            message="Inventory has no live trees with complete measurements.",
            suggestion=(
                "Verify the inventory contains live trees (fia_status_code == 1) "
                "with non-null dbh / height / crown_ratio."
            ),
        )

    dataset = _init_dataset(bands, transform, str(roi.crs), shape)
    kwargs = _core_kwargs(source)
    if kwargs.get("crown_class_column") == _CROWN_CLASS_COLUMN:
        df[_CROWN_CLASS_COLUMN] = _crown_class_letters(df)

    progress("Computing canopy metrics...", 45)
    try:
        compute_canopy_metrics(df, dataset, **kwargs)
    except ValueError as e:
        # The schema constrains every method string, so a ValueError here is a
        # tree-frame input the science rejects — most often a species it cannot
        # price. Terminal: the fix is the request or the inventory, not a retry.
        raise ProcessingError(
            code="CANOPY_FUEL_INPUT_ERROR",
            message=str(e),
            suggestion=(
                "Adjust the inventory or the request so every tree can be "
                "priced — e.g. exclude unpriceable species, supply per-tree "
                "fuel via an inventory column, or set species_inclusion to "
                "exclude hardwoods."
            ),
        ) from e

    # Core leaves NaN for cbh/chm where a cell has no canopy (cbd/cfl/cc come
    # back 0). LANDFIRE encodes non-forest as 0 across every band, so fill NaN
    # with 0 to keep this a drop-in substitute. Fill each band's array in place
    # to preserve float32 and the georeferencing rioxarray wrote.
    progress("Finalizing...", 70)
    for key in bands:
        np.nan_to_num(dataset[key].data, copy=False, nan=0.0)

    return dataset
