"""
api/v2/resources/inventories/tree/allometry/gdam/schema.py

Schema models for GDAM allometry imputation inventory creation.

GDAM fills the morphology columns (dbh, crown ratio, species, crown base
height) of an existing position+height tree inventory. With the default `gdam`
crown base height model GDAM predicts the crown base too; any other model
computes it from the final species, dbh, crown ratio and height. By default all
four are imputed, filling empty cells only; the caller can narrow that with
`impute_columns` and replace existing values with `overwrite`.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from api.resources.input_requirements import input_requirements, inventory
from api.resources.inventories.schema import (
    BASE_INVENTORY_COLUMNS,
    CreateInventoryRequestBase,
)
from api.resources.inventories.tree.allometry.schema import (
    GDAM_CROWN_BASE_HEIGHT_MODEL,
    CrownBaseHeightModel,
    GdamModel,
    default_crown_base_height_model,
)

# Columns GDAM needs on the source inventory to make a prediction.
GDAM_REQUIRED_SOURCE_COLUMNS = ("x", "y", "height")

# The morphology columns GDAM predicts and conditions on when present.
GDAM_COLUMNS = ("dbh", "crown_ratio", "fia_species_code")

# The morphology columns the endpoint can impute. The caller selects a subset.
# GDAM predicts the first three; crown_base_height comes from GDAM or a crown
# base height equation.
ImputableColumn = Literal["dbh", "crown_ratio", "fia_species_code", "crown_base_height"]
_DEFAULT_IMPUTE_COLUMNS: list[ImputableColumn] = [
    "dbh",
    "crown_ratio",
    "fia_species_code",
    "crown_base_height",
]

# Inputs of a crown base height equation: each must be imputed or on the source.
CROWN_BASE_HEIGHT_INPUTS = ("dbh", "crown_ratio", "fia_species_code")


def _validate_impute_columns(v: list[str]) -> list[str]:
    """Reject an empty or duplicated `impute_columns` list."""
    if not v:
        raise ValueError("impute_columns must contain at least one column.")
    if len(set(v)) != len(v):
        raise ValueError("impute_columns must not contain duplicate columns.")
    return v


class GdamAllometrySource(BaseModel):
    """Source metadata stored on the inventory document.

    Records which tree inventory GDAM filled in, its checksum at the time this
    inventory was created (so the source can be detected as stale later), which
    columns were imputed and how, and which columns GDAM conditioned on.
    """

    name: Literal["gdam"] = "gdam"
    source_tree_inventory_id: str
    source_tree_inventory_checksum: str | None = Field(
        default=None,
        description=(
            "The source tree inventory's `checksum` at the time this inventory "
            "was created from it. Compare it against the source inventory's "
            "current `checksum` to tell whether the source has changed since."
        ),
    )
    impute_columns: list[ImputableColumn] = Field(
        default_factory=lambda: list(_DEFAULT_IMPUTE_COLUMNS),
        description="The morphology columns imputed for this inventory.",
    )
    overwrite: bool = Field(
        default=False,
        description=(
            "Whether every tree's value in `impute_columns` was replaced (`true`) "
            "or only empty cells were filled (`false`)."
        ),
    )
    # Older documents lack the field; the backfill gave them the refit.
    crown_base_height_model: CrownBaseHeightModel = Field(
        default_factory=default_crown_base_height_model,
        description="The model that computed `crown_base_height`.",
    )
    conditioned_columns: list[str] | None = Field(
        default=None,
        description=(
            "The source columns GDAM conditioned its predictions on; empty when "
            "GDAM was not called. `null` for inventories created before this was "
            "recorded."
        ),
    )


class CreateGdamAllometryRequest(CreateInventoryRequestBase):
    """Request body for creating an inventory via GDAM allometry imputation."""

    model_config = ConfigDict(
        json_schema_extra=input_requirements(
            inventory("source_tree_inventory_id", columns=GDAM_REQUIRED_SOURCE_COLUMNS),
        ),
    )

    source_tree_inventory_id: str = Field(
        description=(
            "ID of a completed tree inventory whose morphology columns (dbh, "
            "crown ratio, species, crown base height) will be filled in."
        ),
    )
    impute_columns: list[ImputableColumn] = Field(
        default_factory=lambda: list(_DEFAULT_IMPUTE_COLUMNS),
        description=(
            "Which morphology columns to impute. Defaults to all of `dbh`, "
            "`crown_ratio`, `fia_species_code`, `crown_base_height`. Narrow it "
            "(e.g. `['fia_species_code']`) to impute fewer columns and write less "
            "to disk; columns left out are not imputed (they stay as the source "
            "had them). GDAM predicts `dbh`, `crown_ratio` and "
            "`fia_species_code`, and `crown_base_height` with the `gdam` "
            "`crown_base_height_model`. Any other model computes "
            "`crown_base_height` from those and `height`, so each of them must "
            "be imputed too or be a column of the source inventory, and with "
            "`overwrite`, replacing `crown_ratio` without `crown_base_height` "
            "leaves crown bases that no longer match the new crown ratios. Must "
            "contain at least one column, with no duplicates."
        ),
    )
    overwrite: bool = Field(
        default=False,
        description=(
            "`false` keeps existing values and fills only empty cells of "
            "`impute_columns`. `true` replaces every tree's value in "
            "`impute_columns`; the old values are not given to GDAM."
        ),
    )
    crown_base_height_model: CrownBaseHeightModel = Field(
        default_factory=GdamModel,
        description=(
            "The model that gives `crown_base_height`. `gdam` (the default) "
            "predicts it with the other columns; any other model computes it "
            "from each tree's species, dbh, crown ratio and height after GDAM "
            "fills them."
        ),
    )

    @field_validator("impute_columns")
    @classmethod
    def validate_impute_columns(cls, v: list[str]) -> list[str]:
        return _validate_impute_columns(v)


def resolve_gdam_columns(
    source_columns: list[dict], impute_columns: list[str]
) -> list[dict]:
    """Columns an imputed inventory will carry: the source's columns plus the
    columns actually imputed.

    Only `impute_columns` (a subset of `dbh` / `crown_ratio` /
    `fia_species_code` / `crown_base_height`) are filled, preserving existing
    values, so the stored `columns` must reflect exactly the source set plus
    those. Hardcoding the full base
    set instead over-claims columns the parquet never receives — any morphology
    column left un-imputed, and `fia_status_code`, which GDAM never writes.
    """
    imputable_defs = {c.key: c.model_dump() for c in BASE_INVENTORY_COLUMNS}
    resolved = [dict(c) for c in source_columns]
    have = {c["key"] for c in resolved}
    for key in impute_columns:
        if key not in have:
            resolved.append(imputable_defs[key])
            have.add(key)
    return resolved


def gdam_conditioned_columns(
    source_keys: set[str],
    impute_columns: list[str],
    overwrite: bool,
    crown_base_height_model: str,
) -> list[str]:
    """Columns on the source that GDAM receives as conditioning inputs.

    Every GDAM column the source carries, except those being overwritten, and
    `crown_base_height` under the same rule when the model is `gdam`. GDAM is
    called only when it fills an imputed column, so otherwise nothing is
    conditioned on.
    """
    gdam_crown_base = crown_base_height_model == GDAM_CROWN_BASE_HEIGHT_MODEL
    inputs = (*GDAM_COLUMNS, "crown_base_height") if gdam_crown_base else GDAM_COLUMNS
    if not any(key in impute_columns for key in inputs):
        return []
    return [
        key
        for key in inputs
        if key in source_keys and not (overwrite and key in impute_columns)
    ]


def crown_base_height_missing_inputs(
    source_keys: set[str], impute_columns: list[str], crown_base_height_model: str
) -> list[str]:
    """Crown base height equation inputs that are neither imputed nor on the
    source.

    Empty when `crown_base_height` is not requested or the model is `gdam`.
    Checks column presence only; trees with a missing input value get no crown
    base height.
    """
    if (
        "crown_base_height" not in impute_columns
        or crown_base_height_model == GDAM_CROWN_BASE_HEIGHT_MODEL
    ):
        return []
    return [
        key
        for key in CROWN_BASE_HEIGHT_INPUTS
        if key not in impute_columns and key not in source_keys
    ]
