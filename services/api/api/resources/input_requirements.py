"""Input requirements published on request schemas as ``x-input-requirements``.

A request that names input grids, inventories or features lists what each
input must hold. Clients read the list to offer only inputs that meet it; the
router checks the same list (``api.resources.grids.utils.validate_grid_input``).
"""

from collections.abc import Iterable

from pydantic import BaseModel

X_INPUT_REQUIREMENTS = "x-input-requirements"

INPUT_REQUIREMENTS_DOC = """
## Input requirements

A request that names input resources publishes what each input must hold, as
a list under `x-input-requirements` on the request schema. Each entry names a
`field` (a dotted path for a nested field) and the requirements its input must
meet. A field can have several entries; the input must meet all of them.

| Key | Meaning |
|---|---|
| `field` | The request field the entry applies to, e.g. `source_grid_id` or `crown_geometry_source.feature_id`. |
| `resource` | `grid`, `inventory`, `feature`, or `band` (a band key of another input grid). |
| `bands` | Band keys the grid must have. |
| `dims` | `2` (y, x) or `3` (z, y, x). |
| `source` | The `source.name` the grid must come from, e.g. `pim`. |
| `same_lattice_as` | Another grid field whose horizontal lattice (CRS, y/x shape, transform) this grid must share. A 3D grid may share a 2D grid's lattice. |
| `of`, `quantity` | For `band`: a band of grid field `of` whose key has quantity `quantity`. |
| `columns` | Columns the inventory must have, with a value in every row (`columns[].summary.null_count == 0`). |
| `type` | The feature's `type`, e.g. `crown`. |
| `when` | Applies only when the request has these values: `{"<dotted path>": <value>}`. |

Band keys are `<quantity>` or `<quantity>.<label>`; the GeoTIFF upload `bands[].key` description lists the quantities.

Every input must also be completed and in the same domain. A request can still
fail other checks, which the operation's description lists.
"""


def _entry(field: str, resource: str, when: dict | None, **rules) -> dict:
    entry = {"field": field, "resource": resource}
    entry.update({k: v for k, v in rules.items() if v is not None})
    if when:
        entry["when"] = when
    return entry


def grid(
    field: str,
    *,
    bands: Iterable[str] = (),
    dims: int | None = None,
    source: str | None = None,
    same_lattice_as: str | None = None,
    when: dict | None = None,
) -> dict:
    """An input grid's requirements."""
    return _entry(
        field,
        "grid",
        when,
        bands=list(bands) or None,
        dims=dims,
        source=source,
        same_lattice_as=same_lattice_as,
    )


def band(field: str, *, of: str, quantity: str) -> dict:
    """A band of input grid ``of`` whose key has ``quantity``."""
    return _entry(field, "band", None, of=of, quantity=quantity)


def inventory(field: str, *, columns: Iterable[str], when: dict | None = None) -> dict:
    """An input inventory's required columns."""
    return _entry(field, "inventory", when, columns=sorted(columns))


def feature(field: str, *, type: str, when: dict | None = None) -> dict:
    """An input feature's type."""
    return _entry(field, "feature", when, type=type)


def input_requirements(*entries: dict) -> dict:
    """``json_schema_extra`` for a request model."""
    return {X_INPUT_REQUIREMENTS: list(entries)}


def _value_at(body: BaseModel, path: str):
    value = body
    for part in path.split("."):
        value = getattr(value, part, None)
    return value


def requirements_for(body: BaseModel, field: str) -> list[dict]:
    """The entries for ``field`` that apply to ``body``."""
    extra = type(body).model_config.get("json_schema_extra")
    entries = extra.get(X_INPUT_REQUIREMENTS, []) if isinstance(extra, dict) else []
    return [
        e
        for e in entries
        if e["field"] == field
        and all(_value_at(body, p) == v for p, v in e.get("when", {}).items())
    ]
