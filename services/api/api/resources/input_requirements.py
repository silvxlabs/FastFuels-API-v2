"""Input requirements published on request fields as OpenAPI ``x-`` extensions.

A field naming an input grid or inventory declares what that input must hold.
Clients read the declaration to offer only eligible inputs; the router checks
the same declaration (``api.resources.grids.utils.validate_grid_input``).
"""

from collections.abc import Iterable

from pydantic import BaseModel

INPUT_REQUIREMENTS_DOC = """
## Input requirements

A request field that names an input grid or inventory publishes what the input
must hold:

| Extension | On | Meaning |
|---|---|---|
| `x-required-bands` | grid ID | Band keys the grid must have. |
| `x-dims` | grid ID | `2` (y, x) or `3` (z, y, x). |
| `x-source` | grid ID | The `source.name` the grid must come from, e.g. `pim`. |
| `x-same-lattice-as` | grid ID | Another grid ID field whose cells (CRS, shape, transform) this grid must share. |
| `x-band-of` | band key | `{"grid": f, "quantity": q}`: a band of grid field `f` whose key has quantity `q`. |
| `x-required-columns` | inventory ID | Columns the inventory must have, at least. Methods chosen in the request can need more. |

Band keys are `<quantity>` or `<quantity>.<label>`; see `BandKey`.

These cover the common reasons an input is rejected. A request can still fail
other checks, which the operation's description lists.
"""

X_REQUIRED_BANDS = "x-required-bands"
X_DIMS = "x-dims"
X_SOURCE = "x-source"
X_SAME_LATTICE_AS = "x-same-lattice-as"
X_BAND_OF = "x-band-of"
X_REQUIRED_COLUMNS = "x-required-columns"


def grid_input(
    *,
    bands: Iterable[str] = (),
    dims: int | None = None,
    source: str | None = None,
    same_lattice_as: str | None = None,
) -> dict:
    """``json_schema_extra`` for a grid ID field."""
    extra: dict = {}
    if bands:
        extra[X_REQUIRED_BANDS] = list(bands)
    if dims is not None:
        extra[X_DIMS] = dims
    if source is not None:
        extra[X_SOURCE] = source
    if same_lattice_as is not None:
        extra[X_SAME_LATTICE_AS] = same_lattice_as
    return extra


def band_of(grid_field: str, quantity: str) -> dict:
    """``json_schema_extra`` for a field choosing a band of another input grid."""
    return {X_BAND_OF: {"grid": grid_field, "quantity": quantity}}


def inventory_input(columns: Iterable[str]) -> dict:
    """``json_schema_extra`` for an inventory ID field."""
    return {X_REQUIRED_COLUMNS: sorted(columns)}


def declared(model: type[BaseModel], field: str) -> dict:
    """The requirements ``field`` of ``model`` declares (empty if none)."""
    extra = model.model_fields[field].json_schema_extra
    return extra if isinstance(extra, dict) else {}
