"""Band display metadata published in the OpenAPI spec.

Each band enum carries ``x-enum-metadata`` built from its band-definition
table, and the ``Band`` schema carries ``x-band-quantities`` built from
``lib.band_quantities``, so clients can label band keys without copying them.
"""

from collections.abc import Mapping
from enum import StrEnum

from lib.band_quantities import BAND_QUANTITIES

X_ENUM_METADATA = "x-enum-metadata"
X_BAND_QUANTITIES = "x-band-quantities"

ENUM_METADATA_FIELDS = ("name", "description", "unit", "type")

BAND_METADATA_DOC = """
## Band metadata

Every band-key enum (e.g. `UniformBand`, `Fbfm40LookupBand`) publishes a label
for each of its values under `x-enum-metadata`:
`{"<value>": {"name", "description", "unit", "type"}}`. `unit` is left out for
bands without one. These are the same values the API writes to the resulting
grid's `bands[]`.

The `Band` schema publishes the band quantities under `x-band-quantities`:
`{"<quantity>": {"type", "unit"}}`. Any band key `<quantity>.<label>` has its
quantity's type and unit, so a client can label keys that no enum lists, such
as uploaded bands.
"""


def _strip(entry: Mapping) -> dict:
    return {k: str(entry[k]) for k in ENUM_METADATA_FIELDS if entry.get(k) is not None}


_BAND_DEFS: dict[type, Mapping] = {}


class BandEnum(StrEnum):
    """A band-key enum whose schema carries ``x-enum-metadata``.

    Pydantic reads this hook when a model using the enum is built, which can be
    before the definitions exist, so it looks them up at schema-generation time.
    """

    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema, handler):
        json_schema = handler(core_schema)
        defs = _BAND_DEFS[cls]
        json_schema[X_ENUM_METADATA] = {m.value: _strip(defs[m]) for m in cls}
        return json_schema


def publish_band_defs(enum: type[BandEnum], defs: Mapping) -> None:
    """Publish ``defs`` (enum member -> band definition) on ``enum``'s schema."""
    missing = [m.value for m in enum if m not in defs]
    if missing:
        raise ValueError(f"{enum.__name__} has no band definition for {missing}")
    _BAND_DEFS[enum] = defs


BAND_QUANTITIES_SCHEMA = {
    X_BAND_QUANTITIES: {
        name: {k: v for k, v in q._asdict().items() if v is not None}
        for name, q in BAND_QUANTITIES.items()
    }
}
"""``json_schema_extra`` for the ``Band`` model."""
