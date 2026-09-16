"""Canonical unit-string handling. See docs/units.md.

Canonical form is whatever `pint.UnitRegistry()(s).units` formats with
the `~C` (short-compact) spec: ASCII, UDUNITS-2-conformant, with `**`
for exponents. Examples: `kg/m**3`, `1/m`, `m`, `cm`, `kg`, `%`.
"""

from __future__ import annotations

import pint

_ureg = pint.UnitRegistry()


def canonicalize_unit(s: str) -> str:
    """Return the canonical (~C) form of a unit string.

    Raises ValueError only if pint cannot parse `s` as a unit. A
    dimensionless input (a bare number, a ratio like kg/kg, or "") is a
    valid unit and canonicalizes to the empty string. The returned form is
    ASCII, UDUNITS-2-conformant, and uses `**` for exponents.
    """
    # `_ureg(s)` is a pure parse of untrusted input; pint raises a range of
    # types for malformed strings (UndefinedUnitError, DimensionalityError,
    # tokenizer/assertion errors for `(`, `kg/`, `kg**`, ...). Any failure
    # here means "not a valid unit", so normalize them all to ValueError.
    try:
        parsed = _ureg(s)
    except Exception as e:
        raise ValueError(f"Unit {s!r} is not a recognized unit: {e}") from e
    # A bare number parses to a plain int (no `.units`) on pint 0.25.2 and to
    # a dimensionless Quantity (formatting to "") on 0.25.3; both mean "no
    # physical unit" and canonicalize to "".
    units = getattr(parsed, "units", None)
    return f"{units:~C}" if units is not None else ""


def validate_unit(s: str | None) -> None:
    """Raise ValueError if `s` is not None and not already canonical.

    Used at write boundaries (schema validators, stamp_cf, upload handlers).
    Non-None inputs must be exactly the form pint's `~C` formatter would
    emit — this is enforced rather than auto-corrected.
    """
    if s is None:
        return
    canonical = canonicalize_unit(s)
    if canonical != s:
        raise ValueError(
            f"Unit {s!r} is not canonical; expected {canonical!r}. See docs/units.md."
        )
