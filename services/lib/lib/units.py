"""Canonical unit-string handling. See docs/units.md.

Canonical form is whatever `pint.UnitRegistry()(s).units` formats with
the `~C` (short-compact) spec: ASCII, UDUNITS-2-conformant, with `**`
for exponents. Examples: `kg/m**3`, `1/m`, `m`, `cm`, `kg`, `%`.
"""

from __future__ import annotations

import pint

_ureg = pint.UnitRegistry()


def canonicalize_unit(s: str) -> str:
    """Return the canonical form of a unit string.

    Raises ValueError if `s` is not a recognized unit. The returned form
    is ASCII, UDUNITS-2-conformant, and uses `**` for exponents.
    """
    try:
        return f"{_ureg(s).units:~C}"
    except (pint.UndefinedUnitError, pint.DimensionalityError) as e:
        raise ValueError(f"Unit {s!r} is not a recognized unit: {e}") from e


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
