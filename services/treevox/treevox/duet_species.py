"""Resolve inventory FIA species codes onto species DUET and duet-tools both handle.

A tree grid's `spcd` band carries real FIA species codes — any of FIA's ~2,700.
DUET models a curated 287-code subset (its own `FIA_FastFuels…` table), and
duet-tools classifies litter against FIA's reference species table. 274 codes
sit in both; those are written straight through, one correctly-classified litter
layer per species.

A code outside that set is not dropped — it is resolved to the nearest surrogate
so its litter is still modeled:

  1. usable code            -> itself
  2. same genus             -> a usable species of that genus
  3. same softwood/hardwood -> a usable species of that class
  4. unresolvable           -> None (the caller rejects the job)

This is faithful because DUET's litter parameters are genus/group-level: every
species in a genus shares one litter signature, so a same-genus surrogate
deposits the litter DUET would have used anyway. Every usable genus resolves to
a single coniferous/deciduous class, so neither a genus nor a class surrogate
crosses the conifer/hardwood line.

Every *real* FIA species reaches at least tier 3 — MAJOR_SPGRPCD is always
softwood or hardwood — so real inventory data never rejects. Tier 4 is reserved
for codes that are not FIA species at all.

There is deliberately no species *collapse* here. Usable codes keep their own
identity; only unknowns are folded, and only onto a same-class surrogate, so the
wind-driven coniferous/deciduous litter split is never misattributed.
"""

from __future__ import annotations

import functools
from pathlib import Path

DUET_SPECIES_FILE = (
    Path(__file__).parent / "data" / "FIA_FastFuels_fin_fulllist_populated.txt"
)

# Genus is column 4 of DUET's tab-separated table
# (SPCD, group, group_id, epithet, genus, common_name, <litter params…>).
_GENUS_COLUMN = 4


def _load_duet_genus() -> dict[int, str]:
    """Return {spcd: genus} for every species in DUET's table.

    Read from the same file copied into DUET's working directory, so the genera
    can never disagree with what the binary models. Duplicate SPCDs (a handful
    appear twice) keep their first row.
    """
    genus: dict[int, str] = {}
    for line in DUET_SPECIES_FILE.read_text().splitlines():
        if not line.strip():
            continue
        fields = line.split("\t")
        genus.setdefault(int(fields[0]), fields[_GENUS_COLUMN])
    return genus


def _load_duet_tools_classes() -> dict[int, str]:
    """Return {spcd: "coniferous" | "deciduous"} exactly as duet-tools decides it.

    Delegates to duet_tools' own REF_SPECIES table and `_classify_spgrpcd`, so a
    duet-tools upgrade that reclassifies a species moves our grouping with it.
    Codes it cannot classify are absent from the result.
    """
    import pandas as pd
    from duet_tools.calibration import DATA_DIR, _classify_spgrpcd

    ref = pd.read_csv(DATA_DIR / "REF_SPECIES.csv").drop_duplicates(subset="SPCD")
    classes: dict[int, str] = {}
    for spcd, code in zip(ref["SPCD"], ref["MAJOR_SPGRPCD"]):
        group = _classify_spgrpcd(code)
        if group is not None:
            classes[int(spcd)] = group
    return classes


def _load_fia_reference() -> tuple[dict[int, str], dict[int, str]]:
    """Return ({spcd: genus}, {spcd: class}) for every FIA species.

    Sourced from the same REF_SPECIES table duet-tools classifies against, so an
    unknown code's genus and coniferous/deciduous class are read from the
    authority duet-tools will itself use downstream. Every real FIA species has
    both, which is what makes tier 3 total.
    """
    import pandas as pd
    from duet_tools.calibration import DATA_DIR, _classify_spgrpcd

    ref = pd.read_csv(DATA_DIR / "REF_SPECIES.csv").drop_duplicates(subset="SPCD")
    genus = {int(s): g for s, g in zip(ref["SPCD"], ref["GENUS"])}
    classes = {
        int(s): _classify_spgrpcd(code)
        for s, code in zip(ref["SPCD"], ref["MAJOR_SPGRPCD"])
    }
    return genus, classes


@functools.lru_cache(maxsize=1)
def _tables() -> dict:
    """Build the resolution tables once.

    - USABLE: codes DUET models and duet-tools classifies (written through).
    - GENUS_REP / CLASS_REP: the lowest usable SPCD of each genus / class,
      lowest only for determinism since every member is interchangeable.
    """
    duet_genus = _load_duet_genus()
    duet_classes = _load_duet_tools_classes()
    usable = {s for s in duet_genus if s in duet_classes}

    genus_rep: dict[str, int] = {}
    for spcd in usable:
        g = duet_genus[spcd]
        genus_rep[g] = min(spcd, genus_rep.get(g, spcd))

    class_rep: dict[str, int] = {}
    for spcd in usable:
        c = duet_classes[spcd]
        class_rep[c] = min(spcd, class_rep.get(c, spcd))

    fia_genus, fia_class = _load_fia_reference()
    return {
        "usable": usable,
        "genus_rep": genus_rep,
        "class_rep": class_rep,
        "fia_genus": fia_genus,
        "fia_class": fia_class,
    }


def resolve(spcd: int) -> int | None:
    """Resolve one FIA species code to a species DUET and duet-tools both handle.

    Returns the code to write for `spcd` — itself when usable, otherwise a
    same-genus or same-class surrogate — or None when the code is not a FIA
    species that can be placed at all.
    """
    spcd = int(spcd)
    t = _tables()
    if spcd in t["usable"]:
        return spcd
    genus = t["fia_genus"].get(spcd)
    if genus in t["genus_rep"]:
        return t["genus_rep"][genus]
    species_class = t["fia_class"].get(spcd)
    if species_class in t["class_rep"]:
        return t["class_rep"][species_class]
    return None


def resolve_codes(codes: set[int]) -> tuple[dict[int, int], set[int]]:
    """Resolve a set of codes at once.

    Returns ``(mapping, unresolved)`` where ``mapping`` is
    ``{original: surrogate}`` for every code that resolved (including identities)
    and ``unresolved`` is the set that could not be placed. The caller writes the
    mapping onto the spcd array and rejects if ``unresolved`` is non-empty.
    """
    mapping: dict[int, int] = {}
    unresolved: set[int] = set()
    for code in codes:
        surrogate = resolve(code)
        if surrogate is None:
            unresolved.add(int(code))
        else:
            mapping[int(code)] = surrogate
    return mapping, unresolved
