"""Map inventory FIA species codes onto the species DUET and duet-tools agree on.

A tree grid's `spcd` band carries any of FIA's ~2700 species codes. DUET knows
287 of them, and duet-tools can classify a subset of those as coniferous or
deciduous. Both drop what they don't recognize *silently*, so this module
exists to make every SPCD land somewhere accountable before the binary runs.

Two independent drops, both measured against the real binary (DUET v2.1-FF,
duet-tools 1.0.1):

1. **DUET drops codes absent from FIA_FastFuels_fin_fulllist_populated.txt.**
   They never reach `surface_species.dat` and deposit no litter — with
   `returncode == 0` and nothing on stderr. Their canopy silently becomes a
   grass field.

2. **duet-tools drops 13 codes that DUET itself knows** (0, 430, 476-479, 850,
   899, 980, 983-985, 1000) because its bundled REF_SPECIES.csv has no row for
   them. `_group_litter_species` does `lookup_dict.get(spcd)` -> None, which
   joins neither the coniferous nor the deciduous index list, so the litter DUET
   computed is dropped at import. Measured: a 3-species stand containing SPCD
   1000 lost 42.9% of its litter mass between the binary's output and
   `import_duet`'s.

`SPCD_TO_DUET` folds both, mapping every usable code to a canonical
representative. `unmappable()` names the rest so the handler can fail loudly
instead of returning a plausible all-grass grid.

## Why the representative is keyed on (signature, class), not signature alone

DUET's litter physics reads only the parameter columns of its species table, so
two species sharing a parameter signature are interchangeable *to DUET*. Its 287
species collapse to 11 such signatures, which is where the "nsp <= 11" ceiling
in the design notes came from.

That ceiling is wrong, because duet-tools splits litter by a different
authority: FIA's MAJOR_SPGRPCD. DUET's `wo` ("woodland") signature holds
junipers and cypress *and* oaks, maples, and mesquite — one signature, two
duet-tools classes. Collapsing it to a single representative would rewrite every
oak's litter as coniferous. A juniper-Gambel-oak woodland (SPCD 66 + 814, the
common interior-West fuel type) hits this exactly.

Keying on both authorities is inert with respect to each: same signature means
DUET computes the same litter, same class means duet-tools files it under the
same layer. The ceiling becomes 12 rather than 11 — `wo` splits in two, `pi`
already had two signatures — so correctness costs exactly one litter layer.
"""

from __future__ import annotations

import functools
from pathlib import Path

DUET_SPECIES_FILE = (
    Path(__file__).parent / "data" / "FIA_FastFuels_fin_fulllist_populated.txt"
)

# Columns of DUET's species table that its litter model reads. The rest — SPCD,
# genus, epithet, common name — are labels, so two rows agreeing here are
# interchangeable inputs to the binary.
_LABEL_COLUMNS = (0, 3, 4, 5)


def _load_duet_species() -> dict[int, tuple[str, ...]]:
    """Parse DUET's species table into {spcd: litter-parameter signature}.

    Read from the same file that is copied into DUET's working directory, so the
    signatures can never disagree with what the binary uses.

    Three SPCDs appear twice in the table (133 and 408 identically; 764 with
    conflicting `mh` and `wo` parameters). First row wins, matching Fortran's
    read-into-a-keyed-array behavior.
    """
    signatures: dict[int, tuple[str, ...]] = {}
    for line in DUET_SPECIES_FILE.read_text().splitlines():
        if not line.strip():
            continue
        fields = line.split("\t")
        spcd = int(fields[0])
        if spcd in signatures:
            continue
        signatures[spcd] = tuple(
            f for i, f in enumerate(fields) if i not in _LABEL_COLUMNS
        )
    return signatures


def _load_duet_tools_classes() -> dict[int, str]:
    """Return {spcd: "coniferous" | "deciduous"} exactly as duet-tools decides it.

    Delegates to duet_tools' own REF_SPECIES table and `_classify_spgrpcd` rather
    than restating the mapping, so a duet-tools upgrade that reclassifies a
    species moves our grouping with it. Codes it cannot classify are absent from
    the result — those are the 13 whose litter it would silently drop.
    """
    import pandas as pd
    from duet_tools.calibration import DATA_DIR, _classify_spgrpcd

    ref = pd.read_csv(DATA_DIR / "REF_SPECIES.csv")
    ref = ref.drop_duplicates(subset="SPCD")
    classes: dict[int, str] = {}
    for spcd, code in zip(ref["SPCD"], ref["MAJOR_SPGRPCD"]):
        group = _classify_spgrpcd(code)
        if group is not None:
            classes[int(spcd)] = group
    return classes


@functools.lru_cache(maxsize=1)
def _build_remap() -> dict[int, int]:
    """Build {spcd: representative spcd} over the codes both tools accept.

    A code maps to the lowest SPCD sharing its (signature, class) bucket —
    lowest only for determinism, since every member of a bucket is by
    construction interchangeable to both DUET and duet-tools.
    """
    signatures = _load_duet_species()
    classes = _load_duet_tools_classes()

    buckets: dict[tuple, list[int]] = {}
    for spcd, signature in signatures.items():
        species_class = classes.get(spcd)
        if species_class is None:
            # DUET would compute this species' litter and duet-tools would then
            # throw it away. Excluded here so `unmappable` reports it.
            continue
        buckets.setdefault((signature, species_class), []).append(spcd)

    return {spcd: min(members) for members in buckets.values() for spcd in members}


SPCD_TO_DUET: dict[int, int] = _build_remap()


def unmappable(spcds: set[int]) -> set[int]:
    """Return the codes DUET or duet-tools would silently drop."""
    return {int(s) for s in spcds if int(s) not in SPCD_TO_DUET}


def remap(spcds: set[int]) -> dict[int, int]:
    """Return {original spcd: representative spcd} for mappable codes only.

    Callers must have already rejected `unmappable(spcds)`.
    """
    return {int(s): SPCD_TO_DUET[int(s)] for s in spcds if int(s) in SPCD_TO_DUET}
