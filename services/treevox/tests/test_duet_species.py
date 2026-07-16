"""Unit tests for treevox.duet_species — the FIA species resolver.

The resolver exists to give every species DUET can't model directly a faithful
surrogate rather than dropping its litter (see the module docstring). These
tests assert the properties that make the surrogate faithful — a surrogate never
crosses the conifer/hardwood line, real FIA species always resolve, usable codes
keep their own identity — rather than snapshotting a table.

They run without DUET installed; the measured proof that resolution is faithful
against the real binary lives in the integration test.
"""

from __future__ import annotations

from treevox import duet_species


class TestUsableCodesPassThrough:
    def test_a_common_species_resolves_to_itself(self):
        # Ponderosa pine is in DUET's table and duet-tools can classify it.
        assert duet_species.resolve(122) == 122

    def test_usable_codes_are_never_collapsed(self):
        # Two California live oaks that DUET models directly keep separate
        # identities — no collapse, so nothing can misattribute their litter.
        mapping, unresolved = duet_species.resolve_codes({801, 805})
        assert unresolved == set()
        assert mapping == {801: 801, 805: 805}

    def test_juniper_and_oak_stay_distinct(self):
        # Both are DUET `wo`. Because usable codes pass through untouched, the
        # signature-collapse misattribution that this used to guard against
        # cannot arise at all.
        mapping, _ = duet_species.resolve_codes({66, 814})
        assert mapping[66] != mapping[814]
        assert mapping == {66: 66, 814: 814}


class TestGenusFallback:
    def test_unlisted_species_resolves_to_a_same_genus_surrogate(self):
        # Four-leaf pinyon isn't in DUET's table; its genus (Pinus) is, so it
        # borrows a Pinus surrogate.
        surrogate = duet_species.resolve(138)
        assert surrogate is not None
        genus = duet_species._load_duet_genus()
        assert genus[surrogate] == "Pinus"

    def test_genus_surrogate_keeps_the_species_class(self):
        genus = duet_species._load_duet_genus()
        classes = duet_species._load_duet_tools_classes()
        # An unlisted conifer (Chamaecyparis white-cedar) and an unlisted
        # hardwood (Acacia) each borrow a same-genus, same-class surrogate.
        conifer = duet_species.resolve(40)
        hardwood = duet_species.resolve(303)
        assert genus[conifer] == "Chamaecyparis" and classes[conifer] == "coniferous"
        assert genus[hardwood] == "Acacia" and classes[hardwood] == "deciduous"


class TestClassFallback:
    def test_species_with_no_listed_genus_falls_back_to_its_class(self):
        # "Unknown dead hardwood" (998) is a real FIA code with a hardwood
        # major group but no genus DUET lists, so it lands on the hardwood rep.
        surrogate = duet_species.resolve(998)
        classes = duet_species._load_duet_tools_classes()
        assert classes[surrogate] == "deciduous"


class TestEveryRealSpeciesResolves:
    def test_no_real_fia_species_is_rejected(self):
        # The whole point: real inventory data never fails the job. Every FIA
        # species has a softwood/hardwood group, so tier 3 always catches.
        fia_genus, _ = duet_species._load_fia_reference()
        rejected = [s for s in fia_genus if duet_species.resolve(s) is None]
        assert rejected == []


class TestUnresolvable:
    def test_non_species_codes_are_reported_not_placed(self):
        # 1000 ("hardwoods general") and garbage are not FIA species.
        mapping, unresolved = duet_species.resolve_codes({122, 1000, 999999})
        assert mapping == {122: 122}
        assert unresolved == {1000, 999999}


class TestSurrogatesAreThemselvesUsable:
    def test_every_surrogate_is_a_usable_code(self):
        # A surrogate that DUET or duet-tools couldn't handle would defeat the
        # exercise. Check across a broad sweep of real FIA species.
        fia_genus, _ = duet_species._load_fia_reference()
        usable = duet_species._tables()["usable"]
        for spcd in list(fia_genus)[::7]:
            surrogate = duet_species.resolve(spcd)
            assert surrogate in usable
